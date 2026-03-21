import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional

import boto3
import yaml
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw, ImageFont

from src.utils import CourtKeypoints, ImagePoint, VideoData


# ---------------------------
# ffmpeg / ffprobe utilities
# ---------------------------

def _probe_duration(url: str, timeout: int = 45) -> float:
    """Return video duration (seconds) using ffprobe on a URL."""
    proc = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", url],
        capture_output=True, text=True, check=True, timeout=timeout,
    )
    return float(json.loads(proc.stdout)["format"]["duration"])


def _extract_last_frame_from_url(url: str, out_path: Path, seek_from_end_sec: int = 3, timeout: int = 45) -> None:
    """Extract a frame near the end of a URL-accessible video and save to out_path."""
    # Use duration so this works for presigned HTTP URLs (no -sseof support there)
    duration = _probe_duration(url, timeout=timeout)
    seek_time = max(0, duration - seek_from_end_sec)
    subprocess.run(
        ["ffmpeg", "-ss", str(seek_time), "-i", url, "-update", "1", "-frames:v", "1", "-q:v", "2", str(out_path)],
        capture_output=True, check=True, timeout=timeout,
    )


def _extract_last_frame_from_file(file_path: Path, out_path: Path, timeout: int = 60) -> None:
    """Extract the last frame from a local file (uses -sseof for better accuracy)."""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-sseof", "-1", "-i", str(file_path), "-frames:v", "1", "-q:v", "2", "-y", str(out_path)],
        capture_output=True, check=True, timeout=timeout,
    )


# ---------------------------
# S3 adapter
# ---------------------------

class S3Adapter:
    """Small helper around boto3 S3 with presign + convenience ops."""
    def __init__(self):
        self.s3 = boto3.client("s3")

    @staticmethod
    def _parse_s3_uri(s3_uri: str) -> tuple:
        assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
        bucket_key = s3_uri[len("s3://"):]
        return tuple(bucket_key.split("/", 1))

    def object_exists(self, s3_uri: str) -> bool:
        bucket, key = self._parse_s3_uri(s3_uri)
        try:
            self.s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("404", "NotFound"):
                return False
            raise

    def presign_url(self, s3_uri: str, expires_seconds: int = 3600) -> str:
        bucket, key = self._parse_s3_uri(s3_uri)
        return self.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=expires_seconds,
        )

    def download_file_to(self, s3_uri: str, local_path: Path) -> None:
        bucket, key = self._parse_s3_uri(s3_uri)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.s3.download_file(bucket, key, str(local_path))

    # ---- Frame extraction paths (URL first, fallback to full download) ----

    def try_extract_last_frame_via_presign(self, s3_uri: str, out_path: Path) -> bool:
        """Attempt to extract last frame by presigning and streaming; return True on success."""
        try:
            url = self.presign_url(s3_uri)
            _extract_last_frame_from_url(url, out_path)
            return True
        except Exception:
            return False

    def extract_last_frame_via_download(self, s3_uri: str, video_id_hint: Optional[str], out_path: Path, work_dir: Optional[Path] = None) -> None:
        """Download whole file to temp, extract last frame, then delete the file."""
        tmp_root = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
        tmp_root.mkdir(parents=True, exist_ok=True)
        local_video = tmp_root / f"{(video_id_hint or 'video')}.mp4"

        try:
            self.download_file_to(s3_uri, local_video)
            _extract_last_frame_from_file(local_video, out_path)
        finally:
            try:
                if local_video.exists():
                    local_video.unlink()
            except Exception:
                pass


# --------------------------------
# Public helpers (stable behavior)
# --------------------------------

def get_court_keypoints(s3_adapter: S3Adapter, video_data: VideoData) -> CourtKeypoints:
    """
    Download YAML keypoints and return CourtKeypoints.
    S3 path pattern is unchanged.
    """
    s3_uri = (
        "s3://clutchvideostorageios162404-production/"
        f"protected/{video_data.creator_identity_id}/court-keypoints/{video_data.video_id}-court-keypoints.yaml"
    )
    local_yaml = Path(tempfile.gettempdir()) / f"{video_data.video_id}-court-keypoints.yaml"
    s3_adapter.download_file_to(s3_uri, local_yaml)

    with open(local_yaml, "r") as fh:
        data = yaml.safe_load(fh) or {}

    # Normalize list->tuple for dataclass compatibility
    as_tuple = {k: tuple(v) for k, v in data.items()}
    return CourtKeypoints(
        tol=as_tuple["tol"],
        tor=as_tuple["tor"],
        point_7=as_tuple["point_7"],
        point_9=as_tuple["point_9"],
    )


def get_last_video_image_save(s3_adapter: S3Adapter, video_data: VideoData, local_image_path: Path) -> None:
    """
    Extract the last frame to local_image_path (.jpg). Logic preserved:
    1) Try presigned stream on regular video
    2) Try presigned stream on *_stream.mp4
    3) Fallback: download whole file (regular, then streamed) and extract last frame
    """
    assert str(local_image_path).endswith(".jpg"), "Filename must end with .jpg"

    base = "s3://clutchvideostorageios162404-production/public"
    regular_uri = f"{base}/{video_data.creator_identity_id}/{video_data.video_id}.mp4"
    streamed_uri = f"{base}/{video_data.creator_identity_id}/{video_data.video_id}_stream.mp4"

    print(f"{video_data.video_id}: downloading (presign regular)")
    regular_exists = s3_adapter.object_exists(regular_uri)
    if regular_exists and s3_adapter.try_extract_last_frame_via_presign(regular_uri, local_image_path):
        return

    print(f"{video_data.video_id}: try presign streamed")
    streamed_exists = s3_adapter.object_exists(streamed_uri)
    if streamed_exists and s3_adapter.try_extract_last_frame_via_presign(streamed_uri, local_image_path):
        return

    print(f"{video_data.video_id}: try full download fallback")
    if regular_exists:
        s3_adapter.extract_last_frame_via_download(regular_uri, video_data.video_id, local_image_path)
        return
    if streamed_exists:
        s3_adapter.extract_last_frame_via_download(streamed_uri, video_data.video_id, local_image_path)
        return

    raise ValueError(f"Could not get last video image for video {video_data.video_id}")


def add_points_to_image(
    image_path: Path,
    points: Dict[str, ImagePoint],  # {label: (x, y)}
    r: int = 6,
    new_image_path: Optional[str] = None,
) -> None:
    """Draws labeled points onto image. Overwrites original if new_image_path is None."""
    im = Image.open(image_path).convert("RGB")
    W, H = im.size
    draw = ImageDraw.Draw(im)

    # Font: scale with image; fall back to default if system font missing
    try:
        font = ImageFont.truetype("arial.ttf", max(12, min(32, max(W, H) // 60)))
    except Exception:
        font = ImageFont.load_default()

    for label, (x, y) in points.items():
        x, y = int(round(x)), int(round(y))
        x = max(0, min(W - 1, x))
        y = max(0, min(H - 1, y))

        # Dot (white ring + red fill)
        draw.ellipse((x - r - 2, y - r - 2, x + r + 2, y + r + 2), outline="white", width=2)
        draw.ellipse((x - r, y - r, x + r, y + r), fill="red")

        # Label placement (try top-right; flip if out-of-bounds)
        text = str(label)
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        tx, ty = x + r + 4, y - r - 4 - th
        if tx + tw > W:
            tx = x - r - 4 - tw
        if ty < 0:
            ty = y + r + 4

        # Light drop shadow + text
        draw.text((tx + 1, ty + 1), text, font=font, fill="black")
        draw.text((tx, ty), text, font=font, fill="white")

    out_path = Path(new_image_path) if new_image_path else image_path
    im.save(out_path, quality=95)
