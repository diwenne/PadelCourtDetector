"""
Detect if court keypoints (the 'T') have moved.

We store the location of the court's 'T' and alert if it changes or is invalid.
"""

import argparse
import datetime as dt
import math
import os
import random
import signal
import sys
import traceback
from pathlib import Path
from typing import Optional, Tuple

import concurrent.futures
import multiprocessing as mp

from src.compute_t_intersection import find_t_intersection
from src.emails import EmailAdapter
from src.image_extracter import S3Adapter, add_points_to_image, get_last_video_image_save
from src.slack_adapter import SlackBot
from src.supabase_adapter import SupabaseAdapter
from src.utils import DeviceInfo, ImagePoint, VideoData, near_line
from PIL import Image


# -----------------------
# Args / constants / util
# -----------------------

def get_command_line_arg() -> int:
    """--interval_seconds: how far back we look for fresh videos and schedule watchdogs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval_seconds", type=int, default=24 * 60 * 60)
    return parser.parse_args().interval_seconds


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def cutoff_from_interval(seconds: int) -> dt.datetime:
    return now_utc() - dt.timedelta(seconds=int(seconds))


def almost_equal(
    x: ImagePoint,
    y: ImagePoint,
    width: int = 1920,
    height: int = 1080,
    threshold_of_diagonal: float = 0.007,
) -> bool:
    """Within X% of the screen diagonal."""
    dx, dy = x[0] - y[0], x[1] - y[1]
    distance = math.hypot(dx, dy)
    threshold = threshold_of_diagonal * math.hypot(width, height)
    return distance < threshold


# -----------------------
# Slack watchdog utilities
# -----------------------

def update_slack_alert_dormant_message(slack_bot: SlackBot, interval_seconds: int, script_failed_msg: str) -> None:
    """
    We schedule a future Slack message (2x interval) and cancel it each run.
    If it ever posts, the script likely stalled.
    """
    dormant_msg = ":red_circle: The script that checks if the cameras have moved has NOT run in a while. Check on it!"
    slack_bot.cancel_slack_msg_with_txt(dormant_msg)
    slack_bot.schedule_message(dormant_msg, ((2 * interval_seconds) // 60) + 1)

    # Fire a short-fuse message in case the current run crashes (we cancel on success)
    slack_bot.schedule_message(script_failed_msg, 120)


# -----------------------
# Orchestration
# -----------------------

def main(slack_bot: SlackBot, is_test: bool) -> None:
    script_failed_msg = ":red_circle: The court keypoint checker script failed to complete. Check on it please"
    interval_seconds = get_command_line_arg()

    if not is_test:
        update_slack_alert_dormant_message(slack_bot, interval_seconds, script_failed_msg)

    device_infos = SupabaseAdapter(is_test).get_device_info()
    if not device_infos:
        return

    print([dev.camera_name for dev in device_infos])

    max_workers = min(len(device_infos), os.cpu_count() or 8)
    ctx = mp.get_context("spawn")

    # You can get these from the slack api 
    slack_user_ids = {
        "kari": "U01KLFBTKLL",
        "alexk": "U04Q9J4QE6R",
        "kirill": "U08FJDPEM33",
        "brennan": "U08UXES7BRU",
        "paul": "U09C55UKSGG",
}
    people_to_alert = [f"<@{user_id}>" for user_id in slack_user_ids.values()]
    person_to_alert = random.choice(people_to_alert)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as pool:
        futures = [pool.submit(run_worker, is_test, interval_seconds, di, person_to_alert) for di in device_infos]
        for f in concurrent.futures.as_completed(futures):
            _ = f.result()  # surface exceptions

    if not is_test:
        slack_bot.cancel_slack_msg_with_txt(script_failed_msg)


def run_worker(is_test: bool, interval_seconds: int, device_info: DeviceInfo, person_to_alert: str) -> None:
    CameraKeypointsMonitor(is_test, interval_seconds, person_to_alert).thread_worker(device_info)


# -----------------------
# Core monitor class
# -----------------------

class CameraKeypointsMonitor:
    def __init__(self, is_test: bool, interval_seconds: int, person_to_alert: str):
        self.is_test = is_test
        self.interval_seconds = interval_seconds

        self.slack_bot = SlackBot(is_test)
        self.supabase = SupabaseAdapter(is_test)
        self.s3 = S3Adapter()
        self.email = EmailAdapter(self.slack_bot, is_test)
        self.person_to_alert = person_to_alert
        self._camera_mgmt_base_url = "https://clutch-core.fly.dev/clutch-god?tab=camera-management"

    # ---- public entry ----
    def thread_worker(self, device_info: DeviceInfo) -> None:
        if not self._court_keypoints_correct_order(device_info):
            self._handle_court_keypoint_incorrect_order(device_info)
            return

        t_int, img_path = self._get_new_t_intersection(device_info)
        
        # This means we dont have a frame to analyze 
        if img_path is None:
            return 

        if t_int is None:
            self._handle_no_new_t(device_info, img_path)
        elif self._is_serve_line_too_low(t_int, img_path):
            self._handle_serve_line_too_low(device_info, img_path, t_int)
        elif not self._is_t_on_court_line(device_info, t_int):
            self._handle_t_not_on_court_line(device_info, t_int, img_path)
        elif device_info.t_intersection and not almost_equal(t_int, device_info.t_intersection):
            self._handle_camera_moved(device_info, img_path, t_int)
        else:
            # Write new t_intersection (if device_info.t_intersection null) otherwise, 
            # this does not update the t_int (keeps it to device_info.t_intersection)
            # and just updates the timestamp in the database
            self._write_t(device_info, device_info.t_intersection or t_int)
        img_path.unlink(True)

    # ---- shared helpers ----
    def _is_serve_line_too_low(self, t_int: ImagePoint, img_path: Path) -> bool:
        """Checks if the serve line is too close to the bottom of the screen"""
        # What percetnage of the height of the image is the t_int allowed to take up? 
        threshold = 0.91
        with Image.open(img_path) as img:
            _, height = img.size
        # Ratio of t_int y to the image height. I.e., what percentage of screen does it go down
        ratio = t_int[1]/height
        return ratio > threshold

    def _handle_serve_line_too_low(self, device_info: DeviceInfo, img_path: Path, t_intersection: ImagePoint):
        msg = f"*{device_info.camera_name}*: Serve line may be too close to bottom of screen"
        self._annotate_and_notify(device_info, img_path, msg, t_intersection)


    def _is_t_on_court_line(self, device_info: DeviceInfo, t_intersection: ImagePoint) -> bool:
        kp = device_info.court_keypoints
        return near_line(t_intersection, kp.point_7, kp.point_9)

    def _handle_t_not_on_court_line(self, device_info: DeviceInfo, t_intersection: ImagePoint, img_path: Path) -> None:
        msg = f"*{device_info.camera_name}*: T-intersection not between court bottom line"
        msg_id = self._annotate_and_notify(device_info, img_path, msg, t_intersection)
        if msg_id:
            self.email.send_camera_moved_email(device_info, msg_id)

    def _handle_camera_moved(self, device_info: DeviceInfo, img_path: Path, new_t: ImagePoint) -> None:
        msg = (
            f"*{device_info.camera_name}*: CAMERA MOVED (T-intersection updated automatically :white_check_mark:). "
            f"Please update the court keypoints manually and you should be good."
        )
        msg_id = self._annotate_and_notify(device_info, img_path, msg, new_t)
        if msg_id:
            self.email.send_camera_moved_email(device_info, msg_id)

    def _handle_no_new_t(self, device_info: DeviceInfo, img_path: Path):
        # TODO: eventually we want to alert that we couldn't compute the new_t, but 
        # currently, there are a lot of edge cases that prevent us from computing it
        # so no need to notify every time 
        if self.is_test:
            msg = f"*{device_info.camera_name}*: Could not compute new T-intersection :red_circle:"
            self._annotate_and_notify(device_info, img_path, msg, device_info.t_intersection)

    def _write_t(self, device_info: DeviceInfo, t: ImagePoint, did_alert_today: bool = False) -> None:
        self.supabase.write_t_intersection(device_info.camera_id, t[0], t[1], did_alert_today)

    def _download_last_frame(self, video: VideoData) -> Path:
        out = Path(__file__).resolve().parent / f"tmp_image_{video.video_id}.jpg"
        get_last_video_image_save(self.s3, video, out)
        print(f"{video.video_id}: successfully downloaded")
        return out

    def _get_new_t_intersection(self, device_info: DeviceInfo)-> Tuple[Optional[ImagePoint], Optional[Path]]:
        """Returns new t-intersection (if we can find one) and the image of the frame we analyzed.
        
        If no frame returned, that means there's no frame to analyze for this device
        """
        cutoff = None if device_info.t_intersection is None else cutoff_from_interval(self.interval_seconds)
        video = self.supabase.get_latest_video(device_info.camera_id, cutoff)
        print(f"{device_info.camera_name} video data: ", video)
        
        if not video:
            return None, None

        return self._get_t_intersection_from_video(
            video,
            estimated_t=device_info.bottom_line_center,
            preferred_t_estimate=device_info.t_intersection,
        )

    def _get_t_intersection_from_video(
        self,
        video: VideoData,
        estimated_t: ImagePoint,
        preferred_t_estimate: Optional[ImagePoint] = None,
    ) -> Tuple[Optional[ImagePoint], Path]:
        """
        Try to compute T near an estimate; preferes preferred_t_estimate, but defaults to estimated_t
        Returns (t_intersection or None, frame_path).
        """
        last_frame = self._download_last_frame(video)
        t_int = None 
        if preferred_t_estimate:
            t_int = find_t_intersection(last_frame, preferred_t_estimate) 
        if t_int is None:   
            t_int = find_t_intersection(last_frame, estimated_t)
        return t_int, last_frame

    def _annotate_and_notify(self, device_info: DeviceInfo, img_path: Path, msg: str, new_t: Optional[ImagePoint] = None) -> Optional[str]:
        """Annotate image with keypoints and send to Slack with a threaded message.
        
        extra_points: if you want to add additional points to the image

        returns main slack thread id 
        """
        did_alert_today = False
        msg_id = None
        # Only alert on a device every once in a while 
        if device_info.last_alert_ts is None or (dt.datetime.now() - dt.timedelta(days=2) >= device_info.last_alert_ts):
            did_alert_today = True
            points = device_info.court_keypoints.to_dict()
            if device_info.t_intersection:
                points["old_t"] = device_info.t_intersection
            if new_t:
                points["new_t"] = new_t
            add_points_to_image(img_path, points)
            msg_id = self.slack_bot.write(f"{msg}\n{self._camera_management_link(device_info)}", img_path)
            assert msg_id
            self.slack_bot.write(str(points), thread_id=msg_id)
            self.slack_bot.write(f"{self.person_to_alert}: you have been chosen randomly to fix this. Thank you.", thread_id=msg_id)
        if new_t:
            self._write_t(device_info, new_t, did_alert_today)
        return msg_id

    # ---- keypoint sanity ----
    def _court_keypoints_correct_order(self, device_info: DeviceInfo) -> bool:
        kp = device_info.court_keypoints
        tl, tr, bl, br = kp.tol, kp.tor, kp.point_7, kp.point_9
        # left x < right x; top y < bottom y
        return tl[0] < tr[0] and bl[0] < br[0] and tl[1] < bl[1] and tr[1] < br[1]

    def _handle_court_keypoint_incorrect_order(self, device_info: DeviceInfo):
        msg_id = self.slack_bot.write(
            f"*{device_info.camera_name}*: Court keypoints are in incorrect order: "
            f"{device_info.court_keypoints.to_dict()}\n{self._camera_management_link(device_info)}"
        )
        self.slack_bot.write(f"{self.person_to_alert}: you have been chosen randomly to fix this. Thank you.", thread_id=msg_id)
        self.email.send_court_keypoints_ooo_email(device_info, msg_id)

    def _camera_management_link(self, device_info: DeviceInfo) -> str:
        url = f"{self._camera_mgmt_base_url}&cameraId={device_info.camera_id}"
        return f"<{url}|Open camera management>"

# -----------------------
# Signal handling / entry
# -----------------------

def handle_exit(signum, _frame):
    """Handle SIGTERM/SIGINT gracefully (K8s/Docker stop or Ctrl+C)."""
    msg = f":warning: Received signal {signum}, shutting down gracefully.\n"
    try:
        sb.write(msg)
    except Exception:
        print(msg)
    sys.exit(0)


if __name__ == "__main__":
    is_test = False
    in_dockerfile = os.getenv("CAMERA_KEYPOINTS_CHECK_IN_DOCKERFILE")

    if in_dockerfile is None and is_test is False:
        print("You're running this locally, are you sure you want to do is_test = False?")
        sys.exit(0)

    sb = SlackBot(is_test)

    if in_dockerfile and is_test is True:
        sb.write("Running prod camera keypoints scipt in is_test mode. Fix that")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_exit)  # Kubernetes / Docker stop
    signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C

    try:
        main(sb, is_test)
    except Exception:
        tb = traceback.format_exc()
        sb.write(f":red_circle: Error with script that checks if cameras moved : {tb}")
