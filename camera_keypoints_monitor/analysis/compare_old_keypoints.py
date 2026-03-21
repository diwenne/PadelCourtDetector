import datetime 
from pathlib import Path
from typing import Optional

from src.camera_keypoints_main import almost_equal
from src.compute_t_intersection import find_t_intersection
from src.image_extracter import S3Adapter, add_points_to_image, get_court_keypoints, get_last_video_image_save
from src.supabase_adapter import SupabaseAdapter


def main(camera_name: str, how_far_back_get_older_video: datetime.timedelta, first_video_latest_ts: Optional[datetime.datetime] = None):
    """The purpose of this script is to compare the old T-intersection of a camera to the new one 
    
    how_far_back_get_older_video: how many days prior to the last video should we use to fetch the other video 
    from which we should fetch the t-intersection 

    first_video_latest_ts: for the first video, what is the latest ts it can be from? Note that the latest 
    ts for the second video is thus first_video_latest_ts - how_far_back_get_older_video
    """
    # We dont want to write to the db 
    sa = SupabaseAdapter(read_only=True)
    devices = sa.get_device_info()
    for device in devices:
        if device.camera_name == camera_name:
            break 
    else:
        raise ValueError("Could not find camera name")
    
    latest_video = sa.get_latest_video(device.camera_id, max_date=first_video_latest_ts)
    if latest_video is None:
        raise ValueError("Could not get latest video for camera")
    latest_prev_video_time = latest_video.updated_ts - how_far_back_get_older_video
    prev_video = sa.get_latest_video(device.camera_id, max_date=latest_prev_video_time)
    if prev_video is None:
        raise ValueError("Could not find an old video")
    assert latest_video.video_id != prev_video.video_id

    s3 = S3Adapter()
    latest_video_path = Path(f"latest_vid_{latest_video.video_id}.jpg")
    if not latest_video_path.exists():
        get_last_video_image_save(s3, latest_video, latest_video_path)
    old_video_path = Path(f"old_vid_{prev_video.video_id}.jpg")
    if not old_video_path.exists():
        get_last_video_image_save(s3, prev_video, old_video_path)
    new_t = find_t_intersection(latest_video_path, device.bottom_line_center, debug=True)
    if new_t is None:
        raise ValueError("Could not find t intersection in new image")
    old_t = find_t_intersection(old_video_path, device.bottom_line_center, debug=True)
    if old_t is None:
        raise ValueError("Could not find t intersection in old image")
    almost_equal_t = almost_equal(new_t, old_t)
    print(f"Would have alerted?: {not almost_equal_t}")
    comparison_image_path = Path(f"comparison_{latest_video.video_id}_{prev_video.video_id}.jpg")
    if not comparison_image_path.exists():
        add_points_to_image(latest_video_path, {"new_t": new_t, "old_t": old_t}, new_image_path=str(comparison_image_path))
    else:
        print("Comparison image already exists")
    new_court_keypoints = get_court_keypoints(s3, latest_video)
    old_court_keypoints = get_court_keypoints(s3, prev_video)
    print(f"Current court keypoints in db: {device.court_keypoints}")
    print(f"Court keypoints from old video in s3: {old_court_keypoints}")
    print(f"Court keypoints from new video in s3: {new_court_keypoints}")

if __name__ == "__main__":
    # 2 am ET
    md = datetime.datetime(2025, 10, 28, 6, tzinfo=datetime.timezone.utc)
    main("Høyde Teknikk", datetime.timedelta(days = 1), md)