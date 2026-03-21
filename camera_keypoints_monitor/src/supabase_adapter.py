"""Connects us to supabase"""
from collections import defaultdict
import datetime
import time
from typing import Dict, List, Optional
from src.utils import CourtKeypoints, DeviceInfo, ImagePoint, VideoData
from supabase import create_client, Client
import os 

def to_t_intersection_column(x: float, y: float, did_alert_today: bool = False) -> dict:
    t_int_colum = {"x": x, "y" : y, "last_update_ts": time.time()}
    if did_alert_today:
        t_int_colum["last_alert_ts"] = time.time()
    return t_int_colum

def from_t_intersection_column(json_raw: Optional[dict]) -> Optional[ImagePoint]:
    if json_raw is None:
        return None
    return (float(json_raw['x']), float(json_raw['y']))

def get_last_alert_ts_from_t_int_col(json_raw: Optional[dict]) -> Optional[datetime.datetime]:
    if json_raw is None:
        return None
    last_alert_ts = json_raw.get("last_alert_ts", None)
    if last_alert_ts is None:
        return None
    return datetime.datetime.fromtimestamp(float(last_alert_ts))

def get_court_keypoints(json_raw: dict) -> CourtKeypoints:
    return CourtKeypoints(tol=(json_raw['tol'][0], json_raw['tol'][1]), tor=(json_raw['tor'][0], json_raw['tor'][1]), point_7=(json_raw['point_7'][0], json_raw['point_7'][1]), point_9=(json_raw['point_9'][0], json_raw['point_9'][1]))

class SupabaseAdapter():
    def __init__(self, read_only: bool):
        url: str = os.environ.get("APP_SUPABASE_URL", "")
        key: str = os.environ.get("APP_SUPABASE_KEY", "")
        assert url and key
        self.client: Client = create_client(url, key)
        self.read_only = read_only

    def get_device_info(self) -> List[DeviceInfo]:
        """Return info from camera.devices"""

        resp = (
            self.client
            .schema("camera")
            .table("devices")
            .select("id,name,status,t_intersection, court_keypoints,venue_id")
            .in_("status", ['available', 'recording'])
            .in_("sport", ['padel'])
            .execute()
        )

        # Assuming self.client is your Supabase client instance
        club_config_resp = (
            self.client
            .schema("public")
            .table("club_configs")
            .select("venue_id,send_alert_emails")
            .execute()
        )
        send_alert_emails = {row['venue_id'] : row['send_alert_emails'] for row in club_config_resp.data}
        venue_admins_resp = (
            self.client
            .schema("camera")
            .table("venue_admins")
            .select("venue_id,admin_id")
            .execute()
        )
        venue_id_to_admin_ids = defaultdict(list)
        for row in venue_admins_resp.data:
            venue_id_to_admin_ids[row["venue_id"]].append(row["admin_id"])

        public_user_resp = (
            self.client
            .schema("public")
            .table("User")
            .select("id,email")
            .execute()
        )
        admin_id_to_email: Dict[str, str] = {row['id'] : row['email'] for row in public_user_resp.data}

        # Get emails and preferences 
        results = []
        for row in resp.data:
            # We skip the camera if it does not have court keypoints yet
            if row['court_keypoints'] is None:
                continue 
            venue_id = row['venue_id']
            admins_ids = []
            emails = []
            for admin_id in venue_id_to_admin_ids[venue_id]:
                if admin_id in admin_id_to_email:
                    email = admin_id_to_email[admin_id]
                    if not email.endswith("@clutchapp.io") and email != "paul.liu@stanford.edu":
                        admins_ids.append(admin_id)
                        emails.append(email)
            email_pref = send_alert_emails[venue_id]
            results.append(DeviceInfo(
                camera_id=row['id'],
                camera_name=row['name'],
                t_intersection=from_t_intersection_column(row['t_intersection']),
                court_keypoints=get_court_keypoints(row['court_keypoints']),
                venue_id=venue_id,
                admin_ids=admins_ids,
                emails=emails,
                should_send_email=email_pref,
                last_alert_ts=get_last_alert_ts_from_t_int_col(row['t_intersection']),
            ))
        return results 
    
    def write_t_intersection(self, camera_id: str, x: float, y: float, did_alert_today: bool = False):
        if self.read_only:
            print(f"Skipping write to db. Would have written: x = {x} and y = {y} for {camera_id}")
            return 
        self.client.schema("camera").table("devices").update({
            "t_intersection": to_t_intersection_column(x, y, did_alert_today)
        }).eq("id", camera_id).execute()

    def get_latest_video(self, camera_id: str, min_date: Optional[datetime.datetime] = None, max_date: Optional[datetime.datetime] = None) -> Optional[VideoData]:
        """min_date and max_date specify what time range we should look into"""
        allowed = ["OK", "OK_EMPTY_COURT", "UPLOADED", "COMPLETED"]

        q = (
            self.client
            .table("VideoMetadata")
            .select("id,creatorIdentityID,updated_at") 
            .not_.is_("camera_name", "null")
            .not_.is_("creatorIdentityID", "null")
            .in_("device_id", [camera_id])
            .in_("process_status", allowed)
        )

        if min_date is not None:
            q = q.gt("updated_at", min_date.isoformat())
        if max_date is not None: 
            q = q.lt("updated_at", max_date.isoformat())

        res = (
            q.order("updated_at", desc=True)
            .order("id", desc=True)     # tiebreaker for equal timestamps
            .limit(1)
            .execute()
        )
        if res.data is None or len(res.data) == 0:
            return None 
        assert len(res.data) == 1
        row = res.data[0]
        return VideoData(creator_identity_id=row["creatorIdentityID"], video_id=row["id"], updated_ts = datetime.datetime.fromisoformat(row["updated_at"]))
