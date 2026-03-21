from typing import List, Optional
import requests
from src.slack_adapter import SlackBot
from src.utils import DeviceInfo 

def get_camera_camera_moved_email(camera_name: str):
    return f"""Hello, 

The camera may have moved on court {camera_name}. Please check the court keypoints on the dashboard (https://dashboard.clutchapp.io/). 

Best,
The Clutch Team
    """

def get_court_keypoints_out_of_order(camera_name: str):
    return f""""Hello,

The court keypoints are not in the correct order on court {camera_name}. Please double check the court keypoints on the dashboard (https://dashboard.clutchapp.io/).

Best,
The Clutch Team
    """

class EmailAdapter():
    def __init__(self, slack_adapter: SlackBot, is_test: bool):
        self.slack_adapter = slack_adapter
        self.is_test = is_test

    def send_court_keypoints_ooo_email(self, device_info: DeviceInfo, slack_primary_message: Optional[str] = None):
        if device_info.should_send_email:
            subject = f"Court Keypoints Out of Order on {device_info.camera_name}"
            message = get_court_keypoints_out_of_order(device_info.camera_name)
            self.send_email(device_info.emails, subject, message, slack_primary_message)
        else:
            self.slack_adapter.write("Not sending email (prefernece turned off)", thread_id=slack_primary_message)

    def send_camera_moved_email(self, device_info: DeviceInfo, slack_primary_message: Optional[str] = None):
        if device_info.should_send_email:
            subject = f"Court Keypoints May Have Moved on {device_info.camera_name}"
            message = get_camera_camera_moved_email(device_info.camera_name)
            self.send_email(device_info.emails, subject, message, slack_primary_message)
        else:
            self.slack_adapter.write("Not sending email (preference turned off)", thread_id=slack_primary_message)

    def send_email(self, recipient_emails: List[str], subject: str, message: str, slack_primary_message: Optional[str] = None):
        """
        in test mode, instead of sending the email, we will reply to the slack message slack_primary_message
        in regular mode, we send the email and we reply to the slack thread saying we send the email 
        """
        slack_msg = f"Emailed {str(recipient_emails)}:\nSubject: {subject}:\n\n{message}"
        if self.is_test:
            slack_msg = "Would have " + slack_msg
        if not self.is_test:
            for email in recipient_emails:
                url = "https://clutch-eye.fly.dev/api/v1/alert/gmail/send"
                headers = {"Content-Type": "application/json"}
                data = {
                    "email": email,
                    "subject": subject,
                    "message": message
                }
                requests.post(url, headers=headers, json=data)
        self.slack_adapter.write(slack_msg, thread_id=slack_primary_message)
    