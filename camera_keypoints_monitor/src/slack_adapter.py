from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import zoneinfo
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class SlackBot:
    def __init__(self, test: bool = False):
        """
        Initialize the Slack bot client with an OAuth token.
        Usually starts with 'xoxb-' for bot tokens.
        """
        self.client = WebClient(
            token="xoxb-1653580682886-9631855698615-OCakeY1RwJSOCZnyOIaY61KV"
        )
        self.test = test

        if test:
            self.channel = "C09D8L2GHPY"
        else:
            self.channel = "C09G90X0M9C"

    def write(self, message: str, image_path: Optional[Path] = None, thread_id: Optional[str] = None) -> Optional[str]:
        """Send a message; optionally reply with an image if a JPG path is given.

        thread_id: if it's a reply, what thread to reply to
        """
        if self.test:
            print(message)
        try:
            resp = self.client.chat_postMessage(channel=self.channel, text=message, thread_ts=thread_id)
            ts = resp["ts"]

            if image_path and image_path.exists() and image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg"}:
                self.client.files_upload_v2(
                    channel=self.channel,
                    thread_ts=ts,
                    file=str(image_path),
                    filename=image_path.name,
                )
            return ts
        except SlackApiError as e:
            print(f"Slack error: {e.response.get('error', e)}")
            return None

    def schedule_message(self, message: str, minutes_from_now: int):
        """Schedule a message to send `minutes_from_now` minutes from now."""
        ny = zoneinfo.ZoneInfo("America/New_York")
        target_dt = datetime.now(ny) + timedelta(minutes=minutes_from_now)
        post_at = int(target_dt.timestamp())

        try:
            resp = self.client.chat_scheduleMessage(
                channel=self.channel,
                text=message,
                post_at=post_at,
            )
            scheduled_id = resp["scheduled_message_id"]
            print(f"✅ Scheduled for {target_dt} (ID: {scheduled_id})")
            return scheduled_id
        except SlackApiError as e:
            print(f"Error scheduling message: {e.response['error']}")
            return None

    def list_scheduled_messages(self):
        """
        List all scheduled messages for the current channel.
        Returns a list of dicts with message text, post time, and ID.
        """
        try:
            resp = self.client.chat_scheduledMessages_list(channel=self.channel)
            messages = resp.get("scheduled_messages", [])
            if not messages:
                print("No scheduled messages found.")
                return []

            print("📅 Scheduled messages:")
            for msg in messages:
                ts = datetime.fromtimestamp(int(msg["post_at"]))
                text = msg.get("text", "(no text)")
                mid = msg["id"]
                print(f"- {ts} | ID: {mid} | Text: {text[:60]}")
            return messages
        except SlackApiError as e:
            print(f"Error listing scheduled messages: {e.response['error']}")
            return []

    def cancel_scheduled_message(self, scheduled_message_id: str, channel: str):
        """Cancel a previously scheduled Slack message."""
        try:
            resp = self.client.chat_deleteScheduledMessage(
                channel=channel,
                scheduled_message_id=scheduled_message_id,
            )
            resp.validate()
            print(f"🗑️ Canceled scheduled message: {scheduled_message_id}")
        except SlackApiError as e:
            print(f"Error canceling message: {e.response['error']}")

    def cancel_slack_msg_with_txt(self, msg: str):
        scheduled_messages = self.list_scheduled_messages()
        for scheduled_msg in scheduled_messages:
            if scheduled_msg["text"] == msg:
                self.cancel_scheduled_message(scheduled_msg["id"], scheduled_msg["channel_id"])