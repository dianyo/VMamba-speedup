import os
import time
import psutil
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import argparse
import sys
from dotenv import load_dotenv
from pathlib import Path


class ProcessMonitor:
    def __init__(self):
        # Get credentials from environment variables
        self.slack_token = os.getenv("SLACK_BOT_TOKEN")
        self.channel_id = os.getenv("SLACK_CHANNEL_ID")

        if not self.slack_token or not self.channel_id:
            raise ValueError(
                "Missing required environment variables. Please check your .env file."
            )

        self.slack_client = WebClient(token=self.slack_token)

    def is_process_running(self, pid):
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except psutil.NoSuchProcess:
            return False

    def send_slack_message(self, message):
        try:
            response = self.slack_client.chat_postMessage(
                channel=self.channel_id, text=message
            )
            print(f"Message sent: {message}")
        except SlackApiError as e:
            print(f"Error sending message: {e.response['error']}")

    def monitor_process(self, pid):
        process_name = "Unknown"
        process_args = ""
        try:
            process = psutil.Process(pid)
            process_name = process.name()
            # Get command line arguments if available
            try:
                cmdline = process.cmdline()
                if len(cmdline) > 1:
                    process_args = f" (args: {cmdline[1]})"
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_args = ""

            self.send_slack_message(
                f"Started monitoring process {process_name}{process_args} (PID: {pid})"
            )
        except psutil.NoSuchProcess:
            self.send_slack_message(f"Process with PID {pid} not found!")
            return

        while True:
            if not self.is_process_running(pid):
                self.send_slack_message(
                    f"Process {process_name}{process_args} (PID: {pid}) has completed!"
                )
                break
            time.sleep(10)  # Check every 10 seconds


def load_env_file(env_path):
    env_path = Path(env_path)
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    # Load the specified .env file
    load_dotenv(env_path)
    print(f"Loaded environment from: {env_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor a process and send Slack notifications"
    )
    parser.add_argument("pid", type=int, help="PID of the process to monitor")
    parser.add_argument("--env", type=str, help="Path to .env file", default=".env")

    args = parser.parse_args()

    try:
        # Load environment variables from specified file
        load_env_file(args.env)

        monitor = ProcessMonitor()
        monitor.monitor_process(args.pid)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
