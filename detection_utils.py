from datetime import datetime
import os

def log_event(event):
    os.makedirs("logs", exist_ok=True)  

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("logs/activity_log.txt", "a") as log_file:
        log_file.write(f"[{timestamp}] {event}\n")
    print(f"[{timestamp}] {event}")