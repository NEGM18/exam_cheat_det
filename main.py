from video_monitor import monitor_video
from audio_monitor import monitor_audio
from detection_utils import log_event
import threading
import time
import cv2

shared_frame = None

def main():
    print("Starting monitoring system...")
    
    print("Starting video monitoring...")
    video_thread = threading.Thread(target=monitor_video, args=(log_event,))
    video_thread.daemon = True
    video_thread.start()
    
    time.sleep(2)
    
    print("Starting audio monitoring...")
    audio_thread = threading.Thread(target=monitor_audio, args=(log_event,))
    audio_thread.daemon = True
    audio_thread.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitoring system...")

if __name__ == "__main__":
    main()
