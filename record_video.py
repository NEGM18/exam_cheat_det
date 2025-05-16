import cv2
import mediapipe as mp
import time
import os
import numpy as np
from ultralytics import YOLO

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def record_video(output_path="dataset/cheating/cheat2.avi", duration=10):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30

    model = YOLO('yolov8n.pt')

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        print("❌ Failed to initialize VideoWriter.")
        cap.release()
        return

    print(f"📹 Recording to: {output_path}")
    print("Instructions:")
    print("- Make sure your face is clearly visible")
    print("- Press 'q' to stop recording early")
    print(f"- Recording will automatically stop after {duration} seconds")
    
    frame_count = 0
    max_frames = duration * fps
    valid_frames = 0
    multiple_face_frames = 0
    object_detected_frames = 0
    hand_detected_frames = 0

    start_time = time.time()
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = face_mesh.process(frame_rgb)
        face_visible = False
        multiple_faces = False
        
        if face_results.multi_face_landmarks:
            face_visible = True
            valid_frames += 1
            
            if len(face_results.multi_face_landmarks) > 1:
                multiple_faces = True
                multiple_face_frames += 1
            
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        objects_detected = False
        results = model(frame, verbose=False)[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            class_name = results.names[int(class_id)]
            if confidence > 0.5:
                objects_detected = True
                object_detected_frames += 1
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name}", (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        hand_results = hands.process(frame_rgb)
        hands_detected = False
        
        if hand_results.multi_hand_landmarks:
            hands_detected = True
            hand_detected_frames += 1
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                cv2.putText(frame, "Hand movement detected!", (10, 270), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        time_left = duration - (time.time() - start_time)
        status_color = (0, 255, 0) if face_visible else (0, 0, 255)
        cv2.putText(frame, f"Time left: {time_left:.1f}s", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Face detected: {face_visible}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(frame, f"Multiple faces: {multiple_faces}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if multiple_faces else status_color, 2)
        if objects_detected:
            cv2.putText(frame, "Objects detected!", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        border_color = (0, 0, 255) if objects_detected or hands_detected or multiple_faces else status_color
        cv2.rectangle(frame, (0, 0), (width-1, height-1), border_color, 2)

        out.write(frame)
        cv2.imshow("Recording...", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Recording manually stopped.")
            break

        frame_count += 1

    recording_time = time.time() - start_time
    face_detection_rate = (valid_frames / frame_count) * 100 if frame_count > 0 else 0
    multiple_face_rate = (multiple_face_frames / frame_count) * 100 if frame_count > 0 else 0
    object_detection_rate = (object_detected_frames / frame_count) * 100 if frame_count > 0 else 0
    hand_detection_rate = (hand_detected_frames / frame_count) * 100 if frame_count > 0 else 0

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n=== Recording Summary ===")
    print(f"✅ Recording complete:")
    print(f"- Total frames: {frame_count}")
    print(f"- Valid frames (face detected): {valid_frames} ({face_detection_rate:.1f}%)")
    print(f"- Multiple face frames: {multiple_face_frames} ({multiple_face_rate:.1f}%)")
    print(f"- Object detection frames: {object_detected_frames} ({object_detection_rate:.1f}%)")
    print(f"- Hand movement frames: {hand_detected_frames} ({hand_detection_rate:.1f}%)")
    print(f"- Recording duration: {recording_time:.1f} seconds")
    print(f"- Output file: {output_path}")
    
    if face_detection_rate < 80:
        print("\n⚠️ Warning: Low face detection rate. Consider re-recording with better face visibility.")
    
    return face_detection_rate > 80

def get_next_video_number(base_dir, prefix):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1 if prefix == "cheat" else 3
    
    existing_files = [f for f in os.listdir(base_dir) if f.startswith(prefix) and f.endswith('.avi')]
    if not existing_files:
        return 1 if prefix == "cheat" else 3
    
    numbers = []
    for file in existing_files:
        try:
            num = int(file[len(prefix):-4])
            numbers.append(num)
        except ValueError:
            continue
    
    next_num = max(numbers) + 1 if numbers else (1 if prefix == "cheat" else 3)
    return next_num

if __name__ == "__main__":
    BASE_DIR = "D:\\projects\\exam_cheat_det"
    NORMAL_DIR = os.path.join(BASE_DIR, "normal_videos")
    CHEATING_DIR = os.path.join(BASE_DIR, "cheating_videos")
    
    while True:
        print("\n=== Video Recording Menu ===")
        print("1. Record Normal Behavior")
        print("2. Record Cheating Behavior")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "3":
            print("Exiting...")
            break
        elif choice not in ["1", "2"]:
            print("Invalid choice. Please try again.")
            continue
        
        behavior_type = "normal" if choice == "1" else "cheat"
        prefix = "normal" if choice == "1" else "cheat"
        base_dir = NORMAL_DIR if choice == "1" else CHEATING_DIR
        
        next_num = get_next_video_number(base_dir, prefix)
        output_path = os.path.join(base_dir, f"{prefix}{next_num}.avi")
        
        print(f"\n=== Recording {behavior_type.upper()} Behavior ===")
        if behavior_type == "normal":
            print("Instructions:")
            print("- Act normally as if taking an exam")
            print("- Look at the screen")
            print("- Make sure your face is clearly visible")
        else:
            print("Instructions:")
            print("- Simulate phone usage during exam")
            print("- Try different positions and angles")
            print("- Make sure your face is clearly visible")
        
        print(f"\nThis will be saved as: {output_path}")
        print("Press Enter to start recording...")
        input()
        
        record_video(output_path, duration=20)
        
        print("\nRecording complete!")
        print("Would you like to record another video? (y/n)")
        if input().lower() != 'y':
            break
    
    print("\nThank you for recording! All videos have been saved.")