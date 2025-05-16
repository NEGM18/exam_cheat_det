import cv2
import mediapipe as mp
import time
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import os

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

def get_head_pose(landmarks, image_shape):
    key_indices = [1, 152, 33, 263, 287, 57]
    image_points = np.array([landmarks[idx] for idx in key_indices], dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (43.3, 32.7, -26.0),
        (-43.3, 32.7, -26.0),
        (28.9, -28.9, -24.1),
        (-28.9, -28.9, -24.1)
    ])
    focal_length = image_shape[1]
    center = (image_shape[1] / 2, image_shape[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None, None
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    yaw, pitch, roll = [float(angle) for angle in euler_angles]
    yaw = ((yaw + 180) % 360) - 180
    if abs(yaw) > 90:
        print(f"Warning: Unusual yaw value: {yaw:.2f}")
    return yaw, pitch, roll

def monitor_video(log_callback, frame_callback=None, use_recorded=False):
    cheating_detect_dir = "cheating_detect"
    if not os.path.exists(cheating_detect_dir):
        os.makedirs(cheating_detect_dir)
    
    from audio_monitor import monitor_audio
    import threading
    audio_thread = threading.Thread(target=monitor_audio, args=(log_callback, cheating_detect_dir))
    audio_thread.daemon = True
    audio_thread.start()
    
    model = YOLO('yolov8n.pt')
    model.conf = 0.2
    model.iou = 0.2
    
    if use_recorded:
        cap = cv2.VideoCapture("normal_videos/normal1.avi")
        if not cap.isOpened():
            log_callback("Cannot open normal video.")
            return
    else:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            log_callback("Cannot access camera.")
            return

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.3
    )

    target_object = 'cell phone'
    
    prediction_history = []
    history_size = 3
    threshold = 0.3
    save_threshold = 0.4
    
    face_out_count = 0
    face_out_threshold = 3
    consecutive_no_face = 0
    
    last_save_time = 0
    save_cooldown = 2

    global shared_frame

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if use_recorded:
                cap.release()
                cap = cv2.VideoCapture("cheating_videos/cheat1.avi")
                if not cap.isOpened():
                    log_callback("Cannot open cheating video.")
                    break
                continue
            else:
                log_callback("Failed to capture frame.")
                break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        face_results = face_mesh.process(frame_rgb)
        face_visible = False
        multiple_faces = False
        looking_away = False
        
        if face_results.multi_face_landmarks:
            face_visible = True
            consecutive_no_face = 0
            log_callback("Face detected")
            
            if len(face_results.multi_face_landmarks) > 1:
                multiple_faces = True
                log_callback("Multiple faces detected!")
            
            for face_landmarks in face_results.multi_face_landmarks:
                if len(face_landmarks.landmark) > 287:
                    h, w, _ = frame.shape
                    lm = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
                    yaw, pitch, roll = get_head_pose(lm, frame.shape)
                    if yaw is not None and abs(yaw) > 35:
                        looking_away = True
                        log_callback(f"Person is looking away! (Yaw: {yaw:.2f})")
        else:
            consecutive_no_face += 1
            if consecutive_no_face >= 5:
                face_out_count += 1
                consecutive_no_face = 0
                log_callback(f"No face detected! (Count: {face_out_count}/{face_out_threshold})")

        objects_detected = False
        results = model(frame, verbose=False)[0]
        
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            class_name = results.names[int(class_id)]
            if class_name == target_object and confidence > 0.2:
                objects_detected = True
                log_callback(f"Telephone detected! (Confidence: {confidence:.2f})")

        hand_results = hands.process(frame_rgb)
        hands_detected = False
        
        if hand_results.multi_hand_landmarks:
            hands_detected = True
            log_callback("Hand movement detected!")

        cheating_probability = 0.0
        if multiple_faces:
            cheating_probability += 0.5
        if objects_detected:
            cheating_probability += 0.5
        if hands_detected:
            cheating_probability += 0.3
        if looking_away:
            cheating_probability += 0.4
        if face_out_count >= face_out_threshold:
            cheating_probability = 1.0
        
        prediction_history.append(cheating_probability)
        if len(prediction_history) > history_size:
            prediction_history.pop(0)
        
        avg_prediction = sum(prediction_history) / len(prediction_history)
        
        current_time = time.time()
        if avg_prediction >= save_threshold and (current_time - last_save_time) >= save_cooldown:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cheat_{timestamp}_prob{avg_prediction:.2f}.jpg"
            filepath = os.path.join(cheating_detect_dir, filename)
            
            cv2.imwrite(filepath, frame)
            log_callback(f"Saved cheating frame: {filename} (Probability: {avg_prediction:.2f})")
            last_save_time = current_time
        
        if avg_prediction > threshold:
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-1, frame.shape[0]-1), (0, 0, 255), 2)
            log_callback(f"Potential cheating detected! (Confidence: {avg_prediction:.3f})")

        # cv2.imshow("Video Monitor", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     log_callback("Monitoring stopped by user.")
        #     break

        if frame_callback is not None:
            frame_callback(frame)

        shared_frame = frame

    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    return frame

def record_cheating_video(duration=30, output_dir="cheating_videos"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot access camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"cheat_telephone_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
    
    print(f"Recording started. Press 'q' to stop recording or wait for {duration} seconds.")
    print(f"Video will be saved to: {output_file}")
    
    start_time = time.time()
    recording = True
    
    while recording:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        current_time = time.time() - start_time
        remaining_time = max(0, duration - current_time)
        cv2.putText(frame, f"Recording... {remaining_time:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(frame)
        
        cv2.imshow("Recording - Press 'q' to stop", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q') or current_time >= duration:
            recording = False
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Recording completed. Video saved to: {output_file}")

if __name__ == "__main__":
    record_cheating_video(duration=30)
