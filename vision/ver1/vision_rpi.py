import sys
sys.path.append('/usr/lib/python3/dist-packages')
import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import defaultdict
from picamera2 import Picamera2

# Load YOLOv8n model (person detection)
model = YOLO("yolov8n.pt")

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark index and 3D model points for PnP
LANDMARK_IDXS = [33, 263, 1, 61, 291, 199]
model_points = np.array([
    (-30, 0, -30),
    (30, 0, -30),
    (0, 0, 0),
    (-20, -30, -20),
    (20, -30, -20),
    (0, -60, -10)
], dtype=np.float64)

# Camera matrix (normalized)
camera_matrix = np.array([
    [1, 0, 0.5],
    [0, 1, 0.5],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

# Yaw detection state
yaw_timer = defaultdict(lambda: None)
YAW_THRESHOLD = 30  # degrees
TIME_THRESHOLD = 3  # seconds

# Setup PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")
picam2.start()
time.sleep(1)

# Optional: fix overly blue color on Pi Camera v3 (manual white balance)
picam2.set_controls({
    "AwbMode": 3  # 3 = incandescent light (warmer color balance)
})

# OpenCV window
cv2.namedWindow("Exam Monitor", cv2.WINDOW_NORMAL)
cv2.moveWindow("Exam Monitor", 100, 100)

while True:
    frame_bgr = picam2.capture_array()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # for MediaPipe and display

    # YOLO detects people
    results = model.predict(source=frame_bgr, classes=[0], verbose=False)
    suspicious_ids = []

    for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        person_roi = frame_rgb[y1:y2, x1:x2]

        # Face detection with MediaPipe
        face_result = face_mesh.process(person_roi)
        if not face_result.multi_face_landmarks:
            yaw_timer[i] = None
            continue

        # Extract facial landmarks
        face_landmarks = face_result.multi_face_landmarks[0]
        image_points = []
        for idx in LANDMARK_IDXS:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * (x2 - x1)) + x1
            y = int(lm.y * (y2 - y1)) + y1
            image_points.append((x, y))
        image_points = np.array(image_points, dtype=np.float64)

        # Estimate yaw with solvePnP
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw = angles[1] * 180

        # Suspicious head turning logic
        current_time = time.time()
        if abs(yaw) > YAW_THRESHOLD:
            if yaw_timer[i] is None:
                yaw_timer[i] = current_time
            elif current_time - yaw_timer[i] >= TIME_THRESHOLD:
                suspicious_ids.append(i)
        else:
            yaw_timer[i] = None

        # Draw bounding box and yaw info on RGB frame
        label = f"Yaw: {int(yaw)} deg"
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw warning
    if suspicious_ids:
        warning_text = "Suspected Cheating: " + ", ".join([f"Person #{i}" for i in suspicious_ids])
        cv2.putText(frame_rgb, warning_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    # Display RGB frame (needed for correct colors on Pi)
    cv2.imshow("Exam Monitor", frame_rgb)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()