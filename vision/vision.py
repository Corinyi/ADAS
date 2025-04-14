import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
from collections import defaultdict

# 모델 불러오기 (YOLOv8n은 경량화 모델)
model = YOLO("yolov8n.pt")

# MediaPipe Face Mesh 준비
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# 얼굴 3D 모델 포인트
model_points = np.array([
    (-30, 0, -30),   # 왼눈꼬리
    (30, 0, -30),    # 오른눈꼬리
    (0, 0, 0),       # 코끝
    (-20, -30, -20), # 입 왼쪽
    (20, -30, -20),  # 입 오른쪽
    (0, -60, -10)    # 턱
], dtype=np.float64)

LANDMARK_IDXS = [33, 263, 1, 61, 291, 199]

# 카메라 기본값
camera_matrix = np.array([
    [1, 0, 0.5],
    [0, 1, 0.5],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros((4, 1))

# 타이머 저장 (사람 ID : 고개 돌린 시간)
yaw_timer = defaultdict(lambda: None)

# 설정값
YAW_THRESHOLD = 30  # 각도 기준
TIME_THRESHOLD = 3  # 3초 이상
fps = 30  # 웹캠 FPS

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]

    results = model.predict(source=frame, classes=[0], verbose=False)  # 사람만 감지

    suspicious_ids = []

    for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        person_roi = frame_rgb[y1:y2, x1:x2]

        face_result = face_mesh.process(person_roi)
        if not face_result.multi_face_landmarks:
            yaw_timer[i] = None
            continue

        # 랜드마크 추출
        face_landmarks = face_result.multi_face_landmarks[0]
        image_points = []
        for idx in LANDMARK_IDXS:
            lm = face_landmarks.landmark[idx]
            x = int(lm.x * (x2 - x1)) + x1
            y = int(lm.y * (y2 - y1)) + y1
            image_points.append((x, y))

        image_points = np.array(image_points, dtype=np.float64)

        # 동작 감지
        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        yaw = angles[1] * 180

        # 타이머 관리
        current_time = time.time()
        if abs(yaw) > YAW_THRESHOLD:
            if yaw_timer[i] is None:
                yaw_timer[i] = current_time  # 처음 돌린 시간 기록
            elif current_time - yaw_timer[i] >= TIME_THRESHOLD:
                suspicious_ids.append(i)
        else:
            yaw_timer[i] = None

        # 사람마다 바운딩 박스 + 각도 표시
        label = f"Yaw: {int(yaw)} deg"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 경고 메시지
    if suspicious_ids:
        warning_text = "Suspected Cheating: " + ", ".join([f"Person #{i}" for i in suspicious_ids])
        cv2.putText(frame, warning_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

    cv2.imshow("Exam Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
