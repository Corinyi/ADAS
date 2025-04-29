import time
import cv2
from ultralytics import YOLO
from collections import defaultdict
from picamera2 import Picamera2

# ---------------- Picamera2 초기화 ----------------
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(preview_config)
picam2.start()
time.sleep(1)  # 카메라 워밍업

# ---------------- YOLO 모델 로드 ----------------
model = YOLO("yolov8n-pose.pt")

# 사람 ID별 상태 기록
arm_up_start_time = defaultdict(lambda: None)
arm_up_confirmed = defaultdict(lambda: False)

while True:
    # PiCamera로부터 프레임 얻기 (numpy array, RGB 순서)
    frame = picam2.capture_array()

    # YOLOv8 inference (BGR가 아니어도 내부에서 처리 가능)
    results = model(frame, verbose=False)

    for r in results:
        if r.keypoints is not None:
            keypoints_all = r.keypoints.xy
            boxes = r.boxes.xyxy

            for i, keypoints_tensor in enumerate(keypoints_all):
                keypoints = keypoints_tensor.cpu().numpy()
                if len(keypoints) < 11:
                    continue

                # 주요 관절 좌표
                left_shoulder = keypoints[5]; left_wrist = keypoints[9]
                right_shoulder = keypoints[6]; right_wrist = keypoints[10]
                nose = keypoints[0]

                left_arm_up = left_wrist[1] < left_shoulder[1]
                right_arm_up = right_wrist[1] < right_shoulder[1]
                arm_up_now = left_arm_up or right_arm_up

                # 3초 이상 팔 들었으면 트래킹 시작
                if arm_up_now:
                    if arm_up_start_time[i] is None:
                        arm_up_start_time[i] = time.time()
                    elif not arm_up_confirmed[i] and time.time() - arm_up_start_time[i] > 3:
                        arm_up_confirmed[i] = True
                else:
                    arm_up_start_time[i] = None
                    arm_up_confirmed[i] = False

                if arm_up_confirmed[i]:
                    x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "Tracking (Arm Up)", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    head_x, head_y = int(nose[0]), int(nose[1])
                    cv2.circle(frame, (head_x, head_y), 5, (255,0,0), -1)
                    cv2.putText(frame, f"Head: ({head_x},{head_y})",
                                (head_x+10, head_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                    print(f"[ID {i}] Head Position: ({head_x}, {head_y})")

    # 화면에 띄우기 (BGR→RGB 전환 불필요)
    cv2.imshow("YOLOv8 3s Arm Up Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
