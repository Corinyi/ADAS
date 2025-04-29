import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

# 모델 로드
model = YOLO("yolov8n-pose.pt")

# 사람 ID별 상태 기록
arm_up_start_time = defaultdict(lambda: None)
arm_up_confirmed = defaultdict(lambda: False)

# 카메라 오픈
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # 수정 포인트!

if not cap.isOpened():
    print("❌ 카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임 읽기 실패")
        break

    # 이후 동일
    results = model(frame, verbose=False)

    for r in results:
        if r.keypoints is not None:
            keypoints_all = r.keypoints.xy
            boxes = r.boxes.xyxy

            for i, keypoints_tensor in enumerate(keypoints_all):
                keypoints = keypoints_tensor.cpu().numpy()

                if len(keypoints) < 11:
                    continue

                left_shoulder = keypoints[5]
                left_wrist = keypoints[9]
                right_shoulder = keypoints[6]
                right_wrist = keypoints[10]
                nose = keypoints[0]

                left_arm_up = left_wrist[1] < left_shoulder[1]
                right_arm_up = right_wrist[1] < right_shoulder[1]
                arm_up_now = left_arm_up or right_arm_up

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
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking (Arm Up)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    head_x, head_y = int(nose[0]), int(nose[1])
                    cv2.circle(frame, (head_x, head_y), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"Head: ({head_x}, {head_y})", (head_x + 10, head_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    print(f"[ID {i}] Head Position: ({head_x}, {head_y})")

    cv2.imshow("YOLOv8 3s Arm Up Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
