import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

# 모델 로드
model = YOLO("yolov8n-pose.pt")

# 사람 ID별 상태 기록
arm_up_start_time = defaultdict(lambda: None)  # 3초 타이머용
arm_up_confirmed = defaultdict(lambda: False)  # 트래킹 중인지 여부

def find_camera_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

camera_index = find_camera_index()
if camera_index is None:
    print("❌ 사용할 수 있는 카메라를 찾을 수 없습니다.")
    exit()

cap = cv2.VideoCapture(camera_index)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

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
                left_shoulder = keypoints[5]
                left_wrist = keypoints[9]
                right_shoulder = keypoints[6]
                right_wrist = keypoints[10]
                nose = keypoints[0]

                # 팔을 들고 있는지
                left_arm_up = left_wrist[1] < left_shoulder[1]
                right_arm_up = right_wrist[1] < right_shoulder[1]
                arm_up_now = left_arm_up or right_arm_up

                # 팔을 들고 있으면
                if arm_up_now:
                    if arm_up_start_time[i] is None:
                        arm_up_start_time[i] = time.time()
                    elif not arm_up_confirmed[i] and time.time() - arm_up_start_time[i] > 3:
                        arm_up_confirmed[i] = True  # 3초 이상 팔을 든 경우 트래킹 시작
                else:
                    # 팔을 내리면 트래킹 멈춤
                    arm_up_start_time[i] = None
                    arm_up_confirmed[i] = False

                # 현재 트래킹 중인 사람만 박스 + 머리 좌표 표시
                if arm_up_confirmed[i]:
                    x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Tracking (Arm Up)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    head_x, head_y = int(nose[0]), int(nose[1])
                    cv2.circle(frame, (head_x, head_y), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"Head: ({head_x}, {head_y})", (head_x + 10, head_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # 콘솔에도 출력
                    print(f"[ID {i}] Head Position: ({head_x}, {head_y})")

    cv2.imshow("YOLOv8 3s Arm Up Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
