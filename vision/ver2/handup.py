import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

# ✅ YOLOv8-pose 모델 로드 (처음 실행 시 자동 다운로드됨)
model = YOLO("yolov8n-pose.pt")  # 필요에 따라 yolov8s-pose.pt 등으로 교체 가능

# ✅ 팔 든 시간 저장 딕셔너리 (사람 ID → 시작 시간)
arm_up_start_time = defaultdict(lambda: None)

# ✅ 사용할 수 있는 첫 번째 카메라 찾기 (맥 내장 웹캠용)
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

    # YOLO 추론
    results = model(frame, verbose=False)

    # 결과 분석
    for r in results:
        if r.keypoints is not None:
            keypoints_all = r.keypoints.xy  # 각 사람마다 관절 좌표 (N x 17 x 2)
            boxes = r.boxes.xyxy  # 바운딩 박스

            for i, keypoints_tensor in enumerate(keypoints_all):
                keypoints = keypoints_tensor.cpu().numpy()

                # 왼쪽 어깨(5), 왼쪽 손목(9), 오른쪽 어깨(6), 오른쪽 손목(10)
                left_shoulder = keypoints[5]
                left_wrist = keypoints[9]
                right_shoulder = keypoints[6]
                right_wrist = keypoints[10]

                # 팔을 든 조건: 손목 y좌표가 어깨보다 위에 있음 (y값은 작을수록 위)
                left_arm_up = left_wrist[1] < left_shoulder[1]
                right_arm_up = right_wrist[1] < right_shoulder[1]

                # 둘 중 하나라도 팔을 들고 있다면 → 타이머 시작
                if left_arm_up or right_arm_up:
                    if arm_up_start_time[i] is None:
                        arm_up_start_time[i] = time.time()
                    elif time.time() - arm_up_start_time[i] > 2:
                        # 3초 이상 팔을 들고 있음 → 바운딩 박스 표시
                        x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, "Arm Up > 3s", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # 팔을 내렸으면 시간 초기화
                    arm_up_start_time[i] = None

    # 프레임 보여주기
    cv2.imshow("YOLOv8 Arm Raise Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
