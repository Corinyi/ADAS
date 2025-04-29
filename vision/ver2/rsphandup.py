import time
import cv2
from ultralytics import YOLO
from collections import defaultdict
from picamera2 import Picamera2

# ── 설정 영역 ──
# 1) Picamera2 초기화 및 해상도 설정
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(video_config)
picam2.start()

# 2) YOLO 모델 로드 (필요시 device="cpu" 명시)
model = YOLO("yolov8n-pose.pt")  

# 3) 트래킹 상태 저장용
arm_up_start_time = defaultdict(lambda: None)
arm_up_confirmed = defaultdict(lambda: False)
# ── 여기까지 설정 ──

try:
    while True:
        # Picamera2로부터 프레임 획득 (RGB 배열)
        frame_rgb = picam2.capture_array()

        # YOLO가 BGR 주문이기 때문에 변환
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 추론
        results = model(frame, verbose=False)

        for r in results:
            if r.keypoints is None:
                continue

            keypoints_all = r.keypoints.xy
            boxes = r.boxes.xyxy

            for i, keypoints_tensor in enumerate(keypoints_all):
                keypoints = keypoints_tensor.cpu().numpy()
                if len(keypoints) < 11:
                    continue

                # 관절 좌표 추출
                left_shoulder = keypoints[5]
                left_wrist    = keypoints[9]
                right_shoulder= keypoints[6]
                right_wrist   = keypoints[10]
                nose          = keypoints[0]

                # 3초 이상 팔 들기 감지 로직
                arm_up_now = (left_wrist[1] < left_shoulder[1]) or (right_wrist[1] < right_shoulder[1])
                if arm_up_now:
                    if arm_up_start_time[i] is None:
                        arm_up_start_time[i] = time.time()
                    elif not arm_up_confirmed[i] and time.time() - arm_up_start_time[i] > 3:
                        arm_up_confirmed[i] = True
                else:
                    arm_up_start_time[i] = None
                    arm_up_confirmed[i] = False

                # 트래킹 중인 사람만 박스+머리 표시
                if arm_up_confirmed[i]:
                    x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "Tracking (Arm Up)", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    head_x, head_y = int(nose[0]), int(nose[1])
                    cv2.circle(frame, (head_x, head_y), 5, (255,0,0), -1)
                    cv2.putText(frame, f"Head: ({head_x},{head_y})", (head_x+10, head_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                    print(f"[ID {i}] Head Position: ({head_x}, {head_y})")

        cv2.imshow("YOLOv8 3s Arm Up Tracking on Pi5", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
