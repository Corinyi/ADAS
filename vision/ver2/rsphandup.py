import time
import cv2
from ultralytics import YOLO
from collections import defaultdict
from picamera2 import Picamera2

# ── 설정 ──
# 1) Picamera2 초기화 및 해상도 설정
picam2 = Picamera2()
video_config = picam2.create_video_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(video_config)
picam2.start()

# 2) YOLOv8-pose 모델 로드 (필요에 따라 device="cpu" 또는 "cuda:0" 지정)
model = YOLO("yolov8n-pose.pt", device="cpu")

# 3) 트래킹 상태 저장용
arm_up_start_time = defaultdict(lambda: None)
arm_up_confirmed  = defaultdict(lambda: False)
# ── 여기까지 설정 ──

try:
    while True:
        # 1) Picamera2로부터 RGB 프레임 획득
        frame_rgb = picam2.capture_array()
        # 2) BGR로 변환 (OpenCV는 BGR 입력)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 3) YOLO 추론
        results = model(frame, verbose=False)

        for r in results:
            if r.keypoints is None:
                continue

            keypoints_all = r.keypoints.xy
            boxes         = r.boxes.xyxy

            for i, kpt in enumerate(keypoints_all):
                kp = kpt.cpu().numpy()
                if len(kp) < 11:
                    continue

                # 어깨·손목·코 좌표
                ls, rs = kp[5], kp[6]
                lw, rw = kp[9], kp[10]
                nose    = kp[0]

                # 팔 들기 감지
                arm_up = (lw[1] < ls[1]) or (rw[1] < rs[1])
                if arm_up:
                    if arm_up_start_time[i] is None:
                        arm_up_start_time[i] = time.time()
                    elif not arm_up_confirmed[i] and time.time() - arm_up_start_time[i] > 3:
                        arm_up_confirmed[i] = True
                else:
                    arm_up_start_time[i] = None
                    arm_up_confirmed[i]  = False

                # 3초 이상 들고 있을 때만 트래킹
                if arm_up_confirmed[i]:
                    x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, "Tracking (Arm Up)", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    hx, hy = int(nose[0]), int(nose[1])
                    cv2.circle(frame, (hx, hy), 5, (255,0,0), -1)
                    cv2.putText(frame, f"Head: ({hx},{hy})", (hx+10, hy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

                    print(f"[ID {i}] Head Position: ({hx}, {hy})")

        # 4) 결과 화면 표시
        cv2.imshow("YOLOv8 3s Arm Up Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
