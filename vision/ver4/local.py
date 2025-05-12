import cv2
from ultralytics import YOLO
from flask import Flask, Response
import socket
import time
from collections import defaultdict

# Flask 앱 설정
app = Flask(__name__)
model = YOLO('yolov8n-pose.pt')

# 라즈베리파이 IP 및 포트 설정
PI_IP = '172.30.1.56'
PI_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 사람별 상태 기록
arm_up_start_time = defaultdict(lambda: None)
arm_up_confirmed = defaultdict(lambda: False)
tracked_id = None  # 현재 트래킹 대상 인덱스

# 영상 스트림 수신
cap = cv2.VideoCapture('http://172.30.1.56:5000/video_feed')

def gen_frames():
    global tracked_id

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)

        for r in results:
            if r.keypoints is None:
                continue

            keypoints_all = r.keypoints.xy
            boxes = r.boxes.xyxy

            if tracked_id is not None and tracked_id >= len(keypoints_all):
                tracked_id = None  # 유효하지 않은 ID면 초기화

            # 트래킹 대상이 없을 때만 선택 로직 수행
            if tracked_id is None:
                max_area = 0
                selected_id = None

                for i, keypoints_tensor in enumerate(keypoints_all):
                    keypoints = keypoints_tensor.cpu().numpy()
                    if len(keypoints) < 11:
                        continue

                    left_shoulder = keypoints[5]
                    left_wrist = keypoints[9]
                    right_shoulder = keypoints[6]
                    right_wrist = keypoints[10]

                    left_arm_up = left_wrist[1] < left_shoulder[1]
                    right_arm_up = right_wrist[1] < right_shoulder[1]
                    arm_up_now = left_arm_up or right_arm_up

                    if not arm_up_now:
                        continue

                    x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())
                    area = (x2 - x1) * (y2 - y1)

                    if area > max_area:
                        max_area = area
                        selected_id = i

                if selected_id is not None:
                    tracked_id = selected_id

            # 선택된 사람만 처리
            for i, keypoints_tensor in enumerate(keypoints_all):
                if i != tracked_id:
                    continue

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
                    elif time.time() - arm_up_start_time[i] > 3:
                        arm_up_confirmed[i] = True
                else:
                    arm_up_start_time[i] = None
                    arm_up_confirmed[i] = False
                    tracked_id = None  # 트래킹 해제

                x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())

                if arm_up_confirmed[i]:
                    nose_x, nose_y = int(nose[0]), int(nose[1])
                    msg = f"{nose_x},{nose_y}".encode()
                    sock.sendto(msg, (PI_IP, PI_PORT))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (nose_x, nose_y), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"Nose: ({nose_x},{nose_y})", (nose_x + 10, nose_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(frame, "Tracking (3s+)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif arm_up_now:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    cv2.putText(frame, "Tracking...", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "<h2>YOLO Inference Stream</h2><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
