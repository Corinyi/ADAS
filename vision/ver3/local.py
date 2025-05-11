import cv2
from ultralytics import YOLO
from flask import Flask, Response
import socket
import time
from collections import defaultdict

# Flask 앱 설정
app = Flask(__name__)
model = YOLO('yolov8n-pose.pt')

# 라즈베리파이 IP 및 포트 설정 이때!!!!!!!!!!!! 라즈베리파이랑 노트북이랑 같은 공유기에 연결되어 있어야함
PI_IP = '172.30.1.56'  # 라즈베리 파이 IP 주소//예시로 설정해둔 아이피 주소는 연구실 주소임
PI_PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 사람별 추적 상태 기록
arm_up_start_time = defaultdict(lambda: None)
arm_up_confirmed = defaultdict(lambda: False)

# 영상 스트림 수신
cap = cv2.VideoCapture('http://172.30.1.56:5000/video_feed') #아이피 주소 자리에 공유기 아이피 주소 들어가야 함.

def gen_frames():
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

            for i, keypoints_tensor in enumerate(keypoints_all):
                keypoints = keypoints_tensor.cpu().numpy()
                if len(keypoints) < 11:
                    continue

                left_shoulder = keypoints[5]
                left_wrist = keypoints[9]
                right_shoulder = keypoints[6]
                right_wrist = keypoints[10]
                nose = keypoints[0]

                # 팔이 올라가 있는지 판단
                left_arm_up = left_wrist[1] < left_shoulder[1]
                right_arm_up = right_wrist[1] < right_shoulder[1]
                arm_up_now = left_arm_up or right_arm_up

                # 팔을 든 시간 기록 및 확인
                if arm_up_now:
                    if arm_up_start_time[i] is None:
                        arm_up_start_time[i] = time.time()
                    elif time.time() - arm_up_start_time[i] > 3:
                        arm_up_confirmed[i] = True
                else:
                    arm_up_start_time[i] = None
                    arm_up_confirmed[i] = False  #내리면 즉시 트래킹 종료

                # 박스 좌표
                x1, y1, x2, y2 = map(int, boxes[i].cpu().numpy())

                if arm_up_confirmed[i]:
                    # 팔을 3초 이상 든 경우 → 전송
                    nose_x, nose_y = int(nose[0]), int(nose[1])
                    msg = f"{nose_x},{nose_y}".encode()
                    sock.sendto(msg, (PI_IP, PI_PORT))

                    # 시각화
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (nose_x, nose_y), 5, (255, 0, 0), -1)
                    cv2.putText(frame, f"Nose: ({nose_x},{nose_y})", (nose_x + 10, nose_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                    cv2.putText(frame, "Tracking (3s+)", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                elif arm_up_now:
                    # 아직 3초 안 된 경우
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    cv2.putText(frame, "Tracking...", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    # 트래킹 안 함
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (180, 180, 180), 1)
                    cv2.putText(frame, "Not Tracking", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # 프레임 인코딩 및 스트리밍
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 웹 라우팅
@app.route('/')
def index():
    return "<h2>YOLO Inference Stream</h2><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
