from flask import Flask, Response
from picamera2 import Picamera2
import socket
import threading
import cv2
import time

# ------------ Flask MJPEG 스트리밍 설정 ------------
app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)}))
picam2.start()
time.sleep(1)

def gen_frames():
    while True:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "<h2>Pi Camera Stream</h2><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------ UDP 좌표 수신 쓰레드 ------------
def receive_coords():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 9999))
    print("코 좌표 수신 대기 중")

    while True:
        data, addr = sock.recvfrom(1024)
        coords = data.decode().strip()

        try:
            head_x, head_y = map(int, coords.split(","))
            print(f"Head Position → head_x: {head_x}, head_y: {head_y}")
        except ValueError:
            print(f"좌표 파싱 실패: {coords}")

# ------------ 메인 실행 ------------
if __name__ == '__main__':
    threading.Thread(target=receive_coords, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
