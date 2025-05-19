import os
import subprocess
from flask import Flask, Response
from picamera2 import Picamera2
import socket
import threading
import cv2
import time
import pigpio
subprocess.run(["sudo", "pigpiod"], check=True)
time.sleep(0.5)  # 데몬이 뜨는 시간 약간 기다리기

# ------------ 설정값 ------------
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_X = FRAME_WIDTH // 2  # 320
CENTER_Y = FRAME_HEIGHT // 2  # 240

Pitch_GPIO = 19  # DS3240 (pitch)
Yaw_GPIO = 16    # MG996R (yaw)

# PID 파라미터
Kp_pitch, Ki_pitch, Kd_pitch = 0.2, 0.1, 0.1
Kp_yaw, Ki_yaw, Kd_yaw = 0.2, 0.1, 0.1
MAX_CTRL_PITCH = 2.0
MAX_CTRL_YAW = 2.0

# ------------ 초기화 ------------
pi = pigpio.pi()

current_pitch = 90.0
current_yaw = 90.0

integral_pitch = 0.0
integral_yaw = 0.0
last_error_pitch = 0.0
last_error_yaw = 0.0

target_dx = 0  # head_x - center_x
target_dy = 0  # head_y - center_y

# ------------ 함수 정의 ------------
def angle_to_pulse(deg):
    return 600 + 10 * deg

def set_servo(pitch, yaw):
    pi.set_servo_pulsewidth(Pitch_GPIO, angle_to_pulse(pitch))
    pi.set_servo_pulsewidth(Yaw_GPIO, angle_to_pulse(yaw))

def pid_control(current_error, integral, last_error, Kp, Ki, Kd, max_ctrl):
    error = current_error
    integral += error
    derivative = error - last_error
    control = Kp * error + Ki * integral + Kd * derivative
    control = max(-max_ctrl, min(max_ctrl, control))
    return control, integral, error

# ------------ Flask MJPEG 스트리밍 ------------
app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": "RGB888", "size": (FRAME_WIDTH, FRAME_HEIGHT)}))
picam2.start()
time.sleep(1)

def gen_frames():
    while True:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "<h2>Pi Camera Stream</h2><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ------------ UDP 좌표 수신 쓰레드 ------------
def receive_coords():
    global target_dx, target_dy
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 9999))
    print("코 좌표 수신 대기 중...")

    while True:
        data, _ = sock.recvfrom(1024)
        coords = data.decode().strip()
        try:
            head_x, head_y = map(int, coords.split(","))
            target_dx = head_x - CENTER_X
            target_dy = head_y - CENTER_Y
            print(f"dx: {target_dx}, dy: {target_dy}")
        except ValueError:
            print(f"좌표 파싱 실패: {coords}")

# ------------ PID 서보 제어 쓰레드 ------------
def pid_servo_loop():
    global current_pitch, current_yaw
    global integral_pitch, integral_yaw
    global last_error_pitch, last_error_yaw

    while True:
        # 입력값 정규화: 화면 중심 대비 0~±0.5
        norm_dy = target_dy / FRAME_HEIGHT
        norm_dx = target_dx / FRAME_WIDTH
        # PID 제어: 오차 = 정규화된 위치 오차
        control_pitch, integral_pitch, last_error_pitch = pid_control(
            norm_dy, integral_pitch, last_error_pitch,
            Kp_pitch, Ki_pitch, Kd_pitch, MAX_CTRL_PITCH
        )
        control_yaw, integral_yaw, last_error_yaw = pid_control(
            norm_dx, integral_yaw, last_error_yaw,
            Kp_yaw, Ki_yaw, Kd_yaw, MAX_CTRL_YAW
        )
        current_pitch -= control_pitch
        current_yaw += control_yaw

        current_pitch = max(0, min(180, current_pitch))
        current_yaw = max(0, min(180, current_yaw))

        set_servo(current_pitch, current_yaw)
        time.sleep(0.005)

# ------------ 메인 실행 ------------
if __name__ == '__main__':
    pi.set_servo_pulsewidth(Pitch_GPIO, 0)
    pi.set_servo_pulsewidth(Yaw_GPIO, 0)

    threading.Thread(target=receive_coords, daemon=True).start()
    threading.Thread(target=pid_servo_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)