# raspberry_pi_client.py
import time
import cv2
import requests
from picamera2 import Picamera2

# 서버 주소
SERVER_URL = "http://172.20.10.4:8000/upload_frame"  # 여기에 서버 IP 넣기

# Picamera2 초기화
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
)
picam2.configure(preview_config)
picam2.start()
time.sleep(1)  # 카메라 워밍업

while True:
    frame = picam2.capture_array()

    # JPEG로 인코딩
    _, jpeg = cv2.imencode('.jpg', frame)

    try:
        # 서버로 프레임 전송
        response = requests.post(
            SERVER_URL,
            files={"frame": jpeg.tobytes()},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            heads = data.get('heads', [])

            # 받아온 머리 좌표 출력
            for head in heads:
                print(f"[ID {head['id']}] Head Position: ({head['x']}, {head['y']})")
        else:
            print("서버 응답 에러:", response.status_code)
    except requests.exceptions.RequestException as e:
        print("요청 에러:", e)

    # 너무 자주 보내면 부하 걸리니까 살짝 쉬기
    time.sleep(0.1)
