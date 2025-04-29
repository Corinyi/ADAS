# server.py
import time
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO
from collections import defaultdict

app = Flask(__name__)

# YOLOv8-pose 모델 로드
model = YOLO("yolov8n-pose.pt")

# ID별 트래킹 상태 기록
arm_up_start_time = defaultdict(lambda: None)
arm_up_confirmed = defaultdict(lambda: False)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    # 수신한 파일 읽기
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    head_positions = []

    # YOLOv8 Pose Inference
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
                    head_x, head_y = int(nose[0]), int(nose[1])
                    head_positions.append({"id": i, "x": head_x, "y": head_y})

    # 좌표 반환
    return jsonify({"heads": head_positions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, threaded=True)
