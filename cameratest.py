import sys
sys.path.append('/usr/lib/python3/dist-packages')
from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "BGR888"
picam2.configure("preview")

picam2.start()
time.sleep(1)  # 카메라 워밍업 시간

while True:
    frame = picam2.capture_array()
    cv2.imshow("Camera Preview", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()