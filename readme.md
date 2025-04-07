# Active Directional Alert System Using Computer Vision and Parabolic Reflection

This project presents an intelligent alert system that detects cheating behavior during exams and delivers focused warnings to specific individuals without disturbing others. Utilizing a parabolic reflector and computer vision, the system provides directional sound alerts controlled by servo motors.


## 🧭 Purpose

The system is designed to:
- Monitor exam environments for cheating behavior using computer vision.
- Deliver focused auditory warnings to specific individuals via a speaker positioned at the focus of a parabolic reflector.
- Minimize distraction for other examinees by providing targeted alerts only to detected individuals.
- Switch between directional and general announcements depending on the mode.

<p align="center">
  <img src="https://github.com/user-attachments/assets/6d7d48d6-cd15-41e1-b568-4ea6357bb149" alt="System Diagram" width="500"/>
</p>

## 🛠️ Development Tools

- **Hardware**
  - Raspberry Pi (main controller)
  - Camera module (for real-time video processing)
  - 2x Servo motors (for reflector direction control)
  - 1x Servo motor (for rotating the speaker)
  - Parabolic reflector
  - Directional speaker
  
- **Software**
  - Python (main control logic and CV algorithms)
  - OpenCV (for image processing and target detection)
  - GPIO / pigpio library (servo control)
  - Possibly TensorFlow Lite or YOLOv5 (for detection model, depending on performance needs)

## ✨ Features

- **Target Detection:** Detect cheating behavior using vision algorithms.
- **Directional Alert:** Focus sound using a speaker placed at the focal point of a parabolic reflector.
- **Servo Control:** Adjust reflector direction with 2 servo motors for precise targeting.
- **Dual Alert Mode:** 
  - *Directional Mode:* Alerts only the identified individual.
  - *General Mode:* Rotates speaker toward the classroom for general announcements.
- **Quiet Operation:** Designed to minimize distraction to other students.
- **Raspberry Pi Based:** Lightweight and portable system for exam supervision environments.

---

Feel free to contribute or suggest improvements. This project aims to make exam environments more fair and less disruptive.