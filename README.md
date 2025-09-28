# 🎓 Student Monitoring System  
*AI-powered classroom and exam surveillance tool using YOLOv8 and MediaPipe*

---

## 📌 Project Overview
This project is an **AI-powered student monitoring system** that helps institutions detect and prevent distractions or cheating inside classrooms and examination halls.

It operates in two modes:
1. **Classroom Mode** → Detects students using mobile phones during lectures.  
2. **Exam Mode** → Detects students cheating by looking at other students’ papers through suspicious head movements.  

The system uses **YOLOv8 (Ultralytics)** for phone detection and **MediaPipe Face Mesh** for analyzing head pose, combined with **OpenCV** for real-time video processing.

---

## 🚀 Features
- ✅ Dual Operation Modes: *Classroom* / *Exam*  
- ✅ Detects and flags **mobile phone usage** in real time  
- ✅ Detects **suspicious head turns** (cheating attempts)  
- ✅ Supports **multiple students simultaneously**  
- ✅ Automatically **captures snapshots** of suspicious activity  
- ✅ Stores **event logs** with timestamps for later review  
- ✅ Simple setup, runs on any laptop with a webcam  

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) → Mobile phone detection  
- [MediaPipe](https://developers.google.com/mediapipe) → Face mesh & head pose estimation  
- [OpenCV](https://opencv.org/) → Video stream processing, bounding boxes & visualization  

🔮 Future Plans

Deploy on Raspberry Pi 4 or Jetson Nano for real hardware implementation
Add a web dashboard to monitor classrooms remotely
Enable cloud sync / Telegram alerts for instant notifications
Train a custom YOLO model for improved accuracy in real-world classroom environments

📜 License
This project is licensed under the MIT License – you are free to use, modify, and distribute it.

