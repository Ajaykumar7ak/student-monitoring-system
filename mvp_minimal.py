import cv2, time, os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from datetime import datetime

# Ask for mode before starting
print("Select mode:")
print("1. Classroom Mode (phone detection only)")
print("2. Exam Mode (cheating detection with head pose)")
mode = input("Enter 1 or 2: ").strip()

CLASSROOM_MODE = (mode == "1")

# --- models ---
yolo = YOLO("yolov8n.pt")  # light model
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=6, min_detection_confidence=0.5)

# for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),       # nose tip
    (0.0, -63.6, -12.5),   # chin
    (-43.3, 32.7, -26.0),  # left eye left corner
    (43.3, 32.7, -26.0),   # right eye right corner
    (-28.9, -28.9, -24.1), # left mouth corner
    (28.9, -28.9, -24.1)   # right mouth corner
], dtype=np.float32)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    raise SystemExit

suspicion_time = {}
FPS = 30.0

# folder for captures
os.makedirs("captures", exist_ok=True)
last_capture_time = 0

def estimate_head_yaw(landmarks, w, h):
    pts = np.array([
        (landmarks[1][0]*w, landmarks[1][1]*h),
        (landmarks[152][0]*w, landmarks[152][1]*h),
        (landmarks[33][0]*w, landmarks[33][1]*h),
        (landmarks[263][0]*w, landmarks[263][1]*h),
        (landmarks[61][0]*w, landmarks[61][1]*h),
        (landmarks[291][0]*w, landmarks[291][1]*h)
    ], dtype=np.float32)
    focal = w
    center = (w/2, h/2)
    camera_matrix = np.array([[focal, 0, center[0]],
                              [0, focal, center[1]],
                              [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4,1))
    ok, rvec, tvec = cv2.solvePnP(model_points, pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok: return 0,0,0
    rmat, _ = cv2.Rodrigues(rvec)
    pose = cv2.hconcat((rmat, tvec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(pose)
    return float(euler[1][0]), float(euler[0][0]), float(euler[2][0])  # yaw, pitch, roll

print(f"Running in {'Classroom' if CLASSROOM_MODE else 'Exam'} Mode. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    phone_present = False

    # --- Phone detection ---
    yres = yolo(frame, imgsz=640, conf=0.35)[0]
    for box in yres.boxes:
        cls = int(box.cls[0])
        name = yolo.names.get(cls, str(cls)).lower()
        if 'phone' in name or 'cell phone' in name:
            phone_present = True
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame, f"{name}", (x1, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)

    # --- Capture evidence if phone detected ---
    if CLASSROOM_MODE and phone_present:
        now = time.time()
        if now - last_capture_time > 3:  # save once every 3 sec max
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captures/phone_{ts}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[Captured] {filename}")
            last_capture_time = now

    # --- Exam mode (head pose) ---
    if not CLASSROOM_MODE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if res.multi_face_landmarks:
            for fid, fm in enumerate(res.multi_face_landmarks):
                lm = [(p.x, p.y) for p in fm.landmark]
                try:
                    yaw, pitch, roll = estimate_head_yaw(lm, w, h)
                except:
                    continue
                sideways = abs(yaw) > 18
                suspicion_time[fid] = suspicion_time.get(fid, 0.0) + (1/FPS if sideways else -1/FPS)
                suspicion_time[fid] = max(0, suspicion_time[fid])

                color = (0,0,255) if sideways else (255,255,0)
                cv2.putText(frame, f"Face{fid} Yaw:{yaw:.1f}", (10, 30+20*fid), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if suspicion_time[fid] > 1.5:
                    cv2.putText(frame, "SUSPICIOUS!", (10, 60+20*fid), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    # Overlay phone status
    if CLASSROOM_MODE:
        cv2.putText(frame, f"Phone: {'Yes' if phone_present else 'No'}", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0) if phone_present else (0,255,255), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
