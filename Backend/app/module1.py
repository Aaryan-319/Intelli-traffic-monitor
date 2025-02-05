import cv2
import time
import numpy as np
import pandas as pd
import pytesseract
import uuid
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from ultralytics import YOLO
from collections import deque
from flask_cors import CORS

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

model = YOLO("yolov8n.pt")

rider_ids = {}
rider_speeds = {}
total_vehicles = 0
total_violations = 0
violation_data = {}

SPEED_LIMIT = 50 
KNOWN_DISTANCE = 5 
FRAME_SKIP = 2  

frame_queue = deque(maxlen=5)

COLORS = {
    "Rider": (0, 255, 0),
    "Helmet": (255, 0, 0),
    "Motorcycle": (0, 255, 255),
    "Number Plate": (255, 255, 255),
    "Violation": (0, 0, 255),
}

def get_centroid(box):
    """Returns the centroid of a bounding box (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2

def detect_objects(frame):
    """Detects objects and applies bounding boxes, speed detection, and helmet verification."""
    global total_violations, total_vehicles

    frame = cv2.flip(frame, 1)  
    results = model.predict(frame, verbose=False)

    motorcycles, riders, helmets, plates = [], [], [], []
    current_riders = {}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])

            if cls == 0:  
                riders.append((x1, y1, x2, y2))
            elif cls == 1:  
                helmets.append((x1, y1, x2, y2))
            elif cls == 3:  
                motorcycles.append((x1, y1, x2, y2))
            elif cls == 2:  
                plates.append((x1, y1, x2, y2))

    for rider in riders:
        x1, y1, x2, y2 = rider
        centroid = get_centroid(rider)

        rider_id = None
        for existing_id, past_centroid in rider_ids.items():
            if np.linalg.norm(np.array(centroid) - np.array(past_centroid)) < 50:  
                rider_id = existing_id
                break

        if not rider_id:
            rider_id = f"R-{uuid.uuid4().hex[:6]}"
            rider_ids[rider_id] = centroid
            total_vehicles += 1
            socketio.emit('new_vehicle', {'count': total_vehicles})

        helmet_found = any(
            hx1 <= x1 <= hx2 and hy1 <= y1 <= hy2 for (hx1, hy1, hx2, hy2) in helmets
        )

        if not helmet_found:
            if rider_id not in violation_data:
                log_violation(rider_id, "No Helmet")
                cv2.putText(frame, "VIOLATION: NO HELMET", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["Violation"], 2)

        if rider_id in rider_speeds:
            prev_x, prev_y, prev_time = rider_speeds[rider_id]
            distance_traveled = np.linalg.norm(np.array([centroid[0], centroid[1]]) - np.array([prev_x, prev_y]))
            time_elapsed = time.time() - prev_time

            if time_elapsed > 0:
                speed = (distance_traveled / KNOWN_DISTANCE) * 3.6
                if speed > SPEED_LIMIT:
                    log_violation(rider_id, f"Speeding ({speed:.2f} km/h)")
                    cv2.putText(frame, f"VIOLATION: SPEED {speed:.2f} km/h", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["Violation"], 2)

            rider_speeds[rider_id] = (centroid[0], centroid[1], time.time())
        else:
            rider_speeds[rider_id] = (centroid[0], centroid[1], time.time())

    return frame

def log_violation(rider_id, violation_type):
    """Logs violations to CSV and updates SocketIO clients."""
    global total_violations
    total_violations += 1
    violation_data[rider_id] = violation_type
    df = pd.DataFrame(list(violation_data.items()), columns=["Rider ID", "Violation"])
    df.to_csv("violations_log.csv", index=False)
    socketio.emit('new_violation', {
        'rider_id': rider_id,
        'violation': violation_type,
        'total_violations': total_violations
    })
def generate_frames():
    """Reads frames, applies object detection, and streams to frontend."""
    VIDEO_PATH = "/Users/mac_surf/VS-code/ano/Traffic_management/References/traffic_video.mp4"
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FPS, 30)  

    frame_count = 0  

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue

        frame = detect_objects(frame)

        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/clear_log', methods=['POST'])
def clear_log():
    """Clears the violation log file and resets stored data."""
    global total_violations, violation_data

    violation_data.clear()
    total_violations = 0

    df = pd.DataFrame(columns=["Rider ID", "Violation"])
    df.to_csv("violations_log.csv", index=False)

    socketio.emit('clear_log', {})

    return {"success": True}, 200  

CORS(app, origins="http://127.0.0.1:3000")

if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)