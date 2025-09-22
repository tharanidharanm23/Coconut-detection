from flask import Flask, render_template, Response, jsonify
import cv2
import math
import numpy as np
from ultralytics import YOLO
import threading
import time
from datetime import datetime

app = Flask(__name__)

# Global variables
model = None
detection_active = False
detection_thread = None
current_frame = None
current_stats = {
    'total_coconuts': 0,
    'cracked_coconuts': 0,
    'healthy_coconuts': 0,
    'total_weight': 0.0,
    'session_start': None,
    'current_detections': []
}

# Coconut density (g/cm³)
density = 0.9  
pixels_per_cm = 15.0
pixel_to_cm = 1.0 / pixels_per_cm

def load_model():
    """Load the YOLO model"""
    global model
    try:
        model = YOLO("best.pt")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def estimate_weight(box):
    """Estimate coconut weight using bounding box dimensions."""
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    # Convert numpy types to Python native types
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    
    w = (x2 - x1) * pixel_to_cm
    h = (y2 - y1) * pixel_to_cm
    d = (w + h) / 2  # approximate depth

    # Ellipsoid radii
    a, b, c = w/2, h/2, d/2
    volume = (4/3) * math.pi * a * b * c   # cm³

    # Weight (kg)
    weight = float(volume * density / 1000)  # Ensure Python float

    # Center of bounding box
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    return weight, (cx, cy), (int(x1), int(y1), int(x2), int(y2))

def get_direction(cx, cy, frame_w, frame_h, tolerance=50):
    """Returns relative position of coconut compared to frame center."""
    offset_x = cx - frame_w // 2
    offset_y = cy - frame_h // 2

    if abs(offset_x) < tolerance and abs(offset_y) < tolerance:
        return "Center"
    elif abs(offset_x) > abs(offset_y):
        return "Right" if offset_x > 0 else "Left"
    else:
        return "Down" if offset_y > 0 else "Up"

def detection_loop():
    """Main detection loop that runs in a separate thread"""
    global detection_active, current_stats, current_frame
    
    cap = cv2.VideoCapture(0)
    
    while detection_active:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, conf=0.25)
        annotated = frame.copy()
        frame_h, frame_w = frame.shape[:2]
        
        frame_detections = []
        frame_total_weight = 0.0
        frame_coconuts = 0
        frame_cracked = 0

        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    conf = float(box.conf[0])

                    # Estimate weight + position
                    est_weight, (cx, cy), _ = estimate_weight(box)
                    pos_text = get_direction(cx, cy, frame_w, frame_h)

                    # Update counters
                    frame_coconuts += 1
                    frame_total_weight += est_weight
                    if label == "Cracked_Coconut":
                        frame_cracked += 1

                    # Store detection data
                    detection_data = {
                        'label': label,
                        'confidence': float(conf),
                        'weight': float(est_weight),
                        'position': pos_text,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'center': [int(cx), int(cy)]
                    }
                    frame_detections.append(detection_data)

                    # --- Display text ---
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 0.6
                    thickness = 2
                    line_height = 25

                    line1 = f"{label} ({conf:.2f})"
                    line2 = f"Weight: {est_weight:.2f} kg"
                    line3 = f"Position: {pos_text}"

                    color = (0, 255, 0) if label == "Coconut" else (0, 0, 255)

                    # Background rectangle
                    bg_y1 = y1 - 3 * line_height - 5
                    bg_y2 = y1
                    cv2.rectangle(annotated, (x1, bg_y1), (x1 + 250, bg_y2), (0, 0, 0), -1)

                    # Put text
                    cv2.putText(annotated, line1, (x1 + 5, y1 - 2 * line_height),
                                font, scale, color, thickness)
                    cv2.putText(annotated, line2, (x1 + 5, y1 - line_height),
                                font, scale, (255, 255, 0), thickness)
                    cv2.putText(annotated, line3, (x1 + 5, y1),
                                font, scale, (255, 200, 100), thickness)

                    # Draw center point
                    cv2.circle(annotated, (cx, cy), 5, (255, 0, 0), -1)

        # Update global stats (ensure all values are JSON serializable)
        current_stats['total_coconuts'] = int(max(current_stats['total_coconuts'], frame_coconuts))
        current_stats['cracked_coconuts'] = int(max(current_stats['cracked_coconuts'], frame_cracked))
        current_stats['healthy_coconuts'] = int(current_stats['total_coconuts'] - current_stats['cracked_coconuts'])
        current_stats['total_weight'] = float(max(current_stats['total_weight'], frame_total_weight))
        current_stats['current_detections'] = frame_detections

        # Store current frame
        _, buffer = cv2.imencode('.jpg', annotated)
        current_frame = buffer.tobytes()

        time.sleep(0.1)  # Control frame rate

    cap.release()

def generate_frames():
    """Generator function for video streaming"""
    global current_frame
    while detection_active:
        if current_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + current_frame + b'\r\n')
        time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active, detection_thread, current_stats
    
    if not detection_active:
        if not load_model():
            return jsonify({'error': 'Failed to load model'}), 500
            
        detection_active = True
        current_stats['session_start'] = datetime.now().isoformat()
        current_stats['total_coconuts'] = int(0)
        current_stats['cracked_coconuts'] = int(0)
        current_stats['healthy_coconuts'] = int(0)
        current_stats['total_weight'] = float(0.0)
        
        detection_thread = threading.Thread(target=detection_loop)
        detection_thread.start()
        
        return jsonify({'status': 'Detection started'})
    
    return jsonify({'status': 'Detection already running'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({'status': 'Detection stopped'})

@app.route('/get_stats')
def get_stats():
    return jsonify(current_stats)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)