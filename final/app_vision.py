#!/usr/bin/env python3
"""
=============================================================================
SMART CITY AI VISION SERVER: TRAFFIC & SMART WASTE CLASSIFIER
=============================================================================
Server Flask untuk 2 Dashboard AI Vision:
1. CCTV Lalu Lintas Jalan Raya (YouTube Live Stream + YOLO)
2. Smart Waste Sorting (Deteksi Sampah Organik vs Anorganik + Servo D10 & D11)
3. Sinkronisasi Event ke Firebase Realtime Database
=============================================================================
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
import torch
from ultralytics import YOLO

# Coba import streamlink & requests
try:
    from streamlink import Streamlink
    STREAMLINK_AVAILABLE = True
except ImportError:
    STREAMLINK_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Serial opsional jika ingin langsung mengirim perintah ke Arduino dari Python
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

app = Flask(__name__)

# Deteksi Akselerasi Hardware NVIDIA GPU (CUDA)
if torch.cuda.is_available():
    DEVICE = 0  # GPU CUDA ID 0 (NVIDIA RTX 2050)
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_VRAM = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    USE_HALF = True  # FP16 Precision (Akselerasi Tensor Core untuk GPU Ampere RTX 2050)
    # RTX 2050 punya 4GB VRAM — batasi fragmentasi dengan allocator efisien
    print("=" * 70)
    print(f"GPU AKTIF: {GPU_NAME}")
    print(f"VRAM: {GPU_VRAM:.1f} GB | Presisi: FP16 | CUDA: {torch.version.cuda}")
    print("=" * 70)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    DEVICE = "cpu"
    GPU_NAME = "CPU"
    USE_HALF = False
    print("=" * 70)
    print("CPU MODE — NVIDIA GPU (CUDA) tidak terdeteksi.")
    print("Untuk GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("=" * 70)

# Konfigurasi Model & Path:
# 1. CCTV Traffic: Prioritas yolov8s (lebih ringan dari yolov8m untuk RTX 2050)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "yolov8s.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "object detection", "yolov8s.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

print(f"[CCTV] Model: {os.path.basename(MODEL_PATH)} -> {GPU_NAME}")
yolo_model = YOLO(MODEL_PATH)
yolo_model.to(DEVICE)
if DEVICE != "cpu":
    try:
        yolo_model.model.half()
    except Exception:
        pass

# 2. Smart Waste: Gunakan yolov8n (nano) untuk latency minimum di webcam realtime
WASTE_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
if not os.path.exists(WASTE_MODEL_PATH):
    WASTE_MODEL_PATH = os.path.join(BASE_DIR, "object detection", "yolov8s.pt")
if not os.path.exists(WASTE_MODEL_PATH):
    WASTE_MODEL_PATH = MODEL_PATH

print(f"[WASTE] Model: {os.path.basename(WASTE_MODEL_PATH)} -> {GPU_NAME}")
yolo_waste_model = YOLO(WASTE_MODEL_PATH)
yolo_waste_model.to(DEVICE)
if DEVICE != "cpu":
    try:
        yolo_waste_model.model.half()
    except Exception:
        pass

# Warmup model dengan inference dummy agar graph CUDA terkompilasi sebelum frame pertama
print("Warmup model...")
_warmup_f = np.zeros((480, 640, 3), dtype=np.uint8)
yolo_model(_warmup_f, imgsz=640, device=DEVICE, verbose=False)
yolo_waste_model(_warmup_f, imgsz=416, device=DEVICE, verbose=False)
print("Model siap.")

# Link YouTube Live CCTV Jalan Raya
YOUTUBE_URL = "https://www.youtube.com/live/JJ3MWNYVCU4?si=00EZLlAVTAP0caTL"

# Klasifikasi Sampah Berdasarkan Kelas COCO
ORGANIC_CLASSES = {
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'potted plant', 'food'
}

INORGANIC_CLASSES = {
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'can', 'plastic', 'cell phone', 'remote', 'keyboard', 'mouse',
    'laptop', 'scissors', 'toothbrush', 'book', 'paper', 'cardboard',
    'box', 'tin can'
}

VEHICLE_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck'
}

# State Global
traffic_state = {
    "total_vehicles": 0,
    "cars": 0,
    "motorcycles": 0,
    "buses": 0,
    "trucks": 0,
    "density": "Lancar",
    "updated_at": "",
    "status": "Inisialisasi"
}

waste_state = {
    "latest_item": "Menunggu objek...",
    "category": "STANDBY",  # 'ORGANIK' | 'ANORGANIK' | 'STANDBY'
    "confidence": 0.0,
    "target_servo": "None", # 'D10' (Organik) | 'D11' (Anorganik)
    "history": [],
    "last_detection_time": 0
}

latest_traffic_frame = None
latest_waste_frame = None
frame_lock = threading.Lock()

# Firebase Helper
def push_event_to_firebase(path, payload):
    if not REQUESTS_AVAILABLE:
        return
    try:
        # Cek apakah ada firebase_config.json
        cfg_file = os.path.join(BASE_DIR, "firebase_config.json")
        db_url = None
        if os.path.exists(cfg_file):
            with open(cfg_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                db_url = cfg.get("databaseURL", "").rstrip("/")

        if db_url and "YOUR_PROJECT_ID" not in db_url:
            url = f"{db_url}/{path.lstrip('/')}.json"
            # POST membuat child baru dengan push ID; PUT akan menimpa seluruh histori.
            requests.post(url, json=payload, timeout=2)
    except Exception as e:
        pass

# =========================================================================
# THREAD 1: CCTV JALAN RAYA (YOUTUBE STREAMLINK + YOLO)
# =========================================================================
def traffic_worker():
    global latest_traffic_frame, traffic_state
    print("Thread CCTV lalu lintas dimulai...")

    while True:
        cap = None
        stream_url = None

        if STREAMLINK_AVAILABLE:
            try:
                session = Streamlink()
                streams = session.streams(YOUTUBE_URL)
                # Prioritaskan stream kualitas lebih tinggi untuk deteksi yang jelas
                for res in ['720p', '480p', 'best', '360p']:
                    if res in streams:
                        stream_url = streams[res].to_url()
                        print(f"Stream YouTube resolusi: {res}")
                        break
            except Exception as e:
                print(f"Gagal stream YouTube: {e}")

        if stream_url:
            cap = cv2.VideoCapture(stream_url)
            traffic_state["status"] = "Terhubung ke YouTube Live"
        else:
            traffic_state["status"] = "Mode Fallback Video Lalu Lintas"

        # Loop membaca frame
        frame_count = 0
        while cap and cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Stream frame terputus. Reconnect...")
                break

            frame_count += 1
            # Proses deteksi tiap 3 frame untuk menjaga 10-15 FPS streaming di RTX 2050
            if frame_count % 3 == 0:
                h, w = frame.shape[:2]
                # Resolusi inference 640px — cukup untuk deteksi kendaraan, ringan di VRAM
                inf_w = 640
                if w > inf_w:
                    inf_frame = cv2.resize(frame, (inf_w, int(h * (inf_w / w))))
                else:
                    inf_frame = frame

                # Inferensi langsung tanpa stream=True untuk hasil yang bisa di-plot segera
                results = yolo_model(inf_frame, classes=[0, 1, 2, 3, 5, 7], conf=0.25, iou=0.45, imgsz=640, device=DEVICE, verbose=False)
                
                annotated_frame = results[0].plot(line_width=1, font_size=8)
                # Resize kembali ke ukuran stream asli untuk tampilan
                if annotated_frame.shape[1] < w:
                    annotated_frame = cv2.resize(annotated_frame, (w, h))

                cars = 0
                motos = 0
                buses = 0
                trucks = 0

                vehicle_motor_boxes = []
                rider_boxes = []

                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls_id == 2:
                        cars += 1
                    elif cls_id == 3:
                        motos += 1
                        vehicle_motor_boxes.append((x1, y1, x2, y2))
                    elif cls_id == 5:
                        buses += 1
                    elif cls_id == 7:
                        trucks += 1
                    elif cls_id == 1:
                        # Sepeda / Roda Dua
                        motos += 1
                    elif cls_id == 0:
                        # Person di jalan raya (rider motor)
                        rider_boxes.append((x1, y1, x2, y2, conf))

                # Logika asosiasi cerdas:
                # Jika pengendara motor malam hari terdeteksi sebagai 'person' tanpa box motor di bawahnya, hitung sebagai motor
                for rx1, ry1, rx2, ry2, rconf in rider_boxes:
                    is_riding_existing = False
                    for mx1, my1, mx2, my2 in vehicle_motor_boxes:
                        inter_x1 = max(rx1, mx1)
                        inter_y1 = max(ry1, my1)
                        inter_x2 = min(rx2, mx2)
                        inter_y2 = min(ry2, my2)
                        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                            rider_area = (rx2 - rx1) * (ry2 - ry1)
                            if inter_area / max(1, rider_area) > 0.25:
                                is_riding_existing = True
                                break

                    if not is_riding_existing:
                        motos += 1
                        vehicle_motor_boxes.append((rx1, ry1, rx2, ry2))

                total = cars + motos + buses + trucks
                density = "Lancar"
                if total > 15: density = "Padat Merayap"
                elif total > 7: density = "Ramai Lancar"

                traffic_state["total_vehicles"] = total
                traffic_state["cars"] = cars
                traffic_state["motorcycles"] = motos
                traffic_state["buses"] = buses
                traffic_state["trucks"] = trucks
                traffic_state["density"] = density
                traffic_state["updated_at"] = datetime.now().strftime("%H:%M:%S")

                # HUD ringan — 1 rect + 2 text lines
                cv2.rectangle(annotated_frame, (10, 10), (300, 60), (10, 16, 26), -1)
                cv2.putText(annotated_frame, f"TOTAL: {total}  STATUS: {density}", (18, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (66, 184, 181), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f"M:{cars}  Mo:{motos}  B/T:{buses + trucks}", (18, 52),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 215, 225), 1, cv2.LINE_AA)

                with frame_lock:
                    latest_traffic_frame = annotated_frame.copy()

            time.sleep(0.015)

        if cap:
            cap.release()

        placeholder = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(placeholder, "CCTV RECONNECTING...", (180, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (66, 184, 181), 2)
        with frame_lock:
            latest_traffic_frame = placeholder
        time.sleep(3)


# =========================================================================
# THREADED CAMERA CAPTURE (MENGHILANGKAN BUFFER DELAY 100% PADA WINDOWS)
# =========================================================================
class DirectShowWebcam:
    """Threaded Webcam Reader untuk membuang antrian buffer driver Windows.
    Menghasilkan video real-time 0ms delay tanpa lag."""
    def __init__(self, src=0):
        # DirectShow (CAP_DSHOW) di Windows membuka webcam tanpa buffer latency
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.running = True
        self.frame = None
        self.lock = threading.Lock()
        
        if self.cap.isOpened():
            ret, f = self.cap.read()
            if ret:
                self.frame = f
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()

    def _reader(self):
        while self.running:
            ret, f = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = f
            time.sleep(0.005) # Menguras buffer driver secara real-time

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def is_opened(self):
        return self.cap.isOpened() if self.cap else False

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()


# =========================================================================
# THREAD 2: DETEKSI SAMPAH ULTRA-FAST (0-DELAY REALTIME)
# =========================================================================
def waste_worker():
    global latest_waste_frame, waste_state
    print("Thread deteksi sampah dimulai...")

    cam = DirectShowWebcam(0)
    use_webcam = cam.is_opened()
    if use_webcam:
        print("Webcam terhubung (DirectShow, zero buffer delay)")
    else:
        print("Webcam tidak terdeteksi — mode fallback gambar sampel.")

    sample_bottle_path = os.path.join(BASE_DIR, "sampah", "recyclable", "botol.jpg")
    sample_img = None
    if os.path.exists(sample_bottle_path):
        sample_img = cv2.imread(sample_bottle_path)

    cached_frame = None

    while True:
        frame = cam.read() if use_webcam else None

        if frame is None:
            if sample_img is not None:
                frame = sample_img.copy()
            else:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(frame, (120, 160), (520, 320), (35, 42, 54), -1)
                cv2.putText(frame, "KAMERA SAMPAH SIAP", (150, 245),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (66, 184, 181), 1)

        # Inferensi tiap frame dengan imgsz=416 untuk sub-20ms di RTX 2050
        h, w = frame.shape[:2]
        inf_frame = frame
        if w > 640:
            inf_frame = cv2.resize(frame, (640, int(h * (640 / w))))

        results = yolo_waste_model(inf_frame, conf=0.28, imgsz=416, device=DEVICE, verbose=False)
        annotated_frame = results[0].plot(line_width=1, font_size=7)
        
        detected_category = None
        detected_item_name = None
        detected_conf = 0.0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = yolo_waste_model.names[cls_id].lower()
            conf = float(box.conf[0])

            if name in ORGANIC_CLASSES:
                detected_category = "ORGANIK"
                detected_item_name = name
                detected_conf = conf
                break
            elif name in INORGANIC_CLASSES:
                detected_category = "ANORGANIK"
                detected_item_name = name
                detected_conf = conf
                break

        now = time.time()
        if detected_category and (now - waste_state["last_detection_time"] > 3.0):
            waste_state["last_detection_time"] = now
            waste_state["category"] = detected_category
            waste_state["latest_item"] = detected_item_name
            waste_state["confidence"] = round(detected_conf, 2)
            waste_state["target_servo"] = "D10" if detected_category == "ORGANIK" else "D11"

            event_record = {
                "item": detected_item_name,
                "category": detected_category,
                "confidence": round(detected_conf * 100, 1),
                "servo": waste_state["target_servo"],
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
            waste_state["history"].insert(0, event_record)
            if len(waste_state["history"]) > 20:
                waste_state["history"].pop()

            print(f"[WASTE] {detected_item_name} -> {detected_category} | Servo {waste_state['target_servo']}")

            push_event_to_firebase("monitoring/waste_events", event_record)

        hud_bg = (0, 100, 0) if waste_state["category"] == "ORGANIK" else ((120, 80, 0) if waste_state["category"] == "ANORGANIK" else (30, 30, 30))
        cv2.rectangle(annotated_frame, (10, 10), (280, 55), hud_bg, -1)
        cv2.putText(annotated_frame, f"{waste_state['category']} | Servo {waste_state['target_servo']}",
                    (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(annotated_frame, f"{waste_state['latest_item']}",
                    (18, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

        cached_frame = annotated_frame

        with frame_lock:
            latest_waste_frame = cached_frame

        time.sleep(0.005)


# =========================================================================
# FLASK STREAMING GENERATORS & ENDPOINTS
# =========================================================================
def generate_mjpeg(get_frame_fn):
    while True:
        with frame_lock:
            frame = get_frame_fn()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.015)

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    return response

@app.route('/video_feed/traffic')
def video_feed_traffic():
    """Streaming CCTV Jalan Raya dengan Deteksi Kendaraan YOLO"""
    return Response(generate_mjpeg(lambda: latest_traffic_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/waste')
def video_feed_waste():
    """Streaming Kamera Pemilah Sampah Organik & Anorganik"""
    return Response(generate_mjpeg(lambda: latest_waste_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/traffic/stats')
def api_traffic_stats():
    return jsonify(traffic_state)

@app.route('/api/waste/status')
def api_waste_status():
    return jsonify(waste_state)

@app.route('/api/waste/trigger/<category>', methods=['POST', 'GET'])
def api_waste_trigger(category):
    """Memicu deteksi sampah manual dari Web untuk pengujian servo"""
    cat = category.upper()
    if cat not in ['ORGANIK', 'ANORGANIK']:
        return jsonify({"error": "Kategori harus ORGANIK atau ANORGANIK"}), 400

    item_name = "Apel / Buah Organik" if cat == "ORGANIK" else "Botol Plastik Bekas"
    servo = "D10" if cat == "ORGANIK" else "D11"

    waste_state["last_detection_time"] = time.time()
    waste_state["category"] = cat
    waste_state["latest_item"] = item_name
    waste_state["confidence"] = 0.95
    waste_state["target_servo"] = servo

    event_record = {
        "item": item_name,
        "category": cat,
        "confidence": 95.0,
        "servo": servo,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    waste_state["history"].insert(0, event_record)
    if len(waste_state["history"]) > 20:
        waste_state["history"].pop()

    push_event_to_firebase("monitoring/waste_events", event_record)

    return jsonify({
        "status": "success",
        "category": cat,
        "item": item_name,
        "servo": servo,
        "command": f"SAMPAH:{cat}"
    })

@app.route('/api/status')
def api_status():
    return jsonify({
        "server": "Smart City AI Vision",
        "yolo_model": MODEL_PATH,
        "device": str(DEVICE),
        "device_name": GPU_NAME,
        "cuda_available": torch.cuda.is_available(),
        "half_precision": USE_HALF,
        "traffic_stream": YOUTUBE_URL,
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Jalankan background thread
    t_traffic = threading.Thread(target=traffic_worker, daemon=True)
    t_traffic.start()

    t_waste = threading.Thread(target=waste_worker, daemon=True)
    t_waste.start()

    print("AI Vision Server: http://localhost:5000")
    print("  CCTV:  /video_feed/traffic")
    print("  Waste:  /video_feed/waste")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
