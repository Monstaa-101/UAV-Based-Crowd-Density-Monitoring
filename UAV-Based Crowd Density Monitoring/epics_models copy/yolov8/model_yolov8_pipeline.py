import time
import os
import psutil
import torch
import numpy as np
import json
from datetime import datetime
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8n.pt"
IMAGE_DIR = "../images"
OUTPUT_DIR = "outputs"
CONF_THRESH = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- DEVICE ----------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

model = YOLO(MODEL_PATH)
model.to(device)

process = psutil.Process(os.getpid())

times = []
cpu_usages = []
ram_usages = []
gpu_memories = []

# ---------------- INFERENCE ----------------
for img in os.listdir(IMAGE_DIR):
    if img.endswith((".jpg", ".png")):
        img_path = os.path.join(IMAGE_DIR, img)

        start = time.time()

        results = model(
            img_path,
            conf=CONF_THRESH,
            device=device,
            save=True
        )

        end = time.time()

        times.append((end - start) * 1000)

        # -------- COUNT PEOPLE --------
        boxes = results[0].boxes
        count = len(boxes) if boxes is not None else 0

        # -------- DENSITY LOGIC --------
        if count > 50:
            density = "High"
        elif count > 20:
            density = "Medium"
        else:
            density = "Low"

        print(f"{img} → Count: {count} → Density: {density}")

        # -------- STRUCTURED DATA --------
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "zone_id": "Zone_1",
            "people_count": count,
            "density_level": density
        }

        print("Prepared data:", data)

# ---------------- RESULTS ----------------
avg_time = np.mean(times)
fps = 1000 / avg_time
model_size = os.path.getsize(MODEL_PATH) / (1024 ** 2)

print("\n===== BENCHMARK RESULTS =====")
print("Average inference time (ms):", round(avg_time, 2))
print("FPS:", round(fps, 2))
print("Model size (MB):", round(model_size, 2))