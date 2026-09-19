import time
import os
import psutil
import torch
import numpy as np
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

        cpu_before = process.cpu_percent(interval=None)
        ram_before = process.memory_info().rss / (1024 ** 2)

        start = time.time()
        results = model(
            img_path,
            conf=CONF_THRESH,
            device=device,
            save=True
        )
        end = time.time()

        cpu_after = process.cpu_percent(interval=None)
        ram_after = process.memory_info().rss / (1024 ** 2)

        times.append((end - start) * 1000)
        cpu_usages.append(cpu_after)
        ram_usages.append(ram_after)

        if device == "mps":
            gpu_memories.append(torch.mps.current_allocated_memory() / (1024 ** 2))
        elif torch.cuda.is_available():
            gpu_memories.append(torch.cuda.memory_allocated() / (1024 ** 2))

# ---------------- RESULTS ----------------
avg_time = np.mean(times)
fps = 1000 / avg_time
model_size = os.path.getsize(MODEL_PATH) / (1024 ** 2)

print("\n===== BENCHMARK RESULTS =====")
print("Average inference time (ms):", round(avg_time, 2))
print("FPS:", round(fps, 2))
print("Model size (MB):", round(model_size, 2))
print("Average CPU usage (%):", round(np.mean(cpu_usages), 2))
print("Average RAM usage (MB):", round(np.mean(ram_usages), 2))

if gpu_memories:
    print("Average GPU memory usage (MB):", round(np.mean(gpu_memories), 2))