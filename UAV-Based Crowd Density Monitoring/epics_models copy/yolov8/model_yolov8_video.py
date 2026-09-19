import cv2
import time
import json
from datetime import datetime
import torch
from ultralytics import YOLO
from AWSIoTPythonSDK.MQTTLib import AWSIoTMQTTClient

# -------- CONFIG --------
MODEL_PATH = "../yolov8n.pt"
VIDEO_PATH = "../video.mp4"
CONF_THRESH = 0.05

# -------- AWS CONFIG (FILL THESE) --------
ENDPOINT = "a5vzxypita2v-ats.iot.ap-south-1.amazonaws.com"
ROOT_CA = "rootCA.pem"
PRIVATE_KEY = "private.key"
CERTIFICATE = "certificate.pem"

# -------- DEVICE --------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

model = YOLO(MODEL_PATH)
model.to(device)

# -------- MQTT SETUP --------
client = AWSIoTMQTTClient("UAVDevice")
client.configureEndpoint(ENDPOINT, 8883)
client.configureCredentials(ROOT_CA, PRIVATE_KEY, CERTIFICATE)

client.connect()
print("Connected to AWS IoT")

# -------- VIDEO --------
cap = cv2.VideoCapture(VIDEO_PATH)

# -------- SEND CONTROL --------
last_sent_time = 0
SEND_INTERVAL = 10  # seconds

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    results = model(frame, conf=CONF_THRESH, device=device, imgsz=1280)

    end = time.time()
    latency = (end - start) * 1000

    # -------- COUNT --------
    boxes = results[0].boxes
    count = len(boxes) if boxes is not None else 0

    # -------- DENSITY --------
    if count > 50:
        density = "High"
    elif count > 20:
        density = "Medium"
    else:
        density = "Low"

    # -------- DRAW --------
    annotated_frame = results[0].plot()

    # Count
    cv2.rectangle(annotated_frame, (10, 10), (300, 50), (50, 50, 50), -1)
    cv2.putText(annotated_frame, f"Count: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Density color
    if density == "High":
        box_color = (0, 0, 255)
    elif density == "Medium":
        box_color = (0, 165, 255)
    else:
        box_color = (0, 150, 0)

    cv2.rectangle(annotated_frame, (10, 60), (300, 100), box_color, -1)
    cv2.putText(annotated_frame, f"Density: {density}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Latency
    cv2.rectangle(annotated_frame, (10, 110), (300, 150), (50, 50, 50), -1)
    cv2.putText(annotated_frame, f"Latency: {latency:.2f} ms", (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Title
    cv2.rectangle(annotated_frame, (10, 160), (500, 200), (50, 50, 50), -1)
    cv2.putText(annotated_frame, "UAV Crowd Monitoring System", (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # -------- DATA --------
    data = {
        "timestamp": datetime.utcnow().isoformat(),
        "zone_id": "Zone_1",
        "people_count": count,
        "density_level": density
    }

    # -------- SEND TO CLOUD (CONTROLLED) --------
    current_time = time.time()

    if current_time - last_sent_time > SEND_INTERVAL:
        client.publish("crowd/data", json.dumps(data), 1)
        print("Sent to cloud:", data)
        last_sent_time = current_time

    # -------- SHOW --------
    cv2.imshow("Crowd Monitoring", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()