# realtime_infer.py
import time
import threading
import queue
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from main_model import HiFuse_Small
  # giống test.py
import json
import os

# -------- CONFIG ----------
CAM_ID = 0
TARGET_FPS = 30
MODEL_PATH = "model_weight_backup/best_model.pth"
CLASS_IDX_PATH = "class_indices.json"
IMG_SIZE = 224
USE_GPU = torch.cuda.is_available()
DETECT_EVERY_N_FRAMES = 1  # detect every N frames (increase to 2-3 to save time)
# OpenCV DNN face detector files (Res10 SSD)
DEPLOY_PROTOTXT = "deploy.prototxt.txt"   # download from opencv repo if not present
RES10_CAFFEMODEL = "res10_300x300_ssd_iter_140000.caffemodel"

# -------- Helper: ensure detector files exist ----------
if not os.path.exists(DEPLOY_PROTOTXT) or not os.path.exists(RES10_CAFFEMODEL):
    print("Face detector model files not found. Download from OpenCV's GitHub and place:\n"
          " - deploy.prototxt.txt\n - res10_300x300_ssd_iter_140000.caffemodel")
    # We continue, but detection will fail if files missing.

# -------- Load class indices ----------
assert os.path.exists(CLASS_IDX_PATH), f"Missing {CLASS_IDX_PATH}"
with open(CLASS_IDX_PATH, "r") as f:
    class_indices = json.load(f)

# -------- Build model ----------
device = torch.device("cuda:0" if USE_GPU else "cpu")
print("Using device:", device)
num_classes = len(class_indices)
model = HiFuse_Small(num_classes=num_classes).to(device)
assert os.path.exists(MODEL_PATH), f"Model weights not found: {MODEL_PATH}"
state_dict = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(state_dict, strict=True)
print("✅ Loaded HiFuse_Small weights successfully!\n")

model.eval()
# optional: model.half()  # if you test fp16 later

# transforms (same as in your test/train)
data_transform = transforms.Compose([
    transforms.Resize(int(256)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -------- Face detector (OpenCV DNN) ----------
face_net = None
if os.path.exists(DEPLOY_PROTOTXT) and os.path.exists(RES10_CAFFEMODEL):
    face_net = cv2.dnn.readNetFromCaffe(DEPLOY_PROTOTXT, RES10_CAFFEMODEL)
    if USE_GPU:
        try:
            face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            print("OpenCV DNN using CUDA backend.")
        except Exception as e:
            print("Could not set CUDA backend for OpenCV DNN:", e)
else:
    print("Face detector not available; please add the caffemodel and prototxt.")

# -------- Threaded camera capture ----------
frame_queue = queue.Queue(maxsize=4)
stop_event = threading.Event()

def capture_thread():
    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    # optional: adjust frame size to reduce processing
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed")
            break
        # push latest frame (drop old if queue full)
        try:
            if frame_queue.full():
                _ = frame_queue.get_nowait()
            frame_queue.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()

# -------- Inference loop (main thread) ----------
def infer_loop():
    last_time = time.time()
    frame_count = 0
    fps = 0.0
    frame_idx = 0
    tracker = cv2.TrackerKCF_create() if cv2.__version__.startswith("4") else None
    tracking = False
    track_bbox = None

    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        frame_idx += 1
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = []
        # detection every N frames (optional)
        if (frame_idx % DETECT_EVERY_N_FRAMES) == 0 or not tracking or tracker is None:
            if face_net is not None:
                blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,
                                             (300,300), (104.0, 177.0, 123.0))
                face_net.setInput(blob)
                detections = face_net.forward()
                for i in range(detections.shape[2]):
                    conf = float(detections[0,0,i,2])
                    if conf > 0.5:
                        box = detections[0,0,i,3:7] * np.array([w, h, w, h])
                        (x1, y1, x2, y2) = box.astype("int")
                        x1 = max(0, x1); y1 = max(0,y1)
                        x2 = min(w-1, x2); y2 = min(h-1, y2)
                        faces.append((x1,y1,x2,y2))
                # optionally initialize tracker on first face
                if len(faces) > 0 and tracker is not None:
                    tracking = True
                    (x1,y1,x2,y2) = faces[0]
                    track_bbox = (x1, y1, x2-x1, y2-y1)
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, tuple(track_bbox))
            else:
                # fallback: no detector -> treat entire frame as one face
                faces = [(0,0,w-1,h-1)]
        else:
            # tracking path
            ok, bbox = tracker.update(frame)
            if ok:
                x, y, bw, bh = [int(v) for v in bbox]
                faces = [(x, y, x + bw, y + bh)]
            else:
                tracking = False
                faces = []

        # Prepare crops and batch them
        crops = []
        face_boxes = []
        for (x1,y1,x2,y2) in faces:
            if x2-x1 < 10 or y2-y1 < 10:
                continue
            crop = rgb_frame[y1:y2, x1:x2]
            pil = Image.fromarray(crop)
            inp = data_transform(pil)  # tensor C,H,W
            crops.append(inp)
            face_boxes.append((x1,y1,x2,y2))

        pred_labels = []
        pred_scores = []
        if len(crops) > 0:
            batch = torch.stack(crops, dim=0).to(device)
            with torch.no_grad():
                # optionally use amp:
                # with torch.cuda.amp.autocast(enabled=USE_GPU):
                outputs = model(batch)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                for i in range(len(preds)):
                    pred_labels.append(int(preds[i]))
                    pred_scores.append(float(probs[i, preds[i]]))

        # Draw results
        for i, box in enumerate(face_boxes):
            x1,y1,x2,y2 = box
            label = class_indices.get(str(pred_labels[i]), "unk") if i < len(pred_labels) else "unk"
            score = pred_scores[i] if i < len(pred_scores) else 0.0
            text = f"{label}: {score:.2f}"
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # FPS calc
        frame_count += 1
        if frame_count >= 5:
            now = time.time()
            fps = frame_count / (now - last_time)
            last_time = now
            frame_count = 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.imshow("Realtime Inference", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_event.set()
            break

    cv2.destroyAllWindows()

# -------- start threads ----------
t = threading.Thread(target=capture_thread, daemon=True)
t.start()

try:
    infer_loop()
finally:
    stop_event.set()
    t.join()
