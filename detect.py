import argparse
import time
from pathlib import Path
from collections import Counter, deque

import cv2
import torch
from ultralytics import YOLO

# CONFIG
DEFAULT_WEIGHTS = "runs/train/sign_detection/weights/best.pt"
CONF_THRESHOLD  = 0.5
IOU_THRESHOLD   = 0.45
DEVICE          = 0 if torch.cuda.is_available() else "cpu"

TTS_COOLDOWN = 2.0
history = deque(maxlen=10)
 
# INIT TTS
try:
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty("rate", 160)
except:
    engine = None


def speak(text):
    if engine:
        engine.say(text)
        engine.runAndWait()


# DRAW BOXES
def draw_boxes(frame, results, names):
    detected = []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        cls = int(box.cls[0])
        label = names[cls]

        # Clean label formatting
        label = label.lower().capitalize()

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        detected.append(label)

    return frame, detected


# SMOOTHING
def get_stable_prediction(detected):
    if detected:
        history.extend(detected)

    if history:
        return Counter(history).most_common(1)[0][0]
    return None


# WEBCAM MODE
def run_webcam(model):
    cap = cv2.VideoCapture(0)

    last_spoken = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF_THRESHOLD,
                                iou=IOU_THRESHOLD, device=DEVICE, verbose=False)

        frame, detected = draw_boxes(frame, results, model.names)
        stable = get_stable_prediction(detected)

        # FPS calculation
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        cv2.putText(frame, f"FPS: {fps:.2f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if stable:
            cv2.putText(frame, f"Prediction: {stable}", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        cv2.imshow("Sign Detection", frame)

        # Speak stable prediction
        if stable:
            now = time.time()
            if now - last_spoken > TTS_COOLDOWN:
                speak(stable)
                last_spoken = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# IMAGE MODE
def run_image(model, path):
    frame = cv2.imread(str(path))

    results = model.predict(frame, conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD, device=DEVICE)

    frame, detected = draw_boxes(frame, results, model.names)

    print("Detected:", detected if detected else "None")

    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="0 for webcam OR path to image")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)

    args = parser.parse_args()

    if not Path(args.weights).exists():
        print("[ERROR] Model weights not found. Train first using train.py")
        return

    model = YOLO(args.weights)

    if args.source == "0":
        run_webcam(model)
    else:
        run_image(model, Path(args.source))


if __name__ == "__main__":
    main()