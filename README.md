# Sign Language Detection using YOLOv8

A real-time sign detection system built using YOLOv8 that detects hand signs (words/phrases) from images, videos, or webcam input. Detected signs are optionally announced using offline Text-to-Speech (TTS).

---

## Features

* Real-time detection using webcam
* Supports images, videos, and live camera input
* Bounding boxes with labels and confidence scores
* Offline Text-to-Speech output using `pyttsx3`
* Detection summary printed in terminal
* Works entirely on CPU (no GPU required)

---

## Project Structure

```
CV_Project/
├── train.py          # Train YOLOv8 model on dataset
├── detect.py         # Run inference (image/video/webcam)
├── requirements.txt  # Dependencies
├── README.md
```

> Note: The `runs/` folder is automatically generated during training and is not included in the repository.

---

## System Requirements

| Requirement | Details                      |
| ----------- | ---------------------------- |
| OS          | macOS / Windows / Linux      |
| Python      | 3.9+                         |
| GPU         | Not required (CPU supported) |
| RAM         | 8 GB recommended             |

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

---

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset is downloaded manually from Roboflow in YOLOv8 format and placed inside the `data/` directory.

> Due to size constraints, the dataset is not included in this repository.

---

## Training the Model

```bash
python3 train.py
```

* Uses YOLOv8 (Nano version for faster training)
* Runs on CPU
* Epochs are reduced for demonstration purposes
* Best weights are saved automatically after training

---

## Running Detection

### Webcam Detection

```bash
python3 detect.py --source 0
```

---

### Image Detection

```bash
python3 detect.py --source path/to/image.jpg
```

---

### Video Detection

```bash
python3 detect.py --source path/to/video.mp4
```

---

## Optional Arguments

| Argument     | Description                   |
| ------------ | ----------------------------- |
| `--weights`  | Path to trained model weights |
| `--conf`     | Confidence threshold          |
| `--no-tts`   | Disable speech output         |
| `--save-dir` | Output directory              |

---

## Output

* Annotated images/videos saved in `runs/detect/`
* Bounding boxes with labels
* Console summary of detected signs
* Optional speech output

---

## Limitations

* Model trained with very few epochs (for time constraints)
* Dataset contains mixed or inconsistent labels
* Accuracy can be improved with better dataset and longer training

---

## Technologies Used

* YOLOv8 (Ultralytics)
* PyTorch
* OpenCV
* pyttsx3 (Text-to-Speech)

---

## Course Context

This project was developed as part of a Computer Vision course. It demonstrates object detection using deep learning, including training, inference, and real-time deployment.
