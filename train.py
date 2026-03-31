import sys
from pathlib import Path
from ultralytics import YOLO
import yaml

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_YAML   = "data/data.yaml"
MODEL_NAME  = "yolov8n.pt"
EPOCHS      = 1
IMG_SIZE    = 640
BATCH_SIZE  = 4
WORKERS     = 4
PROJECT_DIR = "runs/train"
RUN_NAME    = "sign_detection"
DEVICE      = "cpu"


# ─────────────────────────────────────────────
# CHECK DATASET
# ─────────────────────────────────────────────
def check_dataset():
    if not Path(DATA_YAML).exists():
        print("\n[ERROR] data/data.yaml not found!")
        print("Download dataset from Roboflow in YOLOv8 format and extract into 'data/' folder.\n")
        sys.exit(1)


# ─────────────────────────────────────────────
# FIX PATHS
# ─────────────────────────────────────────────
def patch_data_yaml():
    yaml_path = Path(DATA_YAML)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    base = yaml_path.parent.resolve()

    for key in ("train", "val", "test"):
        if key in cfg:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str((base / p).resolve())

    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f)

    print("[INFO] data.yaml paths fixed")
    return str(yaml_path.resolve())


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(data_yaml):
    print("\n[INFO] Training started...\n")

    model = YOLO(MODEL_NAME)

    model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=15,
        save=True,
        plots=True,
        exist_ok=True,
    )

    best = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    print(f"\n[INFO] Best weights saved at: {best}")
    return str(best)


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(weights, data_yaml):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, device=DEVICE)

    print("\n──────── RESULTS ────────")
    print(f"mAP@0.5      : {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95 : {metrics.box.map:.4f}")
    print(f"Precision    : {metrics.box.mp:.4f}")
    print(f"Recall       : {metrics.box.mr:.4f}")
    print("────────────────────────")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    check_dataset()
    data_yaml = patch_data_yaml()
    best = train(data_yaml)
    evaluate(best, data_yaml)