import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_CANDIDATES = [
    PROJECT_ROOT / "dataset",
    Path("/content/dataset"),
    Path("/content/drive/MyDrive/dataset"),
    Path("/content/drive/MyDrive/emotion_detection/dataset"),
    Path("/kaggle/input/fer-2013-facial-expression-dataset"),
    Path("/kaggle/working/dataset"),
]

DATA_DIR = PROJECT_ROOT / "dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EXPORT_DIR = PROJECT_ROOT / "exported_models"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"

for _d in (CHECKPOINT_DIR, MODELS_DIR, LOG_DIR, OUTPUT_DIR, EXPORT_DIR, SCREENSHOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = CHECKPOINT_DIR / "best.pt"
LAST_MODEL_PATH = CHECKPOINT_DIR / "last.pt"
METRICS_PATH = CHECKPOINT_DIR / "metrics.json"
HISTORY_CSV_PATH = LOG_DIR / "training_history.csv"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASS_NAMES)

MODEL_NAME = "efficientnet_b0"  # efficientnet_b0 | mobilenet_v3_small | mobilenet_v3_large | resnet18
PRETRAINED = True
FREEZE_BACKBONE_EPOCHS = 3
IMG_SIZE = 224
IN_CHANNELS = 3

SEED = 42
EPOCHS = 60
BATCH_SIZE = 64
NUM_WORKERS = min(8, os.cpu_count() or 2)
PIN_MEMORY = True
PERSISTENT_WORKERS = NUM_WORKERS > 0

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
OPTIMIZER = "adamw"  # adamw | sgd
MOMENTUM = 0.9

SCHEDULER = "cosine"  # cosine | plateau | none
WARMUP_EPOCHS = 2
MIN_LR = 1e-6

LABEL_SMOOTHING = 0.1
USE_CLASS_WEIGHTS = True

GRAD_CLIP_NORM = 1.0
USE_AMP = torch.cuda.is_available()
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4
MONITOR_METRIC = "val_f1"

RESUME_FROM_LAST = True

AUG_HFLIP_PROB = 0.5
AUG_ROTATION_DEGREES = 12
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2
AUG_RANDOM_CROP_PADDING = 8
AUG_RANDOM_ERASING_PROB = 0.25

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

import cv2  # noqa: E402

HAAR_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
FACE_DETECT_SCALE_FACTOR = 1.15
FACE_DETECT_MIN_NEIGHBORS = 8
FACE_DETECT_MIN_SIZE = (60, 60)
FACE_ASPECT_RATIO_RANGE = (0.65, 1.5)
FACE_MIN_CONFIDENCE = 0.40

WEBCAM_INDEX = 0
WEBCAM_WIDTH = 640
WEBCAM_HEIGHT = 480
TARGET_INFERENCE_SIZE = IMG_SIZE
SMOOTHING_WINDOW = 4
FACE_MATCH_IOU_THRESHOLD = 0.3
TRACK_TIMEOUT_FRAMES = 5

EMOTION_COLORS = {
    "angry":    (0, 0, 255),
    "disgust":  (0, 140, 0),
    "fear":     (130, 0, 130),
    "happy":    (0, 220, 255),
    "neutral":  (200, 200, 200),
    "sad":      (255, 140, 0),
    "surprise": (255, 0, 255),
}
