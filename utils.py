from __future__ import annotations

import csv
import json
import logging
import random
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models as tv_models
except ImportError as e:
    raise ImportError(
        "PyTorch / torchvision are required but not installed.\n"
        "Run:  pip install -r requirements.txt\n"
        f"Original error: {e}"
    )

import config


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def setup_logger(name: str = "emotion_detection", log_dir: Path = config.LOG_DIR) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "run.log")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def append_history_row(csv_path: Path, row: Dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_model(
    model_name: str = config.MODEL_NAME,
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.PRETRAINED,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.efficientnet_b0(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_small":
        weights = tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "mobilenet_v3_large":
        weights = tv_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif model_name == "resnet18":
        weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            "Choose one of: efficientnet_b0, mobilenet_v3_small, mobilenet_v3_large, resnet18."
        )

    return model


def set_backbone_trainable(model: nn.Module, model_name: str, trainable: bool) -> None:
    model_name = model_name.lower()
    if "efficientnet" in model_name or "mobilenet" in model_name:
        head = model.classifier
    elif "resnet" in model_name:
        head = model.fc
    else:
        head = None

    head_params = set(id(p) for p in head.parameters()) if head is not None else set()
    for p in model.parameters():
        if id(p) not in head_params:
            p.requires_grad = trainable


def save_checkpoint(state: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    map_location: Optional[str] = None,
) -> Dict:
    if not Path(path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=map_location or config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]) -> Dict:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
    per_class_acc = np.nan_to_num(per_class_acc)

    return {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "per_class_accuracy": {c: float(a) for c, a in zip(class_names, per_class_acc)},
        "confusion_matrix": cm.tolist(),
    }


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        is_best = False
        if self.best is None:
            self.best = value
            is_best = True
        else:
            improved = (value > self.best + self.min_delta) if self.mode == "max" else (
                value < self.best - self.min_delta
            )
            if improved:
                self.best = value
                self.counter = 0
                is_best = True
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.should_stop = True
        return is_best


def plot_training_history(history: Dict[str, List[float]], save_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(history["train_loss"], label="train")
    axes[0, 0].plot(history["val_loss"], label="val")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].legend()

    axes[0, 1].plot(history["train_acc"], label="train")
    axes[0, 1].plot(history["val_acc"], label="val")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].legend()

    axes[1, 0].plot(history["val_f1"], color="green")
    axes[1, 0].set_title("Validation F1 (macro)")
    axes[1, 0].set_xlabel("epoch")

    axes[1, 1].plot(history["lr"], color="purple")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_yscale("log")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], save_path: Path, normalize: bool = True) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=float)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


class FaceDetector:
    def __init__(
        self,
        cascade_path: str = config.HAAR_CASCADE_PATH,
        scale_factor: float = config.FACE_DETECT_SCALE_FACTOR,
        min_neighbors: int = config.FACE_DETECT_MIN_NEIGHBORS,
        min_size: Tuple[int, int] = config.FACE_DETECT_MIN_SIZE,
    ):
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(
                f"Failed to load Haar cascade from {cascade_path}. "
                "Verify your OpenCV installation includes the data files."
            )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        raw_faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        ar_lo, ar_hi = config.FACE_ASPECT_RATIO_RANGE
        valid = []
        for f in raw_faces:
            x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
            if h == 0:
                continue
            aspect = w / h
            if ar_lo <= aspect <= ar_hi:
                valid.append((x, y, w, h))
        return valid


def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


@dataclass(eq=False)
class FaceTrack:
    box: Tuple[int, int, int, int]
    prob_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=config.SMOOTHING_WINDOW))
    frames_since_seen: int = 0
    face_id: int = 0

    def smoothed_probs(self) -> np.ndarray:
        return np.mean(self.prob_history, axis=0)


class FaceTracker:
    def __init__(self) -> None:
        self.tracks: List[FaceTrack] = []
        self._next_id = 1

    def update(
        self, boxes: List[Tuple[int, int, int, int]], probs: List[np.ndarray]
    ) -> List[FaceTrack]:
        matched_track_indices = set()
        results: List[FaceTrack] = []

        for box, prob in zip(boxes, probs):
            best_iou, best_idx = 0.0, -1
            for idx, track in enumerate(self.tracks):
                if idx in matched_track_indices:
                    continue
                iou = _iou(box, track.box)
                if iou > best_iou:
                    best_iou, best_idx = iou, idx

            if best_iou >= config.FACE_MATCH_IOU_THRESHOLD and best_idx != -1:
                track = self.tracks[best_idx]
                track.box = box
                track.prob_history.append(prob)
                track.frames_since_seen = 0
                matched_track_indices.add(best_idx)
                results.append(track)
            else:
                new_track = FaceTrack(box=box, face_id=self._next_id)
                self._next_id += 1
                new_track.prob_history.append(prob)
                self.tracks.append(new_track)
                results.append(new_track)

        for track in self.tracks:
            if track not in results:
                track.frames_since_seen += 1
        self.tracks = [t for t in self.tracks if t.frames_since_seen <= config.TRACK_TIMEOUT_FRAMES]

        return results


class FPSCounter:
    def __init__(self, window: int = 30):
        self.timestamps: Deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# Robot phrases for each emotion (displayed below the face box)
_ROBOT_PHRASES: Dict[str, str] = {
    "angry":    ">> AGGRESSION PATTERN DETECTED",
    "disgust":  ">> AVERSION SIGNAL IDENTIFIED",
    "fear":     ">> THREAT RESPONSE ENGAGED",
    "happy":    ">> JOY SIGNATURE CONFIRMED",
    "neutral":  ">> BASELINE CALM STATE",
    "sad":      ">> SORROW SIGNAL PROCESSING",
    "surprise": ">> SHOCK REACTION LOGGED",
}


def _draw_corner_bracket(
    frame: np.ndarray,
    x: int, y: int, w: int, h: int,
    color: Tuple[int, int, int],
    thickness: int = 3,
    arm: int = 22,
) -> None:
    corners = [
        # top-left
        ((x, y + arm), (x, y), (x + arm, y)),
        # top-right
        ((x + w - arm, y), (x + w, y), (x + w, y + arm)),
        # bottom-left
        ((x, y + h - arm), (x, y + h), (x + arm, y + h)),
        # bottom-right
        ((x + w - arm, y + h), (x + w, y + h), (x + w, y + h - arm)),
    ]
    for p1, corner, p2 in corners:
        cv2.line(frame, p1, corner, color, thickness, cv2.LINE_AA)
        cv2.line(frame, corner, p2, color, thickness, cv2.LINE_AA)


def draw_prediction_box(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    face_id: Optional[int] = None,
) -> None:
    x, y, w, h = box
    color = config.EMOTION_COLORS.get(label, (0, 255, 0))
    H, W = frame.shape[:2]

    # --- Semi-transparent face-box fill ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)

    # --- Thin dashed inner rectangle ---
    cv2.rectangle(frame, (x + 2, y + 2), (x + w - 2, y + h - 2), color, 1, cv2.LINE_AA)

    # --- Corner bracket overlay ---
    _draw_corner_bracket(frame, x, y, w, h, color, thickness=3, arm=min(26, w // 4, h // 4))

    # ---- Face-ID badge (top-left corner) ----
    if face_id is not None:
        badge = f" ID:{face_id} "
        (bw, bh), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x, y), (x + bw + 4, y + bh + 6), color, -1)
        cv2.putText(frame, badge, (x + 2, y + bh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # ---- Emotion label pill (above box) ----
    label_text = label.upper()
    pct_text = f"{confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (lw, lh), _ = cv2.getTextSize(label_text, font, 0.75, 2)
    (pw, _), _ = cv2.getTextSize(pct_text, font, 0.5, 1)

    pill_w = max(lw, pw) + 16
    pill_h = lh + 28
    pill_x = x
    pill_y = max(y - pill_h - 4, 0)

    # pill background
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), (10, 10, 10), -1)
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), color, 2)

    # emotion name
    cv2.putText(frame, label_text, (pill_x + 8, pill_y + lh + 4),
                font, 0.75, color, 2, cv2.LINE_AA)
    # percentage
    cv2.putText(frame, pct_text, (pill_x + 8, pill_y + lh + 22),
                font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    # ---- Confidence bar (below box) ----
    bar_x, bar_y = x, y + h + 6
    bar_w, bar_h = w, 8
    if bar_y + bar_h < H:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
        fill_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), color, 1)

    # ---- Robot phrase (below confidence bar) ----
    phrase = _ROBOT_PHRASES.get(label, ">> ANALYZING...")
    phrase_y = bar_y + bar_h + 16
    if phrase_y < H:
        (ph_w, ph_h), _ = cv2.getTextSize(phrase, font, 0.42, 1)
        bg_x2 = min(x + ph_w + 8, W)
        cv2.rectangle(frame, (x, phrase_y - ph_h - 3), (bg_x2, phrase_y + 3), (0, 0, 0), -1)
        cv2.putText(frame, phrase, (x + 4, phrase_y),
                    font, 0.42, color, 1, cv2.LINE_AA)


def draw_overlay_hud(frame: np.ndarray, fps: float, num_faces: int) -> None:
    H, W = frame.shape[:2]
    timestamp = time.strftime("%H:%M:%S")
    date_str  = time.strftime("%Y-%m-%d")

    # ---- Top HUD bar ----
    bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # FPS indicator with color coding
    fps_color = (0, 255, 80) if fps >= 20 else (0, 200, 255) if fps >= 10 else (0, 80, 255)
    cv2.putText(frame, f"FPS {fps:4.1f}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 1, cv2.LINE_AA)

    # Center title
    title = "[[ EMOTION RECOGNITION SYSTEM ]]"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, title, (W // 2 - tw // 2, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

    # Right — faces + time
    right_text = f"FACES:{num_faces}  {date_str} {timestamp}"
    (rw, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, right_text, (W - rw - 10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

    # ---- Bottom status bar ----
    bot_h = 26
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, H - bot_h), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)

    status = "[ PRESS 'q' QUIT ]  [ 's' SCREENSHOT ]  [ SPACE PAUSE ]  [ ROBOT-VISION ACTIVE ]"
    cv2.putText(frame, status, (8, H - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)


def preprocess_face(face_bgr: np.ndarray, img_size: int = config.IMG_SIZE) -> "torch.Tensor":
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    face_arr = face_rgb.astype(np.float32) / 255.0

    mean = np.array(config.NORM_MEAN, dtype=np.float32)
    std = np.array(config.NORM_STD, dtype=np.float32)
    face_arr = (face_arr - mean) / std

    face_arr = np.transpose(face_arr, (2, 0, 1))
    return torch.from_numpy(face_arr)


def load_model_for_inference(checkpoint_path: Path, device: "torch.device") -> Tuple["nn.Module", str, List[str]]:
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'. Train a model first with train.py, "
            "or pass --checkpoint pointing at an existing best.pt / last.pt."
        )
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_name = raw.get("model_name", config.MODEL_NAME)
    class_names = [c.lower() for c in raw.get("class_names", config.CLASS_NAMES)]
    model = build_model(model_name=model_name, num_classes=len(class_names), pretrained=False)
    model.load_state_dict(raw["model_state_dict"])
    model.to(device)
    model.eval()
    return model, model_name, class_names
