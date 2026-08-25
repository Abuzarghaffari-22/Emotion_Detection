from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

import config
import utils

logger = utils.setup_logger()

VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emotion recognition on image file(s).")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--checkpoint", type=str, default=str(config.BEST_MODEL_PATH))
    p.add_argument("--output-dir", type=str, default=str(config.OUTPUT_DIR / "image_inference"))
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
    p.add_argument("--no-save-annotated", action="store_true")
    return p.parse_args()


def process_image(path: Path, model, device, face_detector, class_names, out_dir: Path, save_annotated: bool):
    frame = cv2.imread(str(path))
    if frame is None:
        logger.warning(f"Could not read image (unsupported format or corrupted): {path}")
        return []

    boxes = face_detector.detect(frame)
    results = []

    if not boxes:
        logger.info(f"{path.name}: no face detected.")
    else:
        crops = []
        for (x, y, w, h) in boxes:
            x, y = max(0, x), max(0, y)
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                crop = np.zeros((config.IMG_SIZE, config.IMG_SIZE, 3), dtype=np.uint8)
            crops.append(utils.preprocess_face(crop))

        batch = torch.stack(crops).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        face_num = 0
        for i, (box, prob) in enumerate(zip(boxes, probs)):
            cls_idx = int(np.argmax(prob))
            label = class_names[cls_idx]
            confidence = float(prob[cls_idx])
            if confidence < config.FACE_MIN_CONFIDENCE:
                continue
            face_num += 1
            utils.draw_prediction_box(frame, box, label, confidence, face_id=face_num)
            results.append({
                "file": str(path), "face_index": face_num, "emotion": label, "confidence": confidence,
                "box_x": box[0], "box_y": box[1], "box_w": box[2], "box_h": box[3],
            })
            logger.info(f"{path.name} [face {face_num}]: {label} ({confidence * 100:.1f}%)")

        if face_num == 0:
            logger.info(f"{path.name}: detections found but none above confidence threshold.")

    if save_annotated:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_annotated{path.suffix}"
        cv2.imwrite(str(out_path), frame)

    return results


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else config.DEVICE
    input_path = Path(args.input)
    out_dir = Path(args.output_dir)

    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)

    try:
        model, model_name, class_names = utils.load_model_for_inference(Path(args.checkpoint), device)
        face_detector = utils.FaceDetector()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Model: {model_name} | Device: {device}")

    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in VALID_EXTENSIONS)
        if not image_paths:
            logger.error(f"No supported images found in folder: {input_path}")
            sys.exit(1)
        logger.info(f"Found {len(image_paths)} image(s) in {input_path}")

    all_results = []
    t0 = time.time()
    for path in image_paths:
        try:
            all_results.extend(
                process_image(path, model, device, face_detector, class_names, out_dir, not args.no_save_annotated)
            )
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")

    elapsed = time.time() - t0
    logger.info(f"Processed {len(image_paths)} image(s) in {elapsed:.2f}s "
                f"({len(image_paths) / elapsed if elapsed > 0 else 0:.1f} img/s)")

    if all_results:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "predictions.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        logger.info(f"Predictions summary saved -> {csv_path}")
    else:
        logger.info("No faces detected in any input image — nothing to summarize.")

    logger.info(f"Annotated images (if enabled) saved under -> {out_dir}")


if __name__ == "__main__":
    main()
