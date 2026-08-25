from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    from tqdm import tqdm
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

import config
import utils

logger = utils.setup_logger()

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
FOURCC_BY_EXTENSION = {
    ".mp4": "mp4v",
    ".avi": "XVID",
    ".mov": "mp4v",
    ".mkv": "X264",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Emotion recognition over a video file.")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--checkpoint", type=str, default=str(config.BEST_MODEL_PATH))
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
    p.add_argument("--skip-frames", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else config.DEVICE
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error(f"Video file not found: {input_path}")
        sys.exit(1)
    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        logger.warning(f"Extension '{input_path.suffix}' is untested — attempting to open anyway.")

    try:
        model, model_name, class_names = utils.load_model_for_inference(Path(args.checkpoint), device)
        face_detector = utils.FaceDetector()
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(str(e))
        sys.exit(1)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(
            f"Could not open video '{input_path}'. The file may be corrupted or use a codec "
            "not supported by your OpenCV build."
        )
        sys.exit(1)

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = Path(args.output) if args.output else config.OUTPUT_DIR / f"{input_path.stem}_annotated.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*FOURCC_BY_EXTENSION.get(output_path.suffix.lower(), "mp4v"))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))
    if not writer.isOpened():
        logger.error(f"Could not open VideoWriter for '{output_path}'. Try a different --output extension (.mp4/.avi).")
        cap.release()
        sys.exit(1)

    tracker = utils.FaceTracker()
    fps_counter = utils.FPSCounter()

    logger.info(f"Model: {model_name} | Device: {device}")
    logger.info(f"Input: {input_path} ({width}x{height} @ {fps_in:.1f}fps, {total_frames} frames)")
    logger.info(f"Output: {output_path}")

    frame_idx = 0
    last_boxes, last_probs = [], []
    t_start = time.time()

    try:
        with tqdm(total=total_frames if total_frames > 0 else None, desc="processing", dynamic_ncols=True) as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                run_detection = (args.skip_frames <= 0) or (frame_idx % (args.skip_frames + 1) == 0)
                if run_detection:
                    boxes = face_detector.detect(frame)
                    probs = []
                    if boxes:
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
                            batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                        probs = list(batch_probs)

                        filtered_boxes, filtered_probs = [], []
                        for b, p in zip(boxes, probs):
                            if float(np.max(p)) >= config.FACE_MIN_CONFIDENCE:
                                filtered_boxes.append(b)
                                filtered_probs.append(p)
                        boxes, probs = filtered_boxes, filtered_probs

                    last_boxes, last_probs = boxes, probs

                tracks = tracker.update(last_boxes, last_probs) if last_boxes else []
                visible_count = 0
                for track in tracks:
                    smoothed = track.smoothed_probs()
                    cls_idx = int(np.argmax(smoothed))
                    label = class_names[cls_idx]
                    confidence = float(smoothed[cls_idx])
                    if confidence >= config.FACE_MIN_CONFIDENCE:
                        utils.draw_prediction_box(frame, track.box, label, confidence, track.face_id)
                        visible_count += 1

                processing_fps = fps_counter.tick()
                utils.draw_overlay_hud(frame, processing_fps, visible_count)

                writer.write(frame)
                frame_idx += 1
                pbar.update(1)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user — partial output has been saved.")
    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t_start
    logger.info(f"Processed {frame_idx} frames in {elapsed:.1f}s ({frame_idx / elapsed if elapsed > 0 else 0:.1f} fps).")
    logger.info(f"Annotated video saved -> {output_path}")


if __name__ == "__main__":
    main()
