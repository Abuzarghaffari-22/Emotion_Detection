from __future__ import annotations

import argparse
import sys
import threading
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


class ThreadedVideoStream:
    def __init__(self, source, width: int, height: int):
        self.cap = cv2.VideoCapture(source)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera/source '{source}'. "
                "Check that no other application is using the camera, that the index is correct "
                "(try --source 0, 1, 2 ...), and that camera permissions are granted to this process."
            )

        self.lock = threading.Lock()
        self.frame = None
        self.ok = False
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        time.sleep(0.2)

    def _update(self) -> None:
        while not self.stopped:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame
            if not ok:
                time.sleep(0.05)

    def read(self):
        with self.lock:
            return self.ok, (None if self.frame is None else self.frame.copy())

    def stop(self) -> None:
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time webcam emotion recognition.")
    p.add_argument("--source", default=str(config.WEBCAM_INDEX),
                   help="Camera index (0,1,...) or DroidCam URL e.g. http://192.168.x.x:4747/video")
    p.add_argument("--checkpoint", type=str, default=str(config.BEST_MODEL_PATH))
    p.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda", "mps"])
    p.add_argument("--width", type=int, default=config.WEBCAM_WIDTH)
    p.add_argument("--height", type=int, default=config.WEBCAM_HEIGHT)
    p.add_argument("--skip-frames", type=int, default=0)
    p.add_argument("--no-window", action="store_true")
    return p.parse_args()


def resolve_source(source: str):
    try:
        return int(source)
    except ValueError:
        return source


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else config.DEVICE

    try:
        model, model_name, class_names = utils.load_model_for_inference(Path(args.checkpoint), device)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        face_detector = utils.FaceDetector()
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    tracker = utils.FaceTracker()
    fps_counter = utils.FPSCounter()

    try:
        stream = ThreadedVideoStream(resolve_source(args.source), args.width, args.height)
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Model: {model_name} | Device: {device} | Source: {args.source}")
    logger.info("Press 'q' to quit, 's' to save a screenshot, space to pause.")

    WIN_NAME = "Emotion Recognition | press Q to quit"
    WIN_W, WIN_H = 960, 540

    if not args.no_window:
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, WIN_W, WIN_H)

    # CLAHE for automatic brightness/contrast correction on dark frames
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    frame_idx = 0
    paused = False
    last_boxes, last_probs = [], []
    no_signal_frames = 0

    try:
        while True:
            ok, frame = stream.read()
            if not ok or frame is None:
                logger.warning("Dropped frame / camera returned no data -- retrying...")
                time.sleep(0.05)
                continue

            # ---- Auto-brightness correction with CLAHE ----
            mean_bright = float(frame.mean())
            if mean_bright < 30.0:
                # Frame is very dark: apply CLAHE per channel
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                l_ch = clahe.apply(l_ch)
                lab = cv2.merge([l_ch, a_ch, b_ch])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

            if not paused:
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

                        # Filter out detections where model is not confident
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

                fps = fps_counter.tick()
                utils.draw_overlay_hud(frame, fps, visible_count)
                frame_idx += 1

                # ---- "Scanning..." overlay when no face found ----
                if visible_count == 0 and not paused:
                    H, W = frame.shape[:2]
                    scan_txt = ">> SCANNING FOR FACE..."
                    (sw, sh), _ = cv2.getTextSize(scan_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    sx = W // 2 - sw // 2
                    sy = H // 2
                    cv2.rectangle(frame, (sx - 12, sy - sh - 8), (sx + sw + 12, sy + 10), (0, 0, 0), -1)
                    cv2.putText(frame, scan_txt, (sx, sy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

            if not args.no_window:
                cv2.imshow(WIN_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    config.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
                    out_path = config.SCREENSHOT_DIR / f"screenshot_{int(time.time())}.png"
                    cv2.imwrite(str(out_path), frame)
                    logger.info(f"Screenshot saved -> {out_path}")
                elif key == ord(" "):
                    paused = not paused
                    logger.info("Paused" if paused else "Resumed")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
    finally:
        stream.stop()
        if not args.no_window:
            cv2.destroyAllWindows()
        logger.info("Webcam session ended cleanly.")


if __name__ == "__main__":
    main()
