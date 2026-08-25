# Real-Time Facial Emotion Recognition

A production-structured, transfer-learning-based facial emotion recognition system
trained on **FER2013**, with real-time webcam inference, static image inference,
and video file inference — plus TorchScript/ONNX export for deployment.

7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
---
## Visual Demo
| Original Collage (Before) | Model Detections (After) |
| :---: | :---: |
| <img src="assets/<img width="1498" height="1050" alt="test_emotions" src="https://github.com/user-attachments/assets/8152bef6-6f28-47d0-8c3c-706f32035b7c" />" width="460"> |
| <img src="assets/<img width="1498" height="1050" alt="test_emotions_annotated" src="https://github.com/user-attachments/assets/6f716f90-0ab5-4b99-9a73-efca497c6e88" />" width="460"> |
---

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Dataset Preparation](#dataset-preparation)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Export](#export)
8. [Inference](#inference)
9. [How It Works](#how-it-works)
10. [Testing Procedures](#testing-procedures)
11. [Performance](#performance)
12. [Troubleshooting](#troubleshooting)
13. [Future Improvements](#future-improvements)
14. [License](#license)

---

## Features

- **Transfer learning** on EfficientNet-B0 / MobileNetV3 / ResNet18 (ImageNet-pretrained), swap with one config line
- **Full augmentation stack**: random crop, horizontal flip, rotation, brightness/contrast jitter, random erasing
- **Class-imbalance handling** via inverse-frequency weighted sampling (FER2013's `disgust` class is ~5% the size of `happy`)
- **Mixed precision, gradient clipping, cosine LR schedule, early stopping, checkpoint resume** — all automatic
- **TensorBoard + CSV logging** of every epoch
- **Full evaluation suite**: confusion matrix, classification report, per-class accuracy
- **Three inference modes**: real-time webcam, single image/folder batch, video file
- **Multi-face detection & tracking** with temporal smoothing to eliminate label flicker
- **TorchScript + ONNX export** with automatic output-parity verification
- Runs on **CPU, CUDA, or Apple Silicon (MPS)**

---

## Project Structure

```
emotion_detection/
├── Colab_Training.ipynb     # Full training notebook (GPU detection → export → download)
├── config.py                 # Every path & hyperparameter — single source of truth
├── utils.py                   # Model factory, checkpointing, metrics, face detector, drawing, tracking
├── dataset.py                 # Locate/verify FER2013, corruption check, transforms, DataLoaders
├── train.py                   # Training loop (resumable, early-stopping, AMP, logging)
├── evaluate.py                 # Confusion matrix + classification report on any split
├── export.py                   # PyTorch → TorchScript → ONNX export + verification
├── infer_webcam.py             # Real-time multi-face webcam inference
├── infer_image.py              # Single image / folder / batch inference
├── infer_video.py              # Annotated video-file inference (mp4/avi/mov/mkv)
├── requirements.txt
├── dataset/                    # FER2013: train/ val/ test/, 7 class folders each
├── checkpoints/                # best.pt, last.pt, metrics.json (created by train.py)
├── models/                     # reserved for auxiliary model assets
├── logs/                       # run.log, training_history.csv, tensorboard/, training_curves.png
├── outputs/                    # evaluation reports, confusion matrices, batch inference results
├── exported_models/             # .pth / TorchScript .pt / .onnx exports
└── screenshots/                 # saved via 's' key during webcam inference
```

---

## Installation

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

**GPU users:** the pinned `torch`/`torchvision` versions in `requirements.txt` install a
CPU build by default on some platforms. For CUDA, install PyTorch first using the
command generator at https://pytorch.org/get-started/locally/, e.g.:

```bash
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt   # installs everything else
```

Verify your setup:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python dataset.py     # runs the dataset verification/statistics report
```

---

## Dataset Preparation

This project uses **FER2013**, specifically the pre-split Kaggle mirror:
https://www.kaggle.com/datasets/pankaj4321/fer-2013-facial-expression-dataset

Expected structure (already what this Kaggle mirror provides):

```
dataset/
├── train/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
├── val/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
└── test/{angry,disgust,fear,happy,neutral,sad,surprise}/*.png
```

Folder names are matched **case-insensitively**, so `Angry`, `angry`, or `ANGRY` all work.

### Option A — already have it (this repo)
The dataset you uploaded has already been extracted into `dataset/` for you (35,887
images total — 3,995–7,215 per class in train, matching the standard FER2013 split).
Nothing further to do.

### Option B — Kaggle CLI
```bash
pip install kaggle
# place your kaggle.json API token at ~/.kaggle/kaggle.json (from kaggle.com/settings)
kaggle datasets download -d pankaj4321/fer-2013-facial-expression-dataset -p dataset --unzip
```

### Option C — Google Drive (Colab)
Upload the dataset zip to your Drive, then in `Colab_Training.ipynb` the "Mount Drive
& Locate Dataset" cell will find it automatically under common paths, or you can set
the path explicitly.

`dataset.py` verifies structure, counts images per class, and scans every file for
corruption on each run — see [How It Works](#how-it-works) for details.

---

## Training

```bash
python train.py                                          # uses every default in config.py
python train.py --epochs 40 --batch-size 32 --lr 1e-4
python train.py --model mobilenet_v3_small --img-size 160  # faster, for real-time CPU targets
python train.py --resume                                   # continue from checkpoints/last.pt
```

What happens automatically:
- Backbone is frozen for the first `FREEZE_BACKBONE_EPOCHS` epochs (classifier-head warmup), then unfrozen
- Mixed precision + gradient clipping on CUDA
- Cosine-annealed learning rate
- Class-weighted loss (inverse frequency) to counter FER2013's imbalance
- Best model (by validation macro-F1) saved to `checkpoints/best.pt`; latest epoch to `checkpoints/last.pt`
- Early stopping after `EARLY_STOPPING_PATIENCE` epochs without improvement
- Per-epoch metrics to TensorBoard (`logs/tensorboard/`) and `logs/training_history.csv`
- `Ctrl+C` saves `last.pt` before exiting — resume any time with `--resume`

View training curves live:
```bash
tensorboard --logdir logs/tensorboard
```

---

## Evaluation

```bash
python evaluate.py                                  # best.pt on the test split
python evaluate.py --checkpoint checkpoints/last.pt --split val
```

Produces, in `outputs/`:
- `confusion_matrix_<split>.png` — normalized confusion matrix
- `evaluation_report_<split>.json` — accuracy, macro precision/recall/F1, per-class accuracy, full sklearn classification report

---

## Export

```bash
python export.py                                    # exports best.pt to all 3 formats
python export.py --formats torchscript onnx
```

Writes to `exported_models/`:
- `<model>_weights.pth` — portable state_dict + metadata
- `<model>_torchscript.pt` — for LibTorch/C++/mobile
- `<model>.onnx` — for ONNX Runtime, OpenVINO, TensorRT (output-parity checked against PyTorch automatically)

---

## Inference

### Real-time webcam
```bash
python infer_webcam.py
python infer_webcam.py --source 1 --device cpu
python infer_webcam.py --skip-frames 1              # detect every other frame for extra speed
```
Controls: `q` quit · `s` save screenshot · `space` pause/resume.

### Single image or folder
```bash
python infer_image.py --input photo.jpg
python infer_image.py --input ./photos/ --output-dir outputs/batch_run
```
Writes annotated images plus `predictions.csv` (file, face index, emotion, confidence, box).

### Video file
```bash
python infer_video.py --input clip.mp4
python infer_video.py --input clip.mov --output outputs/clip_annotated.mp4
```
Supports `.mp4 .avi .mov .mkv`.

All three share the same face detector, model, preprocessing, and drawing code
(`utils.py`), so results are consistent across every entry point.

---

## How It Works

**Face detection.** We use OpenCV's Haar cascade frontal-face detector
(`haarcascade_frontalface_default.xml`, bundled with `opencv-python` — no extra
download). It runs on histogram-equalized grayscale frames, which noticeably
improves detection under uneven or low lighting, and is fast enough for real-time
multi-face detection on CPU. A DNN-based detector (e.g. an SSD/RetinaFace ONNX
model) would detect more off-angle/occluded faces at the cost of extra latency and
a separate model download — a reasonable upgrade path if your use case needs it
(see [Future Improvements](#future-improvements)).

**Emotion recognition.** Each detected face is cropped, resized, and classified
independently by the trained CNN — a 7-way softmax over the FER2013 emotion set.
Detection and classification are decoupled on purpose: swapping the face detector
never requires retraining the classifier, and vice versa.

**Transfer learning.** Rather than training a CNN from scratch on ~29K training
images, we start from ImageNet-pretrained weights and replace only the final
classifier layer. The backbone already knows general visual features (edges,
textures, shapes); training only has to learn how those map to 7 emotions, which
converges faster and generalizes better than training from random initialization
on a dataset this size.

**Data augmentation.** Random crop, horizontal flip, rotation, brightness/contrast
jitter, and random erasing are applied to the training split only. They simulate
the pose, lighting, and partial-occlusion variation a real webcam sees, which
directly reduces the training/validation accuracy gap (overfitting).

**Evaluation metrics.** Overall accuracy is reported alongside **macro** precision/
recall/F1 (unweighted mean across classes) specifically because FER2013 is
imbalanced — a model that ignores the rare `disgust` class could still post a high
*overall* accuracy while failing that class completely. Per-class accuracy and the
confusion matrix make that failure mode visible if it happens.

**Model selection.** Preferred order is EfficientNet-B0 → MobileNetV3 → ResNet18:
- **EfficientNet-B0** gives the best accuracy per FLOP of the three and is the
  default — best suited to GPU training and GPU/edge-GPU inference (Jetson, etc.)
  where the ≥30 FPS GPU target has headroom to spare.
- **MobileNetV3-Small**, purpose-built for mobile/CPU inference, is the better
  choice when the ≥20 FPS **CPU** real-time target is the binding constraint —
  roughly 3–4× faster per frame than EfficientNet-B0 for a few points less
  accuracy. Switch to it with `MODEL_NAME = "mobilenet_v3_small"` and consider
  `IMG_SIZE = 160` in `config.py`.
- **ResNet18** is the fallback: slightly larger and slower than EfficientNet-B0
  with no consistent accuracy edge on this dataset, but the safest choice if
  either of the above isn't available in your torchvision version.

There's no single "best" answer independent of your deployment target — pick
based on where you actually need the FPS headroom.

**Optimization techniques.** Mixed precision (CUDA) and gradient clipping keep
training stable and fast; a cosine LR schedule with early stopping avoids both
under- and over-training; inverse-frequency class weighting stops the dominant
`happy`/`neutral` classes from swamping the loss. At inference time, batching all
faces detected in a frame into one forward pass, an optional frame-skip, and a
threaded camera reader (decoupling capture from processing) are what keep the
webcam pipeline smooth instead of trailing behind real time.

**Deployment.** `export.py` produces a TorchScript file for LibTorch/C++/mobile
embedding and an ONNX file for ONNX Runtime / OpenVINO / TensorRT — pick whichever
matches your target runtime. Both are verified against the original PyTorch output
before being written, so a silent export bug can't ship unnoticed.

---

## Testing Procedures

Manual QA checklist to run before considering a trained model deployment-ready.
For each scenario, confirm: the correct number of faces is boxed, the emotion
label is stable (doesn't flicker between two classes frame-to-frame), and FPS
stays within the [Performance](#performance) targets.

| Scenario | How to test | Watch for |
|---|---|---|
| **Single face** | `python infer_webcam.py`, one person facing the camera | Stable single box, correct label |
| **Multiple faces** | 2–4 people in frame simultaneously | Each face gets its own stable ID/box; no swapped IDs when people move |
| **Low light** | Dim room / backlit window | Detector still finds the face (histogram equalization helps); confidence may drop — acceptable if label stays correct |
| **Bright light / overexposure** | Direct light on face or strong backlight | No false double-detections; label stability holds |
| **Occlusion** | Hand over part of face, mask, glasses, hair over eyes | Detector may miss heavily occluded faces — verify graceful skip (no crash), label recovers once unoccluded |
| **Different distances** | Very close to camera, then several meters back | `FACE_DETECT_MIN_SIZE` in `config.py` sets the smallest detectable face — tune if far-distance detection is a requirement |
| **Fast motion** | Turn head quickly / walk across frame | Some dropped detections are expected; tracking should re-acquire within `TRACK_TIMEOUT_FRAMES`, not create duplicate IDs |
| **Camera angles** | Look up/down/sideways, profile view | Haar cascade is frontal-face-only — expect reduced recall at extreme angles (see Future Improvements for a DNN detector upgrade) |
| **Live webcam** | `python infer_webcam.py` | FPS + timestamp overlay update smoothly; `q`/`s`/`space` all respond |
| **Recorded video** | `python infer_video.py --input <clip>` | Output file plays back with boxes/labels correctly synced to source frames; run against a short clip of each scenario above |

---

## Performance

Targets this pipeline is designed to hit with the default EfficientNet-B0 config,
and the levers to reach them if your first run falls short:

| Metric | Target | Primary lever if you're short |
|---|---|---|
| Test accuracy (FER2013) | ≥ 70% | More epochs, `mobilenet_v3_large`→`efficientnet_b0`, verify class weighting is on |
| CPU real-time FPS | ≥ 20 | `--model mobilenet_v3_small`, lower `IMG_SIZE`, `--skip-frames 1` |
| GPU real-time FPS | ≥ 30 | Already comfortable with EfficientNet-B0 on most discrete GPUs; batch faces per frame (already automatic) |
| Val/train accuracy gap | small | Increase augmentation strength, add dropout, more class-weighted sampling |

FER2013 itself is a noisy, low-resolution (48×48) dataset with known label
ambiguity between classes like `fear`/`surprise` and `disgust`/`angry` — human
accuracy on FER2013 is estimated around 65–68%, so treat that as useful context
for the ≥70% target rather than a hard ceiling to chase indefinitely.

---

## Troubleshooting

| Problem | Likely cause / fix |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | Run `pip install -r requirements.txt` inside your activated virtual environment |
| `FileNotFoundError: Could not locate the FER2013 dataset` | Check `dataset/train`, `dataset/val`, `dataset/test` exist with 7 class folders each, or pass `--data-dir` |
| `Dataset split 'train' is missing class folder(s)` | A class folder is missing or misspelled — the check is case-insensitive but the *name* must still match one of the 7 emotions |
| Corrupted image warnings from `dataset.py` | Remove or re-download the listed files; a handful of corrupt files won't meaningfully affect training, but many indicate a bad download |
| `Could not open camera/source '0'` | Another app is using the camera, the index is wrong (try `--source 1`), or OS camera permissions aren't granted to your terminal/Python |
| CUDA out of memory during training | Lower `--batch-size`, or lower `IMG_SIZE` in `config.py` |
| Training seems stuck at low accuracy | Confirm `PRETRAINED = True` in `config.py`, check the dataset report for class imbalance, try more epochs before concluding it's stuck |
| ONNX export skipped / warns "not installed" | `pip install onnx onnxruntime` (already in `requirements.txt`; re-run `pip install -r requirements.txt`) |
| Webcam FPS far below target | Switch to `mobilenet_v3_small`, add `--skip-frames 1`, close other GPU/CPU-heavy applications, confirm you're actually running on the GPU (`--device cuda`) if one is available |
| `KeyboardInterrupt` during training loses progress | It shouldn't — `train.py` saves `last.pt` on Ctrl+C. Resume with `python train.py --resume` |
| Video won't open in `infer_video.py` | Codec unsupported by your OpenCV build — try converting the source to `.mp4` (H.264) first, or install `opencv-python` with a fuller codec set |

---

## Future Improvements

- Swap the Haar cascade for a small DNN face detector (e.g. an ONNX SSD/RetinaFace
  model) to handle profile views and heavier occlusion
- Add temperature-scaled confidence calibration, since raw softmax confidence tends
  to run overconfident on noisy datasets like FER2013
- Multi-task head for facial landmarks alongside emotion, to support gaze/pose-aware
  applications
- Knowledge-distill EfficientNet-B0 into a smaller student model for edge devices
  (Raspberry Pi, mobile) without giving up as much accuracy as MobileNetV3-Small does
- Active-learning loop: log low-confidence real-world predictions for manual review
  and periodic retraining

---

## License

MIT License — see the code headers; free to use, modify, and distribute with
attribution. FER2013 itself is subject to its own dataset license/terms on Kaggle —
review those separately before any commercial use.
