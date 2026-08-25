Real-Time Facial Emotion Recognition

A computer-vision system for facial emotion recognition using transfer learning on FER2013. The project supports model training, evaluation, webcam inference, image and folder inference, video inference, and export to TorchScript and ONNX.

The implementation is organized around a small set of reusable components so that the same preprocessing and model pipeline can be used during training and inference.

Overview

The classifier predicts seven FER2013 emotion classes:

angry

disgust

fear

happy

neutral

sad

surprise

The project includes:

Transfer learning with EfficientNet-B0, MobileNetV3, or ResNet18

Training with augmentation and class-imbalance handling

Checkpointing and resumable training

Validation and test-set evaluation

Confusion matrix and classification reports

Real-time webcam inference

Single-image and batch image inference

Video-file inference

Multi-face detection and temporal smoothing

TorchScript and ONNX export

CPU, CUDA, and Apple Silicon (MPS) support

Note: The performance figures in this README are targets used for development and testing. They should not be interpreted as measured results unless they are backed by an evaluation run from this repository.

Table of Contents

Project Structure

Requirements

Installation

Dataset

Training

Evaluation

Model Export

Inference

Pipeline

Testing

Performance Targets

Troubleshooting

Future Improvements

License

Project Structure

emotion_detection/
├── Colab_Training.ipynb       # GPU training notebook and model export
├── config.py                  # Paths and training/inference configuration
├── utils.py                   # Shared model, metrics, detection and tracking utilities
├── dataset.py                 # Dataset validation, statistics and transforms
├── train.py                   # Training loop and checkpoint management
├── evaluate.py                # Evaluation and classification reports
├── export.py                  # TorchScript and ONNX export
├── infer_webcam.py             # Real-time webcam inference
├── infer_image.py              # Image and folder inference
├── infer_video.py              # Video-file inference
├── requirements.txt            # Python dependencies
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── checkpoints/                # Created during training
├── logs/                       # Training logs and TensorBoard files
├── outputs/                    # Evaluation and inference results
├── exported_models/            # Exported model files
└── screenshots/                # Webcam screenshots saved with `s`

Requirements

Recommended environment:

Python 3.9+

PyTorch

torchvision

OpenCV

NumPy

pandas

scikit-learn

TensorBoard

ONNX / ONNX Runtime for export and runtime validation

A CUDA-capable GPU can be used for training when available. CPU inference is also supported.

Installation

Create and activate a virtual environment:

python3 -m venv venv
source venv/bin/activate

On Windows:

python -m venv venv
venv\Scripts\activate

Install the project dependencies:

pip install -r requirements.txt

Verify the PyTorch installation:

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

Verify the dataset:

python dataset.py

CUDA

If the installed PyTorch package does not provide the CUDA build required by your system, install the appropriate version from the official PyTorch installation instructions before installing the remaining dependencies.

For example:

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

Dataset

This project uses FER2013.

The expected directory layout is:

dataset/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── val/
│   └── ...
│
└── test/
    └── ...

Class-directory matching is case-insensitive.

The project can validate the directory structure, count images by class, and check image files for corruption.

Dataset options

Option A — Existing dataset

If the FER2013 dataset is already available under dataset/, no additional download step is required.

The dataset supplied with the original project contains 35,887 images across the train, validation, and test directories.

Option B — Kaggle

The original project uses the following FER2013 Kaggle mirror:

https://www.kaggle.com/datasets/pankaj4321/fer-2013-facial-expression-dataset

Using the Kaggle CLI:

pip install kaggle
kaggle datasets download -d pankaj4321/fer-2013-facial-expression-dataset -p dataset --unzip

Follow Kaggle's documentation for configuring kaggle.json.

Option C — Google Colab

Upload the dataset to Google Drive and run the dataset-location section of Colab_Training.ipynb.

The notebook can search common Drive locations, or the dataset path can be specified explicitly.

Training

The default configuration is stored in config.py.

Run training with:

python train.py

Common alternatives:

python train.py --epochs 40 --batch-size 32 --lr 1e-4

For a smaller CPU-oriented model:

python train.py --model mobilenet_v3_small --img-size 160

Resume the latest saved training state:

python train.py --resume

Training workflow

The training pipeline includes:

Dataset validation and loading

Data augmentation on the training split

Optional pretrained ImageNet weights

Class-imbalance handling

Backbone warm-up

Fine-tuning

Learning-rate scheduling

Gradient clipping

Checkpoint saving

Validation after each epoch

Early stopping

TensorBoard and CSV logging

The best checkpoint is selected using validation macro-F1, while the most recent checkpoint is stored separately.

Training can also be interrupted safely with Ctrl+C; the latest checkpoint is saved so training can be resumed.

Monitor training

Start TensorBoard with:

tensorboard --logdir logs/tensorboard

Evaluation

Evaluate the best checkpoint on the test split:

python evaluate.py

Evaluate another checkpoint or split:

python evaluate.py --checkpoint checkpoints/last.pt --split val

Evaluation reports include:

Accuracy

Macro precision

Macro recall

Macro F1

Per-class accuracy

Classification report

Confusion matrix

Generated files are stored under:

outputs/
├── confusion_matrix_<split>.png
└── evaluation_report_<split>.json

Macro-averaged metrics are included because FER2013 is class-imbalanced. Looking only at overall accuracy can hide poor performance on less frequent classes.

Model Export

Export the best checkpoint:

python export.py

Export only selected formats:

python export.py --formats torchscript onnx

Exported files are written to:

exported_models/

The export pipeline supports:

PyTorch state dictionaries

TorchScript

ONNX

The ONNX export is checked against the PyTorch model to detect output mismatches before the exported model is written.

ONNX models can be used with compatible runtimes such as ONNX Runtime, OpenVINO, or TensorRT, depending on the deployment environment.

Inference

Webcam

Start real-time webcam inference:

python infer_webcam.py

Select a different camera:

python infer_webcam.py --source 1 --device cpu

For a CPU-oriented configuration:

python infer_webcam.py --skip-frames 1

Controls:

q — quit

s — save a screenshot

space — pause/resume

Image

Run inference on one image:

python infer_image.py --input photo.jpg

Run inference on a folder:

python infer_image.py --input ./photos/ --output-dir outputs/batch_run

Batch inference writes annotated images and a CSV containing the prediction information.

Video

Run inference on a video:

python infer_video.py --input clip.mp4

Specify an output path:

python infer_video.py \
    --input clip.mov \
    --output outputs/clip_annotated.mp4

Supported formats include:

.mp4

.avi

.mov

.mkv

The webcam, image, and video entry points share the same model preprocessing and drawing utilities to keep inference behavior consistent.

Pipeline

The inference pipeline is divided into four main stages.

1. Face detection

OpenCV's Haar cascade detector identifies frontal faces in the input frame.

Frames are converted to grayscale and histogram-equalized before detection. This can improve detection in uneven lighting while keeping the detector lightweight enough for CPU inference.

The main limitation is that Haar cascades are primarily designed for frontal faces. Profile views and heavy occlusion can reduce detection quality.

2. Emotion classification

Each detected face is cropped, resized, and passed independently through the trained CNN.

The classifier produces a seven-class softmax prediction corresponding to the FER2013 labels.

Keeping detection and classification separate makes it possible to replace the face detector without retraining the emotion classifier.

3. Temporal smoothing

For webcam inference, predictions can be stabilized across consecutive frames. This reduces rapid label changes when the model's confidence moves between similar classes.

Multi-face tracking also helps maintain consistent identities while people move through the frame.

4. Deployment

The trained model can be exported to TorchScript or ONNX for integration with compatible deployment environments.

Model Selection

Three backbone options are provided:

EfficientNet-B0

The default option when accuracy and computational cost need to be balanced.

It is suitable for GPU-based training and inference and provides a stronger accuracy-to-compute trade-off than a larger general-purpose model.

MobileNetV3-Small

A smaller option for CPU and edge-oriented inference.

Use it when real-time latency is more important than maximizing classification accuracy:

MODEL_NAME = "mobilenet_v3_small"
IMG_SIZE = 160

ResNet18

A conventional fallback architecture with a relatively small computational footprint.

It is useful when EfficientNet-B0 or MobileNetV3 is not suitable for the target environment.

There is no universally best backbone. The appropriate choice depends on the deployment hardware, latency requirement, and measured validation performance.

Data Augmentation

Training augmentation includes:

Random crop

Horizontal flip

Rotation

Brightness/contrast adjustment

Random erasing

Augmentation is applied to the training split only.

The purpose is to expose the model to variations in pose, lighting, framing, and partial occlusion that may occur during webcam inference while avoiding artificial changes to validation and test data.

Class Imbalance

FER2013 contains substantially different numbers of samples across emotion classes.

The training pipeline therefore supports inverse-frequency weighting/sampling so that frequent classes do not dominate the optimization process.

Macro precision, recall, and F1 are also reported during evaluation to make class-specific performance easier to inspect.

Testing

Before considering a trained model ready for deployment, test the complete inference pipeline under different conditions.

Scenario

Test

Expected behavior

Single face

One person facing the camera

One stable face box and prediction

Multiple faces

2–4 people in frame

Each detected face is processed independently

Low light

Dim room or backlit window

Detection may weaken, but the application should remain stable

Bright light

Direct or uneven lighting

No repeated false detections

Occlusion

Hand, glasses, hair, or mask

Missed detections should fail gracefully and recover

Distance

Move toward and away from camera

Detection should remain within the configured minimum face size

Fast motion

Turn or move quickly

Tracking should recover without duplicate identities

Camera angle

Frontal and profile views

Reduced recall is expected for extreme angles with Haar detection

Webcam

Full live pipeline

FPS and controls should remain responsive

Recorded video

Short test clip

Output timing and annotations should remain aligned

For every test, inspect:

Number of detected faces

Prediction stability

Confidence behavior

FPS

Application stability

Output file correctness

Performance Targets

The original project defines the following development targets for the default EfficientNet-B0 configuration:

Metric

Target

Possible adjustment

FER2013 test accuracy

≥ 70%

Tune training, verify augmentation and class weighting

CPU inference

≥ 20 FPS

Use MobileNetV3-Small, reduce image size, or skip frames

GPU inference

≥ 30 FPS

Batch detected faces and use a suitable CUDA device

Train/validation gap

Small

Review augmentation, regularization, and training duration

These are targets, not guaranteed results. Actual performance depends on the hardware, configuration, dataset split, and trained checkpoint.

FER2013 is a low-resolution dataset with ambiguous labels between visually similar emotions. Results should therefore be interpreted using more than a single accuracy number.

Troubleshooting

PyTorch is not installed

ModuleNotFoundError: No module named 'torch'

Activate the virtual environment and install the project dependencies:

pip install -r requirements.txt

Dataset cannot be found

Check that the following directories exist:

dataset/train
dataset/val
dataset/test

Each split should contain the seven emotion directories.

Dataset class is missing

Verify that the directory name exactly matches one of:

angry
disgust
fear
happy
neutral
sad
surprise

Capitalization is not important, but the class name must be one of the supported labels.

Corrupted images

Run:

python dataset.py

Remove or replace files reported as corrupted.

Camera cannot be opened

If the default camera is unavailable:

python infer_webcam.py --source 1

Also check whether another application is using the camera and whether the operating system has granted camera access to Python.

CUDA out-of-memory

Reduce the batch size:

python train.py --batch-size 16

You can also reduce the image size in the configuration.

Training accuracy remains low

Check:

Dataset structure

Class distribution

Pretrained-weight configuration

Learning rate

Number of epochs

Validation metrics

Avoid increasing training time blindly without checking the training and validation curves.

ONNX export fails

Install the export/runtime dependencies:

pip install onnx onnxruntime

Then rerun:

python export.py

Webcam FPS is too low

Try:

python infer_webcam.py --skip-frames 1

or use the smaller model:

mobilenet_v3_small

Also close other CPU/GPU-intensive applications and verify that the selected device is actually available.

Training is interrupted

The training script saves the latest checkpoint when interrupted. Resume with:

python train.py --resume

Video cannot be opened

If OpenCV cannot decode the source codec, convert the video to a broadly supported H.264 .mp4 file and try again.

Future Improvements

Potential next steps include:

Replace the Haar cascade with a lightweight DNN face detector for better profile and occlusion handling.

Add calibrated confidence scores instead of relying directly on raw softmax confidence.

Add facial landmark estimation for pose- and gaze-aware applications.

Distill EfficientNet-B0 into a smaller model for edge deployment.

Add an active-learning workflow for low-confidence predictions and periodic retraining.

Benchmark all supported backbones on the same hardware and dataset split.

Add automated integration tests for webcam, image, video, and export pipelines.

Track experiment configurations and model versions more systematically.

Reproducibility

For meaningful comparisons between experiments, record:

Model architecture

Image size

Batch size

Learning rate

Number of epochs

Random seed

Dataset split

Class-balancing configuration

Validation and test metrics

Hardware used for training and inference

A model should only be compared with another model when the evaluation setup is equivalent.

License

The project code is released under the MIT License. See the repository's license file for the complete terms.

FER2013 is a separate dataset and may have its own licensing and usage terms. Review the applicable dataset terms before using the dataset or a trained model for commercial purposes.
