# Padel Court Keypoint Detector

Deep learning model for detecting 4 padel court keypoints from broadcast video frames.

Uses a TrackNet-based architecture to predict heatmaps for 4 court corners + 1 center point. Trained on ~22,900 labeled padel court images.

## Model

**Current model:** `exps/padel_v1/model_last.pt` (~127 MB, epoch 95)

- **Architecture:** TrackNet (5-channel heatmap output)
- **Input:** 1920×1088 → resized to 960×544 internally
- **Output:** 4 keypoints (tol, tor, point_7, point_9) + center
- **Accuracy:** median ~3.4px error on validation set (at 960×544 resolution)

## Inference Pipeline

```
Image → Resize to 960×544 → Normalize (0-1) → TrackNet → 5 Sigmoid Heatmaps → HoughCircles → Keypoint (x, y)
```

1. **Load image** and resize to 960×544 (model output resolution)
2. **Normalize** pixel values to 0–1, convert to `[C, H, W]` tensor
3. **Forward pass** through TrackNet → produces 5 heatmap channels (one per keypoint)
4. **Sigmoid** activation to get probability maps (0–1)
5. **Postprocess** each heatmap: threshold at 155 → `cv2.HoughCircles` to find peak → extract `(x, y)`
6. **Scale** coordinates back to original image resolution

Key files: `infer_padel.py` (single image), `run_inference.py` (batch), `postprocess.py` (heatmap → keypoint)

## Keypoints

| ID | Name | Description |
|----|------|-------------|
| 0 | `tol` | Top-left court corner |
| 1 | `tor` | Top-right court corner |
| 2 | `point_7` | Bottom-left court corner |
| 3 | `point_9` | Bottom-right court corner |
| 4 | `center` | Center (diagonal intersection) |

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Run Inference on Validation Images
```bash
python run_inference.py --num_samples 10
```
Results are saved to `inference_results/`. Each output image shows:
- **Colored filled circles** = predicted keypoints
- **Cyan outlined circles** = ground truth
- **White lines** = detected court outline
- **Error text** = per-keypoint pixel distance from GT

### Run Inference on a Single Image
```bash
python infer_padel.py --model_path exps/padel_v1/model_last.pt --input_path your_image.jpg --output_path output.png
```

### Train
```bash
python train_padel.py --exp_id padel_v2 --batch_size 4 --num_epochs 100
```
Expects dataset in `./data/` with `data_train.json`, `data_val.json`, and `images/`.

## Directory Structure

```
├── tracknet.py              # Model architecture (TrackNet)
├── train_padel.py           # Training script
├── infer_padel.py           # Single-image inference
├── run_inference.py         # Batch inference on val set
├── dataset_padel.py         # Dataset loader
├── base_trainer.py          # Training loop
├── base_validator.py        # Validation / metrics
├── postprocess.py           # Heatmap → keypoint extraction
├── postprocess_padel.py     # Padel-specific postprocessing
├── court_reference_padel.py # Padel court reference coordinates
├── homography_padel.py      # Padel homography utils
├── prepare_padel_dataset.py # Dataset preparation from raw annotations
├── utils.py                 # Gaussian drawing, line intersection, etc.
├── requirements.txt
├── exps/padel_v1/           # Trained model checkpoint
├── data/                    # Training data (images + JSON annotations)
├── court-keypoints-dataset/ # Raw dataset from GCS
├── inference_results/       # Inference output images
└── _original_tennis/        # Original tennis court detector code (archived)
```

## Dataset

- **Source:** `gs://clutch-research/court-keypoints-dataset`
- **Format:** Each sample has 4 keypoint annotations `[x, y]` at original image resolution
- **Split:** 22,939 train / 2,549 validation
- **Resolution:** Mostly 1920×1080

## Original Work

Based on [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector). Original tennis court detection code is archived in `_original_tennis/`.
