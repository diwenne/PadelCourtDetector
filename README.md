# Clutch Court Keypoints

A multi-sport court keypoint detection system (Padel, Pickleball) using a heatmap-regression CNN. Includes homography-based inference to recover missing keypoints.


Based on: [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) by yastrebksv.

---

## Keypoints

The model predicts 6 keypoints per frame:

```
  point_7 ────────── bottom_t ────────── point_9
     |                   |                   |
     |          (near/bottom of court)       |
     |                                       |
     |                                       |
     |          (far/top of court)           |
     |                   |                   |
    tol ──────────── tom ──────────────── tor
```

| Index | Name       | Position                      |
|-------|------------|-------------------------------|
| 0     | `tol`      | Top-left (far-side left)      |
| 1     | `tor`      | Top-right (far-side right)    |
| 2     | `point_7`  | Bottom-left (near-side left)  |
| 3     | `point_9`  | Bottom-right (near-side right)|
| 4     | `tom`      | Top-middle (midpoint of tol–tor) |
| 5     | `bottom_t` | Bottom-middle T-junction      |

> **Note**: `tom` and `bottom_t` are computed during training as midpoints. `bottom_t` can also be precomputed from annotation geometry when available.

---

## Architecture

**Model**: Modified TrackNet — a fully-convolutional encoder-decoder.

- **Encoder**: 3×Conv64 → Pool → 2×Conv128 → Pool → 3×Conv256 → Pool → 3×Conv512
- **Decoder**: Upsample → 3×Conv256 → Upsample → 2×Conv128 → Upsample → 2×Conv64 → Conv(out_channels)
- **Output**: 6-channel heatmap at half input resolution
- **Activation**: Sigmoid (applied post-inference in numpy for ONNX)
- **Loss**: MSE between predicted and ground-truth Gaussian heatmaps (radius=55px)
- **Input size**: `960×544` (half of `1920×1088`)
- **Parameters**: ~42 MB (ONNX)

Defined in [`tracknet.py`](tracknet.py).

---

## Pipeline Workflow

1. **Decode** — API receives image, reads into buffer via `cv2.imdecode`
2. **Preprocess** — Resize to `960×544`, normalize to `[0, 1]`, CHW layout, batch dim
3. **Inference** — `onnxruntime.InferenceSession` with CPU provider
4. **Sigmoid** — Manual `1/(1+exp(-x))` in numpy
5. **Postprocess** — Per-channel: threshold heatmap at 155, detect circles via `cv2.HoughCircles`, extract peak `(x, y)`
6. **Homography fallback** — If any keypoint is missing but ≥4 are detected, compute `cv2.findHomography` to a canonical 500×1000 top-down view, then inverse-warp the missing point(s) back to image coordinates
7. **Scale** — Map predictions from `960×544` back to original image dimensions
8. **Response** — Return JSON with keypoint names and `(x, y)` coordinates

---

## Model Versions

| Version | Keypoints | Key Changes | Best Accuracy (7px) |
|---------|-----------|-------------|---------------------|
| `padel_v1` | 4 (corners only) | Initial training from scratch | — |
| `padel_v2` | 5 (+ top_t) | Added top-T anchor, larger dataset | — |
| `padel_v3` | 6 (+ bottom_t) | Added bottom-T, homography inference | 94.16% |
| `padel_v4` | 6 | Fine-tuned on v4 dataset with precomputed bottom_t | **Current deployed** |

Active model: `exps/padel_v4/model_best.onnx`

### Accuracy (v3 benchmark)

| Threshold | Accuracy |
|-----------|----------|
| 7px       | 94.16%   |
| 10px      | 96.92%   |
| 15px      | 98.46%   |

---

## Project Structure

```
clutch-court-keypoints/
├── app.py                    # Multi-sport FastAPI server
├── predictor.py              # ONNX inference wrapper (sport-agnostic)
├── tracknet.py               # CNN architecture
├── postprocess.py            # Heatmap → (x,y) extraction
├── homography.py             # Shared homography utilities
├── court_reference.py        # Base court geometry classes
├── utils.py                  # General utilities
│
├── padel/                    # Padel-specific logic
│   ├── court_reference.py    # Padel dimensions (10m x 20m)
│   ├── dataset.py            # Padel dataset loader
│   ├── train.py              # Padel-specific training loop
│   └── tools/                # Padel dataset & inference tools
│
├── pickleball/               # Pickleball-specific logic
│   ├── court_reference.py    # Pickleball dimensions
│   └── dataset.py            # Pickleball dataset loader
│
├── tools/                    # Shared tools (export, inspect, etc.)
├── data/                     # Training/val data (gitignored)
├── exps/                     # Model checkpoints (.pt, .onnx)
└── Dockerfile                # Production container
```


---

## Installation

### Requirements

- Python 3.10+
- No GPU required for inference (ONNX CPU)
- GPU (CUDA) recommended for training

```bash
git clone <repo-url>
cd clutch-court-keypoints
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Pre-trained Weights

Download the v4 weights to `exps/padel_v4/`:

- **[model_best.pt (PyTorch, 127 MB)](https://drive.google.com/uc?export=download&id=1_h27hakOyMGu7UBv0OuKlShOVS8E5Ix5)**
- **[model_best.onnx (ONNX, 42 MB)](https://drive.google.com/uc?export=download&id=1qQhOGCz2vJrm4m3i6BY3VdKCJLgSLJDX)**

---

## API Usage

### Base URLs

| Environment | URL |
|---|---|
| **Production** | `https://clutch-keypoint.fly.dev` |
| **Local** | `http://localhost:8000` |

### Starting the Local Server

```bash
PYTHONPATH=. python3 app.py
# → Uvicorn running on http://0.0.0.0:8000
```

The model loads on startup (~2s). First request is instant after that.

---

### Endpoints

#### `GET /` — Health Check

```bash
curl -H "X-CALLER-ID: terminal" -H "X-API-KEY: dev_api_key" https://clutch-keypoint.fly.dev/
```

Response:
```json
{
  "message": "Padel Court Detector API is running",
  "model": "exps/padel_v4/model_best.onnx"
}
```

---

#### `POST /predict` — Detect Keypoints

Send an image, get back 6 padel court keypoint coordinates.

**Request**: `multipart/form-data` with a single field `file` containing the image.

**Headers**:
- `X-CALLER-ID`: (Required) String to identify the client application/caller.
- `X-API-KEY`: (Required) The whitelisted API Key for the environment.

**Response fields**:

| Field | Type | Description |
|---|---|---|
| `filename` | string | Original filename from the upload |
| `results` | array | 6 keypoint objects, always in the same order |
| `results[].name` | string | Keypoint identifier (see table below) |
| `results[].x` | int \| null | X pixel coordinate in the original image |
| `results[].y` | int \| null | Y pixel coordinate in the original image |

**Keypoint order** (always returned in this order):

| Index | `name` | Position on court |
|---|---|---|
| 0 | `tol` | Top-left (far-side left corner) |
| 1 | `tor` | Top-right (far-side right corner) |
| 2 | `point_7` | Bottom-left (near-side left corner) |
| 3 | `point_9` | Bottom-right (near-side right corner) |
| 4 | `tom` | Top-middle (midpoint of tol–tor) |
| 5 | `bottom_t` | Bottom-middle T-junction |

> If a keypoint is not detected directly **but** ≥4 other keypoints are found, the missing one is **automatically inferred via homography**. If fewer than 4 are detected, undetected keypoints return `x: null, y: null`.

---

### Examples

#### curl

```bash
# Single image (Headers are required)
curl -X POST \
  -H "X-CALLER-ID: my-app" \
  -H "X-API-KEY: dev_api_key" \
  -F "file=@frame.png" \
  https://clutch-keypoint.fly.dev/predict

# Save response to file
curl -s -X POST \
  -H "X-CALLER-ID: my-app" \
  -H "X-API-KEY: dev_api_key" \
  -F "file=@frame.png" \
  https://clutch-keypoint.fly.dev/predict | jq . > result.json
```

#### Python (requests)

```python
import requests

url = "https://clutch-keypoint.fly.dev/predict"
headers = {
    "X-CALLER-ID": "python-script",
    "X-API-KEY": "dev_api_key"
}

with open("frame.png", "rb") as f:
    resp = requests.post(url, headers=headers, files={"file": f})

if resp.status_code == 200:
    data = resp.json()
    for kp in data["results"]:
        if kp["x"] is not None:
            print(f"{kp['name']}: ({kp['x']}, {kp['y']})")
        else:
            print(f"{kp['name']}: not detected")
else:
    print(f"Error ({resp.status_code}): {resp.json()['detail']}")
```

#### Python (batch processing)

```python
import requests
import glob

url = "https://clutch-keypoint.fly.dev/predict"
headers = {
    "X-CALLER-ID": "batch-processor",
    "X-API-KEY": "dev_api_key"
}

for img_path in glob.glob("frames/*.png"):
    with open(img_path, "rb") as f:
        resp = requests.post(url, headers=headers, files={"file": f})
    
    if resp.status_code == 200:
        kps = {kp["name"]: (kp["x"], kp["y"]) for kp in resp.json()["results"]}
        print(f"{img_path}: tol={kps['tol']}, tor={kps['tor']}")
    else:
        print(f"Failed {img_path}: {resp.status_code}")
```

#### JavaScript (fetch)

```javascript
const formData = new FormData();
formData.append("file", fileInput.files[0]);

const resp = await fetch("https://clutch-keypoint.fly.dev/predict", {
  method: "POST",
  headers: {
    "X-CALLER-ID": "web-ui",
    "X-API-KEY": "dev_api_key",
  },
  body: formData,
});
const data = await resp.json();

if (resp.ok) {
  // data.results = [{name: "tol", x: 482, y: 312}, ...]
  data.results.forEach((kp) => {
    console.log(`${kp.name}: (${kp.x}, ${kp.y})`);
  });
} else {
  console.error("Auth error or invalid request", data);
}
```

---

### Error Responses

| Status | Condition | Response body |
|---|---|---|
| `400` | File is not an image | `{"detail": "File must be an image"}` |
| `400` | Image could not be decoded | `{"detail": "Could not decode image"}` |
| `422` | No `file` field in request | Validation error (FastAPI auto) |
| `500` | Internal model error | `{"detail": "Prediction error: ..."}` |

---

### Interactive Docs

FastAPI auto-generates interactive API docs:

- **Swagger UI**: `https://clutch-keypoint.fly.dev/docs`
- **ReDoc**: `https://clutch-keypoint.fly.dev/redoc`

You can test the `/predict` endpoint directly from the Swagger UI by clicking "Try it out" and uploading an image.

---

## Training

### 1. Prepare Dataset

Annotations come from the custom annotator as YAML per-video-clip in `court-keypoints-dataset/`:

```
court-keypoints-dataset/
  <UUID>/
    frame_001.jpg
    court-keypoints.yaml   # Contains: tol: [x, y], tor: [x, y], ...
```

Convert to training format:

```bash
python tools/prepare_padel_dataset.py \
  --input_dir ./court-keypoints-dataset \
  --output_dir ./data \
  --train_ratio 0.9
```

This produces `data/data_train.json`, `data/data_val.json`, and copies images to `data/images/`.

### Dataset JSON Schema

```json
[
  {
    "id": "UUID_frame_01",
    "kps": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
    "size": [1920, 1080]
  }
]
```

- `kps`: 4 corner keypoints in order `[tol, tor, point_7, point_9]`
- `size`: `[width, height]` of the original image
- `tom` and `bottom_t` are computed dynamically during training (see `dataset_padel.py`)

### 2. Train

```bash
python train_padel.py \
  --exp_id padel_v4 \
  --batch_size 4 \
  --num_epochs 200 \
  --lr 3e-4 \
  --val_intervals 5 \
  --patience 15 \
  --steps_per_epoch 500
```

Key flags:
- `--resume` — Resume from `exps/<exp_id>/model_last.pt`
- `--model_path <path>` — Fine-tune from a different checkpoint

Checkpoints saved to `exps/<exp_id>/`:
- `model_best.pt` — Best validation accuracy
- `model_last.pt` — Latest checkpoint (for resuming)

Training logs TensorBoard events to `exps/<exp_id>/plots/`.

### 3. Export to ONNX

```bash
python tools/export_v3_onnx.py \
  --model_path exps/padel_v4/model_best.pt \
  --output_onnx exps/padel_v4/model_best.onnx
```

### 4. Run Inference on Val Set

```bash
python tools/run_inference.py \
  --model_path exps/padel_v4/model_best.pt \
  --num_samples 20 \
  --output_dir results/inference_results_v4
```

---

## Deployment (Fly.io)

Deployed as `clutch-keypoint` in the `clutch` org on Fly.io.

```bash
fly deploy
```

### Configuration (`fly.toml`)

- **Region**: `ord` (Chicago)
- **VM**: 8 shared CPUs, 2 GB RAM
- **Auto-stop**: Machines stop when idle, auto-start on request
- **Min machines**: 1

### Environment Variables & Secrets

To secure the API in production with a custom `X-API-KEY`, you must set the `KEYPOINTS_API_KEY` secret on Fly.io:

```bash
fly secrets set KEYPOINTS_API_KEY="your_secure_api_key_here"
```

Once set, any client request to the production API will need to include the API key.
If not set, it defaults to the `dev_api_key` placeholder specified in `app.py`.

### Caller ID Validation (Optional)

You can restrict which callers access the API using the `ALLOWED_CALLER_IDS` list in `app.py`.

- **Current Behavior**: Set to empty `[]`. ANY non-empty `X-CALLER-ID` is accepted (as long as they have the correct API key).
- **Restricted Behavior**: Add specific allowed titles to the list in `app.py`:
  ```python
  ALLOWED_CALLER_IDS = ["frontend", "dashboard", "mobile_app"]
  ```

If someone tries to send a request with a header like `-H "X-CALLER-ID: unknown"`, they will receive a `403 Forbidden` response.




### Dockerfile Notes

- Base image: `python:3.10-slim`
- Only production files are copied (no training code, no dataset)
- ONNX weights are baked into the image via `COPY exps/ /app/exps/`
- CPU memory arena is disabled in `predictor.py` to prevent RAM spikes on shared VMs

### Updating the Deployed Model

1. Export new ONNX: `python tools/export_v3_onnx.py --model_path ... --output_onnx exps/padel_v4/model_best.onnx`
2. Run `fly deploy`
3. Verify: `curl https://clutch-keypoint.fly.dev/`

---

## Homography Inference

When fewer than 6 keypoints are detected but at least 4 are available, the system uses `cv2.findHomography` to infer the missing ones:

1. Map detected image points → canonical top-down court coordinates (500×1000 px)
2. Compute 3×3 homography matrix with RANSAC
3. For each missing keypoint, transform its known court position back to image coordinates via the inverse homography

This is critical for `bottom_t` which is often occluded by players near the net.

Implementation: [`homography_padel.py`](homography_padel.py)

---

## Court Reference Geometry

Padel court dimensions (FIP regulations) defined in [`court_reference_padel.py`](court_reference_padel.py):

- **Court**: 10m × 20m
- **Service line**: 6.95m from net
- **Net height**: 0.88m (center), 0.92m (ends)

The reference model maps to millimeter coordinates internally (10000 × 20000 units).

---

## Tools Reference

| Script | Purpose |
|--------|---------|
| `tools/prepare_padel_dataset.py` | Convert YAML annotations → training JSON |
| `tools/export_v3_onnx.py` | Export PyTorch checkpoint → ONNX |
| `tools/run_inference.py` | Batch inference with error metrics on val set |
| `tools/infer_padel.py` | Standalone PyTorch inference script |
| `tools/infer_padel_homography.py` | Inference + homography visualization |
| `tools/compare_models.py` | Compare predictions across model versions |
| `tools/inspect_heatmap.py` | Visualize raw heatmap output for debugging |
| `tools/quick_detect.py` | Quick single-image keypoint detection |
| `tools/precompute_dataset_t.py` | Precompute T-junction coordinates for dataset |
| `tools/crop_bottom_t.py` | Crop and inspect bottom-T image regions |
