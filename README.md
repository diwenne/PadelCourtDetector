# Padel Court Keypoint Detector

Internal documentation for the padel court keypoint detection system.

Based on: [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) by yastrebksv.

---

## Features
- **4 Keypoint Corners**: `tol` (top-left), `tor` (top-right), `point_7` (bottom-left), `point_9` (bottom-right).
- **Postprocessing**: Uses `sympy` line-intersections to extrapolate grid nodes.
- **ONNX Acceleration**: Uses `onnxruntime` with disabled memory arena limiters for deployment.

---

## Pipeline Workflow
1. **Decode**: API receives image and reads into buffer (`cv2.imdecode`).
2. **Preprocess**: Resizes image to `960x544`, normalizes array to `[0, 1]`, and expands to batch dimensions.
3. **Inference**: Runs via `onnxruntime.InferenceSession` across safe core buckets.
4. **Postprocess**: Applies manual sigmoid layout, extracts heatmap maximum-peaks per channel.
5. **Coordinate Scaling**: Adjusts prediction back into uploaded image dimensions.
6. **Response**: Dispatches JSON coordinate coordinates list.

---

## Installation

```bash
git clone <repo-url>
cd PadelCourtDetector
pip install -r requirements.txt
```

### Pre-trained Weights
Download the weights to `exps/padel_v2/model_best.onnx`:
- **[Download model_best.onnx](https://drive.google.com/uc?export=download&id=1Yl1_x4uo_FVJmp2lD-MTG7CqEIJ10mL3)**

---

## Running

### Local API Server
```bash
PYTHONPATH=. python3 app.py
```

Test with `curl`:
```bash
curl -X POST -F "file=@/path/to/image.png" http://localhost:8000/predict
```

---

## Deployment (Fly.io)

```bash
fly deploy
```
Configurations optimize OOM scenarios on shared cores via ONNX pool settings.
