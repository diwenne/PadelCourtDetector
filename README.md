# Padel Court Keypoint Detector

Internal documentation for the padel court keypoint detection system.

Based on: [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) by yastrebksv.

---

## Features
- **6 Keypoint Anchors**: `tol` (top-left), `tor` (top-right), `point_7` (bottom-left), `point_9` (bottom-right), `top_t` (middle-top T), `bottom_t` (middle-bottom T).
- **Homography Inference**: Incorporates cv2 homography mathematically to extrapolate any missing or occluded points automatically if at least 4 valid anchors are found.
- **ONNX Acceleration**: Uses `onnxruntime` with disabled memory arena limiters for deployment.

---

## Pipeline Workflow
1. **Decode**: API receives image and reads into buffer (`cv2.imdecode`).
2. **Preprocess**: Resizes image to `960x544`, normalizes array to `[0, 1]`, and expands to batch dimensions.
3. **Inference**: Runs via `onnxruntime.InferenceSession` across safe core buckets.
4. **Postprocess**: Applies manual sigmoid layout, extracts heatmap maximum-peaks per channel.
5. **Homography Check**: Checks for missing keypoints. Solves transformation matrix via `cv2.findHomography` to infer missing `(x, y)` locations seamlessly.
6. **Coordinate Scaling**: Adjusts prediction back into uploaded image dimensions.
7. **Response**: Dispatches JSON coordinate coordinates list.

---

## Model Metadata
- **Validation Accuracy**: 94.16% (@ Epoch ~104)
- **Architecture**: Modified TrackNet (6-channel single-frame heatmap)

---

## Installation

```bash
git clone <repo-url>
cd PadelCourtDetector
pip install -r requirements.txt
```

### Pre-trained Weights
Download the primary v3 weights from GCS and export into ONNX format for the API:
```bash
gsutil cp gs://clutch-research/padel-model/model.pt exps/padel_v3/model_best.pt
python3 export_v3_onnx.py
```

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
