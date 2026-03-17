# Padel Court Keypoint Detector

Deep learning network to detect padel court keypoints from broadcast videos or images. 

This project was inspired by and built upon the **[TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)** architecture, utilizing a modified **TrackNet** base for single-image heatmap inference, optimized with scalable throughput for deployment.

---

## 📌 Features
- **4 Keypoint Corners**: Standard calculation for top-left (`tol`), top-right (`tor`), bottom-left (`point_7`), and bottom-right (`point_9`).
- **Postprocessing Accuracy**: Uses `sympy` line-intersections to extrapolate high-fidelity grid alignment nodes even with dynamic camera vectors.
- **ONNX Acceleration**: Uses `onnxruntime` with memory-safe limiters with responses breaking **< 2.0s** standard thresholds.

---

## 🛠️ Installation & Setup

### 1. Clone & Dependencies
```bash
git clone <your-repo-url>
cd PadelCourtDetector

# Install packages
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights
Since weights are large, download them directly via the link below:
*   **[Download model_best.onnx (ONNX Weights)](https://drive.google.com/uc?export=download&id=1Yl1_x4uo_FVJmp2lD-MTG7CqEIJ10mL3)**
Save this to `exps/padel_v2/model_best.onnx`.

---

## 🚀 How to Run

### Local FastAPI Server
1. Start the server:
   ```bash
   PYTHONPATH=. python3 app.py
   ```
2. Send a prediction request via `curl`:
   ```bash
   curl -X POST -F "file=@/path/to/image.png" http://localhost:8000/predict
   ```

---

## ☁️ API Deployment (Fly.io)

This repository includes fully optimized Docker configurations addressing OOM (Out Of Memory) issues inside low-ram shared CPU grids by disabling the ONNX execution arena limiters.

To deploy or provision scale heights on Fly.io:
```bash
fly deploy
```

---

## 🤝 Credits
Special thanks to **[yastrebksv](https://github.com/yastrebksv)** for the original `TennisCourtDetector` codebase, architecture, and postprocessing workflow weights.
