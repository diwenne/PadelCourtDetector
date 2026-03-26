"""
Production ONNX inference wrapper for padel court keypoint detection.

This is the primary inference module used by the FastAPI app (app.py).
It loads an ONNX model, runs inference on a single image, and returns
6 keypoint predictions with automatic homography fallback for any
missing keypoints.

Inference pipeline:
    1. Resize input image to 960×544 (output_width × output_height)
    2. Normalize to [0, 1], convert to CHW layout, add batch dimension
    3. Run ONNX inference (CPU provider)
    4. Apply sigmoid manually in numpy: 1/(1+exp(-x))
    5. Per-channel: multiply by 255, cast to uint8, run HoughCircles
    6. Scale detected coordinates back to original image dimensions
    7. If any keypoints are missing but ≥4 detected, use homography
       to infer the missing ones via inverse projection

ONNX session configuration:
    - CPU memory arena disabled (prevents 1.9GB RAM spike on shared VMs)
    - 8 intra/inter-op threads for parallel execution
    - CPUExecutionProvider only (no GPU required for inference)

Used by:
    - app.py (FastAPI production server)
"""
import cv2
import numpy as np
import os
import onnxruntime as ort
from postprocess import postprocess

class CourtPredictor:
    """ONNX-based court keypoint predictor with sport-specific configuration."""
    
    def __init__(self, model_path, sport='padel'):
        self.sport = sport
        if sport == 'padel':
            from padel.court_reference import PadelCourtReference
            self.court_ref = PadelCourtReference()
            self.kp_names = ['tol', 'tor', 'point_7', 'point_9', 'tom', 'bottom_t']
        elif sport == 'pickleball':
            from pickleball.court_reference import PickleballCourtReference
            self.court_ref = PickleballCourtReference()
            self.kp_names = ['tol', 'tor', 'bol', 'bor', 'tom', 'bom']
        else:
            raise ValueError(f"Unknown sport: {sport}")

        self.input_w, self.input_h = 1920, 1088
        self.scale = 2
        self.out_w, self.out_h = self.input_w // self.scale, self.input_h // self.scale
        
        # Initialize ONNX session
        print(f"Initializing ONNX InferenceSession for {sport.upper()} model: {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False
        sess_options.intra_op_num_threads = 8
        sess_options.inter_op_num_threads = 8
        
        self.sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, img):
        h, w = img.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(img, (self.out_w, self.out_h))
        inp = (img_resized.astype(np.float32) / 255.)
        inp = np.rollaxis(inp, 2, 0)
        inp = np.expand_dims(inp, axis=0).astype(np.float32)
        
        # Inference
        out = self.sess.run([self.output_name], {self.input_name: inp})[0]
        pred = 1 / (1 + np.exp(-out[0]))
        
        # Postprocess
        results = []
        detected_pts_tuple = []
        for ch in range(len(self.kp_names)):
            hm = (pred[ch] * 255).astype(np.uint8)
            x_out, y_out = postprocess(hm, scale=1)
            
            kp_info = {"name": self.kp_names[ch], "x": None, "y": None}
            if x_out is not None:
                x_scaled = int(x_out * w / self.out_w)
                y_scaled = int(y_out * h / self.out_h)
                kp_info["x"], kp_info["y"] = x_scaled, y_scaled
                detected_pts_tuple.append((x_scaled, y_scaled))
            else:
                detected_pts_tuple.append(None)
            results.append(kp_info)
            
        # Try homography to infer missing points
        if None in detected_pts_tuple and sum(p is not None for p in detected_pts_tuple) >= 4:
            from homography import compute_homography, warp_point_to_image
            H = compute_homography(detected_pts_tuple, self.court_ref)
            if H is not None:
                # Map reference points to visualization size (500x1000) for consistent projection
                output_w, output_h = 500, 1000
                ref_pts = np.array(self.court_ref.key_points, dtype=np.float32)
                dst_coords_viz = np.zeros_like(ref_pts)
                dst_coords_viz[:, 0] = ref_pts[:, 0] / self.court_ref.court_width * output_w
                dst_coords_viz[:, 1] = ref_pts[:, 1] / self.court_ref.court_length * output_h

                for i, kp in enumerate(results):
                    if kp["x"] is None and i < len(dst_coords_viz):
                        inferred = warp_point_to_image(dst_coords_viz[i], H)
                        if inferred is not None:
                            kp["x"], kp["y"] = inferred
                            
        return results

if __name__ == "__main__":
    # Quick test
    predictor = CourtPredictor('exps/padel_v4/model_best.onnx', sport='padel')
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = predictor.predict(dummy_img)
    print(res)
