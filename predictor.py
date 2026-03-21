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

class PadelPredictor:
    """ONNX-based padel court keypoint predictor with homography fallback.
    
    Loads a pre-exported ONNX model and provides a single `predict()` method
    that returns 6 keypoints per image. Missing keypoints are automatically
    inferred using homography when ≥4 other points are detected.
    
    Attributes:
        kp_names:    List of 6 keypoint names in channel order.
        input_w/h:   Full input resolution (1920×1088).
        scale:       Downscale factor (2), so actual model input = 960×544.
        out_w/h:     Model input/output resolution (960×544).
        sess:        ONNX InferenceSession.
        input_name:  ONNX input tensor name.
        output_name: ONNX output tensor name.
    """
    def __init__(self, model_path, device=None):
        """Initialize the predictor with an ONNX model.
        
        Args:
            model_path: Path to the .onnx model file.
            device:     Unused (kept for API compatibility). Always uses CPU.
        """
        self.kp_names = ['tol', 'tor', 'point_7', 'point_9', 'tom', 'bottom_t']
        self.input_w, self.input_h = 1920, 1088
        self.scale = 2
        self.out_w, self.out_h = self.input_w // self.scale, self.input_h // self.scale
        
        # Initialize ONNX session
        print(f"Initializing ONNX InferenceSession for {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False  # Critical for preventing 1.9GB RAM spike
        sess_options.intra_op_num_threads = 8
        sess_options.inter_op_num_threads = 8
        
        self.sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, img):
        """Run inference on a single OpenCV image (BGR).
        
        Full pipeline: preprocess → ONNX inference → sigmoid → postprocess
        → scale to original coords → homography fallback for missing points.
        
        Args:
            img: OpenCV BGR image of any size (H, W, 3).
        
        Returns:
            list[dict]: 6 dictionaries, each with keys:
                - 'name': Keypoint name (str)
                - 'x': X coordinate in original image pixels (int or None)
                - 'y': Y coordinate in original image pixels (int or None)
                
                If a keypoint is not detected and cannot be inferred via
                homography, x and y will be None.
        """
        h, w = img.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(img, (self.out_w, self.out_h))
        inp = (img_resized.astype(np.float32) / 255.)
        inp = np.rollaxis(inp, 2, 0)
        inp = np.expand_dims(inp, axis=0).astype(np.float32)
        
        # Inference
        out = self.sess.run([self.output_name], {self.input_name: inp})[0]
        
        # Apply Sigmoid manually in numpy
        pred = 1 / (1 + np.exp(-out[0]))
        
        # Postprocess
        results = []
        detected_pts_tuple = []
        for ch in range(pred.shape[0]):
            hm = (pred[ch] * 255).astype(np.uint8)
            x_out, y_out = postprocess(hm, scale=1)
            
            kp_info = {"name": self.kp_names[ch], "x": None, "y": None}
            if x_out is not None:
                # Scale back to original image coords
                x_scaled = int(x_out * w / self.out_w)
                y_scaled = int(y_out * h / self.out_h)
                kp_info["x"] = x_scaled
                kp_info["y"] = y_scaled
                detected_pts_tuple.append((x_scaled, y_scaled))
            else:
                detected_pts_tuple.append(None)
            
            results.append(kp_info)
            
        # Try homography to infer missing points
        if None in detected_pts_tuple and sum(p is not None for p in detected_pts_tuple) >= 4:
            from homography_padel import compute_homography, warp_point_to_image
            H = compute_homography(detected_pts_tuple)
            if H is not None:
                output_w, output_h = 500, 1000
                dst_coords = [
                    (0, output_h),
                    (output_w, output_h),
                    (0, 0),
                    (output_w, 0),
                    (output_w // 2, output_h),
                    (output_w // 2, 0)
                ]
                for i, kp in enumerate(results):
                    if kp["x"] is None:
                        inferred = warp_point_to_image(dst_coords[i], H)
                        if inferred is not None:
                            kp["x"], kp["y"] = inferred
                            
        return results

if __name__ == "__main__":
    # Quick test
    predictor = PadelPredictor('exps/padel_v3/model_best.onnx')
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = predictor.predict(dummy_img)
    print(res)
