import cv2
import numpy as np
import os
import onnxruntime as ort
from postprocess import postprocess

class PadelPredictor:
    def __init__(self, model_path, device=None):
        self.kp_names = ['tol', 'tor', 'point_7', 'point_9', 'center']
        self.input_w, self.input_h = 1920, 1088
        self.scale = 2
        self.out_w, self.out_h = self.input_w // self.scale, self.input_h // self.scale
        
        # Initialize ONNX session
        print(f"Initializing ONNX InferenceSession for {model_path}")
        sess_options = ort.SessionOptions()
        sess_options.enable_cpu_mem_arena = False  # Critical for preventing 1.9GB RAM spike
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 4
        
        self.sess = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

    def predict(self, img):
        """
        Run inference on an OpenCV image (BGR).
        Returns a list of dictionaries with keypoint names and coordinates.
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
        for ch in range(min(4, pred.shape[0])):
            hm = (pred[ch] * 255).astype(np.uint8)
            x_out, y_out = postprocess(hm, scale=1)
            
            kp_info = {"name": self.kp_names[ch], "x": None, "y": None}
            if x_out is not None:
                # Scale back to original image coords
                kp_info["x"] = int(x_out * w / self.out_w)
                kp_info["y"] = int(y_out * h / self.out_h)
            
            results.append(kp_info)
            
        return results

if __name__ == "__main__":
    # Quick test
    predictor = PadelPredictor('exps/padel_v2/model_best.onnx')
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = predictor.predict(dummy_img)
    print(res)
