import torch
import cv2
import numpy as np
import os
from tracknet import BallTrackerNet
from postprocess import postprocess

class PadelPredictor:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.kp_names = ['tol', 'tor', 'point_7', 'point_9', 'center']
        self.input_w, self.input_h = 1920, 1088
        self.scale = 2
        self.out_w, self.out_h = self.input_w // self.scale, self.input_h // self.scale
        
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path):
        model = BallTrackerNet(out_channels=5)
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        else:
            model.load_state_dict(ckpt)
        model = model.to(self.device)
        model.eval()
        return model

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
        inp = torch.tensor(inp).unsqueeze(0).float().to(self.device)
        
        # Inference
        with torch.no_grad():
            out = self.model(inp)
        pred = torch.sigmoid(out).cpu().numpy()[0]
        
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
    predictor = PadelPredictor('exps/padel_v2/model_last.pt')
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = predictor.predict(dummy_img)
    print(res)
