import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sys
import os
import cv2
from predictor import PadelPredictor

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('img_path', type=str, help='Path to image')
    parser.add_argument('--model_path', type=str, default='exps/padel_v3/model_best.onnx', help='Path to ONNX model')
    args = parser.parse_args()
    
    img_path = args.img_path
    model_path = args.model_path
    if not os.path.exists(img_path):
        print(f"Error: File {img_path} not found.")
        sys.exit(1)
        
    print(f"Loading model: {model_path}")
    predictor = PadelPredictor(model_path)
    
    print(f"Processing: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image.")
        sys.exit(1)
        
    results = predictor.predict(img)
    
    # Draw results
    for kp in results:
        if kp['x'] is not None:
            cv2.circle(img, (kp['x'], kp['y']), 10, (0, 255, 0), -1)
            cv2.putText(img, kp['name'], (kp['x'] + 12, kp['y']), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    out_name = "detected_" + os.path.basename(img_path)
    cv2.imwrite(out_name, img)
    print(f"\nDone! Results saved to: {out_name}")
    print("Coordinates:")
    for kp in results:
        print(f"  {kp['name']}: {kp['x']}, {kp['y']}")

if __name__ == "__main__":
    main()
