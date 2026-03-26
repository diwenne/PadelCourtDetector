import cv2

img = cv2.imread('imgs/padel_frame2.png')
if img is not None:
    # Coordinate from inspect_heatmap (942, 931)
    x, y = 942, 931
    h, w = img.shape[:2]
    
    # Safe crop bounds
    crop_size = 100
    y1 = max(y - crop_size, 0)
    y2 = min(y + crop_size, h)
    x1 = max(x - crop_size, 0)
    x2 = min(x + crop_size, w)
    
    crop = img[y1:y2, x1:x2].copy()
    
    # Draw center marker
    center_y = y - y1
    center_x = x - x1
    cv2.circle(crop, (center_x, center_y), 5, (0,0,255), -1)
    
    cv2.imwrite('results/crop_bottom_t.png', crop)
    print("Crop saved to results/crop_bottom_t.png")
