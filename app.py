from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import io
import os
from predictor import PadelPredictor

app = FastAPI(
    title="Padel Court Detector API",
    description="API for detecting padel court keypoints in broadcast video frames.",
    version="1.0.0"
)

# Enable CORS for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Predictor
MODEL_PATH = 'exps/padel_v4/model_best.onnx'
if not os.path.exists(MODEL_PATH):
    # Fallback or error
    MODEL_PATH = 'exps/padel_v4/model_best.onnx'

predictor = None

def get_predictor():
    global predictor
    if predictor is None:
        print(f"Loading model from: {MODEL_PATH}")
        predictor = PadelPredictor(MODEL_PATH)
    return predictor

@app.on_event("startup")
async def startup():
    print("Pre-loading model on startup...")
    get_predictor()

@app.get("/")
async def root():
    return {"message": "Padel Court Detector API is running", "model": MODEL_PATH}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    # Run prediction
    try:
        pred = get_predictor()
        results = pred.predict(img)
        return {
            "filename": file.filename,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
