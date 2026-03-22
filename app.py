from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
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

# --- Basic Auth Setup ---
# In production, the client will set the KEYPOINTS_API_KEY environment variable.
EXPECTED_API_KEY = os.environ.get("KEYPOINTS_API_KEY", "dev_api_key")

# --- Optional: Whitelist of Allowed Callers ---
# If this list is populated, ONLY the caller IDs listed here will be accepted.
# Leave it empty `[]` if you want to allow ANY non-empty caller ID.
ALLOWED_CALLER_IDS = [] 



async def verify_auth(
    x_caller_id: str = Header(..., alias="X-CALLER-ID", description="Identify the caller"),
    x_api_key: str = Header(..., alias="X-API-KEY", description="The whitelisted API Key")
):
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    if not x_caller_id.strip():
        raise HTTPException(status_code=400, detail="X-CALLER-ID cannot be empty")
    
    # Check against whitelist if it's set up
    if ALLOWED_CALLER_IDS and x_caller_id not in ALLOWED_CALLER_IDS:
        raise HTTPException(status_code=403, detail=f"Caller ID '{x_caller_id}' is not authorized")
        
    return x_caller_id



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

@app.get("/", dependencies=[Depends(verify_auth)])
async def root():
    return {"message": "Padel Court Detector API is running", "model": MODEL_PATH}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    caller_id: str = Depends(verify_auth)
):
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
