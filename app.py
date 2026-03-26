from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import io
import os
from predictor import CourtPredictor

app = FastAPI(
    title="Clutch Court Keypoint Detector API",
    description="Multi-sport API for detecting court keypoints (Padel, Pickleball).",
    version="1.1.0"
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

from typing import Optional

# --- Optional: Whitelist of Allowed Callers ---
# If this list is populated, ONLY the caller IDs listed here will be accepted.
# Leave it empty `[]` if you want to allow ANY non-empty caller ID.
ALLOWED_CALLER_IDS = [] 


async def verify_auth(
    x_caller_id: Optional[str] = Header(None, alias="X-CALLER-ID", description="Identify the caller"),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY", description="The whitelisted API Key")
):
    # 1. Check if headers are missing COMPLETELY
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-KEY header")
    if x_caller_id is None:
        raise HTTPException(status_code=401, detail="Missing X-CALLER-ID header")
        
    # 2. Check Key matches
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    
    # 3. Check ID is not empty or blank
    if not x_caller_id.strip():
        raise HTTPException(status_code=400, detail="X-CALLER-ID cannot be empty")
    
    # 4. Check against whitelist if it's set up
    if ALLOWED_CALLER_IDS and x_caller_id not in ALLOWED_CALLER_IDS:
        raise HTTPException(status_code=403, detail=f"Caller ID '{x_caller_id}' is not authorized")
        
    return x_caller_id

# --- Predictor Registry ---
PREDICTORS = {}
MODEL_CONFIGS = {
    "padel": "exps/padel_v4/model_best.onnx",
    "pickleball": "exps/pickleball_v1/model_best.onnx" # Placeholder until trained
}

def get_predictor_for_sport(sport: str):
    if sport not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"Sport '{sport}' not supported")
    
    if sport not in PREDICTORS:
        model_path = MODEL_CONFIGS[sport]
        if not os.path.exists(model_path):
            if sport == "padel":
                raise HTTPException(status_code=500, detail=f"Base Padel model not found at {model_path}")
            # For pickleball, if model doesn't exist yet, we just return error
            raise HTTPException(status_code=503, detail=f"Model for {sport} is still training or not deployed.")
        
        print(f"Loading {sport} model from: {model_path}")
        PREDICTORS[sport] = CourtPredictor(model_path, sport=sport)
    
    return PREDICTORS[sport]

@app.on_event("startup")
async def startup():
    print("Pre-loading models on startup...")
    for sport, path in MODEL_CONFIGS.items():
        if os.path.exists(path):
            get_predictor_for_sport(sport)

@app.get("/", dependencies=[Depends(verify_auth)])
async def root():
    loaded = list(PREDICTORS.keys())
    return {
        "message": "Clutch Court Keypoint Detector API is running",
        "loaded_models": loaded,
        "available_configs": MODEL_CONFIGS
    }

@app.post("/predict")
async def predict(
    sport: str = "padel",
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
        predictor = get_predictor_for_sport(sport)
        results = predictor.predict(img)
        return {
            "filename": file.filename,
            "sport": sport,
            "results": results
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
