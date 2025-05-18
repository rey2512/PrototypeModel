from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from model.model import get_model
import numpy as np
import shutil
import os
import uuid
import logging
import traceback

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# Enable CORS for frontend running anywhere (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["prototype-model-ukhx-9xcvfikl5-prasenjeet-singhs-projects.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files from /static
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

# Serve index.html at root
@app.get("/")
async def root():
    return FileResponse(os.path.join("static", "index.html"))

model = get_model()



@app.post("/detect")
async def detect_deepfake(video: UploadFile = File(...)):
    try:
        temp_video_path = f"temp_{uuid.uuid4().hex}_{video.filename}"
        with open(temp_video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        prediction = process_video_and_predict(temp_video_path)

        os.remove(temp_video_path)

        label = "Fake" if prediction >= 0.5 else "Real"

        return JSONResponse(content={"prediction": prediction, "label": label})

    except Exception as e:
        logger.error(f"Error in /detect: {str(e)}")
        logger.error(traceback.format_exc())
        return JSONResponse(content={"error": str(e)}, status_code=500)


# --------------------------------------------------


import cv2

def process_video_and_predict(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        frame = cv2.resize(frame, (224, 224)) / 255.0
        frames.append(frame)
        if len(frames) >= 10:
            break

    cap.release()

    if len(frames) == 0:
        raise ValueError("No frames extracted from video.")

    frames = np.array(frames)
    frames = frames.reshape((1, frames.shape[0], 224, 224, 3))

    logger.info(f"Input frames shape for model prediction: {frames.shape}")

    pred = model.predict(frames)
    logger.info(f"Model prediction output: {pred}")

    return float(pred[0][0])
