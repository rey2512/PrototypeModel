import cv2
import numpy as np
from model.model import get_model

model = get_model()

def preprocess_frame(frame):
    """
    Preprocess a video frame to fit model input requirements.
    For demo, resize and flatten. Adjust based on your model input shape.
    """
    frame_resized = cv2.resize(frame, (5, 4))  # Example size 20 pixels (5*4=20)
    frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    flat = frame_gray.flatten() / 255.0  # Normalize
    return flat

def detect_deepfake(video_path):
    """
    Process video frame by frame and return prediction (average or last frame).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    preds = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x = preprocess_frame(frame)
        x = np.array(x).reshape(1, -1)  # Shape (1, 20)
        pred = model.predict(x)[0][0]
        preds.append(pred)

    cap.release()

    # Return average prediction as example
    if preds:
        return float(np.mean(preds))
    else:
        return None
