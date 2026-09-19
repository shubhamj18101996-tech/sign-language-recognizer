from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

mp_holistic = mp.solutions.holistic

# Initialize ONCE at startup (moved outside function)
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=0,      # Lite model (already set)
    enable_segmentation=False,
    min_detection_confidence=0.5  # Add this
)

app = FastAPI(title="MediaPipe Landmark Extractor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_landmarks_from_frame(frame_bgr):
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)  # Use global holistic instance
    landmarks = []
    # Pose (33 landmarks x [x,y,z,visibility])
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        landmarks.extend([0.0]*33*4)
    # Left hand (21 landmarks x [x,y,z])
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0]*21*3)
    # Right hand (21 landmarks x [x,y,z])
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0]*21*3)
    return landmarks

def extract_landmarks_from_video_path(video_path, max_frames=None, resize_width=640):
    cap = cv2.VideoCapture(video_path)
    frames_landmarks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if resize_width:
            h, w = frame.shape[:2]
            scale = resize_width / max(w, 1)
            if scale != 1:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
        lm = extract_landmarks_from_frame(frame)
        frames_landmarks.append(lm)
        if max_frames and len(frames_landmarks) >= max_frames:
            break
    cap.release()
    return np.array(frames_landmarks, dtype=np.float32)

@app.post("/extract/video")
async def extract_video(file: UploadFile = File(...), max_frames: int = 300):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_name = tmp.name
        tmp.write(await file.read())
    try:
        landmarks = extract_landmarks_from_video_path(tmp_name, max_frames=max_frames)
        return JSONResponse(content={"frames": landmarks.tolist(), "shape": landmarks.shape})
    finally:
        try:
            os.remove(tmp_name)
        except:
            pass

@app.post("/extract/image")
async def extract_image(file: UploadFile = File(...)):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error":"invalid image"})
    lm = extract_landmarks_from_frame(img)
    return JSONResponse(content={"landmarks": lm})
