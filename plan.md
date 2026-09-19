I'll give a step‑by‑step, copy‑pasteable guide with everything you need (commands, files, and exact next actions) so you can build the MediaPipe extractor, a small PyTorch classifier, test real‑time with a browser → Spring → Python pipeline, and finally export to ONNX for Java. I assume Windows (PowerShell) and minimal prior experience.

Checklist (high level)
- [ ] Install system software: Python 3.9+ (or 3.10/3.11), pip, virtualenv; Git if needed.
- [ ] Create Python workspace and install dependencies.
- [ ] Implement and run MediaPipe extractor microservice (FastAPI) that returns landmarks per frame.
- [ ] Capture small dataset (your own clips) or download public datasets (WLASL, How2Sign) and extract landmarks to .npy files.
- [ ] Implement PyTorch LSTM classifier skeleton, train on extracted landmarks for a small vocabulary.
- [ ] Implement Python inference service that loads the trained model and returns predicted labels for incoming frames/sequences.
- [ ] Create a minimal browser client to capture webcam frames and send to Spring or directly to Python services (WebSocket or REST).
- [ ] (Optional) Export PyTorch model to ONNX and integrate into Java via ONNX Runtime; modify `ModelConfig` so tests don't fail while developing.
- [ ] Evaluate, iterate, and add segmentation & sentence assembly logic.

Important prerequisites (what to install)
- Java & Maven are already in your repo (you have a Spring project).
- Install Python 3.9+ (recommend 3.10 or 3.11) from python.org — choose "Add Python to PATH" during install.
- PowerShell commands below assume you use PowerShell (Windows).

Step 0 — System installs
1. Install Python (if not installed):
    - Download installer from python.org and install 3.10/3.11, enable "Add to PATH".
2. Verify:
   ```powershell
   python --version
   pip --version
   ```
3. (Optional) Install Git if you want to clone datasets/tools:
   https://git-scm.com/download/win

Step 1 — Create Python project and virtual environment
1. In your project root:
   ```powershell
   cd C:\Users\lavib\Downloads\sign-language-recognizer
   mkdir mediapipe_service
   cd mediapipe_service
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Create `requirements.txt` with these lines:
   ```
   fastapi
   uvicorn[standard]
   mediapipe
   opencv-python
   numpy
   torch      # For training and inference (CPU) - if you have GPU, install appropriate torch build
   torchvision
   python-multipart
   ```
   Note: `torch` best installation sometimes requires a special command for GPU; the command above installs CPU-only. If you have an NVIDIA GPU and want GPU acceleration, follow PyTorch instructions on https://pytorch.org for the correct pip URL.

3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

Step 2 — MediaPipe extractor microservice (full minimal code)
Create `extractor.py` in `mediapipe_service` with this content:

```python
# extractor.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

mp_holistic = mp.solutions.holistic

app = FastAPI(title="MediaPipe Landmark Extractor")

def extract_landmarks_from_frame(frame_bgr):
    with mp_holistic.Holistic(static_image_mode=True,
                              model_complexity=1,
                              enable_segmentation=False) as holistic:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        landmarks = []
        # Pose
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
        else:
            landmarks.extend([0.0]*33*4)  # pose has 33 landmarks
        # Left hand
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0]*21*3)
        # Right hand
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
    # Save uploaded file to temp
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_name = tmp.name
        tmp.write(await file.read())
    try:
        landmarks = extract_landmarks_from_video_path(tmp_name, max_frames=max_frames)
        return JSONResponse(content={"frames": landmarks.tolist(), "shape": landmarks.shape})
    finally:
        os.remove(tmp_name)

@app.post("/extract/image")
async def extract_image(file: UploadFile = File(...)):
    # Accept a single image and return landmark vector
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error":"invalid image"})
    lm = extract_landmarks_from_frame(img)
    return JSONResponse(content={"landmarks": lm})
```

Run the extractor:
```powershell
# activate virtualenv if not already
.\.venv\Scripts\Activate.ps1
uvicorn extractor:app --host 127.0.0.1 --port 8000
```

Test the extractor (PowerShell example to POST a small video or image):
```powershell
# send a sample image
curl -X POST "http://127.0.0.1:8000/extract/image" -H "Content-Type: multipart/form-data" -F "file=@C:\path\to\frame.jpg"
# send a sample video (returns per-frame landmarks)
curl -X POST "http://127.0.0.1:8000/extract/video" -H "Content-Type: multipart/form-data" -F "file=@C:\path\to\sample_clip.mp4"
```

Step 3 — Collect data & prepare training examples (very small / simple approach)
Option A — Record your own clips (recommended to start small):
- Use your webcam to record short clips (3–5 seconds) for each sign you want to train (6–20 signs).
- Save each clip as `label_name_01.mp4`, `label_name_02.mp4`, etc. in a folder `data/raw/`.

Option B — Download public datasets (WLASL, How2Sign). Those datasets are large and require extra preprocessing. Start with your own small dataset.

Create a simple script `preprocess.py` that uses the extractor microservice (or call extractor locally) to produce `.npy` data per clip. Example using local function or HTTP call — simplest is to call the extractor function directly if you run in same env; but to keep services decoupled, you can call the endpoint:

Example `download_landmarks.py` (call the extractor endpoint for each local file and save .npy):

```python
# download_landmarks.py
import requests
import json
import os
from pathlib import Path

EXTRACT_URL = "http://127.0.0.1:8000/extract/video"
DATA_DIR = Path("data/raw")
OUT_DIR = Path("data/landmarks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for f in DATA_DIR.glob("*.mp4"):
    print("Processing", f)
    with open(f, "rb") as fh:
        r = requests.post(EXTRACT_URL, files={"file": (f.name, fh, "video/mp4")}, params={"max_frames":300})
    if r.status_code == 200:
        payload = r.json()
        arr = payload["frames"]
        out_path = OUT_DIR / (f.stem + ".npy")
        import numpy as np
        np.save(out_path, np.array(arr, dtype=np.float32))
        print("Saved", out_path)
    else:
        print("Failed", r.status_code, r.text)
```

Run:
```powershell
.\.venv\Scripts\Activate.ps1
python download_landmarks.py
```

You will end up with `.npy` files each containing shape `(T, D)` where T ≈ frames and D is the landmark vector length (33*4 + 21*3 + 21*3 = 132 + 63 + 63 = 258 dimensions in this script).

Step 4 — PyTorch model skeleton (train a small LSTM classifier)
Create `model.py`:

```python
# model.py
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=10, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # take last output
        last = out[:, -1, :]  # (batch, hidden*2)
        logits = self.fc(last)
        return logits
```

Create `train.py` (very small training loop for a toy dataset). This expects `data/landmarks/*.npy` and a `labels.csv` mapping filename -> label index.

```python
# train.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
from model import LSTMClassifier
import torch.nn.functional as F

class LandmarkDataset(Dataset):
    def __init__(self, npy_files, labels_map, seq_len=40):
        self.files = npy_files
        self.labels_map = labels_map
        self.seq_len = seq_len
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        arr = np.load(path)  # shape (T, D)
        # simple sampling/padding to seq_len
        if arr.shape[0] >= self.seq_len:
            start = 0 if self.seq_len==arr.shape[0] else np.random.randint(0, max(1, arr.shape[0]-self.seq_len+1))
            seq = arr[start:start+self.seq_len]
        else:
            pad = np.zeros((self.seq_len-arr.shape[0], arr.shape[1]), dtype=np.float32)
            seq = np.vstack([arr, pad])
        label_name = os.path.basename(path).split("_")[0]  # assumes filename label_01.npy
        label = self.labels_map[label_name]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_labels(labels_csv="data/labels.csv"):
    # CSV with two columns: label_name,label_index
    d = {}
    with open(labels_csv, "r") as f:
        for line in f:
            name, idx = line.strip().split(",")
            d[name] = int(idx)
    return d

def train():
    labels_map = load_labels()
    files = glob.glob("data/landmarks/*.npy")
    dataset = LandmarkDataset(files, labels_map, seq_len=40)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    # infer input_dim
    sample = np.load(files[0])
    input_dim = sample.shape[1]
    model = LSTMClassifier(input_dim=input_dim, num_classes=len(labels_map))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, 21):
        model.train()
        total_loss = 0.0
        for X, y in loader:
            opt.zero_grad()
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss={total_loss/len(loader):.4f}")
    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")

if __name__ == "__main__":
    train()
```

You must create `data/labels.csv` mapping label_name to index, e.g.:
```
hello,0
thankyou,1
yes,2
no,3
help,4
iloveyou,5
```
And create small clips per label; preprocess them to `data/landmarks`.

Run training:
```powershell
.\.venv\Scripts\Activate.ps1
python train.py
```

Step 5 — Simple Python inference service (loads model.pt and receives landmark sequences or video)
Create `inference_service.py`:

```python
# inference_service.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import torch
from model import LSTMClassifier
import json
import os

app = FastAPI(title="SLR Inference Service")

# load labels map (make a file labels_map.json)
with open("data/labels_map.json", "r") as f:
    idx_to_label = json.load(f)  # {"0":"hello",...}

# load model
sample = np.load("data/landmarks/sample.npy")  # any sample to get input_dim
input_dim = sample.shape[1]
num_classes = len(idx_to_label)
model = LSTMClassifier(input_dim=input_dim, num_classes=num_classes)
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

@app.post("/predict/sequence")
async def predict_sequence(seq: UploadFile = File(...)):
    # Accept a multipart .npy file or JSON-serialized sequence
    data = await seq.read()
    arr = np.loadseq = np.load(io.BytesIO(data))  # optional approach; easier: accept JSON in practice
    # For simplicity: expect client to send JSON body containing frames list
    return {"error":"Not implemented in this stub; prefer JSON POST with frames list"}
```

Note: For quick testing you can write a tiny function that loads a `.npy` file and runs prediction locally:

```python
# quick_predict.py
import numpy as np
import torch
from model import LSTMClassifier
import json

with open("data/labels_map.json") as f:
    idx_to_label = json.load(f)
input_dim = np.load("data/landmarks/sample.npy").shape[1]
model = LSTMClassifier(input_dim=input_dim, num_classes=len(idx_to_label))
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

def predict_from_npy(path):
    arr = np.load(path)  # (T, D)
    seq_len = 40
    if arr.shape[0] >= seq_len:
        seq = arr[:seq_len]
    else:
        pad = np.zeros((seq_len-arr.shape[0], arr.shape[1]), dtype=np.float32)
        seq = np.vstack([arr, pad])
    x = torch.tensor(seq[None, ...], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=-1).numpy()[0]
        idx = int(probs.argmax())
        return idx, float(probs.max()), idx_to_label[str(idx)]
print(predict_from_npy("data/landmarks/hello_01.npy"))
```

Step 6 — Quick end-to-end test (browser → extractor → training → predict)
- Use the extractor endpoint to convert your recorded video to `.npy`.
- Use `quick_predict.py` to test the `.npy` file and see predicted label.

Example flow:
1. Record `data/raw/hello_01.mp4`.
2. Run extractor and save `.npy` with `download_landmarks.py`.
3. Run `quick_predict.py` to check prediction.

Step 7 — Add segmentation & sentence assembly (concept)
- If you want sentence-level output, either:
    - Use sliding windows (e.g., 40-frame windows with stride 10), get per-window predictions, then apply smoothing and collapse identical consecutive labels into tokens, or
    - Implement energy/motion-based segmentation: compute hand centroid speed and detect start/end of signs (more robust with training).
- Implementation detail (pseudo):
    - For each window -> predict label & probability.
    - Keep predictions above a confidence threshold (e.g., 0.6).
    - Collapse contiguous identical labels and combine into sentence.

Step 8 — Export to ONNX for Java (when model is stable)
1. Create `export_onnx.py`:
```python
# export_onnx.py
import torch
from model import LSTMClassifier
import numpy as np

model = LSTMClassifier(input_dim=258, num_classes=6)  # use your real dims
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 40, 258)  # batch, seq_len, input_dim
torch.onnx.export(model, dummy, "models/slr_model.onnx", opset_version=14,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0:"batch", 1:"seq_len"}, "output": {0:"batch"}})
print("Exported models/slr_model.onnx")
```
2. Then run:
```powershell
python export_onnx.py
```
3. Place `models/slr_model.onnx` into your Java project `models/` folder (project root)

Step 9 — Make Spring tests not fail while developing
Option 1 — Conditional bean (change `ModelConfig` in Java):
Wrap ONNX session beans with `@ConditionalOnProperty(name="model.enabled", havingValue="true", matchIfMissing=false)`. Then run tests without setting `model.enabled=true`.

Example change (conceptual; paste exact code in your `ModelConfig.java` and adapt to your repo):
```java
@Configuration
public class ModelConfig {

    @Value("${model.path:models/slr_model.onnx}")
    private String modelPath;

    @Bean
    @ConditionalOnProperty(name = "model.enabled", havingValue = "true")
    public OrtEnvironment ortEnvironment() {
        return OrtEnvironment.getEnvironment();
    }

    @Bean(destroyMethod = "close")
    @ConditionalOnProperty(name = "model.enabled", havingValue = "true")
    public OrtSession ortSession(OrtEnvironment env) throws OrtException {
        return env.createSession(modelPath, new OrtSession.SessionOptions());
    }
}
```
Run tests normally (no extra properties) and Spring will not attempt to load ONNX during tests.

Option 2 — Mock the classifier in your test:
Add to `SignLanguageRecognizerApplicationTests.java`:
```java
@MockBean
private com.slr.sign_language_recognizer.service.SignClassifierService signClassifierService;
```

Step 10 — Integrate and run everything
1. Start extractor:
```powershell
cd C:\Users\lavib\Downloads\sign-language-recognizer\mediapipe_service
.\.venv\Scripts\Activate.ps1
uvicorn extractor:app --host 127.0.0.1 --port 8000
```
2. Use your Spring app to receive frames (WebSocket/REST), forward them to the extractor or to the Python inference service, and return predictions to the client. If you prefer rapid dev, have the browser upload small video clips (1–2s) to Spring which forwards them to `http://127.0.0.1:8000/extract/video` and then calls `quick_predict.py` or the inference service.

Simple browser prototype (client HTML + JS)
- Create `client/index.html` with a video element and a button to record ~2 seconds and upload to extractor. There are many ready examples; if you want I can generate a ready-to-use `index.html` and `client.js` next.

Step 11 — Optional TTS (free) for spoken output
- eSpeak (Windows): install eSpeak and call from your Java backend:
```powershell
# example
espeak "Hello how are you" --stdout > output.wav
# serve output.wav to client
```
- Or use `pyttsx3` in Python (offline TTS library).

Step 12 — Iteration & improvement
- More data: collect more clips per label, augment, and retrain.
- Improve architecture: try Transformer encoder or Temporal ConvNet if LSTM underperforms.
- Segmentation: tune thresholds or train segmentation module.

Notes, tips and cautions
- MediaPipe landmarks are normalized coordinates relative to image; maintain the same preprocessing in training and inference.
- Keep metadata: fps and frame ordering must match between training preprocessing and runtime extraction.
- Start with a very small vocabulary (6–10 signs) to get high accuracy quickly; expand later.
- Expect initial accuracy to be limited with small data; more examples/per sign dramatically improves performance.
- ONNX export must preserve input order and normalization; test predictions in Python and Java to ensure parity.

If you want, next I will:
- Generate the complete `client/index.html` + `client.js` recorder that uploads 1–2 second clips to the extractor and shows returned predictions, and
- Provide a ready‑to‑run `download_landmarks.py`, `train.py`, `quick_predict.py`, and `export_onnx.py` in the `mediapipe_service` folder.

Which of those two do you want me to produce now (client UI or the full set of Python scripts for training + export)?