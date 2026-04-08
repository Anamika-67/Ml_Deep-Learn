import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import importlib.util

# Load inference script using importlib because of invalid Python naming convention
def load_inference_module():
    script_path = "Inference Script (inference.py)"
    if not os.path.exists(script_path):
        script_path = os.path.join(os.path.dirname(__file__), "Inference Script (inference.py)")
    spec = importlib.util.spec_from_file_location("inference", script_path)
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    return inference

def load_generate_submission_module():
    script_path = "Generate Submission CSV (generate_submission.py)"
    if not os.path.exists(script_path):
        script_path = os.path.join(os.path.dirname(__file__), "Generate Submission CSV (generate_submission.py)")
    spec = importlib.util.spec_from_file_location("generate", script_path)
    generate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generate)
    return generate

app = FastAPI(title="MLWARE Sherlock Files API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static/frames", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount the static files (UI frontend and extracted images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Expose the ML war / ui folder as the root UI endpoint
app.mount("/ui", StaticFiles(directory="ML war", html=True), name="ui")

print("Initializing AI Pipeline...")
inference = load_inference_module()
generate = load_generate_submission_module()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = generate.CombinedModel().to(device)
if os.path.exists("models/checkpoint.pth"):
    print("Loading weights from checkpoint.pth")
    model.load_state_dict(torch.load("models/checkpoint.pth", map_location=device))
else:
    print("Running with base pre-trained weights.")
model.eval()

# Global state to keep track of uploaded video
UPLOADED_VIDEO_PATH = None

@app.post("/extract_frames")
async def extract_frames(video: UploadFile = File(...)):
    global UPLOADED_VIDEO_PATH
    UPLOADED_VIDEO_PATH = os.path.join("uploads", video.filename)
    
    with open(UPLOADED_VIDEO_PATH, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)
        
    print(f"Video uploaded: {UPLOADED_VIDEO_PATH}")
    
    # Extract frames using OpenCV
    frames = inference.extract_frames_from_video(UPLOADED_VIDEO_PATH)
    
    # Save frames to static folder to serve to frontend
    # Clear old frames natively
    for f in os.listdir("static/frames"):
        os.remove(os.path.join("static/frames", f))
        
    frame_urls = []
    for i, frame in enumerate(frames):
        path = f"static/frames/frame_{i}.jpg"
        frame.save(path)
        frame_urls.append(f"/{path}")
        
    return JSONResponse(content=frame_urls)

@app.get("/predict")
async def predict():
    global UPLOADED_VIDEO_PATH
    if UPLOADED_VIDEO_PATH is None or not os.path.exists(UPLOADED_VIDEO_PATH):
        return {"error": "No video uploaded yet"}
        
    print(f"Predicting timeline for: {UPLOADED_VIDEO_PATH}")
    order = inference.process_video(UPLOADED_VIDEO_PATH, model)
    order_list = order.cpu().squeeze().tolist()
    
    if not isinstance(order_list, list):
        order_list = [order_list]
        
    order_str = " ".join(map(str, order_list))
    print(f"Predicted Output: {order_str}")
    
    return JSONResponse(content={"order": order_str})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
