import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV) to RGB (PIL format)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

def preprocess_frames(frames):
    transform = get_transforms()
    processed = [transform(img) for img in frames]
    return torch.stack(processed)

def predict_order(model, frames):
    # Phase 4: Feature Extraction 
    # (Extracts object shapes, positions, motion clues)
    features = model.cnn(frames)
    
    # Phase 5 & Phase 6: Temporal Modeling and Frame Ranking Prediction 
    # (Understand cause-effect and get a ranking score)
    scores = model.temporal(features)
    
    # Phase 7: Frame Reordering 
    # (Sorted by score to produce predicted order)
    order = torch.argsort(scores)
    return order

def process_video(video_path, model):
    """
    End-to-end pipeline: Video -> Extract Frames -> Preprocess -> Inference
    """
    # 1. Extract frames
    frames = extract_frames_from_video(video_path)
    
    # 2. Preprocess images
    tensor_frames = preprocess_frames(frames)
    
    # Add batch dimension: [1, num_frames, C, H, W]
    tensor_frames = tensor_frames.unsqueeze(0)
    
    # 3. Predict order (with torch.no_grad() to save memory)
    with torch.no_grad():
        order = predict_order(model, tensor_frames)
        
    return order
