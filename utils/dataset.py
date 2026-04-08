import os
import json
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset

# Inline transforms to avoid import errors from `Image Preprocessing(preprocessing.py)`
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
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames

class FrameDataset(Dataset):
    def __init__(self, data_dir, label_file):
        self.data_dir = data_dir
        self.transform = get_transforms()
        with open(label_file) as f:
            self.labels = json.load(f)
        self.videos = list(self.labels.keys())

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video = self.videos[idx]
        video_path = os.path.join(self.data_dir, video)
        
        # Priority: Check if it's a direct .mp4 file
        if os.path.isfile(video_path) and video.endswith('.mp4'):
            pil_frames = extract_frames_from_video(video_path)
        else:
            # Fallback: loading from a directory of extracted frames
            frames = sorted(os.listdir(video_path))
            pil_frames = [Image.open(os.path.join(video_path, f)) for f in frames]
            
        images = []
        for img in pil_frames:
            images.append(self.transform(img))
            
        images = torch.stack(images)
        label = torch.tensor(self.labels[video])
        
        return images, label
