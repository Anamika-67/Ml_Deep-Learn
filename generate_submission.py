import csv
import os
import torch
import torch.nn as nn
import torchvision.models as models
import importlib.util

# --- Import from Inference Script ---
def load_inference_module():
    script_path = "Inference Script (inference.py)"
    if not os.path.exists(script_path):
        script_path = os.path.join(os.path.dirname(__file__), "Inference Script (inference.py)")
    spec = importlib.util.spec_from_file_location("inference", script_path)
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    return inference

# --- Phase 4: Feature Extraction ---
class CNNEncoder(nn.Module):
    def __init__(self, model_type="resnet50"):
        super().__init__()
        self.model_type = model_type
        if model_type == "resnet50":
            base_model = models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.fc = nn.Linear(2048, 512)
        elif model_type == "efficientnet":
            base_model = models.efficientnet_b0(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.fc = nn.Linear(1280, 512)
        elif model_type == "mobilenet":
            base_model = models.mobilenet_v2(pretrained=True)
            self.feature_extractor = base_model.features
            self.fc = nn.Linear(1280, 512)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        features = self.feature_extractor(x)
        
        # Optional pooling depending on base model extraction layers
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            
        features = features.view(B*T, -1)
        features = self.fc(features)
        features = features.view(B, T, 512)
        return features

# --- Phase 5: Temporal Modeling ---
class TemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Phase 6: Frame Ranking Prediction (Scores)
        scores = self.fc(out)
        scores = scores.squeeze(-1)
        return scores

class CombinedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNEncoder()
        self.temporal = TemporalModel()

# --- Phase 8: Output Generation ---
def write_submission(results, output_file="submission.csv"):
    with open(output_file, "w", newline="") as f:
        f.write("video_id,order\n")
        for video, order in results.items():
            order_str = " ".join(map(str, order))
            f.write(f'{video},"{order_str}"\n')

def generate_submission(test_dir="data/test", model_path=None, output_csv="submission.csv"):
    print("Loading AI pipeline components...")
    inference = load_inference_module()
    
    # Initialize the combined full pipeline model
    model = CombinedModel()
    
    # Load weights if specified
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Loaded weights from {model_path}")
    else:
        print("Running with untrained weights for testing pipeline execution.")
    
    model.eval()
    results = {}
    
    if os.path.exists(test_dir):
        print(f"Processing videos in {test_dir}...")
        for file in os.listdir(test_dir):
            if file.endswith(".mp4"):
                video_path = os.path.join(test_dir, file)
                print(f"Running inference on {file}...")
                
                # Complete execution: Video -> Frames -> Preprocess -> CNN -> LSTM -> Sort
                order = inference.process_video(video_path, model)
                
                # Convert predictions into a list and remove file extension
                video_id = os.path.splitext(file)[0]
                results[video_id] = order.cpu().squeeze().tolist()
                
        write_submission(results, output_csv)
        print(f"Successfully generated {output_csv}!")
    else:
        print(f"Test directory not found: {test_dir}")

if __name__ == "__main__":
    generate_submission()
