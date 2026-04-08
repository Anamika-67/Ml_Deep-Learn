import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import importlib.util
from scipy.stats import kendalltau

# Load the dataset dynamically to bypass irregular name characters
def load_dataset_module():
    script_path = "Dataset Loader (dataset.py)"
    if not os.path.exists(script_path):
        script_path = os.path.join(os.path.dirname(__file__), "Dataset Loader (dataset.py)")
    spec = importlib.util.spec_from_file_location("dataset", script_path)
    dataset = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dataset)
    return dataset

# Inline models to bypass missing import links from models/ directory
class CNNEncoder(nn.Module):
    def __init__(self, model_type="resnet50"):
        super().__init__()
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

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        features = self.feature_extractor(x)
        if len(features.shape) > 2:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
        features = features.view(B*T, -1)
        features = self.fc(features)
        features = features.view(B, T, 512)
        return features

class TemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        scores = self.fc(out)
        return scores.squeeze(-1)

class CombinedModel(nn.Module):
    def __init__(self, cnn_type="resnet50"):
        super().__init__()
        self.cnn = CNNEncoder(cnn_type)
        self.temporal = TemporalModel()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}")
    
    dataset_module = load_dataset_module()
    
    # Defaults according to project structure
    labels_file = "data/train_labels.json"
    data_dir = "data/train" # Can be directory full of .mp4s
    
    if not os.path.exists(labels_file):
        print(f"Missing training labels file: {labels_file}")
        return
        
    dataset = dataset_module.FrameDataset(data_dir, labels_file)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model = CombinedModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_tau = 0
        
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            features = model.cnn(images)
            scores = model.temporal(features)
            
            loss = loss_fn(scores, labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Phase 9: Evaluation (Kendall's Tau)
            batch_tau = 0
            s_np = scores.cpu().detach().numpy()
            l_np = labels.cpu().detach().numpy()
            for b in range(s_np.shape[0]):
                tau, _ = kendalltau(l_np[b], s_np[b])
                # Handle NaNs in edge cases where variance is 0
                if not type(tau) is float or tau != tau: 
                    tau = 0.0
                batch_tau += tau
            total_tau += batch_tau / s_np.shape[0]
            
        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
        avg_tau = total_tau / len(loader) if len(loader) > 0 else 0
        print(f"Epoch: {epoch+1}/{epochs} - Avg Loss: {avg_loss:.4f} - Avg Kendall Tau: {avg_tau:.4f}")

    # Save Checkpoint
    os.makedirs("models", exist_ok=True)
    checkpoint_path = "models/checkpoint.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Training complete! Model saved to {checkpoint_path}")

if __name__ == "__main__":
    train()
