import os
import shutil

base_dir = r"c:\Users\anabx\OneDrive\Desktop\ML-war-main"

# Create proper directories
for d in ["models", "utils", "data", "frames"]:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

moves = [
    (r"CNN Feature Extractor (models\cnn_encoder.py)", r"models\cnn_encoder.py"),
    (r"Temporal Model (models\temporal_model.py)", r"models\temporal_model.py"),
    (r"Frame Extraction (utils\extract_frames.py)", r"utils\extract_frames.py"),
    (r"Dataset Loader (dataset.py)", r"utils\dataset.py"),
    (r"Image Preprocessing(preprocessing.py)", r"utils\preprocessing.py"),
    (r"Training Script (train.py)", r"train.py"),
    (r"Inference Script (inference.py)", r"inference.py"),
    (r"Generate Submission CSV (generate_submission.py)", r"generate_submission.py")
]

for src, dst in moves:
    src_path = os.path.join(base_dir, src)
    dst_path = os.path.join(base_dir, dst)
    if os.path.exists(src_path):
        try:
            shutil.move(src_path, dst_path)
            print(f"Moved {src} to {dst}")
        except Exception as e:
            print(f"Failed to move {src}: {e}")

# Clean up empty directories
for d in [
    r"CNN Feature Extractor (models",
    r"Temporal Model (models",
    r"Frame Extraction (utils"
]:
    dir_path = os.path.join(base_dir, d)
    if os.path.exists(dir_path):
        try:
            os.rmdir(dir_path)
            print(f"Removed empty directory {d}")
        except Exception as e:
            print(f"Failed to remove {d}: {e}")
