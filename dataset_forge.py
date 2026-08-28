# dataset_forge.py
# The GPU-Accelerated Omnikon Dataset Architect

import os
import shutil
import random
import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# GLOBAL CONFIGURATION: THE COMMAND CENTER
# =====================================================================
# Temporary extraction zones
TEMP_REAL_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/real_temp"  
TEMP_FAKE_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/fake_temp"

# Final Migration Endpoints for Real Data
TRAIN_REAL_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/train/REAL"
CV_REAL_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/cv/REAL"
TEST_REAL_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/test/REAL"

# Final Migration Endpoints for Fake Data
TRAIN_FAKE_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/train/FAKE"
CV_FAKE_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/cv/FAKE"
TEST_FAKE_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/test/FAKE"

# Target Quotas
REAL_QUOTA = 50000
FAKE_QUOTA = 47000
NUM_CLUSTERS = 50 # The number of geometric regions to define

# Data split ratios (60:20:20)
TRAIN_RATIO = 0.6
CV_RATIO = 0.2
TEST_RATIO = 0.2
BATCH_SIZE = 256
# =====================================================================

class FastImageDataset(Dataset):
    """Pipes images into the GPU without stalling the CUDA cores."""
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            tensor = self.transform(img)
            return tensor, str(path)
        except Exception:
            # Return an empty tensor if the image is violently corrupted
            return torch.zeros(3, 224, 224), ""

def extract_features_and_cluster(image_dir, quota):
    print(f"\n[Oracle] Initiating topological mapping for {image_dir}...")
    
    # Initialize the spatial extractor (ResNet-18 backbone)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1]) # Strip classification head
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_paths = [p for p in Path(image_dir).glob("*") if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    
    if len(image_paths) < quota:
        raise ValueError(f"Not enough images! Found {len(image_paths)}, need {quota}.")

    print(f"[Oracle] Extracting 512-D geometric vectors for {len(image_paths)} images...")
    
    dataset = FastImageDataset(image_paths, preprocess)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    
    features = []
    valid_paths = []
    
    with torch.no_grad():
        for batch_idx, (tensors, paths) in enumerate(dataloader):
            # Filter out any corrupted image paths safely
            valid_mask = [p != "" for p in paths]
            if not any(valid_mask):
                continue
                
            clean_tensors = tensors[[i for i, m in enumerate(valid_mask) if m]].to(device, non_blocking=True)
            clean_paths = [p for p in paths if p != ""]
            
            vecs = model(clean_tensors).squeeze().cpu().numpy()
            
            # Handle single-image batch dimension collapse
            if len(clean_paths) == 1:
                vecs = np.expand_dims(vecs, axis=0)
                
            features.extend(vecs)
            valid_paths.extend(clean_paths)
            
            processed = (batch_idx + 1) * BATCH_SIZE
            if processed % 5000 < BATCH_SIZE:
                print(f"  Mapped {min(processed, len(image_paths))} images...")
                
    features = np.array(features)
    print(f"[Oracle] Mapping complete. Feature matrix shape: {features.shape}")
    
    print("[Oracle] Partitioning the spatial vectors via MiniBatch K-Means...")
    kmeans = MiniBatchKMeans(n_clusters=NUM_CLUSTERS, batch_size=1024, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Group images by their assigned structural cluster
    clusters = {i: [] for i in range(NUM_CLUSTERS)}
    for path, label in zip(valid_paths, labels):
        clusters[label].append(path)
        
    # Harvest equitably across all topological clusters
    print("[Oracle] Harvesting an equitable subset...")
    selected_paths = []
    samples_per_cluster = quota // NUM_CLUSTERS
    
    for c_id, paths in clusters.items():
        take = min(samples_per_cluster, len(paths))
        selected_paths.extend(random.sample(paths, take))
        
    # Pad randomly from the remainder if integer division leaves a tiny shortfall
    remainder = set(valid_paths) - set(selected_paths)
    if len(selected_paths) < quota:
        shortfall = quota - len(selected_paths)
        selected_paths.extend(random.sample(list(remainder), shortfall))
        
    random.shuffle(selected_paths)
    return selected_paths

def construct_and_populate(selected_paths, train_dir, cv_dir, test_dir):
    print(f"\n[Architect] Constructing final repositories and routing files...")
    
    total = len(selected_paths)
    train_split = int(total * TRAIN_RATIO)
    cv_split = int(total * CV_RATIO)
    
    splits = {
        train_dir: selected_paths[:train_split],
        cv_dir: selected_paths[train_split:train_split + cv_split],
        test_dir: selected_paths[train_split + cv_split:]
    }
    
    for dest_dir, paths in splits.items():
        dest_path = Path(dest_dir)
        dest_path.mkdir(parents=True, exist_ok=True)
        
        print(f"  Migrating {len(paths)} images to {dest_path}...")
        for path in paths:
            # We enforce absolute string conversion to avoid Pathlib collision errors
            shutil.copy2(str(path), dest_path / Path(path).name)

if __name__ == "__main__":
    print("=== OMNIKON DATASET FORGE INITIATED ===")
    
    real_subset = extract_features_and_cluster(TEMP_REAL_DIR, REAL_QUOTA)
    construct_and_populate(real_subset, TRAIN_REAL_DIR, CV_REAL_DIR, TEST_REAL_DIR)
    
    fake_subset = extract_features_and_cluster(TEMP_FAKE_DIR, FAKE_QUOTA)
    construct_and_populate(fake_subset, TRAIN_FAKE_DIR, CV_FAKE_DIR, TEST_FAKE_DIR)
    
    print("\n=== FORGE COMPLETE. The Omnikon dataset is primed. ===")