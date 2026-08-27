# dataset_forge.py
# The Omnikon Dataset Architect

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

# =====================================================================
# GLOBAL CONFIGURATION: THE COMMAND CENTER
# =====================================================================
# Direct these paths to where you are extracting the .zip files
REAL_SOURCE_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/REAL"  
FAKE_SOURCE_DIR = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET/FAKE"

# The final destination for the structurally balanced splits.
DATASET_ROOT = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET"

# Target Quotas
REAL_QUOTA = 100
FAKE_QUOTA = 100
NUM_CLUSTERS = 5 # The number of geometric regions to define

# Data split ratios (60:20:20)
TRAIN_RATIO = 0.6
CV_RATIO = 0.2
TEST_RATIO = 0.2
# =====================================================================

def extract_features_and_cluster(image_dir, quota):
    print(f"\n[Oracle] Initiating topological mapping for {image_dir}...")
    
    # Initialize the spatial extractor (ResNet-18 backbone)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model = torch.nn.Sequential(*list(model.children())[:-1]) # Strip classification head
    model.eval()
    
    # Check for hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
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
    
    features = []
    valid_paths = []
    
    with torch.no_grad():
        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert('RGB')
                tensor = preprocess(img).unsqueeze(0).to(device)
                vec = model(tensor).squeeze().cpu().numpy()
                features.append(vec)
                valid_paths.append(path)
            except Exception as e:
                continue
            
            if (i+1) % 5000 == 0:
                print(f"  Mapped {i+1} images...")
                
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

def construct_and_populate(selected_paths, category_name):
    print(f"\n[Architect] Constructing final repositories for {category_name}...")
    
    total = len(selected_paths)
    train_split = int(total * TRAIN_RATIO)
    cv_split = int(total * CV_RATIO)
    
    splits = {
        'train': selected_paths[:train_split],
        'cv': selected_paths[train_split:train_split + cv_split],
        'test': selected_paths[train_split + cv_split:]
    }
    
    for split_name, paths in splits.items():
        dest_dir = Path(DATASET_ROOT) / split_name / category_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Migrating {len(paths)} images to {dest_dir}...")
        for path in paths:
            shutil.copy2(path, dest_dir / path.name)

if __name__ == "__main__":
    print("=== OMNIKON DATASET FORGE INITIATED ===")
    
    real_subset = extract_features_and_cluster(REAL_SOURCE_DIR, REAL_QUOTA)
    construct_and_populate(real_subset, "real")
    
    fake_subset = extract_features_and_cluster(FAKE_SOURCE_DIR, FAKE_QUOTA)
    construct_and_populate(fake_subset, "fake")
    
    print("\n=== FORGE COMPLETE. The Omnikon dataset is primed. ===")