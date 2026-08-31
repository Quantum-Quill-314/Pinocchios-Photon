# omnikon_trainer.py
# The Master Crucible: Omnikon Training & Checkpoint Sentry

import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

# Import your architectural components
from dual_branch_net import Dectector
from omnikon_pipeline import compute_defocus_cpu, compute_specular_cpu, compute_defocus_gpu, compute_specular_gpu

# =====================================================================
# I. GLOBAL CONFIGURATION (The Command Center)
# =====================================================================
USE_GPU       = True  # Toggle: False for Laptop (CPU Physics), True for Workstation (GPU Physics)
DATASET_ROOT  = "/home/mystical-poet/WhisperShade/Codes/Omnikon/DATASET" 
# Point the above to the folder containing REAL and FAKE
BATCH_SIZE    = 32
LEARNING_RATE = 3e-4
PATIENCE      = 15      # Epochs to wait for CV loss improvement before aborting
MAX_EPOCHS    = 50
IMG_SIZE      = 299
FREEZE_EPOCHS = 8
AUX_WEIGHT = 0.5
# =====================================================================

class OmnikonDataset(Dataset):
    def __init__(self, root_dir, split_name, use_gpu_physics=False):
        self.root_dir = root_dir
        self.split_name = split_name
        self.use_gpu_physics = use_gpu_physics
        self.samples = []
        
        # Define the binary labels
        categories = {'REAL': 0.0, 'FAKE': 1.0}
        
        for cat, label in categories.items():
            # This sequence matches DATASET_ROOT/real/train
            cat_dir = os.path.join(root_dir, cat, split_name) 
            
            if os.path.exists(cat_dir):
                for file in os.listdir(cat_dir):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(cat_dir, file), label))
            else:
                print(f"[Warning] Sentry could not find directory: {cat_dir}")
                        
        self.rgb_transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        rgb_tensor = self.rgb_transform(img)
        
        if self.use_gpu_physics:
            # For workstation: just pass the grayscale image to the GPU later
            gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            gray_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
            gray_tensor = torch.from_numpy(gray_img).unsqueeze(0).float() / 255.0
            return rgb_tensor, gray_tensor, torch.tensor([label], dtype=torch.float32)
        else:
            # For laptop: compute OpenCV physics sequentially right now
            gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            gray_img = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE))
            
            defocus_map = compute_defocus_cpu(gray_img)
            specular_map = compute_specular_cpu(gray_img)
            
            phys_tensor = torch.from_numpy(np.stack([defocus_map, specular_map], axis=0)).float()
            return rgb_tensor, phys_tensor, torch.tensor([label], dtype=torch.float32)

def execute_graceful_epilogue(model, optimizer, epoch, best_cv_loss):
    print("\n[Sentry] The crucible has cooled. Training concluded.")
    filename = input("[?] Enter a name for this checkpoint (without extension): ").strip()
    if not filename:
        filename = "omnikon_auto_save"
        
    save_path = f"{filename}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_cv_loss,
    }, save_path)
    print(f"[Architect] Immutable state sealed within: {save_path}")

def main():
    print("=== OMNIKON FORGE INITIATED ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Auto-tunes convolutional algorithms for 299x299 inputs
    print(f"[Oracle] Hardware mapped to: {device} | Native GPU Physics: {USE_GPU}")
    
    # --- Interactive Prologue ---
    start_epoch = 1
    model = Dectector(verbose=False).to(device)
    # Differential Learning Rates
    optimizer = optim.AdamW([
        {'params': model.visual_stream.parameters(), 'lr': 1e-5}, # Microscopic LR for ResNet
        {'params': model.physics_stream.parameters(), 'lr': LEARNING_RATE},
        {'params': model.phys_mlp.parameters(), 'lr': LEARNING_RATE},
        {'params': [model.alpha], 'lr': LEARNING_RATE},
        {'params': model.fc1.parameters(), 'lr': LEARNING_RATE},
        {'params': model.bn2.parameters(), 'lr': LEARNING_RATE},
        {'params': model.fc2.parameters(), 'lr': LEARNING_RATE},
        {'params': model.aux_head.parameters(), 'lr': LEARNING_RATE}
])    
    run_saved = input("[?] Resurrect a saved state? (y/n): ").strip().lower()
    if run_saved == 'y':
        path = input("[?] Enter the exact path to the .pth file: ").strip()
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"[Oracle] Vault unsealed. Resuming from epoch {start_epoch}.")
        else:
            print("[Sentry] File not found. Forging a new architecture instead.")
            
    try:
        user_epochs = int(input(f"[?] How many epochs shall we forge? (Max {MAX_EPOCHS}): ").strip())
        target_epochs = min(user_epochs + start_epoch - 1, MAX_EPOCHS)
    except ValueError:
        target_epochs = start_epoch + 10
        print(f"[Sentry] Invalid input. Defaulting to {target_epochs} epochs.")

    # Generate a unique timestamp (e.g., 20260829-103045)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dynamic_log_dir = f"./runs/omnikon_experiment_{current_time}"

    writer = SummaryWriter(log_dir=dynamic_log_dir)
    print(f"[Oracle] TensorBoard active. Run 'tensorboard --logdir={dynamic_log_dir}' in another terminal.")

    # --- Data Loading ---
    print("[Architect] Assembling the data streams...")
    train_loader = DataLoader(OmnikonDataset(DATASET_ROOT, 'train', USE_GPU), batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    cv_loader = DataLoader(OmnikonDataset(DATASET_ROOT, 'cv', USE_GPU), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(OmnikonDataset(DATASET_ROOT, 'test', USE_GPU), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.BCELoss()
    
    # --- Early Stopping Variables ---
    best_cv_loss = float('inf')
    patience_counter = 0
    optimal_weights = None

    # --- Tri-Phase Engine ---
    try:
        for epoch in range(start_epoch, target_epochs + 1):
            print(f"\n--- Epoch {epoch}/{target_epochs} ---")
            
            # Temporal Cryostasis
            if epoch <= FREEZE_EPOCHS:
                for param in model.visual_stream.parameters():
                    param.requires_grad = False
                print(f"  [Sentry] Visual Stream frozen (Epoch {epoch}/{FREEZE_EPOCHS})")
            else:
                for param in model.visual_stream.parameters():
                    param.requires_grad = True
                    
            # Phase I: Training
            model.train()
            train_loss = 0.0
            all_train_labels, all_train_preds = [], []
            
            for rgb, phys_or_gray, labels in train_loader:
                rgb, labels = rgb.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                phys_or_gray = phys_or_gray.to(device, non_blocking=True)
                
                if USE_GPU:
                    defocus = compute_defocus_gpu(phys_or_gray)
                    specular = compute_specular_gpu(phys_or_gray)
                    phys = torch.cat([defocus, specular], dim=1)
                else:
                    phys = phys_or_gray
                
                optimizer.zero_grad()
                main_prob, aux_prob = model(rgb, phys)
                
                # The Auxiliary Anchor Loss
                loss_main = criterion(main_prob, labels)
                loss_aux = criterion(aux_prob, labels)
                loss = loss_main + AUX_WEIGHT * loss_aux
                
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                all_train_labels.extend(labels.cpu().numpy())
                all_train_preds.extend(main_prob.detach().cpu().numpy())
                
            avg_train_loss = train_loss / len(train_loader)
            train_auc = roc_auc_score(all_train_labels, all_train_preds)
            writer.add_scalar('Loss/Train', avg_train_loss, epoch)
            writer.add_scalar('AUC/Train', train_auc, epoch)
            print(f"  Training Loss: {avg_train_loss:.4f} | AUC: {train_auc:.4f}")

            # Phase II: Cross-Validation
            model.eval()
            cv_loss = 0.0
            all_cv_labels, all_cv_preds = [], []
            with torch.no_grad():
                for rgb, phys_or_gray, labels in cv_loader:
                    rgb, labels = rgb.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    phys_or_gray = phys_or_gray.to(device, non_blocking=True)
                    
                    if USE_GPU:
                        defocus = compute_defocus_gpu(phys_or_gray)
                        specular = compute_specular_gpu(phys_or_gray)
                        phys = torch.cat([defocus, specular], dim=1)
                    else:
                        phys = phys_or_gray
                        
                    main_prob, aux_prob = model(rgb, phys)
                    loss_main = criterion(main_prob, labels)
                    loss_aux = criterion(aux_prob, labels)
                    loss = loss_main + AUX_WEIGHT * loss_aux
                    
                    cv_loss += loss.item()
                    all_cv_labels.extend(labels.cpu().numpy())
                    all_cv_preds.extend(main_prob.cpu().numpy())
                    
            avg_cv_loss = cv_loss / len(cv_loader)
            cv_auc = roc_auc_score(all_cv_labels, all_cv_preds)
            writer.add_scalar('Loss/CV', avg_cv_loss, epoch)
            writer.add_scalar('AUC/CV', cv_auc, epoch)
            print(f"  CV Loss:       {avg_cv_loss:.4f} | AUC: {cv_auc:.4f}")

            # Phase III: Testing
            test_loss = 0.0
            all_test_labels, all_test_preds = [], []
            with torch.no_grad():
                for rgb, phys_or_gray, labels in test_loader:
                    rgb, labels = rgb.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    phys_or_gray = phys_or_gray.to(device, non_blocking=True)
                    
                    if USE_GPU:
                        defocus = compute_defocus_gpu(phys_or_gray)
                        specular = compute_specular_gpu(phys_or_gray)
                        phys = torch.cat([defocus, specular], dim=1)
                    else:
                        phys = phys_or_gray
                        
                    main_prob, aux_prob = model(rgb, phys)
                    loss_main = criterion(main_prob, labels)
                    loss_aux = criterion(aux_prob, labels)
                    loss = loss_main + AUX_WEIGHT * loss_aux
                    
                    test_loss += loss.item()
                    all_test_labels.extend(labels.cpu().numpy())
                    all_test_preds.extend(main_prob.cpu().numpy())
                    
            avg_test_loss = test_loss / len(test_loader)
            test_auc = roc_auc_score(all_test_labels, all_test_preds)
            writer.add_scalar('Loss/Test', avg_test_loss, epoch)
            writer.add_scalar('AUC/Test', test_auc, epoch)
            print(f"  Test Loss:     {avg_test_loss:.4f} | AUC: {test_auc:.4f}")

            # Sentry: Early Stopping Check
            if avg_cv_loss < best_cv_loss:
                best_cv_loss = avg_cv_loss
                patience_counter = 0
                optimal_weights = model.state_dict()
                torch.save(model.state_dict(), 'omnikon_best_checkpoint.pth')
                print("  [Sentry] New optimal state verified and cached.")
            else:
                patience_counter += 1
                print(f"  [Sentry] CV Loss stagnant. Patience: {patience_counter}/{PATIENCE}")
                
            if patience_counter >= PATIENCE:
                print(f"\n[Sentry] Patience threshold exceeded. Halting to prevent overfitting.")
                if optimal_weights:
                    model.load_state_dict(optimal_weights)
                break

    except KeyboardInterrupt:
        print("\n[Oracle] Manual interrupt detected. Freezing current state.")
        if optimal_weights:
            model.load_state_dict(optimal_weights)

    finally:
        writer.close()
        execute_graceful_epilogue(model, optimizer, epoch, best_cv_loss)

if __name__ == "__main__":
    main()