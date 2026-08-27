# dual_branch_net.py
# The Dual-Branch Modulated Architecture (with verbose toggles and test block)

import torch
import torch.nn as nn
import torchvision.models as models
from config import LATENT_DIM, DROPOUT_RATE, BATCH_SIZE, RGB_CHANNELS, PHYSICS_CHANNELS, IMG_SIZE

class VisualStream(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Pre-trained Backbone to extract rich semantic features
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        if self.verbose: print(f"[VisualStream] Input shape: {x.shape}")
        x = self.backbone(x)
        if self.verbose: print(f"[VisualStream] After backbone shape: {x.shape}")
        x = self.adaptive_pool(x)
        if self.verbose: print(f"[VisualStream] After adaptive pool shape: {x.shape}")
        x = x.view(x.size(0), -1)  # Yields V_rgb
        if self.verbose: print(f"[VisualStream] Output V_rgb shape: {x.shape}")
        return x

class PhysicsStream(nn.Module):
    def __init__(self, in_channels=2, latent_dim=LATENT_DIM, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Low-level spatial gradients
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Mid-level structural relationships
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        # Projection and aggregation
        self.conv3 = nn.Conv2d(64, latent_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(latent_dim)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        if self.verbose: print(f"[PhysicsStream] Input shape: {x.shape}")
        x = self.pool1(self.relu1(self.conv1(x)))
        if self.verbose: print(f"[PhysicsStream] After Conv1+Pool1 shape: {x.shape}")
        x = self.relu2(self.conv2(x))
        if self.verbose: print(f"[PhysicsStream] After Conv2 shape: {x.shape}")
        x = self.bn1(self.conv3(x))
        if self.verbose: print(f"[PhysicsStream] After Conv3+BN1 shape: {x.shape}")
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # Yields V_phys
        if self.verbose: print(f"[PhysicsStream] Output V_phys shape: {x.shape}")
        return x

class Dectector(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, dropout_rate=DROPOUT_RATE, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.visual_stream = VisualStream(latent_dim, verbose=self.verbose)
        self.physics_stream = PhysicsStream(PHYSICS_CHANNELS, latent_dim, verbose=self.verbose)
        
        self.sigmoid_gate = nn.Sigmoid()
        
        # Classification & Decision Head
        self.fc1 = nn.Linear(latent_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu_head = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid_out = nn.Sigmoid()
        
    def forward(self, rgb, phys):
        if self.verbose: print("\n=== [Dectector] FORWARD PASS INITIATED ===")
        
        v_rgb = self.visual_stream(rgb)
        v_phys = self.physics_stream(phys)
        
        # Transform physical features into dynamic attention weights
        gate = self.sigmoid_gate(v_phys)
        if self.verbose: print(f"[Dectector] Gate shape: {gate.shape}")
        
        # Hadamard Product Fusion
        v_modulated = v_rgb * gate
        if self.verbose: print(f"[Dectector] Modulated vector shape: {v_modulated.shape}")
        
        # Dense decision space projection
        out = self.fc1(v_modulated)
        out = self.bn2(out)
        out = self.relu_head(out)
        out = self.dropout(out)
        out = self.fc2(out)
        if self.verbose: print(f"[Dectector] Final logit shape: {out.shape}")
        
        final_prob = self.sigmoid_out(out)
        if self.verbose: print("=== [Dectector] FORWARD PASS COMPLETE ===\n")
        
        return final_prob

# --- MOCK DATA FORGE & TEST EXECUTION ---
if __name__ == "__main__":
    print("Initializing Omnikon Detector Test Forge...")
    
    # Initialize the model with verbose logging enabled
    model = Dectector(verbose=True)
    
    # Generate dummy tensors matching the config specifications
    # Using batch size of 2 for testing to ensure dimension handling is correct
    test_batch_size = 2
    mock_rgb = torch.randn(test_batch_size, RGB_CHANNELS, IMG_SIZE, IMG_SIZE)
    mock_phys = torch.randn(test_batch_size, PHYSICS_CHANNELS, IMG_SIZE, IMG_SIZE)
    
    print(f"\nFeeding mock tensors into the network...")
    print(f"RGB Tensor: {mock_rgb.shape}")
    print(f"Physics Tensor: {mock_phys.shape}")
    
    # Run the forward pass
    output_probabilities = model(mock_rgb, mock_phys)
    
    print("\n--- FINAL OUTPUT ---")
    print(f"Predictions Shape: {output_probabilities.shape}")
    print(f"Prediction Values:\n{output_probabilities.detach().numpy()}")
    print("--------------------")