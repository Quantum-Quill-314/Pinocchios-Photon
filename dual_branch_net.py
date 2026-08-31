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
        resnet = models.resnet18(weights='IMAGENET1K_V1')
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

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class PhysicsStream(nn.Module):
    def __init__(self, in_channels=2, latent_dim=LATENT_DIM, verbose=False):
        super().__init__()
        self.verbose = verbose
        
        # Progressive expansion with strided convolutions
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer2 = ResBlock(64, 128, stride=2)
        self.layer3 = ResBlock(128, 256, stride=2)
        self.layer4 = ResBlock(256, 512, stride=2)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, latent_dim)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Dectector(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, dropout_rate=DROPOUT_RATE, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.visual_stream = VisualStream(latent_dim, verbose=self.verbose)
        self.physics_stream = PhysicsStream(PHYSICS_CHANNELS, latent_dim, verbose=self.verbose)
        
        # 1. The Projection MLP for Fusion
        self.phys_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(latent_dim, latent_dim)
        )
        
        # 2. Learnable Scalar Optimization (alpha initialized to 0.05)
        self.alpha = nn.Parameter(torch.tensor([0.05]))
        
        # Classification & Decision Head
        self.fc1 = nn.Linear(latent_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu_head = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 1)
        
        # 3. The Auxiliary Anchor Head
        self.aux_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, rgb, phys):
        v_rgb = self.visual_stream(rgb)
        v_phys = self.physics_stream(phys)
        
        # Additive Residual Fusion
        v_fused = v_rgb + self.alpha * self.phys_mlp(v_phys)
        
        # Main Output
        out = self.fc1(v_fused)
        out = self.bn2(out)
        out = self.relu_head(out)
        out = self.dropout(out)
        main_prob = torch.sigmoid(self.fc2(out))
        
        # Auxiliary Output
        aux_prob = torch.sigmoid(self.aux_head(v_phys))
        
        return main_prob, aux_prob

# --- MOCK DATA FORGE & TEST EXECUTION ---
if __name__ == "__main__":
    print("Initializing Omnikon Detector Test Forge...")
    
    # Initialize the model with verbose logging enabled
    model = Dectector(verbose=False)
    
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