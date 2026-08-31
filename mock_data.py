# Generates synthetic PyTorch tensors to validate the forward pass.

import torch
from config import BATCH_SIZE, RGB_CHANNELS, PHYSICS_CHANNELS, IMG_SIZE

def generate_mock_batch():
    """
    Forges dummy PyTorch tensors representing the RGB and Physical maps.
    Returns a tuple: (rgb_tensor, physics_tensor)
    """
    # Simulates the Visual Stream input
    mock_rgb = torch.randn(BATCH_SIZE, RGB_CHANNELS, IMG_SIZE, IMG_SIZE)
    
    # Simulates the Physics Stream input (Defocus + Specular Maps)
    mock_physics = torch.randn(BATCH_SIZE, PHYSICS_CHANNELS, IMG_SIZE, IMG_SIZE)
    
    print(f"Forged RGB Tensor Shape: {mock_rgb.shape}")
    print(f"Forged Physics Tensor Shape: {mock_physics.shape}")
    
    return mock_rgb, mock_physics

if __name__ == "__main__":
    generate_mock_batch()