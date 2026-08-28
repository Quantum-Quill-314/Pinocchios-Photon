# OMNIKON — Person B Pipeline: Defocus + Specular Map Extraction
# HOW TO USE:
# 1. Only edit the CONFIG block below when the dataset changes.
#    Nothing else in this file needs to be touched.
# 2. Run in Google Colab (opencv/numpy already installed) or
#    locally with: pip install opencv-python numpy
# 3. Output structure created automatically:
#      OUTPUT_DIR/real/rgb/*.png
#      OUTPUT_DIR/real/defocus/*.npy
#      OUTPUT_DIR/real/specular/*.npy
#      OUTPUT_DIR/fake/... (same structure)
# The Transmuted Physics Pipeline (CPU & GPU Duality)

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ---------------- GLOBAL CONFIG ----------------
SIGMA1  = 1.5                     # defocus blur scale 1 
SIGMA2  = 2.0                     # defocus blur scale 2 
F0_REAL = 0.04                    # reference base reflectance 
EPS     = 1e-6
# -----------------------------------------------

# =====================================================================
# I. THE LAPTOP CRUCIBLE (Sequential CPU / OpenCV)
# =====================================================================

def compute_defocus_cpu(gray, sigma1=SIGMA1, sigma2=SIGMA2):
    """Executes Zhuo & Sim's edge-based defocus on the CPU."""
    gray = gray.astype(np.float32) / 255.0
    blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)

    gx1, gy1 = cv2.Sobel(blur1, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blur1, cv2.CV_32F, 0, 1, ksize=3)
    gx2, gy2 = cv2.Sobel(blur2, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blur2, cv2.CV_32F, 0, 1, ksize=3)
    
    grad1 = np.sqrt(gx1**2 + gy1**2)
    grad2 = np.sqrt(gx2**2 + gy2**2)

    R = grad1 / (grad2 + EPS)
    inside = np.clip((R**2 * sigma1**2 - sigma2**2) / (1 - R**2 + EPS), 0, None)
    sigma_map = np.sqrt(inside)
    return (sigma_map / sigma_map.max()).astype(np.float32) if sigma_map.max() > 0 else sigma_map

def compute_specular_cpu(gray, F0=F0_REAL):
    """Calculates GGX specular microfacet response on the CPU."""
    gray = gray.astype(np.float32) / 255.0
    gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    nz = 1.0 / np.sqrt(gx**2 + gy**2 + 1.0)
    nx, ny = -gx * nz, -gy * nz

    V = L = np.array([0.0, 0.0, 1.0])
    H = (V + L) / (np.linalg.norm(V + L) + EPS)
    NdotH = nx * H[0] + ny * H[1] + nz * H[2]
    NdotV = np.clip(nz, EPS, 1.0)

    gxx, gyy, gxy = cv2.GaussianBlur(gx*gx, (0,0), 1.0), cv2.GaussianBlur(gy*gy, (0,0), 1.0), cv2.GaussianBlur(gx*gy, (0,0), 1.0)
    trace = gxx + gyy
    disc = np.sqrt(np.clip((trace / 2)**2 - (gxx*gyy - gxy**2), 0, None))
    alpha = np.clip(np.sqrt(trace / 2 + disc) / (np.sqrt(np.clip(trace / 2 - disc, 0, None)) + EPS), 0.01, 5.0)

    D = (alpha**2) / (np.pi * (NdotH**2 * (alpha**2 - 1) + 1)**2 + EPS)
    G = (2 * NdotV) / (NdotV + np.sqrt(alpha**2 + (1 - alpha**2) * NdotV**2) + EPS)
    F = F0 + (1 - F0) * (1 - NdotH)**5

    s = (D * G * F) / (NdotV + EPS)
    return (s / s.max()).astype(np.float32) if s.max() > 0 else s

# =====================================================================
# II. THE WORKSTATION ORCHESTRA (Batched GPU / PyTorch)
# =====================================================================

def get_sobel_kernels(device):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    return kx, ky

def compute_defocus_gpu(gray_batch, sigma1=SIGMA1, sigma2=SIGMA2):
    """Executes macro-optics across an entire batch natively on the GPU."""
    device = gray_batch.device
    kx, ky = get_sobel_kernels(device)
    
    blur1 = TF.gaussian_blur(gray_batch, kernel_size=[7, 7], sigma=[sigma1, sigma1])
    blur2 = TF.gaussian_blur(gray_batch, kernel_size=[9, 9], sigma=[sigma2, sigma2])
    
    gx1, gy1 = F.conv2d(blur1, kx, padding=1), F.conv2d(blur1, ky, padding=1)
    gx2, gy2 = F.conv2d(blur2, kx, padding=1), F.conv2d(blur2, ky, padding=1)
    
    grad1, grad2 = torch.sqrt(gx1**2 + gy1**2), torch.sqrt(gx2**2 + gy2**2)
    R = grad1 / (grad2 + EPS)
    
    inside = torch.clamp((R**2 * sigma1**2 - sigma2**2) / (1 - R**2 + EPS), min=0)
    sigma_map = torch.sqrt(inside)
    
    batch_max = sigma_map.view(sigma_map.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + EPS
    return sigma_map / batch_max

def compute_specular_gpu(gray_batch, F0=F0_REAL):
    """Executes GGX micro-optics across an entire batch natively on the GPU."""
    device = gray_batch.device
    kx, ky = get_sobel_kernels(device)
    
    gx, gy = F.conv2d(gray_batch, kx, padding=1), F.conv2d(gray_batch, ky, padding=1)
    nz = 1.0 / torch.sqrt(gx**2 + gy**2 + 1.0)
    nx, ny = -gx * nz, -gy * nz
    
    V = L = torch.tensor([0.0, 0.0, 1.0], device=device)
    H = (V + L) / (torch.norm(V + L) + EPS)
    NdotH = nx * H[0] + ny * H[1] + nz * H[2]
    NdotV = torch.clamp(nz, min=EPS, max=1.0)
    
    gxx, gyy, gxy = gx*gx, gy*gy, gx*gy
    gxx = TF.gaussian_blur(gxx, kernel_size=[5, 5], sigma=[1.0, 1.0])
    gyy = TF.gaussian_blur(gyy, kernel_size=[5, 5], sigma=[1.0, 1.0])
    gxy = TF.gaussian_blur(gxy, kernel_size=[5, 5], sigma=[1.0, 1.0])
    
    trace = gxx + gyy
    disc = torch.sqrt(torch.clamp((trace / 2)**2 - (gxx*gyy - gxy**2), min=0))
    alpha = torch.clamp(torch.sqrt(trace / 2 + disc) / (torch.sqrt(torch.clamp(trace / 2 - disc, min=0)) + EPS), min=0.01, max=5.0)
    
    D = (alpha**2) / (torch.pi * (NdotH**2 * (alpha**2 - 1) + 1)**2 + EPS)
    G = (2 * NdotV) / (NdotV + torch.sqrt(alpha**2 + (1 - alpha**2) * NdotV**2) + EPS)
    F_fresnel = F0 + (1 - F0) * (1 - NdotH)**5
    
    s = (D * G * F_fresnel) / (NdotV + EPS)
    batch_max = s.view(s.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + EPS
    return s / batch_max