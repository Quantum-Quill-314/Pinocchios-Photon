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
import cv2
import numpy as np
from pathlib import Path
import pandas as pd 
# ---------------- GLOBAL CONFIG (edit ONLY this block) ----------------
IMG_SIZE    = 299                     # target size for all images (matches Xception input)
REAL_DIR    = "/content/data/real"    # <-- change this when dataset changes
FAKE_DIR    = "/content/data/fake"    # <-- change this when dataset changes
OUTPUT_DIR  = "/content/output"       # where processed data goes
NUM_SAMPLES = 1200                    # images per class to use (subset for hackathon speed)
SIGMA1      = 1.5                     # defocus blur scale 1 (paper-validated best config)
SIGMA2      = 2.0                     # defocus blur scale 2 (paper-validated best config)
F0_REAL     = 0.04                    # reference base reflectance for real skin
VALID_EXTS  = (".png", ".jpg", ".jpeg")
# ------------------------------------------------------------------------


def list_images(folder, limit):
    """Grab up to `limit` image files from a folder. Works regardless of
    how the dataset is organized internally, as long as images sit
    directly in this folder."""
    folder = Path(folder)
    files = [f for f in folder.iterdir() if f.suffix.lower() in VALID_EXTS]
    return sorted(files)[:limit]


def load_and_resize(path, size=IMG_SIZE):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return img


# ---------------- DEFOCUS MAP (Zhuo & Sim, edge-based) ----------------
def compute_defocus_map(gray, sigma1=SIGMA1, sigma2=SIGMA2, eps=1e-6):
    """Returns a 0-1 normalized per-pixel blur intensity map.
    Higher values = more blur at that pixel."""
    gray = gray.astype(np.float32) / 255.0

    blur1 = cv2.GaussianBlur(gray, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(gray, (0, 0), sigma2)

    gx1 = cv2.Sobel(blur1, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(blur1, cv2.CV_32F, 0, 1, ksize=3)
    grad1 = np.sqrt(gx1 ** 2 + gy1 ** 2)

    gx2 = cv2.Sobel(blur2, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(blur2, cv2.CV_32F, 0, 1, ksize=3)
    grad2 = np.sqrt(gx2 ** 2 + gy2 ** 2)

    R = grad1 / (grad2 + eps)
    inside = (R ** 2 * sigma1 ** 2 - sigma2 ** 2) / (1 - R ** 2 + eps)
    inside = np.clip(inside, 0, None)
    sigma_map = np.sqrt(inside)

    if sigma_map.max() > 0:
        sigma_map = sigma_map / sigma_map.max()
    return sigma_map.astype(np.float32)


# ---------------- SPECULAR MAP (simplified GGX / Schlick) ----------------
def compute_specular_map(gray, F0=F0_REAL, eps=1e-6):
    """Returns a 0-1 normalized per-pixel specular reflectance score.
    This is a simplified, hackathon-speed approximation of the full
    microfacet model (assumes a fixed frontal light/view direction —
    a known simplification, mention this in your pitch)."""
    gray = gray.astype(np.float32) / 255.0

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # approximate surface normal from gradients
    nz = 1.0 / np.sqrt(gx ** 2 + gy ** 2 + 1.0)
    nx = -gx * nz
    ny = -gy * nz

    # fixed frontal light + view direction (simplification)
    V = np.array([0.0, 0.0, 1.0])
    L = np.array([0.0, 0.0, 1.0])
    H = (V + L) / (np.linalg.norm(V + L) + eps)

    NdotH = nx * H[0] + ny * H[1] + nz * H[2]
    NdotV = np.clip(nz, eps, 1.0)

    # roughness proxy from local gradient covariance
    gxx = cv2.GaussianBlur(gx * gx, (0, 0), 1.0)
    gyy = cv2.GaussianBlur(gy * gy, (0, 0), 1.0)
    gxy = cv2.GaussianBlur(gx * gy, (0, 0), 1.0)
    trace = gxx + gyy
    det = gxx * gyy - gxy ** 2
    disc = np.sqrt(np.clip((trace / 2) ** 2 - det, 0, None))
    lam1 = trace / 2 + disc
    lam2 = trace / 2 - disc
    alpha = np.clip(np.sqrt(lam1) / (np.sqrt(lam2) + eps), 0.01, 5.0)

    # GGX distribution D
    denom = (NdotH ** 2 * (alpha ** 2 - 1) + 1)
    D = (alpha ** 2) / (np.pi * denom ** 2 + eps)

    # Smith geometric attenuation G (simplified)
    G = (2 * NdotV) / (NdotV + np.sqrt(alpha ** 2 + (1 - alpha ** 2) * NdotV ** 2) + eps)

    # Schlick Fresnel approximation F
    F = F0 + (1 - F0) * (1 - NdotH) ** 5

    s = (D * G * F) / (NdotV + eps)
    if s.max() > 0:
        s = s / s.max()
    return s.astype(np.float32)


# ---------------- MAIN PIPELINE ----------------
def process_class(folder, label, limit):
    files = list_images(folder, limit)
    print(f"[{label}] found {len(files)} images in {folder}")

    out_img_dir = Path(OUTPUT_DIR) / label / "rgb"
    out_defocus_dir = Path(OUTPUT_DIR) / label / "defocus"
    out_specular_dir = Path(OUTPUT_DIR) / label / "specular"
    for d in (out_img_dir, out_defocus_dir, out_specular_dir):
        d.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(files):
        img = load_and_resize(f)
        if img is None:
            print(f"  skipped unreadable file: {f}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        defocus = compute_defocus_map(gray)
        specular = compute_specular_map(gray)

        name = f"{label}_{i:05d}"
        cv2.imwrite(str(out_img_dir / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        np.save(out_defocus_dir / f"{name}.npy", defocus)
        np.save(out_specular_dir / f"{name}.npy", specular)

        if i % 100 == 0:
            print(f"  processed {i}/{len(files)}")

    print(f"[{label}] done — {len(files)} images processed.\n")


if __name__ == "__main__":
    process_class(REAL_DIR, "real", NUM_SAMPLES)
    process_class(FAKE_DIR, "fake", NUM_SAMPLES)
    print("All done! Check:", OUTPUT_DIR)



