# Pinocchio's Photon: A Physics-Augmented Deepfake Inspection Platform

> *Where the laws of optics become the ultimate arbiters of digital truth.*

## 📜 1. Executive Summary & Product Vision

Generative AI models and face-synthesis algorithms have achieved photorealistic fidelity in the spatial pixel domain, rendering traditional RGB-only visual inspection obsolete. Conventional neural network classifiers tend to overfit to superficial texture statistics or specific generator fingerprints, causing them to fail when presented with new, unseen generative architectures or compressed media.

Our solution is a **Physics-Augmented Deepfake Inspection Platform** that evaluates the physical legitimacy of human images. Rather than guessing based on visual patterns, the product acts as an optical forensics inspector that simultaneously audits two physical phenomena that generative algorithms struggle to replicate:
- **Macro-Optics (Defocus Blur):** The optical laws governing camera aperture, depth-of-field, and focal plane sharpness.
- **Micro-Optics (Surface Reflectance):** The microscopic physics of light scattering and specular highlights across biological human skin surfaces (microfacet theory).

---

## 🏗️ 2. Core Methodology & Architecture

To prevent visual RGB artifacts from overpowering the physical laws, the model adopts an asymmetric **Dual-Branch Modulated Architecture**.

### I. Visual Stream (RGB Branch)
Extracts rich, high-level semantic features and visual context from standard RGB pixel channels using a pre-trained backbone, compressing it into a fixed-size 512-dimensional vector ($V_{rgb}$).

### II. Physics Extraction Stream
Processes a 2-Channel Optical Tensor (Defocus Map + Specular Map) to extract low-level spatial gradients and structural relationships, projecting them into a 512-dimensional latent space ($V_{phys}$).

### III. Physics Modulation & Gating Mechanism
The physics vector is mapped through a Sigmoid activation to create dynamic attention weights ($Gate = \sigma(V_{phys})$). These weights are fused with the visual stream using a Hadamard product:
$$V_{modulated} = V_{rgb} \odot Gate$$
This enforces a strict physical "veto"—preventing the network from relying solely on superficial RGB artifacts.

---

## 📂 3. Dataset Description

To maintain computational feasibility and scale within the strict time constraints of the hackathon, our current dataset relies on targeted, high-quality subsets to forge our optical baselines:

*   **Authentic (Real) Data:** The genuine human baseline is sourced from the **Flickr-Faces** dataset (via Kaggle). This provides a rich diversity of natural lighting conditions, skin textures, and genuine microfacet reflectance profiles.
*   **Synthetic (Fake) Data:** The manipulated samples are exclusively sourced from the **SFHQ Dataset Part 1** (via Kaggle). Constraining the synthetic data to this subset allows for a focused and rapid training phase for our dual-branch architecture without compromising the integrity of the physics extraction.
*   **Sample Test Data (Real and Fake):** The real image were scraped from the internet, while the fake images were generated from Gemini and GPT.  

### 🚀 Future Scope: Architectural Agnosticism
While the current prototype focuses on the SFHQ dataset for feasibility, the ultimate vision of the Pinocchio's Photon  platform is to be entirely agnostic to the generator's origin. Future iterations will significantly scale the training pipeline to ingest fake images forged by a wide variety of Generative AI architectures—including diverse Generative Adversarial Networks (GANs), advanced Latent Diffusion Models, and auto-regressive synthesizers. By exposing the network to a broader spectrum of synthesis techniques, the physics-veto mechanism will become robust against the ever-evolving landscape of digital deception.

---

## 🛠️ 4. Technical Stack & Required Libraries

Ensure your environment is equipped with the following before forging the pipeline:

- **Core:** `Python 3.10+`
- **Deep Learning Framework:** `torch`, `torchvision`
- **Computer Vision & Matrix Math:** `opencv-python` (cv2), `numpy`, `Pillow` (PIL)
- **Machine Learning & Clustering:** `scikit-learn` (MiniBatchKMeans)
- **Data Manipulation:** `pandas`
- **Hardware Acceleration:** CUDA-compatible GPU (Optional but highly recommended for training)

---

## ⚙️ 5. Installation & Execution Instructions

### Step 1: Clone and Setup
```bash
git clone https://github.com/Omnikon-Org/<your-repo-name>.git
cd <your-repo-name>
pip install torch torchvision opencv-python numpy scikit-learn pandas Pillow
```

### Step 2: Optical Map Extraction
To generate the Defocus and Specular physical maps from your raw datasets, execute the pipeline script. Update the `REAL_DIR` and `FAKE_DIR` paths in the config block before running.
```bash
python omnikon_pipeline.py
```

### Step 3: Dataset Forging & Clustering
Once the data is pre-processed, use the topological mapper to partition the spatial vectors and construct a balanced dataset split (Train/CV/Test).
```bash
python dataset_forge.py
```

### Step 4: Network Verification & Training
To verify the dual-branch modulated architecture and test the forward pass with dummy tensors:
```bash
python dual_branch_net.py
```
*(Use `config.py` to adjust hyperparameters like `BATCH_SIZE`, `LATENT_DIM`, and `DROPOUT_RATE` as needed).*

---

## 👥 6. Team & Contributors

*   **Dhawal Mehrotra** - [Quantum-Quill-314]
    *   *Role:* Dataset Curation, Topological Clustering, and Neural Network Architecture Design.
*   **[Divya Mangtani]** - [divyamangtani27]
    *   *Role:* Physical Optics Pipeline (Defocus & Specular extraction), UI/UX Development, and General Project Integration.

---

## 🤖 7. Generative AI Disclosure

In accordance with the Omnikon Hackathon rules, we formally disclose our use of Generative AI tools during the development of this repository. 

*   **Ideation & Conceptualization:** Exclusively human-driven. The core concept of the Pinocchio's Photon's frame and the physics-gated methodology was conceptualized entirely by the team; with inspiration from the following references mentioned below.
*   **Development & Implementation:** Generative AI was extensively utilized across all subsequent phases, including coding assistance, scripting the data processing pipelines, debugging neural network tensor dimensionalities, and structuring this technical documentation.
---
## 📚 8. References

1. Kumari, K., Behrouzi, S., Pegoraro, A., & Sadeghi, A.-R. (2026). Light2Lie: Detecting deepfake images using physical reflectance laws. Network and Distributed System Security (NDSS) Symposium 2026, San Diego, CA. https://dx.doi.org/10.14722/ndss.2026.230923

2. Jeon, M., & Woo, S. S. (2025). Seeing through the blur: Unlocking defocus maps for deepfake detection. Proceedings of the 34th ACM International Conference on Information and Knowledge Management (CIKM '25), Seoul, Republic of Korea. https://doi.org/10.1145/3746252.3761260
---

*License: MIT 
