# Pinocchios-Photon
# Project Source of Truth: Dual-Physical Optics Deepfake Detection Framework

## 1. Executive Summary & Product Vision

### The Problem

Generative AI models and face-synthesis algorithms have achieved photorealistic fidelity in the spatial pixel domain, rendering traditional RGB-only visual inspection obsolete. Conventional neural network classifiers tend to overfit to superficial texture statistics or specific generator fingerprints, causing them to fail when presented with new, unseen generative architectures or compressed media.

  

### The Product

Our solution is a **Physics-Augmented Deepfake Inspection Platform** that evaluates the physical legitimacy of human images. Rather than guessing based on visual patterns, the product acts as an optical forensics inspector that simultaneously audits two physical phenomena that generative algorithms struggle to replicate:

  

- **Macro-Optics (Defocus Blur):** The optical laws governing camera aperture, depth-of-field, and focal plane sharpness.
    
      
    
- **Micro-Optics (Surface Reflectance):** The microscopic physics of light scattering and specular highlights across biological human skin surfaces (microfacet theory).
    
      
    

The end deliverable is a responsive, web-hosted diagnostic tool that produces a tamper confidence score alongside transparent, visual explanation heatmaps highlighting exactly where the laws of optics were violated.

  

## 2. Core Methodology & Theoretical Grounding


![[mermaid-diagram-2026-08-18-202030.png|700]]

### I. Optical Defocus Discrepancy (Macro-Optics)

- Real cameras produce natural, depth-dependent defocus blur determined by lens aperture and distance from the focal plane.
    
      
    
- Generative models tend to generate all-in-focus faces or spatially inconsistent blur along facial boundaries.
    
      
    
- The system computes a continuous Defocus Map by evaluating gradient magnitude ratios at dual Gaussian scales ($\sigma_1, \sigma_2$) smoothed via edge-preserving guided filters, isolating focal discrepancies.
    
      
    

### II. Microfacet Specular Reflectance (Micro-Optics)

- Biological human skin is not an ideal smooth surface; it consists of countless microfacets that scatter light based on roughness, geometric masking, and angle of incidence.
    
      
    
- Using microfacet theory and the Cook-Torrance/Blinn formulation, the physical specular response $s$ is modeled directly from the image geometry:
    
      
    
    $$s = \frac{D \cdot G \cdot F}{N \cdot V}$$
    
    where $D$ represents the GGX microfacet normal distribution, $G$ is the geometric attenuation factor (shadowing/masking), $F$ is the angle-dependent Fresnel reflectance, and $N \cdot V$ denotes surface-view alignment.
    
      
    
- Deepfakes exhibit unnaturally smooth or mathematically impossible specular highlights, exposing synthetic skin regions.
    
      
    

## 3. End-to-End Methodology Flowchart

![[mermaid-diagram-2026-08-18-202230.png]]

## 4. Proposed Deep Learning Neural Network Architecture

To prevent visual RGB artifacts from overpowering the physical laws, the model adopts an asymmetric **Dual-Branch Modulated Architecture** where physics feature maps directly modulate the visual latent embeddings via a Hadamard product.

  

### Neural Network Symbolic Flowchart

![[mermaid-diagram-2026-08-18-202322.png]]

### Detailed Neural Network Layer Breakdown

**I. VISUAL STREAM (RGB BRANCH)**

  

- **Pre-trained Backbone (EfficientNet / ResNet):** Extracts rich, high-level semantic features and visual context from standard RGB pixel channels.
    
      
    
- **Adaptive Global Average Pooling (GAP):** Compresses the multi-channel 2D spatial feature maps into a fixed-size 512-dimensional vector ($V_{rgb}$), removing spatial dimension dependencies and reducing parameter count.
    
      
    

**II. PHYSICS EXTRACTION STREAM (CUSTOM LIGHTWEIGHT CNN)**

  

- **Input Layer:** 2-Channel Optical Tensor (Defocus Map + Specular S-Map).
    
      
    
- **Conv2D (In: 2, Out: 32, Kernel: 3x3) + ReLU:** Extracts low-level spatial gradients, edge transitions, and localized reflectance inconsistencies from the optical maps.
    
      
    
- **MaxPool2D (Kernel: 2x2):** Reduces spatial resolution by half (112x112), summarizing prominent optical anomalies while providing translational invariance.
    
      
    
- **Conv2D (In: 32, Out: 64, Kernel: 3x3) + ReLU:** Learns mid-level structural relationships between lens defocus and skin microfacet scattering patterns.
    
      
    
- **Conv2D (In: 64, Out: 512, Kernel: 1x1) + Batch Normalization:** Projects the extracted physics features into the identical 512-dimensional latent channel space as the RGB branch for fusion.
    
      
    
- **Adaptive Global Average Pooling (GAP):** Aggregates the 2D physics representations into a clean 512-dimensional vector ($V_{phys}$).
    
      
    

**III. PHYSICS MODULATION & GATING MECHANISM**

  

- **Sigmoid Activation Layer ($Gate = \sigma(V_{phys})$):** Maps the 512-D physics vector strictly into the range $[0, 1]$, transforming the raw physical features into dynamic attention weights.
    
      
    
- **Hadamard Product Fusion ($V_{modulated} = V_{rgb} \odot Gate$):** Performs element-wise multiplication between the visual vector and the physics attention gate. This enforces a strict physical "veto": if the microfacet or defocus physics indicates a synthetic anomaly, the gate attenuates or amplifies specific visual feature channels, preventing the network from relying solely on superficial RGB artifacts.
    
      
    

**IV. CLASSIFICATION & DECISION HEAD**

  

- **Fully Connected Linear Layer (In: 512, Out: 128):** Compresses the modulated multimodal representations into a dense latent decision space.
    
      
    
- **Batch Normalization (1D) + ReLU Activation:** Stabilizes feature distributions across mini-batches and introduces non-linear decision boundaries.
    
      
    
- **Dropout Layer ($p = 0.3$):** Randomly deactivates 30% of neurons during training to prevent overfitting and encourage generalized forensic representations.
    
      
    
- **Final Linear Layer (In: 128, Out: 1):** Projects the condensed 128-D representation to a single logit.
    
      
    
- **Sigmoid Output Function:** Squashes the final logit into a normalized scalar probability between $0.0$ (Authentic Human) and $1.0$ (Manipulated Deepfake).
    
      
    

## 5. Technical Stack

|**Layer / Component**|**Technology / Library**|**Purpose & Rationale**|
|---|---|---|
|**Core Language**|Python 3.10+|Standardized scientific computing and machine learning ecosystem.|
|**Deep Learning Framework**|PyTorch / Torchvision|Dynamic graph execution, modular dual-branch definitions, and GPU acceleration.|
|**Computer Vision & Optics**|OpenCV & NumPy / SciPy|Deterministic 2D spatial gradients, Sobel operators, and guided filter implementations.|
|**Model Backbone**|`timm` / Torchvision (`EfficientNet-B0` / `ResNet-18`)|Highly optimized, lightweight visual feature extractors for minimal inference latency.|
|**Explainability & Metrics**|Captum / SHAP & Scikit-Learn|Extraction of attribution heatmaps, ROC-AUC, Precision, and Recall validation.|
|**Web Interface**|Gradio / Streamlit|Rapid, lightweight interactive UI supporting image uploads, confidence meters, and visual map tabs.|
|**Hosting & Deployment**|Hugging Face Spaces (CPU/T4 Instance)|Free, publicly shareable cloud hosting with low cold-start latency.|


## 6. Project Implementation Phases

**Phase 1: Mathematical Engine & Optical Feature Extractors**
* Implement edge-based Defocus Blur estimation pipeline.
* Implement Microfacet Reflectance normal & GGX specular equations.
* Validate feature extractors across standard authentic and fake samples.

**Phase 2: Data Pipeline & Tensor Generation**
* Curate balanced face datasets (Authentic vs. GAN/Diffusion Deepfakes).
* Construct automated batch pre-caching for Defocus and Specular maps.
* Establish cross-validation splits.

**Phase 3: Dual-Branch Network Training & Optimization**
* Assemble the PyTorch Dual-Branch Modulation module.
* Train classifier head using Binary Cross-Entropy with early stopping.
* Perform cross-generator evaluation to verify generalization.

**Phase 4: Explainability Layer & Diagnostic Heatmaps**
* Implement gradient-weighted or SHAP-aligned saliency generation.
* Create visual overlays comparing genuine skin reflection vs. fake zones.

**Phase 5: Packaging, User Interface & Public Deployment**
* Construct the interactive Gradio dashboard.
* Optimize inference execution for single-image CPU/GPU evaluation.
* Deploy model artifacts to Hugging Face Spaces with a shareable URL.
## 7. Deliverable Specifications

- **Model Checkpoint:** A compact model  (<150MB) capable of executing inference in under 5 seconds per sample on modern hardware.
    
      
    
- **Interactive Dashboard:**
    
      
    - Clean file drop-zone for standard image formats (JPEG, PNG).
        
          
        
    - Direct probability score gauge (Real vs. Deepfake).
        
          
        
    - Multi-tab visual diagnostic inspector showing:
        
          
        1. The Original Cropped Face.
            
            
        2. The Computed Defocus Blur Discrepancy Map.
                        
        3. The Microfacet Specular Reflectance Map.
            
        4. The Final Decision Attribution Saliency Map.
            
              
            
- **Shareable Link:** Fully hosted web URL allowing external evaluators to test images in real time without local setup.



###Contributors:
1. Quantum-Quill-314 (Dhawal Mehrotra)
2. divyamangtani27 (Divya Mangtani)
