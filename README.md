# Color-Invariant Saree Design Recognition using Deep Learning

This repository focuses on building a deep learning framework capable of recognizing intricate saree designs and patterns while remaining invariant to changes in color. Traditional garment and textile retrieval systems often rely heavily on color spaces, which fails when users look for identical patterns, weaves, or motifs available in alternative colorways. 

By stripping color dependency, this model extracts pure texture, structural geometry, and stylistic design features to ensure robust pattern matching.

## 🚀 Key Features
* **Color-Invariant Representation:** Focuses on structural details, weave types, and layout geometry rather than RGB values.
* **Feature Extraction Pipeline:** Implements image pre-processing transformations designed to reduce color bias.
* **Deep Feature Matching:** Built for deep retrieval tasks to accurately map queries against an extensive textile gallery.

## 🛠️ Tech Stack & Environment
* **Framework:** PyTorch / torchvision
* **Data Augmentation:** Albumentations
* **Evaluation:** Scikit-Learn (Metrics), Seaborn, Matplotlib
* **Development Environment:** Kaggle Notebooks (GPU Accelerated T4 x2)

## 📂 Project Workflow

### 1. Image Transformations
To enhance robust design extraction and alleviate color dominance during training, the pipeline leverages advanced data augmentations:
* Resizing & Aspect Ratio Padding
* Grayscale / Color Jittering (to reduce RGB weight dependency)
* Random Flips and Spatial Distortions
* ImageNet Normalization Matrix

### 2. Model Pipeline
The core framework employs an end-to-end Deep Convolutional Network (or Vision Transformer) tailored for feature extraction. The model evaluates similarity matching across three split types:
* **Train Set:** Used to optimize pattern feature weights.
* **Gallery Set:** The database housing known saree design structures.
* **Query Set:** The test instances used to evaluate retrieval accuracy against the gallery.

### 3. Evaluation Metrics
The framework analyzes model outputs using evaluation components including:
* **Confusion Matrices:** Tracking design class distributions.
* **Rank-N Accuracy / mAP:** Evaluating multi-class image retrieval performance.

## 📊 Sample Usage

```python
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix

# Example Evaluation Configuration
IMAGE_SIZE = (256, 128)
eval_transforms = T.Compose([
    T.Resize(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 📈 Future Roadmaps
* Implementing **Contrastive Learning Architecture** (e.g., Triplet Loss or SupCon) to tighten feature distances for matching designs regardless of color palettes.
* Scaling up spatial attention maps to visualize localized textile weaves.
