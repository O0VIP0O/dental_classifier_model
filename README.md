# 🦷 Dental Disease Classification using Deep Learning


## 📋 Project Overview

Advanced deep learning model for classifying dental diseases from images using **ResNet-50** architecture with balanced training techniques and regularization to prevent overfitting.

### 🎯 Supported Disease Classes

1. **Calculus** - Dental tartar buildup
2. **Dental Caries** - Tooth decay/cavities
3. **Gingivitis** - Gum inflammation
4. **Hypodontia** - Missing teeth (congenital)
5. **Mouth Ulcer** - Oral lesions
6. **Tooth Discoloration** - Abnormal tooth coloring

---

## 🚀 Model Performance

| Metric | Score |
|--------|-------|
| **Training Accuracy** | 94.17% |
| **Validation Accuracy** | 90.83% |
| **Balanced Accuracy** | ~90% |
| **Overfitting Gap** | 3.34% ✅ |

> ✅ **Low overfitting gap** indicates excellent generalization!

---

## 📁 Project Structure

```
dental-classification/
├── data/
│   ├── train/           # Training images (organized by class folders)
│   ├── valid/           # Validation images
│   └── test/            # Test images
├── dental_classifier_balanced.pth    # 🔴 Trained model weights
├── train.py             # Training script
├── confusion_matrix_balanced.png     # Evaluation results
├── training_analysis_complete.png    # Training curves
└── README.md
```

---

## 📥 Model Download

**Due to GitHub's file size limit (25MB), the trained model is hosted externally:**

### Download Link:
📦 **[Download dental_classifier_balanced.pth](YOUR_GOOGLE_DRIVE_LINK_HERE)**

**File Size:** ~XMB  
**SHA-256:** `(optional: add checksum here)`

### 📍 How to Use:
1. Download the model file from the link above
2. Place it in the project root directory
3. Follow the inference instructions below

---

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
CUDA-compatible GPU (recommended)
```

### Install Dependencies
```bash
pip install torch torchvision
pip install pillow numpy scikit-learn
pip install matplotlib seaborn tqdm
```

Or use:
```bash
pip install -r requirements.txt
```

---

## 🎓 Training the Model

### Dataset Preparation
Organize your data as follows:
```
data/
├── train/
│   ├── Calculus/
│   ├── Data caries/
│   ├── Gingivitis/
│   ├── hypodontia/
│   ├── Mouth Ulcer/
│   └── Tooth Discoloration/
├── valid/ (same structure)
└── test/ (same structure)
```

### Run Training
```bash
python train.py
```

### Key Features:
- ⚖️ **Balanced training** with weighted loss
- 🔒 **Strong regularization** (dropout 0.6, weight decay 5e-4)
- 🧊 **Frozen early layers** to prevent overfitting
- 📉 **Learning rate scheduling** with ReduceLROnPlateau
- ⏱️ **Early stopping** (patience=7 epochs)
- 🚀 **Mixed precision training** (AMP) for GPU efficiency

---

## 🔮 Inference / Prediction

```python
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load model
checkpoint = torch.load('dental_classifier_balanced.pth')
model = RegularizedDentalClassifier(num_classes=6)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/dental_image.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    
class_names = checkpoint['class_names']
print(f"Predicted: {class_names[predicted.item()]}")
```

---

## 🏗️ Model Architecture

```
RegularizedDentalClassifier
├── Backbone: ResNet-50 (pretrained on ImageNet)
│   ├── Frozen layers: First 70% of parameters
│   └── Fine-tuned layers: Last 30 layers
│
└── Classifier Head:
    ├── Dropout (0.6)
    ├── Linear (2048 → 256)
    ├── ReLU + BatchNorm
    ├── Dropout (0.42)
    └── Linear (256 → 6 classes)
```

**Total Parameters:** ~25M  
**Trainable Parameters:** ~8M

---


**Key Observations:**
- ✅ Smooth convergence without oscillations
- ✅ Validation loss closely tracks training loss
- ✅ No signs of overfitting (small gap between train/val)
- ✅ Balanced accuracy confirms good performance on all classes

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | ResNet-50 (pretrained) |
| Input Size | 224×224×3 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Weight Decay | 5e-4 |
| Dropout Rate | 0.6 |
| Epochs | 15 (with early stopping) |
| Loss Function | Weighted Cross-Entropy |

---

## 🧪 Handling Imbalanced Data

The project implements multiple strategies:

1. **Class Weighting** - Weighted loss function based on class frequencies
2. **Undersampling/Oversampling** - Balance validation set
3. **Balanced Accuracy Metric** - Better evaluation for imbalanced data
4. **Per-class Monitoring** - Track performance for minority classes

---

## 📝 Citation

If you use this model in your research, please cite:

```bibtex
@software{dental_classifier_2025,
  author = {Your Name},
  title = {Dental Disease Classification using Deep Learning},
  year = {2025},
  url = {https://github.com/yourusername/dental-classification}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- ResNet-50 architecture by Microsoft Research
- PyTorch team for the excellent framework
- Dataset contributors (add source if applicable)

---


## ⚠️ Disclaimer

This model is for **educational and research purposes only**. It should NOT be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for dental health concerns.

---

<div align="center">
  <b>⭐ If you find this project useful, please give it a star! ⭐</b>
</div>
