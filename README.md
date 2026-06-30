# 🫁 Asthma vs COPD Classification using Deep Learning

> A deep learning framework for automated differentiation of Asthma and Chronic Obstructive Pulmonary Disease (COPD) using respiratory sound recordings.

---

## 📖 Overview

Asthma and Chronic Obstructive Pulmonary Disease (COPD) are among the most prevalent chronic respiratory diseases worldwide. Due to overlapping clinical symptoms such as wheezing, coughing, and shortness of breath, differentiating between these conditions can be challenging using traditional diagnostic methods alone.

This project presents an end-to-end deep learning pipeline that automatically classifies respiratory sound recordings into **Asthma** or **COPD**. The proposed framework leverages **MFCC feature extraction** along with a hybrid **CNN + BiGRU + Attention** architecture to learn both local acoustic characteristics and long-term temporal dependencies from respiratory sounds.

---

## 🎯 Objectives

- Develop an automated respiratory disease classification system.
- Differentiate Asthma and COPD using lung auscultation recordings.
- Learn discriminative acoustic features through deep learning.
- Evaluate the model using clinically relevant performance metrics.

---

## 📂 Datasets

The model is trained using respiratory sound recordings collected from multiple datasets:

- **ICBHI Respiratory Sound Database**
- **KAUH Respiratory Sound Dataset**
- **AIIMS Raipur Respiratory Sound Dataset**

These datasets provide a diverse collection of respiratory sound recordings, improving the model's robustness and generalization.

---

# 🏗️ Project Pipeline

```
Respiratory Audio
        │
        ▼
Audio Preprocessing
        │
        ├── Resampling (16 kHz)
        ├── Noise Reduction
        ├── Silence Removal
        ├── Respiratory Cycle Segmentation
        ▼
MFCC Feature Extraction
        ▼
CNN Feature Extractor
        ▼
BiGRU
        ▼
Attention Mechanism
        ▼
Fully Connected Layer
        ▼
Softmax Classifier
        ▼
Asthma / COPD
```

---

## ⚙️ Audio Preprocessing

The following preprocessing techniques are applied before model training:

- Resampling all audio recordings to **16 kHz**
- Noise reduction and signal denoising
- Silence removal
- Respiratory cycle segmentation
- Audio normalization
- MFCC feature extraction

---

## 📈 Data Augmentation

To improve model robustness and reduce overfitting, the following augmentation techniques are applied:

- Time Shifting
- Time Stretching
- Pitch Shifting
- Additive Gaussian Noise

---

## 🧠 Model Architecture

The proposed architecture combines Convolutional Neural Networks, Bidirectional GRUs, and an Attention mechanism.

```
Input MFCC
      │
Conv Block 1
      │
Conv Block 2
      │
Conv Block 3
      │
BiGRU
      │
Attention Layer
      │
Fully Connected Layer
      │
Softmax
```

### CNN

- Learns local acoustic patterns from MFCC representations.
- Extracts respiratory sound features such as wheezes and crackles.

### BiGRU

- Captures temporal dependencies across respiratory cycles.
- Processes sequential information in both forward and backward directions.

### Attention Mechanism

- Learns to focus on diagnostically important regions of respiratory sounds.
- Improves classification performance by emphasizing informative temporal features.

---

## ⚙️ Training Configuration

| Parameter | Value |
|------------|--------|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 100 |

---

## 📊 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Sensitivity
- Specificity
- Confusion Matrix

---

## 📈 Results

| Metric | Score |
|----------|---------|
| Accuracy | **95.2%** |
| Precision | **94.6%** |
| Recall | **93.2%** |
| F1-score | **93.8%** |
| ROC-AUC | **96.1%** |

---

## 📁 Project Structure

```
Asthma-COPD-Classification/
│
├── datasets/
│
├── preprocessing/
│   ├── resample.py
│   ├── denoise.py
│   ├── segmentation.py
│   ├── mfcc.py
│   └── preprocessing.py
│
├── augmentation/
│
├── models/
│   ├── cnn.py
│   ├── bigru.py
│   ├── attention.py
│   └── model.py
│
├── training/
│   ├── train.py
│   └── trainer.py
│
├── evaluation/
│   ├── metrics.py
│   └── evaluate.py
│
├── inference/
│   └── predict.py
│
├── notebooks/
│
├── saved_models/
│
├── results/
│
├── requirements.txt
│
├── README.md
│
└── LICENSE
```

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/Asthma-COPD-Classification.git

cd Asthma-COPD-Classification

pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

---

## 🔍 Run Inference

```bash
python predict.py --audio sample.wav
```

---

## 🛠️ Tech Stack

- Python
- PyTorch
- NumPy
- Librosa
- Scikit-learn
- Pandas
- Matplotlib
- OpenCV

---

## 🔮 Future Work

- Vision Transformer (ViT)
- Self-Supervised Learning
- Explainable AI (Grad-CAM / Attention Visualization)
- FastAPI Deployment
- Streamlit Web Application
- Docker Containerization
- Real-time Respiratory Disease Detection

---

## 📜 License

This project is developed for educational and research purposes.
