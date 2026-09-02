# 🌱 AI Plant Health Assistant

An AI-powered web application that analyzes plant leaf images and predicts whether the plant is **healthy or affected by a disease**. The application uses a deep learning image-classification model and provides the predicted disease, confidence level, and plant-health information through an interactive Streamlit interface.

## 🚀 Live Demo

👉 **[Try the AI Plant Health Assistant](https://ai-plant-ha.streamlit.app/)**

---

## 📌 Project Overview

Plant diseases can significantly reduce crop productivity when they are not identified early. This project aims to provide a simple and accessible AI-based tool that can help users identify common plant diseases from leaf images.

The user uploads an image of a plant leaf, and the trained deep learning model analyzes the image and produces a prediction.

### The application can:

* 📷 Accept plant leaf images
* 🔍 Analyze the uploaded image using a trained CNN model
* 🌿 Identify the predicted plant/disease class
* 📊 Display prediction confidence
* 🚦 Apply confidence-based prediction safeguards
* 💡 Provide plant-health information
* 🖥️ Present the results through an interactive Streamlit web interface

---

## 🧠 Machine Learning Model

The project uses a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.

The model was trained using plant leaf images from a PlantVillage-based dataset.

### Model configuration

| Component         | Details                            |
| ----------------- | ---------------------------------- |
| Framework         | TensorFlow / Keras                 |
| Model Type        | Convolutional Neural Network       |
| Input Size        | 128 × 128 pixels                   |
| Dataset           | PlantVillage-based dataset         |
| Number of Classes | 15                                 |
| Data Augmentation | Random Flip, Rotation, Zoom        |
| Output            | Multi-class disease classification |
| Model Format      | `.keras`                           |

### Validation Performance

The trained model achieved approximately:

**92.56% validation accuracy**

> Validation accuracy represents performance on the validation dataset and does not guarantee the same accuracy on every real-world photograph.

---

## 🌿 Supported Classes

The current model recognizes 15 classes:

1. Pepper Bell Bacterial Spot
2. Pepper Bell Healthy
3. Potato Early Blight
4. Potato Late Blight
5. Potato Healthy
6. Tomato Bacterial Spot
7. Tomato Early Blight
8. Tomato Late Blight
9. Tomato Leaf Mold
10. Tomato Septoria Leaf Spot
11. Tomato Spider Mites
12. Tomato Target Spot
13. Tomato Yellow Leaf Curl Virus
14. Tomato Mosaic Virus
15. Tomato Healthy

---

## 🏗️ Project Architecture

```text
User
  │
  ▼
Upload Plant Leaf Image
  │
  ▼
Streamlit Web Application
  │
  ▼
Image Preprocessing
  │
  ▼
TensorFlow/Keras CNN Model
  │
  ▼
Class Prediction + Confidence
  │
  ├── High Confidence
  │       ▼
  │   Disease/Health Result
  │
  └── Low Confidence
          ▼
      Uncertain / Out-of-Scope Result
```

---

## 🛡️ Confidence Guard

A confidence-based safeguard was added to reduce misleading predictions when the model is uncertain.

Instead of treating every prediction as reliable, the application checks the model's confidence before displaying a disease diagnosis.

This helps distinguish between:

* ✅ High-confidence predictions
* ⚠️ Low-confidence predictions
* ❌ Images that may be outside the model's supported classes

This is especially important because a machine-learning model trained on a specific dataset cannot reliably identify every possible plant, disease, or photograph.

---

## 🖥️ Technology Stack

### Frontend / Web Application

* Streamlit

### Machine Learning

* TensorFlow
* Keras
* NumPy

### Image Processing

* Pillow

### Development

* Python
* Jupyter Notebook
* Visual Studio Code

### Deployment

* Streamlit Community Cloud

---

## 📂 Project Structure

```text
AI-Plant-Health-Assistant/
│
├── app.py
├── plant_disease_baseline.keras
├── requirements.txt
├── README.md
│
└── notebooks/
    └── myproject.ipynb
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI-Plant-Health-Assistant
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📋 Requirements

Example `requirements.txt`:

```text
streamlit
tensorflow
numpy
pillow
```

---

## 🧪 How to Use

1. Open the **[live application](https://ai-plant-ha.streamlit.app/)**.
2. Upload a clear image of a plant leaf.
3. Wait for the AI model to analyze the image.
4. Review the predicted class.
5. Check the displayed confidence level.
6. If the model is uncertain, follow the application's recommendation rather than treating the result as a confirmed diagnosis.

---

## 📊 Model Development

The model-development workflow included:

```text
Dataset Collection
       ↓
Dataset Inspection
       ↓
Image Preprocessing
       ↓
Training / Validation Split
       ↓
CNN Model Development
       ↓
Data Augmentation
       ↓
Model Training
       ↓
Model Evaluation
       ↓
Model Testing
       ↓
Streamlit Integration
       ↓
Deployment
```

---

## ⚠️ Limitations

This application is an AI-based classification tool and should not be considered a professional agricultural diagnosis system.

Important limitations include:

* The model only recognizes the classes it was trained on.
* Real-world images may differ significantly from training images.
* Lighting, image quality, backgrounds, and leaf orientation can affect predictions.
* A high confidence score does not necessarily mean the prediction is correct.
* The model may perform poorly on plant diseases that are not represented in the training dataset.

For important agricultural decisions, AI predictions should be verified using reliable agricultural expertise.

---

## 🔮 Future Improvements

Potential improvements include:

* 🌾 Add more crops and disease classes
* 📸 Improve performance on real-world photographs
* 🧠 Use transfer learning models such as MobileNet or EfficientNet
* 🔬 Add explainable AI with Grad-CAM
* 📱 Improve mobile responsiveness
* 🌍 Add multilingual support
* 📈 Track prediction statistics
* 🗺️ Add region-specific plant-health recommendations
* 🧪 Expand and diversify the training dataset
* 🎯 Improve out-of-distribution image detection

---

## 👨‍💻 Project Goal

The goal of this project is to demonstrate how **artificial intelligence and computer vision can be applied to agriculture** to make plant-health analysis more accessible.

It combines:

**Machine Learning + Computer Vision + Web Development + Agriculture**

into a practical AI application.

---

## 🌐 Live Application

### 🌱 AI Plant Health Assistant

**https://ai-plant-ha.streamlit.app/**

---

## 📜 License

This project is intended for educational and research purposes.
