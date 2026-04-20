# Skin-Disease-Prediction-AI-
Skin Disease Prediction AI is a deep learning project that classifies skin conditions using CNNs. It features a Streamlit dashboard and a Flask API for seamless use. Unique for its Explainable AI (XAI), it generates Heatmaps to highlight infected areas, ensuring transparency and trust in AI-driven medical diagnostics.

# 🩺 Skin Disease Prediction AI with Explainable Heatmaps

A comprehensive Machine Learning system designed to identify skin diseases from images and provide visual explanations using Heatmaps (Grad-CAM).

## 📂 Project Structure
- `dataset/`: Training and testing images.
- `models/`: Pre-trained weight files and saved models.
- `streamlit_app.py`: Interactive user dashboard for easy testing.
- `app.py / test_api.py`: Backend API implementation.
- `predict.py`: Core prediction script.
- `utils.py`: Helper functions for image processing and heatmap generation.
- `class_names.json`: List of disease categories the model can identify.

## ✨ Features
- **Accurate Diagnosis:** Multi-class classification of skin diseases.
- **Explainable AI (XAI):** Generates heatmaps to highlight the exact skin lesion area analyzed by the AI.
- **Full-Stack ML:** Includes training scripts, prediction logic, and web deployment.
- **Disease Information:** Integrated `disease_info.py` to provide details about the detected condition.

## 🚀 How to Run

### 1. Setup Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
