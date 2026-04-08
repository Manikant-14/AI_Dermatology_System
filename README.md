# 🩺 AI Dermatology Intelligence System

A full-stack clinical AI pipeline for skin lesion analysis.

## 🔬 What it does
- **Classification** — MobileNetV2 fine-tuned on HAM10000 (10K images, 7 classes) → **77% accuracy**
- **Segmentation** — U-Net on ISIC 2018 → **0.86 Dice Score**, validated on PH2
- **XAI** — Grad-CAM heatmaps for clinical explainability
- **Severity Scoring** — ABCD rule-based engine
- **Deployment** — Streamlit + FastAPI + MongoDB, ~250ms inference

## 📦 Stack
Python · TensorFlow · Keras · MobileNetV2 · U-Net · OpenCV · Grad-CAM · Streamlit · FastAPI · MongoDB

## 📊 Datasets
HAM10000 · ISIC 2018 · ISIC 2019 (~25K images) · PH2

## 📁 Project Status
🚧 In active development — classification and segmentation modules complete, deployment pipeline in progress.
- PH2

## Future Improvements
- Improve accuracy
- Add more datasets
- Enhance UI/UX
