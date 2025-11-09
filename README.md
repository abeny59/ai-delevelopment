# 🏥 AI-Based Patient Readmission Prediction System

This project builds and serves an **AI system** that predicts whether a patient is at risk of being **readmitted to the hospital within 30 days** after discharge.  
It includes data preprocessing, model training, evaluation, and a REST API built with **Flask**.

---

## 🚀 Features
- Generates synthetic hospital data (can be replaced with real EHR data)
- Trains a **Random Forest** model on patient features
- Returns predictions and probabilities through a `/predict` endpoint
- Prints evaluation metrics: **Confusion Matrix**, **Precision**, and **Recall**

---

## ⚙️ Installation

1. Create a new folder (e.g., `hospital-readmission-ai`)
2. Inside it, create a file named `index.py` and paste the Python code below.
3. Create another file named `requirements.txt` and copy the dependencies listed at the end of this document.
4. In the terminal, install dependencies:
   ```bash
   pip install -r requirements.txt
