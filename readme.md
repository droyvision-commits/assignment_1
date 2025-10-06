# 🧠 ML Ops Assignment 1 — Boston Housing Price Prediction

## 📌 Overview
This project implements a complete ML workflow to predict **house prices** using classical ML models.  
Two models are trained:
- **Decision Tree Regressor** (`train.py`)
- **Kernel Ridge Regressor** (`train2.py`)

Automation is handled via **GitHub Actions**, which automatically trains and evaluates models on code push.

---

## ⚙️ Setup Instructions

### 1️⃣ Create and activate Conda environment
```bash
conda create --name mlops_a1 python=3.10 -y
conda activate mlops_a1
