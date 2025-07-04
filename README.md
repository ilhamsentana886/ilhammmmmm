# Prediksi Diabetes Menggunakan Deep Learning

Repositori ini berisi kode Python untuk membangun model deep learning dalam memprediksi apakah seseorang berpotensi terkena diabetes berdasarkan data medis.

## 📦 Dataset
- Sumber: [Kaggle – Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- Fitur: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- Target: Outcome (0 = Tidak Diabetes, 1 = Diabetes)

## 🧠 Model
- Framework: TensorFlow + Keras
- Arsitektur:
  - Dense(16, ReLU)
  - Dense(8, ReLU)
  - Dense(1, Sigmoid)
- Fungsi Aktivasi: ReLU + Sigmoid
- Loss Function: Binary Crossentropy
- Optimizer: Adam

## 🎯 Evaluasi
- Accuracy: ±78%
- Metrik: Confusion Matrix, Classification Report

## ▶️ Cara Menjalankan
```bash
pip install -r requirements.txt
python model.py
