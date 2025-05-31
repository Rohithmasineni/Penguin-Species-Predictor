# 🐧 Penguin-Species-Predictor

A machine learning project to predict the species of penguins—**Adelie**, **Chinstrap**, or **Gentoo**—based on physical characteristics using a **K-Nearest Neighbors (KNN)** classifier. This project includes exploratory data analysis, preprocessing, model training, evaluation, and deployment via a Streamlit web app hosted on Hugging Face Spaces.

---

## Objective

To build a multi-class classification model that predicts the species of a penguin based on numeric and categorical features using **KNeighborsClassifier**.

---

## Features Used

### Numerical Features:
- `culmen_length_mm`
- `culmen_depth_mm`
- `flipper_length_mm`
- `body_mass_g`

### Categorical Features:
- `island` (Torgersen, Biscoe, Dream)
- `sex` (MALE, FEMALE, NA)

### 🎯 Target Variable:
- `species` (Adelie, Chinstrap, Gentoo)

---

## Process Followed

1. **Dataset Exploration**
2. **Initial Data Cleaning** (handling missing values)
3. **Exploratory Data Analysis**
   - Univariate and Bivariate analysis
4. **Preprocessing**
   - Encoding categorical variables
   - Feature scaling (for KNN)
5. **Model Training**
   - K-Nearest Neighbors (K=optimal via cross-validation)
6. **Model Evaluation**
   - Accuracy Score: **98.5%**
   - Confusion Matrix:
     ```
     [[30  1  0]
      [ 0 13  0]
      [ 0  0 23]]
     ```
7. **Deployment**
   - Built a **Streamlit web app**
   - Deployed on **Hugging Face Spaces**
  
---

## 🧰 Tech Stack

- Python
- Jupyter Notebook
- Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- Streamlit
- Hugging Face Spaces (for deployment)

---

## 🌐 Deployment

👉 **Live Demo:** Penguin Species Predictor on Hugging Face Spaces https://huggingface.co/spaces/rohithmasineni/PenguinSpeciesPredictor
