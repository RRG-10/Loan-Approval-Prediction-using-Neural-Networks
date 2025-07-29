# 🧠 Loan Approval Prediction using Neural Networks

A machine‑learning project that predicts loan approval (Y/N) using a feedforward neural network. By automating application evaluation, this model helps financial institutions speed up decisions and reduce bias.

---

## 📋 Table of Contents
1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Features & Target](#features--target)  
4. [Preprocessing](#preprocessing)  
5. [Model Architecture](#model-architecture)  
6. [Installation & Usage](#installation--usage)  
7. [Results](#results)  
8. [Future Improvements](#future-improvements)  
9. [License](#license)  
10. [Contact](#contact)  

---

## 🔍 Project Overview
- **Objective:** Build a binary classifier to predict loan approval  
- **Approach:**  
  1. Clean & preprocess raw application data  
  2. Train a Keras‑based neural network  
  3. Evaluate on hold‑out test set  
  4. Visualize performance (confusion matrix, loss/accuracy curves)

---

## 🗄️ Dataset
- **Source:** [Kaggle Loan Prediction Problem Dataset](https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset)  
- **Size:** ~614 rows × 13 columns  

---

## 🔑 Features & Target

| Type         | Columns                                      |
| ------------ | -------------------------------------------- |
| **Demographic** | Gender, Married, Dependents, Education, Self_Employed |
| **Financial**   | ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History |
| **Geographic**  | Property_Area                              |
| **Target**      | Loan_Status (Y = Approved, N = Rejected)  |

---

## 🧹 Preprocessing
1. **Missing Values**: Impute numerical with median, categorical with mode  
2. **Encoding**: LabelEncoder for all categorical features  
3. **Scaling**: MinMaxScaler for numeric columns  
4. **Split**: 80% train / 20% test  

---

## 🧠 Model Architecture
- **Framework:** TensorFlow 2.x / Keras  
- **Layers:**  
  - Input: 11 features  
  - Dense(16) → ReLU  
  - Dense(8) → ReLU  
  - Dense(1) → Sigmoid  
- **Compile:**  
  - Loss: `binary_crossentropy`  
  - Optimizer: `Adam` (lr=0.001)  
  - Metrics: `accuracy`  

---

## 📊 Results

- **Train Accuracy**: ~98%  
- **Test Accuracy**: ~81%

### ✅ Evaluation Includes:
- Confusion Matrix  
- ROC Curve  
- Loss & Accuracy vs. Epochs  

---

## 🔮 Future Improvements

- 🔧 Hyperparameter tuning using **GridSearch** or **RandomSearch**
- 🔁 Implement **K-fold cross-validation** to improve robustness
- 🌲 Try ensemble methods like **Random Forest** and **XGBoost**
- 🌐 Build a **REST API** using **Flask** or **FastAPI**
- 🖥️ Deploy an interactive UI using **Streamlit** for real-time predictions
