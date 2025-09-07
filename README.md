# 🫀 Heart Disease Prediction using Machine Learning

## 📌 Overview
This project predicts the likelihood of a person having heart disease based on their medical attributes. It uses a **Random Forest Classifier** to analyze patient health data and classify whether heart disease is present or not.

The goal is to demonstrate how data-driven models can assist in early detection of heart disease, which is one of the leading causes of death worldwide.

---

## ⚙️ Tech Stack
- **Programming Language**: Python
- **Libraries**: NumPy, Pandas, Scikit-learn, Matplotlib, Joblib
- **Model**: RandomForestClassifier
- **Tools**: Jupyter Notebook / VS Code / Google Colab

---

## 📂 Project Structure
heart-disease-ml/
│── data/
│   └── heart.csv
│── artifacts/
│   ├── model.joblib
│   └── scaler.joblib
│── train.py # Training and saving the ML model
│── predict.py # Making predictions
│── requirements.txt # Dependencies
│── README.md

---

## 🚀 Features
- Data preprocessing (handling missing values, normalization).
- **Random Forest Classifier** model for classification.
- Accuracy score: **~90%** on test data (or your actual score).
- Data visualization (correlation heatmap, bar plots).
- **Model and scaler saved** using Joblib for reusability.

---

## 📊 Dataset
The dataset used is the **Heart Disease UCI dataset** from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset). It contains patient attributes such as:
- Age, Sex, Chest pain type, Cholesterol, Resting BP, Max heart rate, etc.

---

## ▶️ How to Run
1.  Clone this repository:
    ```bash
    git clone [https://github.com/kavya-2404/heart-disease-prediction.git](https://github.com/kavya-2404/heart-disease-prediction.git)
    ```
2.  Navigate to the project folder:
    ```bash
    cd heart-disease-prediction
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the training file to create the model and scaler artifacts:
    ```bash
    python train.py
    ```
5.  Make predictions using the saved model and scaler:
    ```bash
    python predict.py
    ```

---

## 📈 Results
Achieved a test accuracy of **~90%** using the RandomForestClassifier.
Visualized feature correlations to understand important risk factors.

## 🔮 Future Improvements
- Explore advanced models (XGBoost, Neural Networks).
- Fine-tune hyperparameters to improve model performance.
- Deploy as a **Streamlit web app** for real-time predictions.
- Add more visualizations for interpretability.

---

## 👩‍💻 Author
Kavya Sri Maddula
