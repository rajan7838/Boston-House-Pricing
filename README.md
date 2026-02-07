Boston House Price Prediction

An end-to-end Machine Learning regression project to predict **Boston house prices** using trained models and a deployed **Streamlit web application**.

This project demonstrates the complete ML workflow:

- Data preprocessing  
- Feature scaling  
- Model training  
- Model serialization (`.pkl`)  
- Web app deployment using Streamlit Cloud  

---

## 🚀 Live Demo

🔗 **Deployed App:** *(Add your Streamlit Cloud link here)*  
Example: https://your-app-name.streamlit.app

---

## 📌 Project Overview

The goal of this project is to predict the **median value of owner-occupied homes** in Boston using 13 input features such as:

- Crime rate  
- Number of rooms  
- Property tax rate  
- Student-teacher ratio  
- Lower status population percentage  

---

## 🛠️ Tech Stack Used

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **Streamlit**
- **Pickle (Model Saving)**
- **Git & GitHub**

---

## 📂 Project Structure

Boston-House-Pricing/
│
├── app.py # Streamlit web application
├── lr.pkl # Trained Linear Regression model
├── scaler.pkl # StandardScaler used in preprocessing
├── requirements.txt # Required dependencies
├── boston.csv # Dataset
├── linear_regression_ml.ipynb # Training notebook
└── README.md # Project documentation


---

## ⚙️ Installation & Setup

Follow these steps to run the project locally:

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/rajan7838/Boston-House-Pricing.git
cd Boston-House-Pricing
2️⃣ Create Virtual Environment
python -m venv venv
Activate the environment:

Windows

venv\Scripts\activate
Linux/Mac

source venv/bin/activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Streamlit App
streamlit run app.py
The app will start at:

http://localhost:8501
📊 Model Details
Algorithm Used: Linear Regression

Feature Scaling: StandardScaler

Model Stored As: lr.pkl

Scaler Stored As: scaler.pkl

🎯 Input Features (All 13)
The model uses the full Boston Housing dataset features:

CRIM

ZN

INDUS

CHAS

NOX

RM

AGE

DIS

RAD

TAX

PTRATIO

B

LSTAT

🌐 Deployment
This project is deployed using Streamlit Cloud:

Steps:

Push code to GitHub

Connect repository to Streamlit Cloud

Deploy directly from app.py

📌 Future Improvements
Add RandomForest and XGBoost model selection

Improve UI with sliders and columns

Add Docker support for AWS deployment

Create CI/CD pipeline using GitHub Actions

🤝 Author
A Rajan
📌 GitHub: https://github.com/rajan7838
'''

