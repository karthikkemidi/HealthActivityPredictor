# 🏃‍♂️  Health Activity Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)

**AI-powered health analytics platform for personalized fitness predictions**

[**🚀 Live Demo**](https://healthactivitypredictor.streamlit.app) • [Report Issue](https://github.com/YOUR_USERNAME/healthactivitypredictor/issues)

</div>

---

## 📖 Overview

Health Activity Predictor is a machine learning web application that analyzes your health and fitness data to predict future outcomes. Using advanced XGBoost algorithms trained on 687,000+ health records from 3,000 participants, it provides personalized predictions for:

- **Fitness Growth** - How much your fitness level will improve tomorrow
- **Daily Steps** - Expected step count for the next day
- **Sleep Hours** - Predicted sleep duration based on your activity

The application features secure JWT authentication, interactive dashboards, and real-time predictions - making health tracking both insightful and actionable.

---
## 📁 Project Structure

healthactivitypredictor/
│
├── 📂 data/
│ └── health_fitness_dataset.csv # 687K health records, 3000 participants
│
├── 📂 models/
│ └── saved_models/
│ ├── xgboost_fitness_growth.pkl # Fitness prediction model (R²=0.9872)
│ ├── xgboost_daily_steps.pkl # Steps prediction model (R²=0.9845)
│ ├── xgboost_hours_sleep.pkl # Sleep prediction model (R²=0.9801)
│ └── feature_names.pkl # Feature list for model input
│
├── 📂 src/
│ ├── init.py # Makes src a Python package
│ ├── auth.py # JWT authentication & token management
│ ├── database.py # SQLite user database operations
│ └── model_training_xgboost.py # XGBoost model training pipeline
│
├── 📂 .streamlit/
│ ├── config.toml # Streamlit app configuration
│ └── secrets.toml # Secret keys (not in Git)
│
├── app_authenticated.py # Main Streamlit application
├── requirements.txt # Python dependencies
├── runtime.txt # Python version specification
├── .gitignore # Git ignore rules
├── README.md # Project documentation
├── LICENSE # MIT License
└── users.db # SQLite user database (auto-generated)


---
## ✨ Key Features

### 🔐 **Secure Authentication**
- JWT token-based login system
- Bcrypt password encryption
- User data isolation (each user sees only their own data)
- 100 pre-created demo accounts (user1-user100)

### 📊 **Personal Dashboard**
- Key health metrics overview (avg steps, sleep, fitness level)
- Recent activity summary
- BMI and vital statistics
- Account information

### 🤖 **AI-Powered Predictions**
- **Input**: Your current activity data (duration, calories, heart rate, etc.)
- **Output**: Predictions for tomorrow's fitness, steps, and sleep
- **Accuracy**: 98.7% R² score on test data
- **Speed**: Real-time predictions (<100ms)

### 📈 **Trend Analysis**
- Interactive Plotly charts
- Time-series visualization of:
  - Daily steps over time
  - Sleep patterns
  - Fitness progression
  - Heart rate trends
- Zoom, pan, and explore your data

### 👤 **Profile Management**
- View account details
- Update personal information
- Change password
- Track login history

---

## 🚀 Live Demo

**Visit:** [healthactivitypredictor.streamlit.app](https://healthactivitypredictor.streamlit.app)

### Demo Credentials

| Username | Password | Participant ID |
|----------|----------|----------------|
| user1 | health1 | 1 |
| user50 | health50 | 50 |
| user100 | health100 | 100 |

*All accounts from user1 to user100 are available*

---

## 💻 Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/healthactivitypredictor.git
cd healthactivitypredictor

# Install
pip install -r requirements.txt

# Run
streamlit run app_authenticated.py
```

## 🛠️ Built With
Python • Streamlit • XGBoost • Plotly

📊 Model Accuracy
R² Score: 98.7%

Dataset: 687K health records

Predictions: Fitness, Steps, Sleep

## 👤 Author
Karthik Kemidi, CBIT Hyderabad

