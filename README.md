# 📊 Sales Forecasting Project

## 🔍 Problem Statement
The goal of this project is to predict product sales based on various factors such as store type, discount, holiday, and order volume.

---

## 📂 Dataset
The dataset contains:
- Store Type
- Region Code
- Date
- Discount
- Orders
- Sales

---

## 📊 Exploratory Data Analysis (EDA)
- Sales show strong fluctuations over time
- Discounts lead to higher variability in sales
- Certain store types consistently outperform others
- Strong correlation between orders and sales

---

## 🧪 Hypothesis Testing
- Tested whether discounts significantly impact sales
- Found that discounted sales tend to be higher

---

## 🤖 Machine Learning Models
Trained multiple models:
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost

### 📈 Best Model:
XGBoost

---

## 📊 Model Performance
- MAE:  2,178.68 
- RMSE: 3,180.51 
- R² Score: 0.9701  

---

## 🌐 Deployment
- Built using Flask
- User inputs features → model predicts sales

Run locally:
```bash
python app.py
