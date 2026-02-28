# 🏍️ Bike Rental Sharing Prediction System

A machine learning–based application that predicts bike rental demand using user-provided inputs such as weather conditions and calendar features.  
The system compares multiple regression models and provides instant predictions through a Streamlit web interface.

---
## 🎯 Business Need (Problem Statement)
The business problem is to ensure a stable supply of rental bikes in urban cities by predicting the demand for bikes. By providing a stable supply of rental bikes, the system can enhance mobility comfort for the public and reduce waiting time, leading to greater customer satisfaction and accurately predicting bike demand can help bike sharing companies optimize operations including bike availability, pricing, strategies, and marketing efforts by considering demand Based on various external factors such as weather, season, holiday etc..,

---
## 📌 Objective
To analyze historical bike rental data and build a predictive model to estimate rental demand based on environmental and seasonal factors.

---

## 🧠 Machine Learning Models

The following regression models are implemented and compare 
- Decision Tree Regressor  
- Random Forest Regressor  
- Gradient Boosting Regressor
  
## 📊 Dataset Information

- Dataset: Bike Sharing Dataset

#### Key Features:

- Season
- Weather situation
- Temperature
- Humidity
- Wind speed
- Working day
- Holiday

#### Target Variable:
- Total number of bike rentals(cnt)

## 🛠️Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib, Seaborn
- Joblib

## 🔎 Approach
- Understood the business problem of demand fluctuation in bike-sharing systems.
- Cleaned and prepared raw rental data to ensure accuracy and reliability.
- Performed detailed EDA to identify key drivers such as temperature, season, holidays, and        working days.
- Applied feature engineering to enhance predictive performance.
- Built and compared multiple regression models (Linear Regression, Random Forest).
- Selected the best model using RMSE and R² evaluation metrics.
I- nterpreted model outputs to explain which factors most influenced rental demand.

## ✅ Conclusion 
- This project showcases my ability to translate a real-world operational problem into a structured data science solution. I demonstrated end-to-end workflow capability from data preprocessing to model evaluation and applied predictive modeling to generate actionable insights for demand forecasting.
