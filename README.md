# 🏍️ Bike Rental Sharing Prediction System
## Project Overview
This project focuses on predicting the number of bikes that will be
rented at a given time based on historical data. Bike sharing systems
generate a large amount of usage data every day. By analyzing this data,
we can understand the factors that influence bike rentals and build
predictive models.

The aim is to use machine learning techniques to estimate future demand
and help companies manage bike availability more effectively.

---
## Business Need (Problem Statement)
The business problem is to ensure a stable supply of rental bikes in urban cities by predicting the demand for bikes. By providing a stable supply of rental bikes, the system can enhance mobility comfort for the public and reduce waiting time, leading to greater customer satisfaction and accurately predicting bike demand can help bike sharing companies optimize operations including bike availability, pricing, strategies, and marketing efforts by considering demand Based on various external factors such as weather, season, holiday etc..,

---
## Objective
-  Analyze historical rental data to uncover demand patterns
- Identify key factors driving bike rental volume
- Train and compare multiple regression models
- Apply hyperparameter tuning to maximise accuracy
- Deploy the best model as an interactive Streamlit web application
  
---
## Dataset

The dataset includes information about bike rentals along with weather
and seasonal data.
| Feature | Description |
|---|---|
|`datetime` | Hourly timestamp |
| `season` | 1=Spring, 2=Summer, 3=Fall, 4=Winter |
| `holiday` | Whether the day is a public holiday |
| `workingday` | Whether the day is a working day |
| `weathersit` | Weather condition (1=Clear → 4=Heavy Rain) |
| `temp` | Normalised temperature |
| `humidity` | Relative humidity (%) |
| `windspeed` | Wind speed |
| `casual` | Rentals by unregistered users |
| `registered` | Rentals by registered users |

The **target variable** is the total number of bike rentals.

--- 

## Data Preprocessing

Before training the model, the dataset is prepared using the following
steps:

-   Handling missing values
-   Converting datetime into useful features (hour, day, month)
-   Encoding categorical variables
-   Checking feature distributions

---

## Exploratory Data Analysis

Key insights uncovered during EDA:

- **Peak hours** are 8 AM and 5–6 PM on working days (commute-driven demand)
- **Summer and Fall** see the highest rental volumes across seasons
- **Temperature** has a strong positive correlation with rentals
- **Heavy rain and storms** significantly reduce demand
- Registered users dominate weekday rentals; casual users peak on weekends

---

## Feature Engineering

New features extracted from the `datetime` column:

- `hour` — captures time-of-day demand patterns
- `day_of_week` — differentiates weekday vs weekend behaviour
- `month` — captures seasonal trends
- `is_weekend` — binary flag for weekend days
- `workingday` — derived from holiday and weekend flags

---

## Models & Results

### Before Hyperparameter Tuning

| Model | RMSE (Train) | RMSE (Test) | R² Train | R² Test |
|---|---|---|---|---|
| Decision Tree Regressor | 43.325 | 46.773 | 89.17% | 86.77% |
| Random Forest Regressor | 27.737 | 36.407 | 95.56% | 91.98% |
| Gradient Boosting Regressor | 35.245 | 46.773 | 92.83% | 91.56% |

### After Hyperparameter Tuning (RandomSearchCV)

| Model | MAE (Test) | RMSE (Test) | R² Train | R² Test |
|---|---|---|---|---|
| Decision Tree Regressor | 26.237 | 41.808 | 93.63% | 89.43% |
| Random Forest Regressor | 26.469 | 39.535 | 98.65% | 90.55% |
| **Gradient Boosting Regressor** | **22.321** | **33.805** | **94.75%** | **93.09%** |

**Gradient Boosting Regressor** was selected as the final deployed model — achieving the lowest Test RMSE of **33.81** and the highest Test R² of **93.09%** after tuning.

---

##  Streamlit Web App

The trained Gradient Boosting model is deployed via **Streamlit**, allowing users to interactively predict bike demand.

**To run locally:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements

-   Deep learning models (BERT / Transformers)
-   Resume ranking system
-   eal‑time demand prediction
-   Integration with weather APIs
-   Cloud deployment
---

## Conclusion

This project demonstrates how machine learning can be used to analyze
historical bike-sharing data and predict future rental demand. Such
predictions can help companies optimize bike availability, reduce
shortages, and improve overall service efficiency.

---
