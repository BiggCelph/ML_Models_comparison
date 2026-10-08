# Used Car Price Prediction

Predicting the market value of used cars for a fictional used-car service (Rusty Bargain) that wants an app that instantly estimates a car's value.

## Business goal
The service cares about three things: prediction quality, prediction speed, and training time. I compared five models on all three.

## Results
Validation set (60,368 cars), error measured as RMSE in price units:

| Model | RMSE | Training time |
|---|---|---|
| Linear Regression (baseline) | 2,692 | 3 s |
| Random Forest | 2,348 | 442 s |
| XGBoost | **1,565** | 145 s |
| LightGBM | 1,583 | **68 s** |
| CatBoost | 1,603 | 132 s |

XGBoost gave the most accurate predictions. LightGBM was nearly as accurate and trained about twice as fast, so I recommend it as the best balance for this app.

## Approach
- **Cleaning:** Dropped rows with missing values and cars priced at 100 or less, leaving 241k of 354k records.
- **Features:** Brand, model, vehicle type, gearbox, fuel type, repair status, power, mileage, registration year, postal code.
- **Preprocessing:** One-hot encoding for categories and standardization for numbers, fit on the training set only to avoid leakage.
- **Tuning:** Grid search with 5-fold cross-validation for each model. Random Forest used a smaller grid because the full search was too slow.

## Project structure
- `sp14.ipynb`: full analysis, from cleaning to model comparison

## Requirements
Python 3, pandas, numpy, scikit-learn, lightgbm, xgboost, catboost, matplotlib
