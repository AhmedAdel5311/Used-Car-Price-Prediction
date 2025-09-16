# Used-Car-Price-Prediction
Predict the price of used cars

A machine learning project that predicts the estimated price of used cars based on their features such as brand, mileage, engine size, fuel type, and more.

This project uses XGBoost with preprocessing pipelines and hyperparameter tuning, and provides an interactive Streamlit web app for predictions.
## Project Structure
├── ML.py              # Training script (builds and saves the model)
├── app.py             # Streamlit app for predictions
├── used_car_price.csv # Dataset (not included in repo for size/privacy)
├── car_price_model.joblib # Saved trained model

## Features 
Data preprocessing with scikit-learn Pipelines & ColumnTransformer

Model training using XGBoost Regressor + GridSearchCV

Performance evaluation with RMSE, MAE, R²

Saved trained model for deployment

Streamlit app for user-friendly predictions

## How It Works
1.Training (ML.py)

Loads dataset (used_car_price.csv)

Splits into train/test sets

Applies preprocessing (imputation, encoding, scaling)

Performs hyperparameter tuning with GridSearchCV

Saves the best model as car_price_model.joblib

2.Prediction (app.py)

Loads the trained model

Accepts user inputs (car details)

Predicts and displays estimated price in USD
