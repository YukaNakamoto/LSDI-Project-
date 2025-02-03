import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from prophet import Prophet

def init_prophet_model(FEATURES):

    # Initialize Prophet model
    model = Prophet(
        changepoint_prior_scale = 0.5,
        seasonality_prior_scale = 1.0,
        holidays_prior_scale = 1.0,
        seasonality_mode= 'multiplicative',
        
        
    )

    #Add fourier order for weekly and monthly seasonality
    model.add_seasonality(name='weekly', period=7, fourier_order=20)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=20)
    model.add_seasonality(name='yearly', period=365, fourier_order=20)

    # Add time-based regressors

    for feature in FEATURES:
        model.add_regressor(feature)

    return model


def predict_mse(best_params, X_train, y_train, X_val, y_val, X_test):
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)
    y_predicted = model.predict(X_test)

    return y_predicted, model

def predict_mae(best_params, X_train, y_train, X_val, y_val, X_test):
    best_params.update({"objective": "reg:squarederror", "eval_metric": "mae"})

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)
    y_predicted = model.predict(X_test)

    return y_predicted, model


def linear_regression(X_train, y_train, X_test, y_test, scalar):
    merged = pd.concat([X_train, y_train], axis=1)
    merged_cleaned = merged.dropna()

    X_train_cleaned = merged_cleaned.iloc[:, :-1]
    y_train_cleaned = merged_cleaned.iloc[:, -1]

    merged = pd.concat([X_test, y_test], axis=1)
    merged_cleaned = merged.dropna()

    X_test_cleaned = merged_cleaned.iloc[:, :-1]
    y_test_cleaned = merged_cleaned.iloc[:, -1]


    #Fit the model
    model = LinearRegression()
    model.fit(X_train_cleaned, y_train_cleaned)

    predictions = model.predict(X_test_cleaned)

    return predictions


def calculate_error_metrics(y_true, y_preds):
    results = []
    models = ["Linear Regression", "Prophet", "XGBoost"]

    for model_name, y_pred in zip(models, y_preds):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        results.append({
            "Model": model_name,
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "MAPE": mape
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    