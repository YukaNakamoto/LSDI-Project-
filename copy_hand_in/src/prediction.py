import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb


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
    # Make copies to avoid modifying the original data
    X_train_copy = X_train.copy()
    y_train_copy = y_train.copy()
    X_test_copy = X_test.copy()
    y_test_copy = y_test.copy()

    # Handle missing values
    merged_train = pd.concat([X_train_copy, y_train_copy], axis=1)
    merged_train_cleaned = merged_train.dropna()

    X_train_cleaned = merged_train_cleaned.iloc[:, :-1]
    y_train_cleaned = merged_train_cleaned.iloc[:, -1]

    merged_test = pd.concat([X_test_copy, y_test_copy], axis=1)
    merged_test_cleaned = merged_test.dropna()

    X_test_cleaned = merged_test_cleaned.iloc[:, :-1]
    y_test_cleaned = merged_test_cleaned.iloc[:, -1]

    # Save index for plotting
    test_index = X_test_cleaned.index

    # Train model
    model = LinearRegression()
    model.fit(X_train_cleaned, y_train_cleaned)

    # Predict
    predictions_scaled = model.predict(X_test_cleaned)

    # Initialize unscaled variables
    y_test_unscaled = y_test_cleaned
    predictions_unscaled = predictions_scaled

    # Inverse-transform the scaled data to original scale if needed
    if scalar:
        y_test_unscaled = scalar.inverse_transform(y_test_cleaned.values.reshape(-1, 1)).ravel()
        predictions_unscaled = scalar.inverse_transform(predictions_scaled.reshape(-1, 1)).ravel()

    return predictions_unscaled


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

    