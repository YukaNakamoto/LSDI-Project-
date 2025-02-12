import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.scraping import download_smard_energy_mix_prediction, fetch_forecast


def get_estimations(df, last_date, col_name, count = None, final_date=None) -> pd.DataFrame: 
    last_24h_from_last_week = df[col_name].iloc[-24 * 7: -24 * 6]
    
    last_24h_from_last_week_mean = last_24h_from_last_week.mean()
    last_24h_from_last_week_std= last_24h_from_last_week.std()

    if final_date:
        count = int((final_date - last_date).total_seconds() / 3600)
    elif count:
        count = count
    else:
        count = 48

    sampled = np.random.normal(last_24h_from_last_week_mean, last_24h_from_last_week_std, size=count) # assuming stationary distribution of the last 24h
    new_indices = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=count, freq="H")
    estimated_df = pd.DataFrame({col_name: sampled}, index=new_indices)
    
    return estimated_df


def get_by_copy(df, last_date, n):
    """
    Copies the last n rows of a DataFrame and appends them to the end,
    ensuring the new index continues from last_date.

    Parameters:
        df (pd.DataFrame): DataFrame with a DateTimeIndex.
        last_date (pd.Timestamp): The last known timestamp in the dataset.
        n (int): Number of hours to extend.

    Returns:
        pd.DataFrame: Updated DataFrame with n additional rows, correctly indexed.
    """
    if len(df) < n:
        raise ValueError("DataFrame must have at least n rows to extend.")

    last_n_rows = df.iloc[-n:].copy()

    # Generate new timestamps starting from last_date + 1 hour
    new_index = pd.date_range(
        start=last_date + pd.Timedelta(hours=1), periods=n, freq="H"
    )

    # Ensure new timestamps match expected future values
    last_n_rows.index = new_index

    return last_n_rows  # Return only the extended rows


def extend_by_predictions_and_samples(df, last_date, n=24) -> pd.DataFrame:

    price_df = df[["Price"]]
    copy_mix_df = df[["Hydro", "Pumped storage generation"]]
    pred_mix_df = df[["Solar", "Wind offshore", "Wind onshore"]]
    weather_df = df[
        ["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"]
    ]

    price_df = get_by_estimations(price_df, last_date, "Price", n)

    copy_mix_df = get_by_copy(copy_mix_df, last_date, n)

    pred_mix_df = download_smard_energy_mix_prediction(last_date, n, pred_mix_df)

    weather_df = fetch_forecast(last_date, n)

    extended_merged_df = pd.concat(
        [price_df, copy_mix_df, pred_mix_df, weather_df], axis=1, join="inner"
    )

    return pd.concat([df, extended_merged_df])


def print_date_ranges(
    X_train_prophet, X_train, y_train, y_actual, prophet_X_predict, X_predict
):
    """
    Prints the date ranges of various datasets in a structured table format.

    Parameters:
    - X_train_prophet: DataFrame with 'ds' column containing timestamps.
    - X_train: DataFrame or Series with a DateTimeIndex.
    - y_train: DataFrame or Series with a DateTimeIndex.
    - y_actual: DataFrame or Series with a DateTimeIndex.
    - prophet_X_predict: DataFrame with 'ds' column containing timestamps.
    - X_predict: DataFrame or Series with a DateTimeIndex.
    """

    data = [
        [
            "X_train_prophet",
            str(X_train_prophet["ds"].iloc[0]),
            str(X_train_prophet["ds"].iloc[-1]),
        ],
        ["X_train", str(X_train.index[0]), str(X_train.index[-1])],
        ["y_train", str(y_train.index[0]), str(y_train.index[-1])],
        ["y_actual", str(y_actual.index[0]), str(y_actual.index[-1])],
        [
            "prophet_X_predict",
            str(prophet_X_predict["ds"].iloc[0]),
            str(prophet_X_predict["ds"].iloc[-1]),
        ],
        ["X_predict", str(X_predict.index[0]), str(X_predict.index[-1])],
    ]

    # Print in table format with correct alignment
    print(f"{'Dataset':<20} {'Start Date':<22} {'End Date'}")
    print("=" * 70)
    for row in data:
        print(f"{row[0]:<20} {row[1]:<22} {row[2]}")
