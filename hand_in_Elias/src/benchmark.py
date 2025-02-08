import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta

from .scraping import download_smard_energy_mix_prediction, fetch_forecast


def get_by_estimations(df, last_date, col_name, count) -> pd.DataFrame:
    last_weeks_val = df[col_name].iloc[-24 * 7 : -24 * 6]

    last_weeks_val_mean = last_weeks_val.mean()
    last_weeks_val_std = last_weeks_val.std()

    sampled = np.random.normal(
        last_weeks_val_mean, last_weeks_val_std, size=count
    )  # assuming stationary distribution of the last 24h
    new_indices = pd.date_range(
        start=last_date + pd.Timedelta(hours=1), periods=count, freq="H"
    )
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
    pred_mix_df = download_smard_energy_mix_prediction(
        last_date + timedelta(hours=1), n
    )

    weather_df = fetch_forecast(last_date, n)

    extended_merged_df = pd.concat(
        [price_df, copy_mix_df, pred_mix_df, weather_df], axis=1, join="inner"
    )

    return pd.concat([df, extended_merged_df])
