import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime, timedelta

from src.scraping import download_smard_energy_mix_prediction, fetch_past_and_predicted_weather

def get_by_estimations(df, last_date, col_name, count) -> pd.DataFrame: 
    last_24h = df[col_name].iloc[-24:]
    
    last_24h_mean = last_24h.mean()
    last_24h_std= last_24h.std()

    sampled = np.random.normal(last_24h_mean, last_24h_std, size=count) # assuming stationary distribution of the last 24h
    new_indices = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=count, freq="H")
    estimated_df = pd.DataFrame({col_name: sampled}, index=new_indices)
    
    print(f"{count} estimations added to {col_name} column.")
    
    return estimated_df

def get_by_copy(df, n):
    """
    Copies the last n rows of a DataFrame and appends them to the end,
    updating the datetime index accordingly.

    Parameters:
        df (pd.DataFrame): DataFrame with a DateTimeIndex.

    Returns:
        pd.DataFrame: Updated DataFrame with n additional rows.
    """
    if len(df) < n:
        raise ValueError("DataFrame must have at least n rows to extend.")

    last_n_rows = df.iloc[-n:].copy()
    
    # Shift the index by n hours
    last_n_rows.index += pd.DateOffset(hours=n)

    return pd.concat([df, last_n_rows])

def extend_by_predictions_and_samples(df, last_date, n = 24) -> pd.DataFrame: 
    
    price_df = df[["Price"]]
    copy_mix_df = df[["Hydro", "Pumped storage generation"]]
    pred_mix_df = df[["Solar","Wind offshore","Wind onshore"]]
    weather_df = df[["temperature_2m","precipitation","wind_speed_100m","direct_radiation"]]


    price_df = get_by_estimations(price_df, last_date, "Price", n)
    copy_mix_df = get_by_copy(copy_mix_df, n)
    pred_mix_df = download_smard_energy_mix_prediction(last_date + timedelta(hours=1), n)
    weather_df = get_by_copy(weather_df, n) # TODO: replace with scraper

    extended_merged_df = pd.concat([price_df, copy_mix_df, pred_mix_df, weather_df], axis=1, join='inner')
    return pd.concat([df, extended_merged_df])



