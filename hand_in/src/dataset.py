import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

dir = "./data/"

def get_by_estimations(df, prediction_date_start, col_name, count = None) -> pd.DataFrame: 
    last_24h = df[col_name].iloc[-24:]
    last_date = df.index[-1]

    time_diff = prediction_date_start - last_date
    hours_diff = count if count else int(time_diff.total_seconds() / 3600) 

    if hours_diff <= 0:
         return None
    
    last_24h_mean = last_24h.mean()
    last_24h_std= last_24h.std()

    sampled = np.random.normal(last_24h_mean, last_24h_std, size=hours_diff) # assuming stationary distribution of the last 24h
    new_indices = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=hours_diff, freq="H")
    estimated_df = pd.DataFrame({col_name: sampled}, index=new_indices)
    
    print(f"{hours_diff} estimations added to {col_name} column.")
    
    return estimated_df

def get_e_price_df() -> pd.DataFrame:
    e_price_df = pd.read_csv(dir +'day_ahead_energy_prices.csv', delimiter=",")

    e_price_df = e_price_df.set_index('Datetime')
    e_price_df.index = pd.to_datetime(e_price_df.index)
    e_price_df.rename(columns={"hourly day-ahead energy price": "Price"}, inplace=True)

    return e_price_df

def get_mix_df() -> pd.DataFrame:
    mix_df = pd.read_csv(dir +'hourly_market_mix_cleaned.csv', usecols=["Timestamp", "Biomass",
        "Hard Coal",
        "Hydro",
        "Lignite",
        "Natural Gas",
        "Nuclear",
        "Other",
        "Pumped storage generation",
        "Solar",
        "Wind offshore",
        "Wind onshore"], delimiter=",")
    mix_df.rename(columns={"Timestamp": "Datetime"}, inplace=True)

    mix_df.set_index("Datetime", inplace=True)
    mix_df.index = pd.to_datetime(mix_df.index)

    return mix_df

def get_avg_weather_data() -> pd.DataFrame:   
    avg_weather_df = pd.read_csv(dir +'germany_weather_average.csv', delimiter=",")

    avg_weather_df.set_index('date', inplace=True)
    avg_weather_df.index = pd.to_datetime(avg_weather_df.index)
    avg_weather_df.index = avg_weather_df.index.tz_localize(None)
    avg_weather_df.rename(columns={
        'Precipitation (rain/snow)': 'precipitation',
        'wind_speed_100m': 'wind_speed_100m',
        'Global Horizontal Irradiance': 'shortwave_radiation',
        'direct_radiation': 'direct_radiation', 
        }, inplace=True)

    return avg_weather_df

def get_datasets() -> pd.DataFrame: # 12:00 of every day -> add 1.5 days of values to e.g. 17.02 12:00 -> 19.02 00:00
    dfs_to_merge = []

    e_price_df = get_e_price_df()
    dfs_to_merge.append(e_price_df)

    mix_df = get_mix_df()
    dfs_to_merge.append(mix_df)

    avg_weather_df = get_avg_weather_data()
    dfs_to_merge.append(avg_weather_df)

    dfs_to_merge = [df[~df.index.duplicated(keep='first')]for df in dfs_to_merge]

    merged_df = pd.concat(dfs_to_merge, axis=1, join='inner').dropna(axis=0)
    return merged_df, e_price_df.copy()

def normalize_data(merged_df, e_price_df, interval_df):
    merged_scaler = MinMaxScaler(feature_range=(0, 1))
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    merged_interval_scaler = MinMaxScaler(feature_range=(0, 1))
    
    z_merged_df = pd.DataFrame(
        merged_scaler.fit_transform(merged_df.copy()),
        columns=merged_df.columns,
        index=merged_df.index,
    )

    z_price_df = pd.DataFrame(
        price_scaler.fit_transform(e_price_df.copy()),
        columns=e_price_df.columns,
        index=e_price_df.index,
    )

    z_merged_interval_df = pd.DataFrame(
        merged_interval_scaler.fit_transform(interval_df.copy()),
        columns=interval_df.columns,
        index=interval_df.index,
    )

    return z_merged_df, z_price_df, z_merged_interval_df, price_scaler

def unstandardized_actual_price(z_y_test, z_y_pred_e, actual_price_scaler):
        y_test = actual_price_scaler.inverse_transform([z_y_test]).flatten()
        y_predicted = actual_price_scaler.inverse_transform([z_y_pred_e]).flatten()

        return y_test, y_predicted

