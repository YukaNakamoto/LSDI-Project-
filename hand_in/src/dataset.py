import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
import pytz
import requests

def get_by_estimations(df, prediction_date, col_name, count = None):
    last_24h = df[col_name].iloc[-24:]
    last_date = df.index[-1]

    time_diff = prediction_date - last_date
    hours_diff = count if count else int(time_diff.total_seconds() / 3600) 
    
    if hours_diff <= 0:
         return df
    
    last_24h_mean = last_24h.mean()
    last_24h_std= last_24h.std()
    sampled = np.random.normal(last_24h_mean, last_24h_std, size=hours_diff) # assuming stationary distribution of the last 24h

    new_indices = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=hours_diff, freq="H")

    estimated_df = pd.DataFrame({col_name: sampled}, index=new_indices)
    
    print(f"{hours_diff} estimations added to {col_name} column.")
    
    return estimated_df

def get_e_price_df(prediction_date):
    e_price_df = pd.read_csv('../data/day_ahead_energy_prices.csv', delimiter=",")

    # new_e_prices = get_new_e_prices() // TODO:
    # e_price_df = e_price_df.stack(new_e_prices) // TODO:

    e_price_df = e_price_df.set_index('Datetime')
    e_price_df.index = pd.to_datetime(e_price_df.index)
    e_price_df.rename(columns={"hourly day-ahead energy price": "Price"}, inplace=True)

    e_price_estimated_df = get_by_estimations(e_price_df.copy(), prediction_date, "Price")
    e_price_df = pd.concat([e_price_df, e_price_estimated_df])

    return e_price_df

def get_mix_df(prediction_date):
    mix_df = pd.read_csv('../data/hourly_market_mix_cleaned.csv', usecols=["Timestamp", "Hydro","Pumped storage generation","Solar","Wind offshore","Wind onshore"], delimiter=",")
    mix_df.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    # new_mix_df = get_new_mix_df() // TODO:
    # e_price_df = e_price_df.stack(new_mix_df) // TODO:

    mix_df.set_index("Datetime", inplace=True)
    mix_df.index = pd.to_datetime(mix_df.index)
    mix_df = mix_df.loc[:, mix_df.max() >= 0.01]

    pred_mix_df = download_smard_energy_mix_prediction(mix_df.index[-1])
    pred_mix_df = pred_mix_df[mix_df.index[-1] + pd.Timedelta(hours=1):]

    n_predictions = len(pred_mix_df)


    hydro_estimated_df = get_by_estimations(mix_df.copy(), prediction_date, "Hydro", n_predictions)
    pumped_df = get_by_estimations(mix_df.copy(), prediction_date, "Pumped storage generation", n_predictions)

    extended_mix_df = pd.concat([hydro_estimated_df, pumped_df, pred_mix_df], axis=1)
    mix_df = pd.concat([mix_df, extended_mix_df], axis=0)

    print(mix_df)

    return mix_df

def get_datasets(prediction_date) -> pd.DataFrame:
    prediction_date = datetime.combine(prediction_date, datetime.min.time()).replace(hour=23)
    dfs_to_merge = []

    e_price_df = get_e_price_df(prediction_date)
    dfs_to_merge.append(e_price_df)

    mix_df = get_mix_df(prediction_date)
    dfs_to_merge.append(mix_df)


    avg_weather_df = pd.read_csv('../data/germany_weather_average.csv', delimiter=",")

    # new_avg_weather_df = get_new_avg_weather_df() // TODO:
    # avg_weather_df = avg_weather_df.stack(new_avg_weather_df) // TODO:

    avg_weather_df.set_index('date', inplace=True)
    avg_weather_df.index = pd.to_datetime(avg_weather_df.index)
    avg_weather_df.index = avg_weather_df.index.tz_localize(None)
    avg_weather_df.rename(columns={
        'Precipitation (rain/snow)': 'precipitation',
        'wind_speed_100m': 'wind_speed_100m',
        'Global Horizontal Irradiance': 'shortwave_radiation',
        'direct_radiation': 'direct_radiation', 
        }, inplace=True)

    dfs_to_merge.append(avg_weather_df)


    merged_df = pd.concat(dfs_to_merge, axis=1, join='inner')
    
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

def download_smard_energy_mix_prediction(target_date: str):
    print("Fetching predicted energy mix")
    local_timezone = pytz.timezone("Europe/Berlin")
    target_date = target_date.strftime('%Y-%m-%d')
    date_string = f"{target_date} 00:00:00"
    local_date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    localized_date_object = local_timezone.localize(local_date_object)
    epoch_timestamp = int(local_date_object.timestamp())

    epoch_timestamp = int(local_date_object.timestamp())
    # Calculate the offset based on the weekday (0 = Monday, 6 = Sunday)
    weekday = localized_date_object.weekday() # Monday = 0, Sunday = 6

    hour_offset = weekday * 24

    timestamp_in_milliseconds = epoch_timestamp * 1000 - (hour_offset * 3600 * 1000)
    # Fetch data from the SMARD.de API (returns 168 hours of data starting from the time
    pred_wind_offs_url = f"https://www.smard.de/app/chart_data/3791/DE/3791_DE_hour_{timestamp_in_milliseconds}.json" # w
    pred_wind_ons_url = f"https://www.smard.de/app/chart_data/123/DE/123_DE_hour_{timestamp_in_milliseconds}.json"
    pred_solar_url = f"https://www.smard.de/app/chart_data/125/DE/125_DE_hour_{timestamp_in_milliseconds}.json"
    
    pred_wind_offs_response = requests.get(pred_wind_offs_url)
    pred_wind_ons_response = requests.get(pred_wind_ons_url)
    pred_solar_response = requests.get(pred_solar_url)

    responses = [pred_wind_offs_response, pred_wind_ons_response,pred_solar_response]

    dfs = []
    
    for r, name in zip(responses, ("Wind onshore", "Wind offshore","Solar")):
        if r.status_code != 200:
            print(f"Failed to fetch {name} data: {r.status_code}")
    
        data = r.json()
    
        if "series" not in data or not data["series"]:
            print(f"No {name} data available for the specified date.")
    
        series = data["series"]
        # Extract the 48-hour data for the given date
        start_index = hour_offset
        end_index = hour_offset + 48
        day_series = series[start_index:end_index]
        dts = [datetime.utcfromtimestamp(dt[0] / 1000).strftime('%Y-%m-%d %H:%M:%S') for dt in day_series]
        observed_output = [item[1] for item in day_series]

        df = pd.DataFrame({
            "Datetime": dts,
            name: observed_output
        })

        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")
        df.set_index("Datetime", inplace=True)

        dfs.append(df)
    df_merged = pd.concat(dfs, axis=1)
    return df_merged