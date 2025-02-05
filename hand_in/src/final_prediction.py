import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import pytz
import requests

dir = "./data/"
prediction_date = datetime(2025, 2, 18, 23, 0, 0)

def fill_up_e_prices():

    path = dir + 'day_ahead_energy_prices.csv'
    e_price_df = pd.read_csv(path, delimiter=",")
    e_price_df = e_price_df.set_index('Datetime')
    e_price_df.index = pd.to_datetime(e_price_df.index)
    
    
    estimations_df = get_estimations(e_price_df.copy(), e_price_df.index[-1], col_name="hourly day-ahead energy price", count=None, final_date=prediction_date)
    e_price_df = pd.concat([e_price_df, estimations_df])
    
    e_price_df.index = e_price_df.index.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Save to CSV with the modified index
    e_price_df.to_csv(path, index_label="Datetime")

def get_estimations(df, last_date, col_name, count = None, final_date=None) -> pd.DataFrame: 
    last_24h = df[col_name].iloc[-24:]
    
    last_24h_mean = last_24h.mean()
    last_24h_std= last_24h.std()

    if final_date:
        count = int((final_date - last_date).total_seconds() / 3600)
    elif count:
        count = count
    else:
        count = 48

    sampled = np.random.normal(last_24h_mean, last_24h_std, size=count) # assuming stationary distribution of the last 24h
    new_indices = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=count, freq="H")
    estimated_df = pd.DataFrame({col_name: sampled}, index=new_indices)
    
    return estimated_df


def fill_up_energy_mix():

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

    mix_df.set_index("Timestamp", inplace=True)
    mix_df.index = pd.to_datetime(mix_df.index)


    copy_mix_df = mix_df[["Hydro", "Pumped storage generation"]]
    pred_mix_df = mix_df[["Solar","Wind offshore","Wind onshore"]]


    last_date = mix_df.index[-1]
    n = int((prediction_date - last_date).total_seconds() / 3600)
    
    copy_mix_df = get_by_copy(copy_mix_df, last_date, n)
    pred_mix_df = download_smard_energy_mix_prediction(last_date + timedelta(hours=1), n)

    estimations_df = pd.concat([copy_mix_df, pred_mix_df], axis=1, join='inner')

    
    mix_extended_df = pd.concat([mix_df, estimations_df])
    mix_extended_df.index = mix_extended_df.index.strftime('%Y-%m-%dT%H:%M:%S')
    mix_extended_df.to_csv(dir +'hourly_market_mix_cleaned.csv', index_label="Timestamp")


def get_by_copy(df, last_date, n):
    """
    Copies the last n rows of a DataFrame one week ago and appends them to the end,
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
    idx_one_week_ago = pd.date_range(start=last_date + pd.Timedelta(hours=1) - pd.Timedelta(weeks=1), periods=n, freq='H')
    values_one_week_ago = df.copy().reindex(idx_one_week_ago)
    
    print(values_one_week_ago)

    # Generate new timestamps starting from last_date + 1 hour
    new_index = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=n, freq='H')

    # Ensure new timestamps match expected future values
    values_one_week_ago.index = new_index
    return values_one_week_ago


def preprocess_smard_energy_mix_prediction_dates(start_date: datetime, n=None, end_date=None):
    local_timezone = pytz.timezone("Europe/Berlin")
    start_date_str = start_date.strftime('%Y-%m-%d')
    date_string = f"{start_date_str} 00:00:00"
    local_date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    localized_date_object = local_timezone.localize(local_date_object)
    epoch_timestamp = int(local_date_object.timestamp())

    weekday = localized_date_object.weekday()  # Monday = 0, Sunday = 6
    hour_offset = weekday * 24
    timestamp_in_milliseconds = epoch_timestamp * 1000 - (hour_offset * 3600 * 1000)
    if end_date:
        delta_pred_values = (end_date - local_date_object).days * 24 + 1  # Adjusted to include the end hour
    elif n:
        delta_pred_values = n + 1 # Adjusted to include the end hour
    else:
        raise ValueError("Either n or end_date must be provided")

    return timestamp_in_milliseconds, hour_offset, delta_pred_values


def fetch_smard_energy_mix_prediction_data(timestamp_in_milliseconds):
    urls = {
        "Wind onshore": f"https://www.smard.de/app/chart_data/3791/DE/3791_DE_hour_{timestamp_in_milliseconds}.json",
        "Wind offshore": f"https://www.smard.de/app/chart_data/123/DE/123_DE_hour_{timestamp_in_milliseconds}.json",
        "Solar": f"https://www.smard.de/app/chart_data/125/DE/125_DE_hour_{timestamp_in_milliseconds}.json"
    }

    responses = {}
    for name, url in urls.items():
        response = requests.get(url)
        if response.status_code == 200:
            responses[name] = response.json()
        else:
            print(f"Failed to fetch {name} data: {response.status_code}")

    return responses


def postprocess_smard_energy_mix_prediction_data(responses, hour_offset, delta_pred_values):
    dfs = []
    for name, data in responses.items():
        if "series" not in data or not data["series"]:
            print(f"No {name} data available for the specified date.")
            continue

        series = data["series"]
        start_index = hour_offset + 1
        end_index = hour_offset + delta_pred_values
        day_series = series[start_index:end_index]
        dts = [datetime.fromtimestamp(dt[0] / 1000, tz=pytz.utc).astimezone(pytz.timezone("Europe/Berlin")).strftime('%Y-%m-%d %H:%M:%S') for dt in day_series]
        
        observed_output = []

        for ts, value in day_series:
            date_time_ts = datetime.fromtimestamp(ts / 1000, tz=pytz.utc).astimezone(pytz.timezone("Europe/Berlin")).strftime('%Y-%m-%d %H:%M:%S')
            if value is None:
                print("None value detected for ts: ", date_time_ts)
                observed_output.append(None)
            else:
                val = value / 1000
                observed_output.append(val)

        df = pd.DataFrame({
            "Datetime": dts,
            name: observed_output
        })

        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")
        df.set_index("Datetime", inplace=True)

        df.dropna(inplace=True)
        additional_rows = get_by_copy(df, df.index[-1], 24)
        df = pd.concat([df, additional_rows])
        dfs.append(df)

    df_merged = pd.concat(dfs, axis=1)

    return df_merged


def download_smard_energy_mix_prediction(start_date: datetime, n, end_date=None):
    timestamp_in_milliseconds, hour_offset, delta_pred_values = preprocess_smard_energy_mix_prediction_dates(start_date, n, end_date)
    responses = fetch_smard_energy_mix_prediction_data(timestamp_in_milliseconds)
    df_merged = postprocess_smard_energy_mix_prediction_data(responses, hour_offset, delta_pred_values)
    return df_merged
