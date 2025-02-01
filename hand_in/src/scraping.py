import pandas as pd
from datetime import datetime, timedelta
import pytz
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import time

def preprocess_dates(start_date: datetime, n=None, end_date=None):
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
        delta_pred_values = (end_date - local_date_object).days * 24 # TODO maybe +1
    elif n:
        delta_pred_values = n
    else:
        raise ValueError("Either n or end_date must be provided")

    return timestamp_in_milliseconds, hour_offset, delta_pred_values


def fetch_smard_data(timestamp_in_milliseconds):
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


def postprocess_data(responses, hour_offset, delta_pred_values):
    dfs = []
    for name, data in responses.items():
        if "series" not in data or not data["series"]:
            print(f"No {name} data available for the specified date.")
            continue

        series = data["series"]
        start_index = hour_offset
        end_index = hour_offset + delta_pred_values
        day_series = series[start_index:end_index]
        dts = [datetime.fromtimestamp(dt[0] / 1000, tz=pytz.utc).astimezone(pytz.timezone("Europe/Berlin")).strftime('%Y-%m-%d %H:%M:%S') for dt in day_series]

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


def download_smard_energy_mix_prediction(start_date: datetime, n, end_date=None):
    print("Fetching predicted energy mix")
    timestamp_in_milliseconds, hour_offset, delta_pred_values = preprocess_dates(start_date, n, end_date)
    responses = fetch_smard_data(timestamp_in_milliseconds)
    df_merged = postprocess_data(responses, hour_offset, delta_pred_values)
    return df_merged


def fetch_past_and_predicted_weather(end_date: datetime, csv_file): # scrapes historical data + 48h of predictions
    directory = '../data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    start_date = "2018-01-01"
    end_date = (end_date - timedelta(days=2)).strftime("%Y-%m-%d")
    # print(start_date, end_date)
    if os.path.exists(csv_file):
        existing_data = pd.read_csv(csv_file, parse_dates=["date"], index_col="date")
        last_entry_date = existing_data.index.max().replace(tzinfo=None)
        required_last_date = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=2)
        
        if last_entry_date >= required_last_date:
            print("Data already up to date. Skipping scraping.")
            return

    # Setup the Open-Meteo API client with caching and retries
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # Define the base URL for the weather API
    historical_url = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    # Define the list of representative coordinates for Germany
    coordinates = [
        {"latitude": 52.5200, "longitude": 13.4050},  # Berlin
        {"latitude": 48.1351, "longitude": 11.5820},  # Munich
        {"latitude": 50.1109, "longitude": 8.6821},   # Frankfurt
        {"latitude": 51.1657, "longitude": 10.4515},  # Central Germany (approximate)
        {"latitude": 53.5511, "longitude": 9.9937},   # Hamburg
        {"latitude": 51.2277, "longitude": 6.7735},   # Düsseldorf
        {"latitude": 51.0504, "longitude": 13.7373},  # Dresden
        {"latitude": 50.9375, "longitude": 6.9603},   # Cologne
        {"latitude": 49.4875, "longitude": 8.4660},   # Mannheim
        {"latitude": 48.7758, "longitude": 9.1829},   # Stuttgart
        {"latitude": 51.3397, "longitude": 12.3731},  # Leipzig
        {"latitude": 50.0782, "longitude": 8.2398},   # Wiesbaden
        {"latitude": 49.0069, "longitude": 8.4037},   # Karlsruhe
        {"latitude": 51.5128, "longitude": 7.4633},   # Dortmund
        {"latitude": 50.1211, "longitude": 8.4965},   # Offenbach
        {"latitude": 50.3569, "longitude": 7.5886},   # Koblenz
        {"latitude": 50.7753, "longitude": 6.0839},   # Aachen
        {"latitude": 49.4521, "longitude": 11.0767},  # Nuremberg
        {"latitude": 52.3759, "longitude": 9.7320},   # Hanover
        {"latitude": 51.4818, "longitude": 7.2162},   # Bochum
        {"latitude": 51.4556, "longitude": 7.0116},   # Essen
        {"latitude": 51.4344, "longitude": 6.7623},   # Duisburg
        {"latitude": 51.9607, "longitude": 7.6261},   # Münster
    ]

    # Convert end_date to a datetime object
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Calculate forecast start and end dates
    forecast_start_date_dt = end_date_dt + timedelta(days=1)
    forecast_end_date_dt = end_date_dt + timedelta(days=2)

    # Convert back to string format
    forecast_start_date = forecast_start_date_dt.strftime("%Y-%m-%d")
    forecast_end_date = forecast_end_date_dt.strftime("%Y-%m-%d")

    # Define the weather variables and date range for historical data
    params_template = {
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_100m",
            "direct_radiation"
        ]
    }

    # Store data for all locations
    all_data = []

    for coord in coordinates:
        params = params_template.copy()
        params.update({
            "latitude": coord["latitude"],
            "longitude": coord["longitude"],
        })

        while True:
            try:
                # Fetch historical weather data for the current location
                responses = openmeteo.weather_api(historical_url, params=params)
                response = responses[0]

                # Extract hourly data for this location
                hourly = response.Hourly()
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    )
                }
                hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
                hourly_data["precipitation"] = hourly.Variables(1).ValuesAsNumpy()
                hourly_data["wind_speed_100m"] = hourly.Variables(2).ValuesAsNumpy()
                hourly_data["direct_radiation"] = hourly.Variables(3).ValuesAsNumpy()

                # Convert to DataFrame and append to the list
                hourly_dataframe = pd.DataFrame(data=hourly_data)
                all_data.append(hourly_dataframe)
                break  # Exit the loop if data is fetched successfully

            except Exception as e:
                print(f"Error fetching historical data for coordinates {coord}: {e}")
                if "Minutely API request limit exceeded" in str(e):
                    print("Waiting for one minute before retrying...")
                    time.sleep(60)  # Wait for one minute before retrying
                else:
                    break  # Exit the loop if the error is not related to the request limit

    # Fetch forecast data for the next 2 days after the end_date
    for coord in coordinates:
        params = {
            "latitude": coord["latitude"],
            "longitude": coord["longitude"],
            "hourly": ["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"],
            "start_date": forecast_start_date,
            "end_date": forecast_end_date
        }

        while True:
            try:
                # Fetch forecast weather data for the current location
                responses = openmeteo.weather_api(forecast_url, params=params)
                response = responses[0]

                # Extract hourly data for this location
                hourly = response.Hourly()
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    )
                }
                hourly_data["temperature_2m"] = hourly.Variables(0).ValuesAsNumpy()
                hourly_data["precipitation"] = hourly.Variables(1).ValuesAsNumpy()
                hourly_data["wind_speed_100m"] = hourly.Variables(2).ValuesAsNumpy()
                hourly_data["direct_radiation"] = hourly.Variables(3).ValuesAsNumpy()

                # Convert to DataFrame and append to the list
                hourly_dataframe = pd.DataFrame(data=hourly_data)
                all_data.append(hourly_dataframe)
                break  # Exit the loop if data is fetched successfully

            except Exception as e:
                print(f"Error fetching forecast data for coordinates {coord}: {e}")
                if "Minutely API request limit exceeded" in str(e):
                    print("Waiting for one minute before retrying...")
                    time.sleep(60)  # Wait for one minute before retrying
                else:
                    break  # Exit the loop if the error is not related to the request limit

    # Combine all data into one DataFrame
    combined_df = pd.concat(all_data)

    # Group by date and calculate the mean for all variables
    averaged_data = combined_df.groupby("date").mean()

    # Rename columns for better understanding
    averaged_data = averaged_data.rename(columns={
        "precipitation": "Precipitation (rain/snow)"
    })

    # Save the averaged data to a CSV file
    averaged_data.to_csv(csv_file, index=True)

def update_e_price_data():
    """
    Fetch day-ahead energy prices from SMARD.de and append any new data to
    '../data/day_ahead_energy_prices.csv' without skipping or duplicating rows.
    """
    print("Starting update_smard_data function...")
    
    # Define Berlin timezone
    tz_berlin = pytz.timezone("Europe/Berlin")

    # Read existing data
    print("Loading existing data...")
    e_price_df = pd.read_csv('../data/day_ahead_energy_prices.csv', delimiter=",")
    e_price_df = e_price_df.set_index('Datetime')
    e_price_df.index = pd.to_datetime(e_price_df.index)
    
    print(f"Loaded {len(e_price_df)} existing records.")

    now = datetime.now(tz_berlin)

    # Calculate how many weeks to go back
    try:
        int_weeks = abs(int((e_price_df.index[-1].tz_localize("Europe/Berlin") - now).days / 7)) + 1
        print(f"Fetching data for the past {int_weeks} weeks...")
    except Exception as e:
        print(f"Error calculating weeks to fetch: {e}")
        return

    # Calculate last Monday in Berlin time
    days_since_monday = now.weekday()
    last_monday_berlin = now - timedelta(days=days_since_monday)
    last_monday_berlin = last_monday_berlin.replace(hour=0, minute=0, second=0, microsecond=0)

    # Convert Berlin time to UTC and get the timestamp in milliseconds
    last_monday_utc = last_monday_berlin.astimezone(pytz.UTC)
    last_monday_utc_ms = int(last_monday_utc.timestamp() * 1000)

    delay = 0.5  # seconds
    base_url = "https://www.smard.de/app/chart_data/4169/DE/4169_DE_hour_{}.json"

    # Use a dictionary to store unique timestamps and prices
    energy_ts_data = {}

    for i in range(int_weeks):
        print(f"Fetching data for week {i+1} (starting {last_monday_berlin.date()})...")
        
        last_monday_berlin = last_monday_utc.astimezone(tz_berlin)
        last_monday_utc = last_monday_berlin.astimezone(pytz.UTC)
        last_monday_utc_ms = int(last_monday_utc.timestamp() * 1000)

        # Adjust timestamp if DST is in effect
        if last_monday_berlin.dst() != timedelta(0):
            last_monday_utc_ms -= 60 * 60 * 1000

        try:
            response = requests.get(base_url.format(last_monday_utc_ms))
            response.raise_for_status()
            json_data = response.json()
            print("Successfully fetched data.")
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            continue
        except requests.exceptions.JSONDecodeError as json_err:
            print(f"JSON decode error: {json_err}")
            continue

        # Parse the JSON response
        parsed_json = dict(json_data)

        for ts, price in parsed_json.get("series", []):
            try:
                price_float = float(price)
                ts_datetime = datetime.fromtimestamp(ts / 1000).replace(tzinfo=None).isoformat()
                energy_ts_data[ts_datetime] = price_float
            except TypeError:
                print(f"Skipping invalid data point: ts={ts}, price={price}")
                continue

        # Move one week back
        last_monday_utc = last_monday_utc - timedelta(weeks=1)
        time.sleep(delay)

    print(f"Fetched {len(energy_ts_data)} new data points.")
    
    # Convert the dictionary to a sorted list of tuples
    energy_ts_data_sorted = sorted(energy_ts_data.items())
    
    # Create a DataFrame from the new data
    new_data_df = pd.DataFrame(energy_ts_data_sorted, columns=["Datetime", "hourly day-ahead energy price"])
    new_data_df["Datetime"] = pd.to_datetime(new_data_df["Datetime"])
    new_data_df.set_index("Datetime", inplace=True)

    # Remove any duplicates in the newly fetched data
    new_data_df = new_data_df[~new_data_df.index.duplicated(keep='first')]

    print(f"New data contains {len(new_data_df)} unique timestamps.")

    # Combine with existing data, making sure to remove duplicates
    combined_df = pd.concat([e_price_df, new_data_df])
    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
    combined_df.sort_index(inplace=True)
    combined_df.dropna(inplace=True)

    print(f"Final dataset contains {len(combined_df)} total records.")
    
    # Save to CSV
    combined_df.to_csv('../data/day_ahead_energy_prices.csv', date_format='%Y-%m-%dT%H:%M:%S')
    print("Data successfully updated and saved.")



def update_e_mix_data(csv_path="../data/hourly_market_mix_cleaned.csv"):
    
    mix_categories = [
        "Biomass", "Hard Coal", "Hydro", "Lignite", "Natural Gas", "Nuclear",
        "Other", "Pumped storage generation", "Solar", "Wind offshore", "Wind onshore"
    ]
    
    url = "https://api.agora-energy.org/api/raw-data"
    headers = {
        "Content-Type": "application/json", "Accept": "*/*", "Api-key": "agora_live_62ce76dd202927.67115829"
    }
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        existing_timestamps = set(df["Timestamp"])
        fetch_start = pd.to_datetime(df["Timestamp"]).max() + pd.Timedelta(hours=1)
    else:
        df = pd.DataFrame(columns=["Timestamp"] + mix_categories)
        existing_timestamps = set()
        fetch_start = datetime(2018, 1, 1)
    
    fetch_end = datetime.now()
    if fetch_start >= fetch_end:
        print("No new data to fetch.")
        return
    
    slices = [(max(fetch_start, datetime(y, 1, 1)), min(fetch_end, datetime(y+1, 1, 1))) for y in range(fetch_start.year, fetch_end.year + 1)]
    
    data_dict = {}
    for start, end in slices:
        payload = {
            "filters": {"from": start.strftime("%Y-%m-%d"), "to": end.strftime("%Y-%m-%d"), "generation": mix_categories},
            "x_coordinate": "date_id", "y_coordinate": "value", "view_name": "live_gen_plus_emi_de_hourly",
            "kpi_name": "power_generation", "z_coordinate": "generation"
        }
        response = requests.post(url, headers=headers, json=payload)
        time.sleep(0.3)
        if response.status_code != 200:
            print(f"Request failed for {start} to {end}")
            continue
        
        for ts, value, category in response.json().get("data", {}).get("data", []):
            if category in mix_categories:
                if value is None:
                    print(f"Warning: Null value for {ts} - {category}, replacing with 0.0")
                data_dict.setdefault(ts, {}).update({category: float(value) if value is not None else 0.0})

        # Ensure every timestamp has all 11 categories
        for ts in data_dict.keys():
            for cat in mix_categories:
                if cat not in data_dict[ts]:
                    print(f"Warning: Missing category {cat} for {ts}, filling with 0.0")
                    data_dict[ts][cat] = 0.0  # Fill missing values

    
    new_rows = [[ts] + [data_dict.get(ts, {}).get(cat, 0.0) for cat in mix_categories] for ts in sorted(data_dict) if ts not in existing_timestamps]
    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=["Timestamp"] + mix_categories)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"Added {len(new_rows)} new rows.")
    else:
        print("No new data added.")
