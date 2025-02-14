import pandas as pd
from datetime import timedelta, datetime, timezone
import numpy as np
import pytz
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
import time
import os

dir = "./data/"

prediction_date_end = datetime(2025, 2, 18, 23, 0, 0)

def fill_up_e_prices(end_date):

    prediction_date_end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")

    path_real = dir + 'day_ahead_energy_prices.csv'
    path_filled_debug = dir + 'debug_filled_day_ahead_energy_prices.csv'
    path_filled = dir + 'filled_day_ahead_energy_prices.csv'
    e_price_df = pd.read_csv(path_real, delimiter=",")
    e_price_df = e_price_df.set_index('Datetime')
    e_price_df.index = pd.to_datetime(e_price_df.index)
    
    estimations_df = get_estimations(e_price_df.copy(), e_price_df.index[-1], col_name="hourly day-ahead energy price", count=None, final_date=prediction_date_end)
    estimations_df.to_csv(path_filled_debug, index_label="Datetime", mode='w')
    e_price_df = pd.concat([e_price_df, estimations_df])
    
    e_price_df.index = e_price_df.index.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Save to CSV with the modified index
    if not os.path.exists(path_filled):
        open(path_filled, 'w').close()  # Create the file if it doesn't exist

    # Write the DataFrame to the file, replacing its content
    e_price_df.to_csv(path_filled, index_label="Datetime", mode='w')

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

    if count <= 0:
        return pd.DataFrame()

    sampled = np.random.normal(last_24h_from_last_week_mean, last_24h_from_last_week_std, size=count).round(2) # assuming stationary distribution of the last 24h
    new_indices = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=count, freq="H")
    estimated_df = pd.DataFrame({col_name: sampled}, index=new_indices)
    return estimated_df


def fill_up_energy_mix(end_date):
    """
    Fills up missing future values in the energy mix dataset by copying past values 
    and supplementing them with SMARD energy mix predictions.
    """
    prediction_date_end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
    input_path = os.path.join(dir, 'hourly_market_mix_cleaned.csv')
    debug_path = os.path.join(dir, 'debug_hourly_market_mix_cleaned.csv')
    output_path = os.path.join(dir, 'filled_hourly_market_mix_cleaned.csv')
    
    # Load the dataset
    new_columns = ["Pumped storage generation", "Solar", "Wind offshore", "Wind onshore", "Hydro"]

    mix_df = pd.read_csv(input_path, delimiter=",")
    
    # Convert timestamp column to datetime index
    mix_df["Timestamp"] = pd.to_datetime(mix_df["Timestamp"])
    mix_df.set_index("Timestamp", inplace=True)
    
    last_date = mix_df.index[-1]
    hours_to_fill = int((prediction_date_end - last_date).total_seconds() / 3600)
    
    if hours_to_fill > 0:
        # Split dataset into relevant subsets
        old_mix_df = mix_df[new_columns]
        copy_mix_df = get_by_copy(old_mix_df, last_date, hours_to_fill)
        
        # Combine estimated and predicted values
        copy_mix_df.to_csv(debug_path, index_label="Timestamp", mode='w')
        
        # Extend the original dataset
        mix_df = pd.concat([old_mix_df, copy_mix_df])
        mix_df.index = mix_df.index.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Ensure the output file exists before writing
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the extended dataset
    mix_df.to_csv(output_path, index_label="Timestamp", mode='w')




def get_by_copy(df, last_date, n):
    """
    Copies the last n rows of a DataFrame from last week and appends them to the end,
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

    last_n_rows = df.iloc[-24:].copy()

    # Generate new timestamps starting from last_date + 1 hour
    new_index = pd.date_range(
        start=last_date + pd.Timedelta(hours=1), periods=n, freq="H"
    )

    # Ensure new timestamps match expected future values
    last_n_rows.index = new_index

    return last_n_rows  # Return only the extended rows



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
                # print("None value detected for ts: ", date_time_ts)
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
        dfs.append(df)

    df_merged = pd.concat(dfs, axis=1)

    return df_merged


def download_smard_energy_mix_prediction(start_date: datetime, n, end_date=None):
    timestamp_in_milliseconds, hour_offset, delta_pred_values = preprocess_smard_energy_mix_prediction_dates(start_date, n, end_date)
    responses = fetch_smard_energy_mix_prediction_data(timestamp_in_milliseconds)
    df_merged = postprocess_smard_energy_mix_prediction_data(responses, hour_offset, delta_pred_values)
    return df_merged


def setup_client():
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)

def convert_dates(start_date, end_date=None, hours=None):
    if end_date:
        end_dt = end_date
    elif hours:
        end_dt = start_date + timedelta(hours=hours)
    else:
        raise ValueError("Either end_date or hours must be provided")
    return start_date.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def fetch_data(client, coords, params_template, forecast_url):
    all_data = []
    for coord in coords:
        params = params_template.copy()
        params.update({"latitude": coord["latitude"], "longitude": coord["longitude"]})
        while True:
            try:
                responses = client.weather_api(forecast_url, params=params)
                response = responses[0]
                hourly = response.Hourly()
                hourly_data = {
                    "date": pd.date_range(
                        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                        freq=pd.Timedelta(seconds=hourly.Interval()),
                        inclusive="left"
                    )
                }
                for var in params_template["hourly"]:
                    hourly_data[var] = hourly.Variables(params_template["hourly"].index(var)).ValuesAsNumpy()
                hourly_dataframe = pd.DataFrame(data=hourly_data)
                hourly_dataframe["weight"] = coord.get("weight", 1)
                all_data.append(hourly_dataframe)
                break
            except Exception as e:
                print(f"Error fetching forecast data for coordinates {coord}: {e}")
                if "Minutely API request limit exceeded" in str(e):
                    print("Waiting for one minute before retrying...")
                    time.sleep(60)
                else:
                    break
    return pd.concat(all_data)

def calculate_weighted_averages(data, parks, variable, weight_column):
    total_weight = sum(park["weight"] for park in parks)
    data[weight_column] = data[variable] * data["weight"] / total_weight
    return data.groupby("date")[weight_column].sum()

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
wind_parks = [
        {"latitude": 54.008333, "longitude": 6.598333, "weight": 60},      # Alpha Ventus
        {"latitude": 54.358333, "longitude": 5.975, "weight": 400},        # BARD Offshore I
        {"latitude": 53.690, "longitude": 6.480, "weight": 113.4},         # Riffgat
        {"latitude": 54.15, "longitude": 7.25, "weight": 295},             # Amrumbank West
        {"latitude": 54.53, "longitude": 6.25, "weight": 200},             # Butendiek
        {"latitude": 54.367, "longitude": 6.467, "weight": 295},           # DanTysk
        {"latitude": 54.480, "longitude": 7.370, "weight": 288},           # Meerwind Süd|Ost
        {"latitude": 54.4, "longitude": 6.6, "weight": 576},               # Gode Wind 1 & 2
        {"latitude": 54.30, "longitude": 6.65, "weight": 400},             # Global Tech I
        {"latitude": 53.88, "longitude": 6.59, "weight": 450},             # Borkum Riffgrund 1
        {"latitude": 53.88, "longitude": 6.59, "weight": 215},             # Borkum Riffgrund 2
        {"latitude": 54.00, "longitude": 6.58, "weight": 342},             # Trianel Windpark Borkum
        {"latitude": 54.22, "longitude": 6.63, "weight": 332},             # Nordsee Ost
        {"latitude": 54.25, "longitude": 7.25, "weight": 385},             # Hohe See
        {"latitude": 54.28, "longitude": 7.30, "weight": 252},             # Albatros
        {"latitude": 54.48, "longitude": 6.78, "weight": 350},             # Wikinger
        {"latitude": 54.55, "longitude": 6.37, "weight": 402},             # Arkona
        {"latitude": 54.45, "longitude": 6.58, "weight": 600},             # Veja Mate
        {"latitude": 54.33, "longitude": 7.18, "weight": 300},             # Deutsche Bucht
        {"latitude": 54.25, "longitude": 7.18, "weight": 402},             # Kaskasi
        {"latitude": 53.610278, "longitude": 7.429167, "weight": 318.2},  # Windpark Holtriem-Dornum
        {"latitude": 53.973889, "longitude": 8.933333, "weight": 302.45},  # Windpark Friedrichskoog
        {"latitude": 54.611111, "longitude": 8.903611, "weight": 293.4},  # Bürgerwindpark Reußenköge
        {"latitude": 53.338333, "longitude": 13.764444, "weight": 242.5},  # Windfeld Uckermark
        {"latitude": 53.715278, "longitude": 13.319722, "weight": 202.85},  # RH2-Werder/Kessin/Altentreptow
        {"latitude": 51.131667, "longitude": 11.964167, "weight": 188.1},  # Windpark Stößen-Teuchern
        {"latitude": 52.539722, "longitude": 12.871667, "weight": 175.2},  # Windpark Ketzin
        {"latitude": 52.515833, "longitude": 11.780833, "weight": 151.3},  # Windpark Hüselitz
        {"latitude": 51.031667, "longitude": 10.629722, "weight": 152.25},  # Windfeld Wangenheim-Hochheim-Ballstädt-Westhausen
        {"latitude": 52.354722, "longitude": 14.373056, "weight": 133.9},  # Windpark Odervorland
        {"latitude": 51.640278, "longitude": 8.912222, "weight": 129.445},  # Windpark Asseln
        {"latitude": 52.001389, "longitude": 12.830833, "weight": 128.2},  # Windpark Feldheim
        {"latitude": 51.395556, "longitude": 11.709167, "weight": 122.1},  # Windpark Esperstedt-Obhausen
        {"latitude": 51.960833, "longitude": 11.606389, "weight": 114.45},  # Windpark Biere-Borne
        {"latitude": 53.3375, "longitude": 7.095833, "weight": 106.25},  # Windpark Wybelsumer Polder
        {"latitude": 53.388056, "longitude": 7.377778, "weight": 102.34},  # Windpark Ihlow
        {"latitude": 52.015556, "longitude": 13.193333, "weight": 98.8},  # Windpark Heidehof
        {"latitude": 51.546389, "longitude": 13.868611, "weight": 93.1},  # Windpark Klettwitz
        {"latitude": 52.662778, "longitude": 11.709167, "weight": 93.5},  # Windpark Schinne-Grassau
        {"latitude": 51.989722, "longitude": 10.833333, "weight": 92.4},  # Windpark Druiberg
        {"latitude": 51.579722, "longitude": 11.708611, "weight": 89.3},  # Windpark Beesenstedt-Rottelsdorf
        {"latitude": 52.123333, "longitude": 11.160000, "weight": 87.65},  # Windpark Ausleben-Badeleben-Wormsdorf
        {"latitude": 53.070833, "longitude": 7.739167, "weight": 86.5},  # Windpark Saterland
        {"latitude": 51.721111, "longitude": 11.644167, "weight": 83.35},  # Windpark Alsleben
        {"latitude": 51.798611, "longitude": 11.491944, "weight": 83.05},  # Windpark Blaue Warthe
        {"latitude": 51.474167, "longitude": 13.249722, "weight": 82.8},  # Windfeld Randowhöhe
        {"latitude": 51.173056, "longitude": 11.350556, "weight": 82.65},  # Windpark Rastenberg-Olbersleben
        {"latitude": 51.975833, "longitude": 11.451944, "weight": 79.1},  # Windpark Egeln-Nord
        {"latitude": 53.363056, "longitude": 7.705000, "weight": 77.4},  # Windpark Wiesmoor
        {"latitude": 51.774444, "longitude": 12.700833, "weight": 77.1},  # Windpark Dorna-Kemberg-Schnellin
        {"latitude": 52.027778, "longitude": 11.367778, "weight": 76.9},  # Windfeld Sonnenberg
        {"latitude": 53.320833, "longitude": 12.026944, "weight": 75.2},  # Windpark Jännersdorf
        {"latitude": 51.617222, "longitude": 8.803333, "weight": 75.05},  # Windpark Altenautal
        {"latitude": 52.192500, "longitude": 11.368056, "weight": 71.3},  # Windpar Bornstedt-Nordgermersleben-Rottmersleben-Schackensleben
        {"latitude": 51.642500, "longitude": 11.658333, "weight": 72},  # Windpark Gerbstedt-Ihlewitz
        {"latitude": 49.964722, "longitude": 7.652500, "weight": 70.5},  # Hunsrück-Windpark Ellern
        {"latitude": 52.867500, "longitude": 7.138889, "weight": 70.1},  # Windpark Haren
        {"latitude": 51.041111, "longitude": 6.530000, "weight": 67.2},  # Windpark Königshovener Höhe
        {"latitude": 51.445278, "longitude": 8.696944, "weight": 65.95},  # Windpark Madfeld-Bleiwäsche
        {"latitude": 53.817778, "longitude": 8.078889, "weight": 65.6},  # Windpark Altenbruch
        {"latitude": 52.176389, "longitude": 11.300000, "weight": 64.1},  # Windpark Hakenstedt
        {"latitude": 51.946111, "longitude": 14.462778, "weight": 64},  # Windpark Cottbuser Halde
        {"latitude": 51.707778, "longitude": 12.239167, "weight": 62.7},  # Windpark Thurland
        {"latitude": 49.689167, "longitude": 8.106944, "weight": 61},  # Windfeld Rheinhessen-Pfalz
        {"latitude": 50.003333, "longitude": 7.386667, "weight": 59.8},  # Windpark Kirchberg im Faas
        {"latitude": 51.040556, "longitude": 11.620278, "weight": 59.1},  # Windpark Eckolstädt
        {"latitude": 51.247500, "longitude": 10.283889, "weight": 58.1},  # Windpark Büttstedt
        {"latitude": 51.072778, "longitude": 11.789444, "weight": 58},  # Windpark Molau-Leislau
        {"latitude": 54.483333, "longitude": 11.110000, "weight": 57.5},  # Windpark Fehmarn-Mitte
        {"latitude": 49.830000, "longitude": 8.138889, "weight": 55},  # Windpark Wörrstadt
        {"latitude": 49.296111, "longitude": 9.415556, "weight": 54.9},  # Windpark Harthäuser Wald
        {"latitude": 53.373333, "longitude": 9.496944, "weight": 52.9},  # Windpark Ahrenswohlde-Wohnste
        {"latitude": 48.980833, "longitude": 11.102500, "weight": 52.8},  # Windpark Raitenbucher Forst
        {"latitude": 48.740000, "longitude": 9.889722, "weight": 52.25},  # Windpark Lauterstein
        {"latitude": 49.721111, "longitude": 7.721944, "weight": 52.2},  # Windpark Lettweiler Höhe
        {"latitude": 50.603056, "longitude": 9.243889, "weight": 49.65},  # Windpark Goldner Steinrück
        {"latitude": 50.516944, "longitude": 6.373611, "weight": 49.45},  # Windpark Schleiden-Schöneseiffen
        {"latitude": 53.538889, "longitude": 8.952778, "weight": 48.8},  # Windpark Köhlen
        {"latitude": 49.764167, "longitude": 8.059722, "weight": 47.1},  # Windpark Heimersheim
        {"latitude": 53.396667, "longitude": 14.169167, "weight": 46.3},  # Windfeld Wolfsmoor
        {"latitude": 53.684167, "longitude": 8.646111, "weight": 46},  # Windpark Holßel
        {"latitude": 51.838333, "longitude": 12.875278, "weight": 44.9},  # Windpark Elster
        {"latitude": 52.002222, "longitude": 12.123056, "weight": 44.4},  # Windpark Zerbst
        {"latitude": 52.178333, "longitude": 11.886111, "weight": 43.6},  # Windpark Stegelitz-Ziepel-Tryppehna
        {"latitude": 53.606944, "longitude": 8.793056, "weight": 43.2},  # Windpark Kührstedt-Alfstedt
        {"latitude": 52.060111, "longitude": 14.381000, "weight": 43.2},  # Windpark Ullersdorf
        {"latitude": 49.813333, "longitude": 8.017778, "weight": 42.4},  # Windpark Gau-Bickelheim
        {"latitude": 51.422778, "longitude": 11.834444, "weight": 42},  # Windpark Holleben-Bad Lauchstädt
        {"latitude": 54.648611, "longitude": 9.176389, "weight": 41.8},  # Bürgerwindpark Löwenstedt
        {"latitude": 50.623861, "longitude": 9.153528, "weight": 41.6},  # Windpark Feldatal
        {"latitude": 51.413056, "longitude": 11.587222, "weight": 41},  # Windpark Farnstädt
        {"latitude": 52.976667, "longitude": 7.415833, "weight": 40.9},  # Windpark Dörpen-Ost
        {"latitude": 52.878056, "longitude": 10.042778, "weight": 40.5},  # Windpark Hermannsburg
        {"latitude": 52.900000, "longitude": 12.384167, "weight": 40.4},  # Windpark Kyritz-Plänitz-Zernitz
        {"latitude": 52.597222, "longitude": 12.266667, "weight": 40},  # Windpark Stüdenitz
    ]
sun_parks = [
        {"latitude": 51.3167, "longitude": 12.3667, "weight": 605},   # Witznitz
        {"latitude": 51.3236, "longitude": 12.6511, "weight": 52},    # Waldpolenz Solar Park
        {"latitude": 51.7625, "longitude": 13.6000, "weight": 52.3},  # Walddrehna Solar Park
        {"latitude": 53.9239, "longitude": 13.2256, "weight": 52},    # Tutow Solar Park
        {"latitude": 53.0290, "longitude": 13.5336, "weight": 128.5}, # Templin Solar Park
        {"latitude": 48.8031, "longitude": 12.7669, "weight": 54},    # Strasskirchen Solar Park
        {"latitude": 53.6391, "longitude": 12.3643, "weight": 76},    # Solarpark Zietlitz
        {"latitude": 52.6475, "longitude": 13.6916, "weight": 187},   # Solarpark Weesow-Willmersdorf
        {"latitude": 53.5267, "longitude": 11.6609, "weight": 172},   # Solarpark Tramm-Göhten
        {"latitude": 48.6490, "longitude": 11.2782, "weight": 120},   # Solarpark Schornhof
        {"latitude": 51.5450, "longitude": 13.9800, "weight": 166},   # Solarpark Meuro
        {"latitude": 50.5960, "longitude": 9.3690, "weight": 54.7},   # Solarpark Lauterbach
        {"latitude": 52.6413, "longitude": 14.1923, "weight": 150},   # Solarpark Gottesgabe
        {"latitude": 53.3818, "longitude": 12.2688, "weight": 65},    # Solarpark Ganzlin
        {"latitude": 53.4148, "longitude": 12.2470, "weight": 90},    # Solarpark Gaarz
        {"latitude": 52.8253, "longitude": 13.6983, "weight": 84.7},  # Solarpark Finow Tower
        {"latitude": 52.6975, "longitude": 14.2300, "weight": 150},   # Solarpark Alttrebbin
        {"latitude": 53.2000, "longitude": 12.5167, "weight": 67.8},  # Solarpark Alt Daber
        {"latitude": 52.6139, "longitude": 14.2425, "weight": 145},   # Neuhardenberg Solar Park
        {"latitude": 51.9319, "longitude": 14.4072, "weight": 71.8},  # Lieberose Photovoltaic Park
        {"latitude": 51.5686, "longitude": 13.7375, "weight": 80.7},  # Finsterwalde Solar Park
        {"latitude": 54.6294, "longitude": 9.3433, "weight": 83.6},   # Eggebek Solar Park
        {"latitude": 52.4367, "longitude": 12.4514, "weight": 91}     # Brandenburg-Briest Solarpark
    ]


#Fetch Forecast Data and append to CSV. Last day till now() + 5 days

def fetch_forecast_and_update_csv(end_date):
    prediction_date_end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    # Read the existing weather data CSV
    csv_file_path = dir + "germany_weather_average.csv"
    output_path_debug = os.path.join(dir, 'debug_filled_germany_weather_average.csv')
    output_path = os.path.join(dir, 'filled_germany_weather_average.csv')
    df_weather = pd.read_csv(csv_file_path, parse_dates=["date"], index_col="date")

    # Check if the last row has the date attribute 00:00:00+0000
    if df_weather.index[-1].time() == datetime.strptime("00:00:00+0000", "%H:%M:%S%z").time():
        df_weather = df_weather.iloc[:-1]  # Drop the last row

    # Calculate the start date for the forecast
    start_date = df_weather.index[-1] + timedelta(hours=1)
    end_date = prediction_date_end

    if end_date >= start_date:
        # Setup the client and fetch the forecast data
        client = setup_client()
        forecast_url = "https://api.open-meteo.com/v1/forecast"

        forecast_start_date, forecast_end_date = convert_dates(start_date, end_date)

        params_template = {
            "start_date": forecast_start_date,
            "end_date": forecast_end_date,
            "hourly": ["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"]
        }
        coordinates_data = fetch_data(client, coordinates, params_template, forecast_url)

        params_template["hourly"] = ["wind_speed_100m"]
        wind_parks_data = fetch_data(client, wind_parks, params_template, forecast_url)

        params_template["hourly"] = ["direct_radiation"]
        sun_parks_data = fetch_data(client, sun_parks, params_template, forecast_url)

        weighted_wind_speed = calculate_weighted_averages(wind_parks_data, wind_parks, "wind_speed_100m", "weighted_wind_speed")
        weighted_radiation = calculate_weighted_averages(sun_parks_data, sun_parks, "direct_radiation", "weighted_radiation")

        coordinates_avg = coordinates_data.groupby("date").mean()
        predicted_weather_df = pd.concat([coordinates_avg, weighted_wind_speed, weighted_radiation], axis=1)
        predicted_weather_df.to_csv(output_path_debug, index_label="date", date_format="%Y-%m-%d %H:%M:%S", mode="w")

    # Select only the required columns
    df_weather = df_weather[["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"]]
    df_weather.index.name = "date"
    df_weather.index = df_weather.index.tz_convert("UTC")

    # Append the new forecast data to the existing CSV
    updated_df = pd.concat([df_weather, predicted_weather_df])
    updated_df.to_csv(output_path, index_label="date", date_format="%Y-%m-%d %H:%M:%S%z", mode="w")

def fetch_forecast_and_update_csv(end_date):
    # Read the existing weather data CSV
    csv_file_path = dir + "germany_weather_average.csv"
    output_path_debug = os.path.join(dir, 'debug_filled_germany_weather_average.csv')
    output_path = os.path.join(dir, 'filled_germany_weather_average.csv')
    df_weather = pd.read_csv(csv_file_path, parse_dates=["date"], index_col="date")

    # Check if the last row has the date attribute 00:00:00+0000
    if df_weather.index[-1].time() == datetime.strptime("00:00:00+0000", "%H:%M:%S%z").time():
        df_weather = df_weather.iloc[:-1]  # Drop the last row


    start_date = df_weather.index[-1]
    end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    if end_date >= start_date:
        # Setup the client and fetch the forecast data
        client = setup_client()
        forecast_url = "https://api.open-meteo.com/v1/forecast"

        forecast_start_date, forecast_end_date = convert_dates(start_date, end_date)

        params_template = {
            "start_date": forecast_start_date,
            "end_date": forecast_end_date,
            "hourly": ["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"]
        }
        coordinates_data = fetch_data(client, coordinates, params_template, forecast_url)

        params_template["hourly"] = ["wind_speed_100m"]
        wind_parks_data = fetch_data(client, wind_parks, params_template, forecast_url)

        params_template["hourly"] = ["direct_radiation"]
        sun_parks_data = fetch_data(client, sun_parks, params_template, forecast_url)

        weighted_wind_speed = calculate_weighted_averages(wind_parks_data, wind_parks, "wind_speed_100m", "weighted_wind_speed")
        weighted_radiation = calculate_weighted_averages(sun_parks_data, sun_parks, "direct_radiation", "weighted_radiation")

        coordinates_avg = coordinates_data.groupby("date").mean()
        combined_df = pd.concat([coordinates_avg, weighted_wind_speed, weighted_radiation], axis=1)
        combined_df = combined_df[["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"]]

    else:
        combined_df = pd.DataFrame(columns=["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"])
    
    combined_df.index.name = "date"
    combined_df.index = combined_df.index.tz_convert("UTC")
    combined_df.to_csv(output_path_debug, date_format="%Y-%m-%d %H:%M:%S%z", mode="w")


    # Append the new forecast data to the existing CSV
    updated_df = pd.concat([df_weather, combined_df])
    updated_df.to_csv(output_path, date_format="%Y-%m-%d %H:%M:%S%z", mode="w")



