import pandas as pd
from datetime import datetime, timedelta
import pytz
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
import os
import time

dir = "./data/"


def preprocess_smard_energy_mix_prediction_dates(
    start_date: datetime, n=None, end_date=None
):

    local_timezone = pytz.timezone("Europe/Berlin")
    start_date_str = start_date.strftime("%Y-%m-%d")
    date_string = f"{start_date_str} 00:00:00"
    local_date_object = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    localized_date_object = local_timezone.localize(local_date_object)
    epoch_timestamp = int(local_date_object.timestamp())

    weekday = localized_date_object.weekday()  # Monday = 0, Sunday = 6
    hour_offset = weekday * 24
    timestamp_in_milliseconds = epoch_timestamp * 1000 - (hour_offset * 3600 * 1000)

    return timestamp_in_milliseconds


def fetch_smard_energy_mix_prediction_data(timestamp_in_milliseconds):
    urls = {
        "Wind onshore": f"https://www.smard.de/app/chart_data/3791/DE/3791_DE_hour_{timestamp_in_milliseconds}.json",
        "Wind offshore": f"https://www.smard.de/app/chart_data/123/DE/123_DE_hour_{timestamp_in_milliseconds}.json",
        "Solar": f"https://www.smard.de/app/chart_data/125/DE/125_DE_hour_{timestamp_in_milliseconds}.json",
    }

    responses = {}
    for name, url in urls.items():
        response = requests.get(url)
        if response.status_code == 200:
            responses[name] = response.json()
        else:
            print(f"Failed to fetch {name} data: {response.status_code}")

    return responses


def postprocess_smard_energy_mix_prediction_data(
    responses, start_date, end_date, backup_df
):
    dfs = []
    for name, data in responses.items():
        if "series" not in data or not data["series"]:
            print(f"No {name} data available for the specified date.")
            continue

        series = data["series"]

        dts = []
        observed_output = []
        last_date_with_value = None
        missing_values = []

        for ts, value in series:
            try:
                observed_output.append(value / 1000)
                dt = datetime.fromtimestamp(ts / 1000, tz=pytz.utc).astimezone(
                    pytz.timezone("Europe/Berlin")
                )
                last_date_with_value = dt
                dts.append(dt.strftime("%Y-%m-%d %H:%M:%S"))
            except TypeError:
                next_date = last_date_with_value + timedelta(hours=1)
                last_date_with_value = next_date
                missing_values.append(next_date.strftime("%Y-%m-%d %H:%M:%S"))

        df = pd.DataFrame({"Datetime": dts, name: observed_output})
        df["Datetime"] = pd.to_datetime(df["Datetime"], format="%Y-%m-%d %H:%M:%S")
        df.set_index("Datetime", inplace=True)

        df = df.loc[start_date:end_date]

        hours_diff = (end_date - start_date).total_seconds() / 3600
        if df.shape[0] < hours_diff:
            print("Sampling missing values for dates {}".format(missing_values))
            for ts in missing_values:
                df.loc[ts] = backup_df.loc[ts - timedelta(weeks=1), name]

        dfs.append(df)

    df_merged = pd.concat(dfs, axis=1)
    return df_merged


def download_smard_energy_mix_prediction(
    start_date: datetime, n, backup_df, end_date=None
):
    start_date = start_date + timedelta(hours=1)
    timestamp_in_milliseconds = preprocess_smard_energy_mix_prediction_dates(
        start_date, n, end_date
    )
    responses = fetch_smard_energy_mix_prediction_data(timestamp_in_milliseconds)

    if n:
        end_date = start_date + timedelta(hours=n - 1)
    elif not end_date:
        raise ValueError("Either n or end_date must be provided")

    df_merged = postprocess_smard_energy_mix_prediction_data(
        responses, start_date, end_date, backup_df
    )

    return df_merged


def setup_client():
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
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
                        inclusive="left",
                    )
                }
                for var in params_template["hourly"]:
                    hourly_data[var] = hourly.Variables(
                        params_template["hourly"].index(var)
                    ).ValuesAsNumpy()
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
    {"latitude": 50.1109, "longitude": 8.6821},  # Frankfurt
    {"latitude": 51.1657, "longitude": 10.4515},  # Central Germany (approximate)
    {"latitude": 53.5511, "longitude": 9.9937},  # Hamburg
    {"latitude": 51.2277, "longitude": 6.7735},  # Düsseldorf
    {"latitude": 51.0504, "longitude": 13.7373},  # Dresden
    {"latitude": 50.9375, "longitude": 6.9603},  # Cologne
    {"latitude": 49.4875, "longitude": 8.4660},  # Mannheim
    {"latitude": 48.7758, "longitude": 9.1829},  # Stuttgart
    {"latitude": 51.3397, "longitude": 12.3731},  # Leipzig
    {"latitude": 50.0782, "longitude": 8.2398},  # Wiesbaden
    {"latitude": 49.0069, "longitude": 8.4037},  # Karlsruhe
    {"latitude": 51.5128, "longitude": 7.4633},  # Dortmund
    {"latitude": 50.1211, "longitude": 8.4965},  # Offenbach
    {"latitude": 50.3569, "longitude": 7.5886},  # Koblenz
    {"latitude": 50.7753, "longitude": 6.0839},  # Aachen
    {"latitude": 49.4521, "longitude": 11.0767},  # Nuremberg
    {"latitude": 52.3759, "longitude": 9.7320},  # Hanover
    {"latitude": 51.4818, "longitude": 7.2162},  # Bochum
    {"latitude": 51.4556, "longitude": 7.0116},  # Essen
    {"latitude": 51.4344, "longitude": 6.7623},  # Duisburg
    {"latitude": 51.9607, "longitude": 7.6261},  # Münster
]
wind_parks = [
    {"latitude": 54.008333, "longitude": 6.598333, "weight": 60},  # Alpha Ventus
    {"latitude": 54.358333, "longitude": 5.975, "weight": 400},  # BARD Offshore I
    {"latitude": 53.690, "longitude": 6.480, "weight": 113.4},  # Riffgat
    {"latitude": 54.15, "longitude": 7.25, "weight": 295},  # Amrumbank West
    {"latitude": 54.53, "longitude": 6.25, "weight": 200},  # Butendiek
    {"latitude": 54.367, "longitude": 6.467, "weight": 295},  # DanTysk
    {"latitude": 54.480, "longitude": 7.370, "weight": 288},  # Meerwind Süd|Ost
    {"latitude": 54.4, "longitude": 6.6, "weight": 576},  # Gode Wind 1 & 2
    {"latitude": 54.30, "longitude": 6.65, "weight": 400},  # Global Tech I
    {"latitude": 53.88, "longitude": 6.59, "weight": 450},  # Borkum Riffgrund 1
    {"latitude": 53.88, "longitude": 6.59, "weight": 215},  # Borkum Riffgrund 2
    {"latitude": 54.00, "longitude": 6.58, "weight": 342},  # Trianel Windpark Borkum
    {"latitude": 54.22, "longitude": 6.63, "weight": 332},  # Nordsee Ost
    {"latitude": 54.25, "longitude": 7.25, "weight": 385},  # Hohe See
    {"latitude": 54.28, "longitude": 7.30, "weight": 252},  # Albatros
    {"latitude": 54.48, "longitude": 6.78, "weight": 350},  # Wikinger
    {"latitude": 54.55, "longitude": 6.37, "weight": 402},  # Arkona
    {"latitude": 54.45, "longitude": 6.58, "weight": 600},  # Veja Mate
    {"latitude": 54.33, "longitude": 7.18, "weight": 300},  # Deutsche Bucht
    {"latitude": 54.25, "longitude": 7.18, "weight": 402},  # Kaskasi
    {
        "latitude": 53.610278,
        "longitude": 7.429167,
        "weight": 318.2,
    },  # Windpark Holtriem-Dornum
    {
        "latitude": 53.973889,
        "longitude": 8.933333,
        "weight": 302.45,
    },  # Windpark Friedrichskoog
    {
        "latitude": 54.611111,
        "longitude": 8.903611,
        "weight": 293.4,
    },  # Bürgerwindpark Reußenköge
    {
        "latitude": 53.338333,
        "longitude": 13.764444,
        "weight": 242.5,
    },  # Windfeld Uckermark
    {
        "latitude": 53.715278,
        "longitude": 13.319722,
        "weight": 202.85,
    },  # RH2-Werder/Kessin/Altentreptow
    {
        "latitude": 51.131667,
        "longitude": 11.964167,
        "weight": 188.1,
    },  # Windpark Stößen-Teuchern
    {"latitude": 52.539722, "longitude": 12.871667, "weight": 175.2},  # Windpark Ketzin
    {
        "latitude": 52.515833,
        "longitude": 11.780833,
        "weight": 151.3,
    },  # Windpark Hüselitz
    {
        "latitude": 51.031667,
        "longitude": 10.629722,
        "weight": 152.25,
    },  # Windfeld Wangenheim-Hochheim-Ballstädt-Westhausen
    {
        "latitude": 52.354722,
        "longitude": 14.373056,
        "weight": 133.9,
    },  # Windpark Odervorland
    {
        "latitude": 51.640278,
        "longitude": 8.912222,
        "weight": 129.445,
    },  # Windpark Asseln
    {
        "latitude": 52.001389,
        "longitude": 12.830833,
        "weight": 128.2,
    },  # Windpark Feldheim
    {
        "latitude": 51.395556,
        "longitude": 11.709167,
        "weight": 122.1,
    },  # Windpark Esperstedt-Obhausen
    {
        "latitude": 51.960833,
        "longitude": 11.606389,
        "weight": 114.45,
    },  # Windpark Biere-Borne
    {
        "latitude": 53.3375,
        "longitude": 7.095833,
        "weight": 106.25,
    },  # Windpark Wybelsumer Polder
    {"latitude": 53.388056, "longitude": 7.377778, "weight": 102.34},  # Windpark Ihlow
    {
        "latitude": 52.015556,
        "longitude": 13.193333,
        "weight": 98.8,
    },  # Windpark Heidehof
    {
        "latitude": 51.546389,
        "longitude": 13.868611,
        "weight": 93.1,
    },  # Windpark Klettwitz
    {
        "latitude": 52.662778,
        "longitude": 11.709167,
        "weight": 93.5,
    },  # Windpark Schinne-Grassau
    {
        "latitude": 51.989722,
        "longitude": 10.833333,
        "weight": 92.4,
    },  # Windpark Druiberg
    {
        "latitude": 51.579722,
        "longitude": 11.708611,
        "weight": 89.3,
    },  # Windpark Beesenstedt-Rottelsdorf
    {
        "latitude": 52.123333,
        "longitude": 11.160000,
        "weight": 87.65,
    },  # Windpark Ausleben-Badeleben-Wormsdorf
    {
        "latitude": 53.070833,
        "longitude": 7.739167,
        "weight": 86.5,
    },  # Windpark Saterland
    {
        "latitude": 51.721111,
        "longitude": 11.644167,
        "weight": 83.35,
    },  # Windpark Alsleben
    {
        "latitude": 51.798611,
        "longitude": 11.491944,
        "weight": 83.05,
    },  # Windpark Blaue Warthe
    {
        "latitude": 51.474167,
        "longitude": 13.249722,
        "weight": 82.8,
    },  # Windfeld Randowhöhe
    {
        "latitude": 51.173056,
        "longitude": 11.350556,
        "weight": 82.65,
    },  # Windpark Rastenberg-Olbersleben
    {
        "latitude": 51.975833,
        "longitude": 11.451944,
        "weight": 79.1,
    },  # Windpark Egeln-Nord
    {"latitude": 53.363056, "longitude": 7.705000, "weight": 77.4},  # Windpark Wiesmoor
    {
        "latitude": 51.774444,
        "longitude": 12.700833,
        "weight": 77.1,
    },  # Windpark Dorna-Kemberg-Schnellin
    {
        "latitude": 52.027778,
        "longitude": 11.367778,
        "weight": 76.9,
    },  # Windfeld Sonnenberg
    {
        "latitude": 53.320833,
        "longitude": 12.026944,
        "weight": 75.2,
    },  # Windpark Jännersdorf
    {
        "latitude": 51.617222,
        "longitude": 8.803333,
        "weight": 75.05,
    },  # Windpark Altenautal
    {
        "latitude": 52.192500,
        "longitude": 11.368056,
        "weight": 71.3,
    },  # Windpar Bornstedt-Nordgermersleben-Rottmersleben-Schackensleben
    {
        "latitude": 51.642500,
        "longitude": 11.658333,
        "weight": 72,
    },  # Windpark Gerbstedt-Ihlewitz
    {
        "latitude": 49.964722,
        "longitude": 7.652500,
        "weight": 70.5,
    },  # Hunsrück-Windpark Ellern
    {"latitude": 52.867500, "longitude": 7.138889, "weight": 70.1},  # Windpark Haren
    {
        "latitude": 51.041111,
        "longitude": 6.530000,
        "weight": 67.2,
    },  # Windpark Königshovener Höhe
    {
        "latitude": 51.445278,
        "longitude": 8.696944,
        "weight": 65.95,
    },  # Windpark Madfeld-Bleiwäsche
    {
        "latitude": 53.817778,
        "longitude": 8.078889,
        "weight": 65.6,
    },  # Windpark Altenbruch
    {
        "latitude": 52.176389,
        "longitude": 11.300000,
        "weight": 64.1,
    },  # Windpark Hakenstedt
    {
        "latitude": 51.946111,
        "longitude": 14.462778,
        "weight": 64,
    },  # Windpark Cottbuser Halde
    {
        "latitude": 51.707778,
        "longitude": 12.239167,
        "weight": 62.7,
    },  # Windpark Thurland
    {
        "latitude": 49.689167,
        "longitude": 8.106944,
        "weight": 61,
    },  # Windfeld Rheinhessen-Pfalz
    {
        "latitude": 50.003333,
        "longitude": 7.386667,
        "weight": 59.8,
    },  # Windpark Kirchberg im Faas
    {
        "latitude": 51.040556,
        "longitude": 11.620278,
        "weight": 59.1,
    },  # Windpark Eckolstädt
    {
        "latitude": 51.247500,
        "longitude": 10.283889,
        "weight": 58.1,
    },  # Windpark Büttstedt
    {
        "latitude": 51.072778,
        "longitude": 11.789444,
        "weight": 58,
    },  # Windpark Molau-Leislau
    {
        "latitude": 54.483333,
        "longitude": 11.110000,
        "weight": 57.5,
    },  # Windpark Fehmarn-Mitte
    {"latitude": 49.830000, "longitude": 8.138889, "weight": 55},  # Windpark Wörrstadt
    {
        "latitude": 49.296111,
        "longitude": 9.415556,
        "weight": 54.9,
    },  # Windpark Harthäuser Wald
    {
        "latitude": 53.373333,
        "longitude": 9.496944,
        "weight": 52.9,
    },  # Windpark Ahrenswohlde-Wohnste
    {
        "latitude": 48.980833,
        "longitude": 11.102500,
        "weight": 52.8,
    },  # Windpark Raitenbucher Forst
    {
        "latitude": 48.740000,
        "longitude": 9.889722,
        "weight": 52.25,
    },  # Windpark Lauterstein
    {
        "latitude": 49.721111,
        "longitude": 7.721944,
        "weight": 52.2,
    },  # Windpark Lettweiler Höhe
    {
        "latitude": 50.603056,
        "longitude": 9.243889,
        "weight": 49.65,
    },  # Windpark Goldner Steinrück
    {
        "latitude": 50.516944,
        "longitude": 6.373611,
        "weight": 49.45,
    },  # Windpark Schleiden-Schöneseiffen
    {"latitude": 53.538889, "longitude": 8.952778, "weight": 48.8},  # Windpark Köhlen
    {
        "latitude": 49.764167,
        "longitude": 8.059722,
        "weight": 47.1,
    },  # Windpark Heimersheim
    {
        "latitude": 53.396667,
        "longitude": 14.169167,
        "weight": 46.3,
    },  # Windfeld Wolfsmoor
    {"latitude": 53.684167, "longitude": 8.646111, "weight": 46},  # Windpark Holßel
    {"latitude": 51.838333, "longitude": 12.875278, "weight": 44.9},  # Windpark Elster
    {"latitude": 52.002222, "longitude": 12.123056, "weight": 44.4},  # Windpark Zerbst
    {
        "latitude": 52.178333,
        "longitude": 11.886111,
        "weight": 43.6,
    },  # Windpark Stegelitz-Ziepel-Tryppehna
    {
        "latitude": 53.606944,
        "longitude": 8.793056,
        "weight": 43.2,
    },  # Windpark Kührstedt-Alfstedt
    {
        "latitude": 52.060111,
        "longitude": 14.381000,
        "weight": 43.2,
    },  # Windpark Ullersdorf
    {
        "latitude": 49.813333,
        "longitude": 8.017778,
        "weight": 42.4,
    },  # Windpark Gau-Bickelheim
    {
        "latitude": 51.422778,
        "longitude": 11.834444,
        "weight": 42,
    },  # Windpark Holleben-Bad Lauchstädt
    {
        "latitude": 54.648611,
        "longitude": 9.176389,
        "weight": 41.8,
    },  # Bürgerwindpark Löwenstedt
    {"latitude": 50.623861, "longitude": 9.153528, "weight": 41.6},  # Windpark Feldatal
    {"latitude": 51.413056, "longitude": 11.587222, "weight": 41},  # Windpark Farnstädt
    {
        "latitude": 52.976667,
        "longitude": 7.415833,
        "weight": 40.9,
    },  # Windpark Dörpen-Ost
    {
        "latitude": 52.878056,
        "longitude": 10.042778,
        "weight": 40.5,
    },  # Windpark Hermannsburg
    {
        "latitude": 52.900000,
        "longitude": 12.384167,
        "weight": 40.4,
    },  # Windpark Kyritz-Plänitz-Zernitz
    {"latitude": 52.597222, "longitude": 12.266667, "weight": 40},  # Windpark Stüdenitz
]
sun_parks = [
    {"latitude": 51.3167, "longitude": 12.3667, "weight": 605},  # Witznitz
    {"latitude": 51.3236, "longitude": 12.6511, "weight": 52},  # Waldpolenz Solar Park
    {
        "latitude": 51.7625,
        "longitude": 13.6000,
        "weight": 52.3,
    },  # Walddrehna Solar Park
    {"latitude": 53.9239, "longitude": 13.2256, "weight": 52},  # Tutow Solar Park
    {"latitude": 53.0290, "longitude": 13.5336, "weight": 128.5},  # Templin Solar Park
    {
        "latitude": 48.8031,
        "longitude": 12.7669,
        "weight": 54,
    },  # Strasskirchen Solar Park
    {"latitude": 53.6391, "longitude": 12.3643, "weight": 76},  # Solarpark Zietlitz
    {
        "latitude": 52.6475,
        "longitude": 13.6916,
        "weight": 187,
    },  # Solarpark Weesow-Willmersdorf
    {
        "latitude": 53.5267,
        "longitude": 11.6609,
        "weight": 172,
    },  # Solarpark Tramm-Göhten
    {"latitude": 48.6490, "longitude": 11.2782, "weight": 120},  # Solarpark Schornhof
    {"latitude": 51.5450, "longitude": 13.9800, "weight": 166},  # Solarpark Meuro
    {"latitude": 50.5960, "longitude": 9.3690, "weight": 54.7},  # Solarpark Lauterbach
    {"latitude": 52.6413, "longitude": 14.1923, "weight": 150},  # Solarpark Gottesgabe
    {"latitude": 53.3818, "longitude": 12.2688, "weight": 65},  # Solarpark Ganzlin
    {"latitude": 53.4148, "longitude": 12.2470, "weight": 90},  # Solarpark Gaarz
    {
        "latitude": 52.8253,
        "longitude": 13.6983,
        "weight": 84.7,
    },  # Solarpark Finow Tower
    {"latitude": 52.6975, "longitude": 14.2300, "weight": 150},  # Solarpark Alttrebbin
    {"latitude": 53.2000, "longitude": 12.5167, "weight": 67.8},  # Solarpark Alt Daber
    {
        "latitude": 52.6139,
        "longitude": 14.2425,
        "weight": 145,
    },  # Neuhardenberg Solar Park
    {
        "latitude": 51.9319,
        "longitude": 14.4072,
        "weight": 71.8,
    },  # Lieberose Photovoltaic Park
    {
        "latitude": 51.5686,
        "longitude": 13.7375,
        "weight": 80.7,
    },  # Finsterwalde Solar Park
    {"latitude": 54.6294, "longitude": 9.3433, "weight": 83.6},  # Eggebek Solar Park
    {
        "latitude": 52.4367,
        "longitude": 12.4514,
        "weight": 91,
    },  # Brandenburg-Briest Solarpark
]


# Return Dataframe containing Forecast Data for the given start_date and end_date or hours
def fetch_forecast(start_date, hours=None, end_date=None):
    client = setup_client()
    forecast_url = "https://api.open-meteo.com/v1/forecast"

    forecast_start_date, forecast_end_date = convert_dates(start_date, end_date, hours)

    params_template = {
        "start_date": forecast_start_date,
        "end_date": forecast_end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_100m",
            "direct_radiation",
        ],
    }
    coordinates_data = fetch_data(client, coordinates, params_template, forecast_url)

    params_template["hourly"] = ["wind_speed_100m"]
    wind_parks_data = fetch_data(client, wind_parks, params_template, forecast_url)

    params_template["hourly"] = ["direct_radiation"]
    sun_parks_data = fetch_data(client, sun_parks, params_template, forecast_url)

    weighted_wind_speed = calculate_weighted_averages(
        wind_parks_data, wind_parks, "wind_speed_100m", "weighted_wind_speed"
    )
    weighted_radiation = calculate_weighted_averages(
        sun_parks_data, sun_parks, "direct_radiation", "weighted_radiation"
    )

    coordinates_avg = coordinates_data.groupby("date").mean()
    combined_df = pd.concat(
        [coordinates_avg, weighted_wind_speed, weighted_radiation], axis=1
    )
    # Select only the required columns
    final_df = combined_df[
        ["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"]
    ]
    final_df.index.name = "Datetime"
    final_df.index = final_df.index.tz_localize(None)

    return final_df


# Fetch historical Data and append to CSV or create new CSV
def fetch_historical_weather():
    client = setup_client()
    forecast_url = "https://archive-api.open-meteo.com/v1/archive"
    historical_csv_file = dir + "germany_weather_average.csv"

    # Read the last date from the existing CSV file
    try:
        df_existing = pd.read_csv(historical_csv_file)
        df_existing = df_existing.iloc[:-1]  # Drop the last row
        last_date = pd.to_datetime(df_existing["date"]).max()
        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    except FileNotFoundError:
        df_existing = pd.DataFrame()
        start_date = "2018-01-01"  # Default start date if file does not exist

    end_date = datetime.utcnow().strftime("%Y-%m-%d")

    # Ensure end_date is greater than or equal to start_date
    if datetime.strptime(end_date, "%Y-%m-%d") < datetime.strptime(
        start_date, "%Y-%m-%d"
    ):
        end_date = start_date

    params_template = {
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "precipitation",
            "wind_speed_100m",
            "direct_radiation",
        ],
    }

    coordinates_data = fetch_data(client, coordinates, params_template, forecast_url)
    params_template["hourly"] = ["wind_speed_100m"]
    wind_parks_data = fetch_data(client, wind_parks, params_template, forecast_url)
    params_template["hourly"] = ["direct_radiation"]
    sun_parks_data = fetch_data(client, sun_parks, params_template, forecast_url)

    weighted_wind_speed = calculate_weighted_averages(
        wind_parks_data, wind_parks, "wind_speed_100m", "weighted_wind_speed"
    )
    weighted_radiation = calculate_weighted_averages(
        sun_parks_data, sun_parks, "direct_radiation", "weighted_radiation"
    )

    coordinates_avg = coordinates_data.groupby("date").mean()
    combined_df = pd.concat(
        [coordinates_avg, weighted_wind_speed, weighted_radiation], axis=1
    )

    # Select only the required columns
    final_df = combined_df[
        ["temperature_2m", "precipitation", "wind_speed_100m", "direct_radiation"]
    ]
    final_df.reset_index(inplace=True)

    # Filter out data that already exists in the CSV
    if not df_existing.empty:
        final_df = final_df[final_df["date"] > last_date]

    # Drop rows with all NaN values
    final_df.dropna(how="all", inplace=True)

    # Check if the file exists and is not empty
    file_exists = os.path.isfile(historical_csv_file)
    if file_exists:
        with open(historical_csv_file, "r") as f:
            file_empty = len(f.read().strip()) == 0
    else:
        file_empty = True

    # Append new data to the existing CSV file
    final_df.to_csv(
        historical_csv_file, index=False, mode="a", header=not file_exists or file_empty
    )
    df = pd.read_csv(historical_csv_file)

    # Drop all rows with any NaN values
    df_cleaned = df.dropna()

    # Drop duplicate rows
    df_cleaned.drop_duplicates(inplace=True)

    # Save the cleaned data back to CSV
    df_cleaned.to_csv(historical_csv_file, index=False)
    print(f"Historical data appended to {historical_csv_file}.")

def update_e_price_data(csv_dir: str = dir,csv_filename: str = "day_ahead_energy_prices.csv",start_date_str: str = "2018-10-01"):
    """
    Fetches entire weeks of day-ahead energy price data from SMARD.de.
    - If the CSV does not exist or is empty, fetches all weeks from start_date to today.
    - If the CSV exists, fetches only missing weeks.
    - Ensures correct handling of DST shifts.
    - Saves data without duplicates.

    Parameters
    ----------
    csv_dir : str
        Directory where the CSV file is stored.
    csv_filename : str
        Name of the CSV file.
    start_date_str : str
        Earliest date to fetch if the file is missing or empty.
    """
    csv_file = os.path.join(csv_dir, csv_filename)
    file_exists = os.path.isfile(csv_file)

    # Check if the file exists and has data
    if file_exists and os.path.getsize(csv_file) > 0:
        existing_df = pd.read_csv(csv_file, parse_dates=["Datetime"], index_col="Datetime")
        existing_df.sort_index(inplace=True)
        last_date_in_file = existing_df.index[-1].date()
        fetch_start_date = last_date_in_file + timedelta(days=1)  # Start from the next missing day
        print(f"CSV found. Last date in file: {last_date_in_file}. Fetching missing weeks from {fetch_start_date}...")
    else:
        print(f"CSV not found or empty. Fetching all data from {start_date_str}...")
        fetch_start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        existing_df = pd.DataFrame()  # No existing data

    # Fetch full weeks at a time
    today = datetime.now().date()
    list_of_weekly_dfs = []

    current_date = fetch_start_date
    while current_date <= today:
        try:
            weekly_df = fetch_full_week_data(current_date)
            if weekly_df is not None and not weekly_df.empty:
                list_of_weekly_dfs.append(weekly_df)
        except Exception as e:
            print(f"Failed to fetch data for week starting {current_date}: {e}")

        time.sleep(0.5)  # Pause to avoid spamming API
        current_date += timedelta(weeks=1)

    # Merge new data if available
    if not list_of_weekly_dfs:
        print("No new data fetched. Exiting...")
        return

    new_data_df = pd.concat(list_of_weekly_dfs, axis=0)
    new_data_df.sort_index(inplace=True)

    # Merge with existing data
    combined_df = pd.concat([existing_df, new_data_df], axis=0)
    combined_df = combined_df[~combined_df.index.duplicated(keep="first")]
    combined_df.sort_index(inplace=True)

    save_to_csv(combined_df, csv_file)

    print(f"Appended {len(new_data_df)} new rows. Final dataset contains {len(combined_df)} rows.")
    print("Update complete.")


def fetch_full_week_data(target_date):
    """
    Fetches an entire week's day-ahead energy prices from SMARD.de,
    converting timestamps to Berlin local time.

    Parameters
    ----------
    target_date : datetime.date
        A date in the week to fetch (any date will be rounded to the start of its week).

    Returns
    -------
    pd.DataFrame or None
        A DataFrame with index=Datetime (local Berlin time) and one column:
        "hourly day-ahead energy price". The DataFrame will contain 168 hours
        unless there are missing hours from the API.
    """
    local_tz = pytz.timezone("Europe/Berlin")

    # Determine the Monday 00:00 of the target week
    dt_midnight_local = local_tz.localize(datetime(target_date.year, target_date.month, target_date.day, 0, 0))
    days_to_subtract = dt_midnight_local.weekday()
    monday_local = (dt_midnight_local - timedelta(days=days_to_subtract)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    # Convert Monday local time to UTC for SMARD request
    monday_utc = monday_local.astimezone(pytz.UTC)
    monday_ms = int(monday_utc.timestamp() * 1000)  # SMARD requires milliseconds

    # Fetch data from SMARD API
    url = f"https://www.smard.de/app/chart_data/4169/DE/4169_DE_hour_{monday_ms}.json"
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch data from SMARD API for week starting {monday_local.date()}: {response.status_code}")
        return None

    json_data = response.json()
    if "series" not in json_data or not json_data["series"]:
        print(f"No 'series' data found for week starting {monday_local.date()}.")
        return None

    # Convert timestamps from UTC → Berlin local time
    data_rows = []
    for (ts_ms, price_val) in json_data["series"]:
        if price_val is None or price_val == "":
            price_val = float("nan")  # Mark as NaN instead of dropping

        dt_utc = datetime.utcfromtimestamp(ts_ms / 1000.0).replace(tzinfo=pytz.UTC)
        dt_local = dt_utc.astimezone(local_tz).replace(tzinfo=None)
        data_rows.append((dt_local, price_val))

    if not data_rows:
        return None

    # Create DataFrame, indexed by local time
    df_week = pd.DataFrame(data_rows, columns=["Datetime", "hourly day-ahead energy price"])
    df_week.sort_values("Datetime", inplace=True)
    df_week.set_index("Datetime", inplace=True)

    print(f"Fetched {len(df_week)} rows for week starting {monday_local.date()}.")
    return df_week


def save_to_csv(df: pd.DataFrame, csv_path: str):
    """
    Saves a DataFrame to CSV with a standardized date format.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing hourly day-ahead energy price data.
    csv_path : str
        File path where the CSV should be saved.
    """
    df.sort_index(inplace=True)
    df.to_csv(csv_path, date_format="%Y-%m-%dT%H:%M:%S")


def update_e_mix_data(csv_path=dir + "hourly_market_mix_cleaned.csv"):

    mix_categories = [
        "Biomass",
        "Hard Coal",
        "Hydro",
        "Lignite",
        "Natural Gas",
        "Nuclear",
        "Other",
        "Pumped storage generation",
        "Solar",
        "Wind offshore",
        "Wind onshore",
    ]

    url = "https://api.agora-energy.org/api/raw-data"
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Api-key": "agora_live_62ce76dd202927.67115829",
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

    slices = [
        (max(fetch_start, datetime(y, 1, 1)), min(fetch_end, datetime(y + 1, 1, 1)))
        for y in range(fetch_start.year, fetch_end.year + 1)
    ]

    data_dict = {}
    for start, end in slices:
        payload = {
            "filters": {
                "from": start.strftime("%Y-%m-%d"),
                "to": end.strftime("%Y-%m-%d"),
                "generation": mix_categories,
            },
            "x_coordinate": "date_id",
            "y_coordinate": "value",
            "view_name": "live_gen_plus_emi_de_hourly",
            "kpi_name": "power_generation",
            "z_coordinate": "generation",
        }
        response = requests.post(url, headers=headers, json=payload)
        time.sleep(0.3)
        if response.status_code != 200:
            print(f"Request failed for {start} to {end}")
            continue

        for ts, value, category in response.json().get("data", {}).get("data", []):
            if category in mix_categories:
                if value is None:
                    print(
                        f"Warning: Null value for {ts} - {category}, replacing with 0.0"
                    )
                data_dict.setdefault(ts, {}).update(
                    {category: float(value) if value is not None else 0.0}
                )

        # Ensure every timestamp has all 11 categories
        for ts in data_dict.keys():
            for cat in mix_categories:
                if cat not in data_dict[ts]:
                    print(f"Warning: Missing category {cat} for {ts}, filling with 0.0")
                    data_dict[ts][cat] = 0.0  # Fill missing values

    new_rows = [
        [ts] + [data_dict.get(ts, {}).get(cat, 0.0) for cat in mix_categories]
        for ts in sorted(data_dict)
        if ts not in existing_timestamps
    ]
    if new_rows:
        df_new = pd.DataFrame(new_rows, columns=["Timestamp"] + mix_categories)
        df = pd.concat([df, df_new], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"Added {len(new_rows)} new rows.")
    else:
        print("No new data added.")
