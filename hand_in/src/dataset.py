from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

from src.plot import plot_correlation_line

dir = "./data/"


def get_e_price_df(predict) -> pd.DataFrame:
    file = "day_ahead_energy_prices.csv" if not predict else "filled_day_ahead_energy_prices.csv"
    e_price_df = pd.read_csv(dir + file, delimiter=",")

    e_price_df = e_price_df.set_index("Datetime")
    e_price_df.index = pd.to_datetime(e_price_df.index)
    e_price_df.rename(columns={"hourly day-ahead energy price": "Price"}, inplace=True)

    return e_price_df


def get_mix_df(predict) -> pd.DataFrame:
    file = "hourly_market_mix_cleaned.csv" if not predict else "filled_hourly_market_mix_cleaned.csv"

    cols = [
            "Timestamp",
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
    
    if predict:
        cols = [
             "Timestamp", "Hydro", "Pumped storage generation", "Solar", "Wind offshore", "Wind onshore",
        ]




    mix_df = pd.read_csv(
        dir + file,
        usecols=cols,
        delimiter=",",
    )
    mix_df.rename(columns={"Timestamp": "Datetime"}, inplace=True)

    mix_df.set_index("Datetime", inplace=True)
    mix_df.index = pd.to_datetime(mix_df.index)

    return mix_df


def get_avg_weather_data(predict) -> pd.DataFrame:
    file = "germany_weather_average.csv" if not predict else "filled_germany_weather_average.csv"
    avg_weather_df = pd.read_csv(dir + file, delimiter=",")

    avg_weather_df.set_index("date", inplace=True)
    avg_weather_df.index = pd.to_datetime(avg_weather_df.index)
    avg_weather_df.index = avg_weather_df.index.tz_localize(None)
    avg_weather_df.rename(
        columns={
            "Precipitation (rain/snow)": "precipitation",
            "wind_speed_100m": "wind_speed_100m",
            "Global Horizontal Irradiance": "shortwave_radiation",
            "direct_radiation": "direct_radiation",
        },
        inplace=True,
    )

    return avg_weather_df


def get_datasets(predict=False) -> (
    pd.DataFrame
):  # 12:00 of every day -> add 1.5 days of values to e.g. 17.02 12:00 -> 19.02 00:00
    dfs_to_merge = []

    e_price_df = get_e_price_df(predict)
    dfs_to_merge.append(e_price_df)

    mix_df = get_mix_df(predict)
    dfs_to_merge.append(mix_df)

    avg_weather_df = get_avg_weather_data(predict)
    dfs_to_merge.append(avg_weather_df)

    dfs_to_merge = [df[~df.index.duplicated(keep="first")] for df in dfs_to_merge]

    merged_df = pd.concat(dfs_to_merge, axis=1, join="inner").dropna(axis=0)
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


def unnormalize(predictions, scaler):
    if isinstance(predictions, (pd.DataFrame, pd.Series)):
        index = predictions.index  # Store the index
        values = predictions.values.reshape(
            -1, 1
        )  # Ensure correct shape for inverse transform
        unnormalized = scaler.inverse_transform(values)
        return (
            pd.Series(unnormalized.ravel(), index=index, name=predictions.name)
            if isinstance(predictions, pd.Series)
            else pd.DataFrame(unnormalized, index=index, columns=predictions.columns)
        )
    else:
        return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))


def correlation_analysis():
    price_df = pd.read_csv("../data/day_ahead_energy_prices.csv")
    price_df.rename(
        columns={"Datetime": "Timestamp", "hourly day-ahead energy price": "Price"},
        inplace=True,
    )

    min_price = -74.44
    max_price = 222.01

    # Filter the dataframe to remove rows with energy prices outside the range
    price_df = price_df[
        (price_df["Price"] >= min_price) & (price_df["Price"] <= max_price)
    ]

    # energy_mix_df = pd.read_csv("../data/hourly_market_mix_delta_abs.csv")
    energy_mix_df = pd.read_csv("../data/hourly_market_mix_cleaned.csv")
    # Merge datasets on timestamp
    merged_df = pd.merge(energy_mix_df, price_df, on="Timestamp", how="inner")

    # Set the Timestamp column as the index
    merged_df.set_index("Timestamp", inplace=True)

    # Calculate the sum of the specified columns
    min_price = -74.44
    max_price = 222.01

    # Filter the dataframe to remove rows with energy prices outside the range
    merged_df = merged_df[
        (merged_df["Price"] >= min_price) & (merged_df["Price"] <= max_price)
    ]

    merged_df["Renewable_Sum"] = merged_df[
        ["Solar", "Wind offshore", "Wind onshore", "Hydro"]
    ].sum(axis=1)
    # Plot the relationship with a correlation line
    plot_correlation_line(merged_df)

    r, p_two_tailed = pearsonr(
        merged_df[["Solar", "Wind offshore", "Wind onshore", "Hydro"]].sum(axis=1),
        merged_df["Price"],
    )

    if p_two_tailed == 0.0:
        p_two_tailed = np.finfo(float).eps  # Assign the smallest positive float value

    p_one_tailed = p_two_tailed / 2

    print("Test Score: ", r)
    print("One Tailed P-Value: ", p_one_tailed)
