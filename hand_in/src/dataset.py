import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def adjust_set_sizes(train, val, test):
    global train_val_test_sets_sizes  # Access the global variable
    total = train + val + test
    train_norm, val_norm, test_norm = np.array([train, val, test]) / total
    train_val_test_sets_sizes = [train_norm, val_norm, test_norm]
    return train_val_test_sets_sizes

def get_e_price_dataset() -> pd.DataFrame:
    e_price_df = pd.read_csv('../data/day_ahead_energy_prices.csv', delimiter=",")

    # new_e_prices = get_new_e_prices() // TODO:
    # e_price_df = e_price_df.stack(new_e_prices) // TODO:

    e_price_df = e_price_df.set_index('Datetime')
    e_price_df.index = pd.to_datetime(e_price_df.index)
    e_price_df.rename(columns={"hourly day-ahead energy price": "Price"}, inplace=True)

    return e_price_df

def get_merged_dataset() -> pd.DataFrame:
    dfs_to_merge = []

    dfs_to_merge.append(get_e_price_dataset())
    
    mix_df = pd.read_csv('../data/hourly_market_mix_cleaned.csv', delimiter=",")

    # new_mix_df = get_new_mix_df() // TODO:
    # e_price_df = e_price_df.stack(new_mix_df) // TODO:

    mix_df.set_index("Timestamp", inplace=True)
    mix_df.index = pd.to_datetime(mix_df.index)
    mix_df.rename_axis("Datetime", inplace=True) 
    mix_df = mix_df.loc[:, mix_df.max() >= 0.01]
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
    
    return merged_df


def standardize_data(merged_df, e_price_df):
    scaler = StandardScaler() #z-transformation
    z_merged_df = pd.DataFrame(
        scaler.fit_transform(merged_df.copy()),
        columns=merged_df.columns,
        index=merged_df.index,
    )

    actual_price_scaler = StandardScaler() #z-transformation
    z_actual_price_df = pd.DataFrame(
        actual_price_scaler.fit_transform(e_price_df.copy()),
        columns=e_price_df.columns,
        index=e_price_df.index,
    )

    return z_merged_df, z_actual_price_df, actual_price_scaler

def unstandardized_actual_price(z_y_test, z_y_pred_e, actual_price_scaler):
        y_test = actual_price_scaler.inverse_transform([z_y_test]).flatten()
        y_predicted = actual_price_scaler.inverse_transform([z_y_pred_e]).flatten()

        return y_test, y_predicted




    
