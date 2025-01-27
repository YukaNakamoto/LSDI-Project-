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


def plot_merged_datasets(merged_df):
    sns.set_theme()

    fig, axs = plt.subplots(nrows=5, figsize=(15, 55))

    train, eval, test, SPLIT_DATE_EVAL, SPLIT_DATE_TEST = split(merged_df)

    train["Price"].plot(ax=axs[0], label='Training Set', title='Hourly Next-Day Energy Price Train/Evaluation/Test Split')
    test["Price"].plot(ax=axs[0], label='Test Set', color="red")
    eval["Price"].plot(ax=axs[0], label='Evaluation Set', color="orange")

    axs[0].set_yticks(np.arange(-500, 900, 50))
    axs[0].axvline(SPLIT_DATE_EVAL, color='orange', ls='--')
    axs[0].axvline(SPLIT_DATE_TEST, color='red', ls='--')
    axs[0].legend(['Training Set', 'Test Set', 'Evaluation Set'])


    cols = [
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
        "Wind onshore"
        ]

    plt.style.use('default')

    filtered_train = train[train.index.hour == 12][::-1]
    filtered_train[cols].plot(ax=axs[1], kind="bar", stacked=True, title='Energy Mix at 12:00 - Train Split', width=1.0)
    axs[1].set_xticks(np.arange(0, len(filtered_train), 30))
    axs[1].set_xticklabels(filtered_train.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')

    filtered_eval = eval[eval.index.hour == 12][::-1]
    filtered_eval[cols].plot(ax=axs[2], kind="bar", stacked=True, title='Daily Energy Mix at 12:00  - Eval Split', width=1.0)
    axs[2].set_xticks(np.arange(0, len(filtered_eval), 30))
    axs[2].set_xticklabels(filtered_eval.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')


    filtered_test = test[test.index.hour == 12][::-1]
    filtered_test[cols].plot(ax=axs[3], kind="bar", stacked=True, title='Daily Energy Mix at 12:00  - Test Split', width=1.0)
    axs[3].set_xticks(np.arange(0, len(filtered_test), 30))
    axs[3].set_xticklabels(filtered_test.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')

    filtered_test = test.head(25)[::-1]
    # print(filtered_test)
    filtered_test[cols].plot(ax=axs[4], kind="bar", stacked=True, title='Energy Mix on the 29. October 2024 - Test Split', width=1.0)
    axs[4].set_xticks(np.arange(0, len(filtered_test), 1))
    axs[4].set_xticklabels(filtered_test.index.strftime('%H:%M'), rotation=45, ha='right')

    plt.subplots_adjust(hspace=0.3)
    plt.show()
