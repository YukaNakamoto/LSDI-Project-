import pandas as pd
from datetime import timedelta, datetime
import numpy as np

dir = "./data/"

def fill_up_e_prices():

    path = dir + 'day_ahead_energy_prices.csv'
    e_price_df = pd.read_csv(path, delimiter=",")
    e_price_df = e_price_df.set_index('Datetime')
    e_price_df.index = pd.to_datetime(e_price_df.index)
    
    
    estimations_df = get_estimations(e_price_df.copy(), e_price_df.index[-1], col_name="hourly day-ahead energy price", count=None, final_date=datetime(2025, 2, 18, 23, 0, 0))
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
    