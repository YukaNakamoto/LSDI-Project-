import pandas as pd


#  hourly_index = None
#     if prediction_date:
#         prediction_date = prediction_date.strftime("%Y-%m-%d")
        
#         hourly_index = pd.date_range(
#             start=f"{prediction_date} 00:00:00",
#             end=f"{prediction_date} 23:59:59",
#             freq="h"
#         )


def split(DATASET, eval_size, test_ratio=0.15):
    """
    Split time series data into train, eval, and test sets, ensuring:
    - Train set includes t0.
    - Test set includes the final 15% of the dataset (excluding the last date).
    - Eval set is positioned between train and test sets.
    
    Parameters:
    - DATASET: Pandas DataFrame or Series, indexed by datetime.
    - eval_size: Proportion of the dataset to use for the eval set (relative to training size).
    - test_ratio: Proportion of the dataset to use for the test set (default: 0.15).
    
    Returns:
    - train: Training set
    - eval: Evaluation set
    - test: Test set
    - SPLIT_DATE_EVAL: Date where train and eval split occurs
    - SPLIT_DATE_TEST: Date where eval and test split occurs
    """
    # Ensure dataset is sorted by datetime index
    DATASET = DATASET.sort_index()
    
    # Get unique dates in dataset
    unique_dates = DATASET.index.date
    last_month = 24 * 31
    last_date = unique_dates[-last_month]  # Exclude the last month for benchmarking
    print(last_date)
    
    # Exclude last date from test set allocation
    dataset_excl_last_month = DATASET[DATASET.index.date < last_date]
    
    n = len(dataset_excl_last_month)
    # test_size = int(test_ratio * n)
    remainder_size = n - 24
    eval_size = int(eval_size * remainder_size)
    train_size = remainder_size - eval_size
    
    SPLIT_DATE_EVAL = dataset_excl_last_month.index[train_size]
    SPLIT_DATE_TEST = dataset_excl_last_month.index[train_size + eval_size]
    
    train = dataset_excl_last_month.iloc[:train_size]
    eval = dataset_excl_last_month.iloc[train_size:train_size + eval_size]
    test = dataset_excl_last_month.iloc[train_size + eval_size:]
    
    return train, eval, test, SPLIT_DATE_EVAL, SPLIT_DATE_TEST


def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    for window in range(2, 24): # hours
        df[f'ma_{int(window)}_hours'] = df['Price'].rolling(window=window).mean()
        df[f'ma_{int(window)}_hours_pumped_storage_generation'] = df['Pumped storage generation'].rolling(window=window).mean()
    
    for window in [24, 72, 168]:  # 1 day 3 days, and 1 week
        df[f'ma_{int(window / 24)}_days'] = df['Price'].rolling(window=window).mean()

    return df