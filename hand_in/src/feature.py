def split(DATASET, eval_size):
    """
    Split time series data into train, eval, and test sets, ensuring:
    - Train set includes t0.
    - Test set includes tn.
    - Eval set is positioned between train and test sets.
    
    Parameters:
    - DATASET: Pandas DataFrame or Series, indexed by datetime.
    - TEST_SIZE: Proportion of the dataset to use for the test set (default: 0.16).
    - VAL_SIZE: Proportion of the dataset to use for the eval set (default: 0.02).
    - predict_last_day: If True, use the last 24 hours as the test set.
    
    Returns:
    - train: Training set
    - eval: Evaluation set
    - test: Test set
    - SPLIT_DATE_EVAL: Date where train and eval split occurs
    - SPLIT_DATE_TEST: Date where eval and test split occurs
    """
    n = len(DATASET)
    
    test_size = 24
    remainder_size = n - test_size
    eval_size = int(eval_size * remainder_size)
    train_size = remainder_size - eval_size
    
    SPLIT_DATE_EVAL = DATASET.index[train_size]
    SPLIT_DATE_TEST = DATASET.index[-24]
    
    train = DATASET.iloc[:train_size]
    eval = DATASET.iloc[train_size:-24]
    test = DATASET.iloc[-24:]
    
    return train, eval, test, SPLIT_DATE_EVAL, SPLIT_DATE_TEST


def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    for window in [3, 6, 12]: # hours
        df[f'ma_{int(window)}_hours'] = df['Price'].rolling(window=window).mean()
    
    for window in [24, 72, 168]:  # 1 day 3 days, and 1 week
        df[f'ma_{int(window / 24)}_days'] = df['Price'].rolling(window=window).mean()

    return df


def create_normalized_features(df, scalar):
    df = df.copy()
    df['hour'] = scalar.transform(df.index.hour.to_numpy()[:, None])
    df['dayofweek'] = scalar.transform(df.index.dayofweek.to_numpy()[:, None])
    df['dayofyear'] = scalar.transform(df.index.dayofyear.to_numpy()[:, None])
    df['weekofyear'] = scalar.transform(df.index.isocalendar().week.to_numpy()[:, None])

    for window in [3, 6, 12]: # hours
        df[f'ma_{int(window)}_hours'] = df['Price'].rolling(window=window).mean()
    
    for window in [24, 72, 168]:  # 1 day 3 days, and 1 week
        df[f'ma_{int(window / 24)}_days'] = df['Price'].rolling(window=window).mean()

    return df