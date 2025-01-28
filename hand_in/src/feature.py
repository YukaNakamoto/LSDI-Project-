def split(DATASET, eval_size, prediction_date):
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
    if prediction_date:
        hourly_index = pd.date_range(
            start=f"{prediction_date} 00:00:00",
            end=f"{prediction_date} 23:59:59",
            freq="H"
        )

        # Adjust dataset to remove any timestamps beyond hourly_index
        max_hourly_time = hourly_index[-1]
        DATASET = DATASET[DATASET.index <= max_hourly_time]

    n = len(DATASET)
    test_size = 24

    # Ensure there's enough data for splitting
    if n < test_size:
        raise ValueError("Not enough data for splitting into train, eval, and test sets.")

    remainder_size = n - test_size
    eval_size = int(eval_size * remainder_size)
    train_size = remainder_size - eval_size

    SPLIT_DATE_EVAL = DATASET.index[train_size]
    SPLIT_DATE_TEST = DATASET.index[-test_size]

    train = DATASET.iloc[:train_size]
    eval = DATASET.iloc[train_size:-test_size]
    test = DATASET.iloc[-test_size:]

    # Use hourly_index for the test set
    test = test.loc[test.index.intersection(hourly_index)]

    # Adjust eval and train sets to account for changes to test set
    if not test.empty:
        latest_test_time = test.index[0]
        eval = eval[eval.index < latest_test_time]
        train = train[train.index < latest_test_time]

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