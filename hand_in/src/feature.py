def split(DATASET, TEST_SIZE=0.16, VAL_SIZE=0.02, predict_last_day=False):
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
    
    if predict_last_day:
        # Use last 24 hours as test set
        test_size = 24
        train_size = n - test_size
        eval_size = int(VAL_SIZE * train_size)
        
        SPLIT_DATE_EVAL = DATASET.index[train_size - eval_size]
        SPLIT_DATE_TEST = DATASET.index[-24]
        
        train = DATASET.iloc[:train_size - eval_size]
        eval = DATASET.iloc[train_size - eval_size:train_size]
        test = DATASET.iloc[-24:]
    
    else:
        # Calculate train, eval, and test sizes based on proportions
        train_size = int((1 - TEST_SIZE - VAL_SIZE) * n)
        eval_size = int(VAL_SIZE * n)
        test_size = n - train_size - eval_size
        
        SPLIT_DATE_EVAL = DATASET.index[train_size]
        SPLIT_DATE_TEST = DATASET.index[train_size + eval_size]
        
        train = DATASET.iloc[:train_size]
        eval = DATASET.iloc[train_size:train_size + eval_size]
        test = DATASET.iloc[train_size + eval_size:]
    
    return train, eval, test, SPLIT_DATE_EVAL, SPLIT_DATE_TEST


def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week

    german_holidays = holidays.Germany(years=[2018,2019,2020,2021,2022,2023,2024,2025])
    df['is_public_holiday'] = df.index.to_series().apply(lambda date: int(date in german_holidays))

    for window in [3, 6, 12]: # hours
        df[f'ma_{int(window)}_hours'] = df['Price'].rolling(window=window).mean()
        df[f'biomass_ma_{int(window)}_hours'] = df['Biomass'].rolling(window=window).mean()
        df[f'pumped_storage_generation_ma_{int(window)}_hours'] = df['Pumped storage generation'].rolling(window=window).mean()
        df[f'hard_coal_ma_{int(window)}_hours'] = df['Hard Coal'].rolling(window=window).mean()
        df[f'lignite_ma_{int(window)}_hours'] = df["Lignite"].rolling(window=window).mean()
        df[f'natural_gas_ma_{int(window)}_hours'] = df["Natural Gas"].rolling(window=window).mean()
        df[f'nuclear_ma_{int(window)}_hours'] = df["Nuclear"].rolling(window=window).mean()
        df[f'other_ma_{int(window)}_hours'] = df[ "Other"].rolling(window=window).mean()
        df[f'solar_ma_{int(window)}_hours'] = df['Solar'].rolling(window=window).mean()
        df[f'wind_offshore_ma_{int(window)}_hours'] = df["Wind offshore"].rolling(window=window).mean()
        df[f'wind_onshore_ma_{int(window)}_hours'] = df["Wind onshore"].rolling(window=window).mean()
    
    for window in [24, 72, 168]:  # 1 day 3 days, and 1 week
        df[f'ma_{int(window / 24)}_days'] = df['Price'].rolling(window=window).mean()
        df[f'biomass_ma_{int(window / 24)}_days'] = df['Biomass'].rolling(window=window).mean()
        df[f'pumped_storage_generation_ma_{int(window / 24)}_days'] = df['Pumped storage generation'].rolling(window=window).mean()
        df[f'hard_coal_ma_{int(window / 24)}_days'] = df['Hard Coal'].rolling(window=window).mean()

    return df