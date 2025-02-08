import pandas as pd

from src.dataset import normalize_data


def create_features_and_split_datasets(
    merged_df,
    default_features,
    selected_features,
    min_price,
    max_price,
    e_price_df,
    normalize,
    eval_size,
):

    features_df = create_features(merged_df)
    features_interval_df = features_df.copy()[
        (features_df["Price"] >= min_price) & (features_df["Price"] <= max_price)
    ]

    (
        normalized_features_df,
        normalized_e_price_df,
        normalized_interval_df,
        e_price_scaler,
    ) = normalize_data(features_df, e_price_df, features_interval_df.copy())

    TARGET = merged_df["Price"]
    TARGET_INTERVAL = features_interval_df["Price"]
    TARGET_NORMALIZED = normalized_features_df["Price"]
    TARGET_NORMALIZED_INTERVAL = normalized_interval_df["Price"]

    FEATURES = default_features if not selected_features else selected_features

    combined_df = pd.concat([features_df[FEATURES], TARGET], axis=1)
    combined_interval_df = pd.concat(
        [features_interval_df[FEATURES], TARGET_INTERVAL], axis=1
    )
    combined_normalized_df = pd.concat(
        [normalized_features_df[FEATURES], TARGET_NORMALIZED], axis=1
    )
    combined_normalized_interval_df = pd.concat(
        [normalized_interval_df[FEATURES], TARGET_NORMALIZED_INTERVAL], axis=1
    )

    if normalize:
        (
            train,
            eval,
            test,
            benchmark,
            SPLIT_DATE_EVAL,
            SPLIT_DATE_TEST,
            SPLIT_DATE_BENCHMARK,
        ) = split(combined_normalized_df, eval_size)
        (
            train_interval,
            eval_interval,
            test_interval,
            benchmark_interval,
            SPLIT_DATE_EVAL_INTERVAL,
            SPLIT_DATE_TEST_INTERVAL,
            SPLIT_DATE_BENCHMARK,
        ) = split(combined_normalized_interval_df, eval_size)
        scalar = e_price_scaler

    else:
        (
            train,
            eval,
            test,
            benchmark,
            SPLIT_DATE_EVAL,
            SPLIT_DATE_TEST,
            SPLIT_DATE_BENCHMARK,
        ) = split(combined_df.copy(), eval_size)
        (
            train_interval,
            eval_interval,
            test_interval,
            benchmark_interval,
            SPLIT_DATE_EVAL_INTERVAL,
            SPLIT_DATE_TEST_INTERVAL,
            SPLIT_DATE_BENCHMARK,
        ) = split(combined_interval_df, eval_size)
        scalar = None
    return (
        train,
        eval,
        test,
        benchmark,
        SPLIT_DATE_EVAL,
        SPLIT_DATE_TEST,
        SPLIT_DATE_BENCHMARK,
        train_interval,
        eval_interval,
        test_interval,
        benchmark_interval,
        SPLIT_DATE_EVAL_INTERVAL,
        SPLIT_DATE_TEST_INTERVAL,
        SPLIT_DATE_BENCHMARK,
        scalar,
        FEATURES,
    )


def split(DATASET, eval_size=0.1, test_ratio=0.2):
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
    LAST_DATE = unique_dates[-last_month]  # Exclude the last month for benchmarking

    # Exclude last date from test set allocation
    dataset_excl_last_month = DATASET[DATASET.index.date < LAST_DATE]
    benchmark = DATASET[DATASET.index.date >= LAST_DATE]

    n = len(dataset_excl_last_month)
    test_size = int(test_ratio * n)
    remainder_size = n - test_size
    eval_size = int(eval_size * remainder_size)
    train_size = remainder_size - eval_size

    SPLIT_DATE_EVAL = dataset_excl_last_month.index[train_size]
    SPLIT_DATE_TEST = dataset_excl_last_month.index[train_size + eval_size]

    train = dataset_excl_last_month.iloc[:train_size]
    eval = dataset_excl_last_month.iloc[train_size : train_size + eval_size]
    test = dataset_excl_last_month.iloc[train_size + eval_size :]

    return train, eval, test, benchmark, SPLIT_DATE_EVAL, SPLIT_DATE_TEST, LAST_DATE


def create_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["dayofyear"] = df.index.dayofyear
    df["weekofyear"] = df.index.isocalendar().week

    for window in range(2, 24):  # hours
        df[f"ma_{int(window)}_hours"] = df["Price"].rolling(window=window).mean()
        df[f"ma_{int(window)}_hours_pumped_storage_generation"] = (
            df["Pumped storage generation"].rolling(window=window).mean()
        )

    for window in [24, 72, 168]:  # 1 day 3 days, and 1 week
        df[f"ma_{int(window / 24)}_days"] = df["Price"].rolling(window=window).mean()

    return df


def get_sets(train, eval, test, train_interval, eval_interval, test_interval, FEATURES):
    TARGET = "Price"
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    X_val = eval[FEATURES]
    y_val = eval[TARGET]

    X_train_interval = train_interval[FEATURES]
    y_train_interval = train_interval[TARGET]

    X_test_interval = test_interval[FEATURES]
    y_test_interval = test_interval[TARGET]

    X_val_interval = eval_interval[FEATURES]
    y_val_interval = eval_interval[TARGET]
    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        X_train_interval,
        y_train_interval,
        X_val_interval,
        y_val_interval,
        X_test_interval,
        y_test_interval,
    )
