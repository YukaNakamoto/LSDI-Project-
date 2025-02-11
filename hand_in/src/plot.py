from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


COLS = [
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


def plot_price_split(
    train,
    eval,
    test,
    benchmark,
    SPLIT_DATE_EVAL,
    SPLIT_DATE_TEST,
    SPLIT_DATE_BENCHMARK,
    title,
):
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(20, 11))
    train["Price"].plot(
        ax=ax,
        label="Training Set",
        title=f"Hourly Next-Day Energy Price Train/Evaluation/Test Split - {title}",
    )
    test["Price"].plot(ax=ax, label="Test Set", color="red")
    eval["Price"].plot(ax=ax, label="Evaluation Set", color="orange")
    benchmark["Price"].plot(ax=ax, label="Benchmark Set", color="green")

    ax.axvline(SPLIT_DATE_EVAL, color="orange", ls="--")
    ax.axvline(SPLIT_DATE_TEST, color="red", ls="--")
    ax.axvline(SPLIT_DATE_BENCHMARK, color="green", ls="--")
    ax.legend()
    plt.show()


def feature_importance(models):
    sns.set_theme()
    fig, axs = plt.subplots(ncols=2, figsize=(16, 6))

    for i, objective in enumerate(["MSE", "MAE"]):
        fi = pd.DataFrame(
            data=models[i].feature_importances_,
            index=models[i].feature_names_in_,
            columns=["importance"],
        ).sort_values("importance")

        fi.plot(ax=axs[i], kind="barh", legend=False)

        axs[i].set_title(f"Feature Importance - Objective: {objective}", fontsize=14)
        axs[i].set_xlabel("Importance", fontsize=12)
        axs[i].set_ylabel("")  # Remove redundant ylabel
        axs[i].tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()  # Adjust layout to prevent overlapping
    plt.show()


def plot_predicted(test_set, predictions):
    fig, ax = plt.subplots(figsize=(20, 6))
    test_set = test_set.iloc[-24:]
    # Extract hours from the datetime index
    hours = test_set.index.hour

    ax.step(hours, test_set, lw=3, where="mid", alpha=1, label="Actual")

    for pred, name, rf in predictions:
        ax.step(hours, pred[-24:], where="mid", alpha=0.6, label=f"{name} - {rf}")

    # ax.set_xlim(0, 23)  # Set x-axis to range from 0 to 23 hours
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24))  # Set ticks every 3 hours for readability
    ax.set_xlabel("Hours")
    ax.set_ylabel("Hourly Next-Day Energy Price")

    plt.legend()
    ax.set_title(f"Test Set vs. Predictions -  Last 24h")
    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid for better readability
    plt.show()


def plot_actual_vs_predicted(results_df, specific_date):
    # Filter the results for the specific date
    filtered_results = results_df[
        results_df["ds"].dt.date == pd.to_datetime(specific_date).date()
    ]

    # Check if there are any rows in filtered_results
    if filtered_results.empty:
        print(f"No data available for the date {specific_date}")
    else:
        # Plot the actual and predicted data
        plt.figure(figsize=(16, 6))
        plt.plot(
            filtered_results["ds"],
            filtered_results["y"],
            label="Actual Prices",
            color="blue",
            linewidth=2,
        )
        plt.plot(
            filtered_results["ds"],
            filtered_results["yhat"],
            label="Forecasted Prices",
            color="red",
            linestyle="dashed",
            linewidth=2,
        )
        plt.title(f"Actual vs Forecasted Energy Prices for {specific_date}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.legend(loc="upper left")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def plot_linear_regression(index, y_test, predictions):
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.plot(index, y_test, color="blue", label="Actual Data")
    plt.plot(index, predictions, color="red", linewidth=2, label="Predictions")
    plt.xlabel("Index")
    plt.ylabel("Target Variable")
    plt.title("Regression Line vs Actual Data")
    plt.legend()
    plt.show()

    # Metrics
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")


def plot_benchmark_predictions(
    y_actual,
    X_predict,
    prophet_X_predict=None,
    prophet_prediction=None,
    lr_prediction=None,
    xgb_prediction=None,
):
    """
    Plots energy price predictions from different models against actual prices.

    Parameters:
    - y_actual: Pandas Series containing actual energy prices.
    - prophet_X_predict: DataFrame with 'ds' column for Prophet's timestamps.
    - prophet_prediction: Series or array containing Prophet's predicted prices.
    - X_predict: DataFrame with DateTimeIndex for the predictions.
    - lr_prediction: Series or array containing Linear Regression predictions.
    - xgb_prediction: Series or array containing XGBoost predictions.
    - title: (Optional) Title of the plot.
    """
    plt.figure(figsize=(12, 6))
    plt.grid(True, linestyle="--", alpha=0.6)

    # Plot actual prices and model predictions
    plt.step(
        y_actual.index,
        y_actual,
        linewidth=3,
        where="mid",
        label="Actual Price",
        color="navy",
    )
    if prophet_X_predict is not None:
        plt.step(
            prophet_X_predict["ds"],
            prophet_prediction,
            linewidth=1.5,
            where="mid",
            label="Prophet",
            color="orange",
        )
    if lr_prediction is not None:
        plt.step(
            X_predict.index,
            lr_prediction,
            linewidth=1.5,
            where="mid",
            label="Linear Regression",
            color="green",
        )
    if xgb_prediction is not None:
        plt.step(
            X_predict.index,
            xgb_prediction,
            linewidth=1.5,
            where="mid",
            label="XGBoost",
            color="brown",
        )

    # Formatting
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.title(f"Hourly Energy Price Predictions", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(fontsize=11)

    plt.show()


def plot_energy_mix():
    # Rename the timestamp column and parse dates
    df_raw = pd.read_csv("../data/hourly_market_mix_cleaned.csv")

    df_raw.rename(columns={"Timestamp": "Datetime"}, inplace=True)
    df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"])

    # Ensure the dataframe is sorted by datetime
    df_raw = df_raw.sort_values(by="Datetime")

    # Select the last 72 hours
    df_last_72h = df_raw.tail(72)

    # Set Datetime as index
    df_last_72h.set_index("Datetime", inplace=True)

    # Normalize the energy sources to get proportions
    df_normalized = df_last_72h.div(df_last_72h.sum(axis=1), axis=0)

    # Plot the stacked area chart
    plt.figure(figsize=(14, 6))
    df_normalized.plot.area(figsize=(14, 6), alpha=0.8, cmap="tab10")
    plt.xlabel("Datetime")
    plt.ylabel("Proportion")
    plt.title("Energy Mix - Last 72 Hours")
    plt.xticks(rotation=45)
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def plot_correlation_line(merged_df):
    plt.figure(figsize=(15, 7))
    sns.regplot(
        x="Renewable_Sum",
        y="Price",
        data=merged_df,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red"},
    )
    plt.title("Relationship between Renewable Energy Sum and Price")
    plt.xlabel("Sum of Solar, Wind offshore, Wind onshore, Hydro")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()
