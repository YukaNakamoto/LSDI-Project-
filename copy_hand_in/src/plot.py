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
    "Wind onshore"
]

def plot_price_split(train, eval, test, SPLIT_DATE_EVAL, SPLIT_DATE_TEST, title):
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(20, 11))
    train["Price"].plot(ax=ax, label='Training Set', title=f'Hourly Next-Day Energy Price Train/Evaluation/Test Split - {title}')
    test["Price"].plot(ax=ax, label='Test Set', color="red")
    eval["Price"].plot(ax=ax, label='Evaluation Set', color="orange")

    ax.axvline(SPLIT_DATE_EVAL, color='orange', ls='--')
    ax.axvline(SPLIT_DATE_TEST, color='red', ls='--')
    ax.legend(['Training Set', 'Test Set', 'Evaluation Set'])
    plt.show()

def plot_energy_mix_at_noon(train, eval, test):
    plt.style.use('default')

    fig, axs = plt.subplots(nrows=3, figsize=(20, 33))

    filtered_train = train[train.index.hour == 12]
    filtered_train[COLS].plot(ax=axs[0], kind="bar", stacked=True, title='Energy Mix at 12:00 - Train Split', width=1.0)
    axs[0].set_xticks(np.arange(0, len(filtered_train), 30))
    axs[0].set_xticklabels(filtered_train.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')

    filtered_eval = eval[eval.index.hour == 12]
    filtered_eval[COLS].plot(ax=axs[1], kind="bar", stacked=True, title='Daily Energy Mix at 12:00  - Eval Split', width=1.0)
    axs[1].set_xticks(np.arange(0, len(filtered_eval), 30))
    axs[1].set_xticklabels(filtered_eval.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')

    filtered_test = test[test.index.hour == 12]
    filtered_test[COLS].plot(ax=axs[2], kind="bar", stacked=True, title='Daily Energy Mix at 12:00  - Test Split', width=1.0)
    axs[2].set_xticks(np.arange(0, len(filtered_test), 30))
    axs[2].set_xticklabels(filtered_test.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')

    plt.subplots_adjust(hspace=0.3)
    plt.show()

def plot_energy_mix_on_date(test, date):
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(20, 11))

    filtered_test = test.loc[date]
    filtered_test[COLS].plot(ax=ax, kind="bar", stacked=True, title=f'Energy Mix on {date.strftime("%d. %B %Y")} - Test Split', width=1.0)
    ax.set_xticks(np.arange(0, len(filtered_test), 1))
    ax.set_xticklabels(filtered_test.index.strftime('%H:%M'), rotation=45, ha='right')

    plt.show()



def feature_importance(reg, objective):
    sns.set_theme()
    fig, axs = plt.subplots(ncols=2, figsize=(16, 6))

    idx = np.argsort(reg.feature_importances_)[::-2]
    fi_sorted = reg.feature_importances_[idx]
    fn_sorted = reg.feature_names_in_[idx]
    idx_limited = np.argsort(reg.feature_importances_)[::-2][1:]
    fi_sorted_limited = reg.feature_importances_[idx_limited]
    fn_sorted_limited = reg.feature_names_in_[idx_limited]

    fi = pd.DataFrame(data=fi_sorted,
                index=fn_sorted,
                columns=['importance'], )
    fi.sort_values('importance').plot(ax=axs[0], kind='barh', title=f'Feature Importance - {objective}')
    fi = pd.DataFrame(data=fi_sorted_limited,
                index=fn_sorted_limited,
                columns=['importance'], )
    fi.sort_values('importance').plot(ax=axs[1], kind='barh', title=f'Feature Importance (ex best feature) - {objective}')
    plt.tight_layout()
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    plt.show()
    # buf.seek(0)

    # return Image.open(buf)

def plot_predicted(test_set, predictions, date):
    fig, ax = plt.subplots(figsize=(20, 10))

    # Extract hours from the datetime index
    hours = test_set.index.hour  

    ax.step(hours, test_set, lw=3, where="mid", alpha=1, label="Actual")
    
    for (pred, name, rf) in predictions:
        ax.step(hours, pred, where="mid", alpha=0.6, label=f"{name} - {rf}")

    ax.set_xlim(0, 23)  # Set x-axis to range from 0 to 23 hours
    ax.set_xticks(range(0, 24))  # Set ticks every 3 hours for readability
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Hourly Next-Day Energy Price")
    
    plt.legend()
    ax.set_title(f'Test Set vs. Predictions ({date})')
    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid for better readability
    plt.show()

def plot_actual_vs_predicted(results_df, specific_date):
    # Filter the results for the specific date
    filtered_results = results_df[results_df['ds'].dt.date == pd.to_datetime(specific_date).date()]

    # Check if there are any rows in filtered_results
    if filtered_results.empty:
        print(f"No data available for the date {specific_date}")
    else:
        # Plot the actual and predicted data
        plt.figure(figsize=(16, 6))
        plt.plot(filtered_results['ds'], filtered_results['y'], label='Actual Prices', color='blue', linewidth=2)
        plt.plot(filtered_results['ds'], filtered_results['yhat'], label='Forecasted Prices', color='red', linestyle='dashed', linewidth=2)
        plt.title(f"Actual vs Forecasted Energy Prices for {specific_date}")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_linear_regression(index, y_test, predictions):
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.plot(index, y_test, color='blue', label='Actual Data')
    plt.plot(index, predictions, color='red', linewidth=2, label='Predictions')
    plt.xlabel('Index')
    plt.ylabel('Target Variable')
    plt.title('Regression Line vs Actual Data')
    plt.legend()
    plt.show()

    # Metrics
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")




    


