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
    fig, axs = plt.subplots(nrows=2, figsize=(15, 10))

    idx = np.argsort(reg.feature_importances_)[::-1]
    fi_sorted = reg.feature_importances_[idx]
    fn_sorted = reg.feature_names_in_[idx]
    idx_limited = np.argsort(reg.feature_importances_)[::-1][1:]
    fi_sorted_limited = reg.feature_importances_[idx_limited]
    fn_sorted_limited = reg.feature_names_in_[idx_limited]

    fi = pd.DataFrame(data=fi_sorted,
                index=fn_sorted,
                columns=['importance'], )
    fi.sort_values('importance').plot(ax=axs[0], kind='barh', title=f'Feature Importance - {objective}')
    fi = pd.DataFrame(data=fi_sorted_limited,
                index=fn_sorted_limited,
                columns=['importance'], )
    fi.sort_values('importance').plot(ax=axs[1], kind='barh', title=f'Feature Importance ex ma_3_hours - {objective}')
    plt.tight_layout()
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    plt.show()
    plt.close()
    # buf.seek(0)

    # return Image.open(buf)

def plot_predicted(test_set, predictions, objective_name):
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.plot(test_set.index, test_set["Price"],  alpha=0.6, label="Actual")
    ax.plot(test_set.index, predictions,  alpha=0.6, label="Predicitons")
    ax.set_xlim(test_set.index.min(), test_set.index.max())
    ax.set_ylabel("Hourly Next-Day Energy Price")
    plt.legend()    
    ax.set_title(f'Test Data vs. Predictions (Full) - {objective_name}')


def plot_linear_regression(index, y_test, predictions):
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
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




    


