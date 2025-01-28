import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_merged_datasets( train, eval, test, SPLIT_DATE_EVAL, SPLIT_DATE_TEST):
    sns.set_theme()

    fig, axs = plt.subplots(nrows=5, figsize=(20, 55))

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

    filtered_train = train[train.index.hour == 12]
    filtered_train[cols].plot(ax=axs[1], kind="bar", stacked=True, title='Energy Mix at 12:00 - Train Split', width=1.0)
    axs[1].set_xticks(np.arange(0, len(filtered_train), 30))
    axs[1].set_xticklabels(filtered_train.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')

    filtered_eval = eval[eval.index.hour == 12]
    filtered_eval[cols].plot(ax=axs[2], kind="bar", stacked=True, title='Daily Energy Mix at 12:00  - Eval Split', width=1.0)
    axs[2].set_xticks(np.arange(0, len(filtered_eval), 30))
    axs[2].set_xticklabels(filtered_eval.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')


    filtered_test = test.head(25)
    filtered_test[cols].plot(ax=axs[3], kind="bar", stacked=True, title='Energy Mix on the 29. October 2024 - Test Split', width=1.0)
    axs[3].set_xticks(np.arange(0, len(filtered_test), 1))
    axs[3].set_xticklabels(filtered_test.index.strftime('%H:%M'), rotation=45, ha='right')

    filtered_test = test[test.index.hour == 12]
    filtered_test[cols].plot(ax=axs[4], kind="bar", stacked=True, title='Daily Energy Mix at 12:00  - Test Split', width=1.0)
    axs[4].set_xticks(np.arange(0, len(filtered_test), 30))
    axs[4].set_xticklabels(filtered_test.index[::30].strftime('%Y-%m-%d'), rotation=45, ha='right')

    plt.subplots_adjust(hspace=0.3)
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


    


