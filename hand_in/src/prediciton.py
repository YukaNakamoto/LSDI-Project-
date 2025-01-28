def predict_return_mse(best_params, X_train, y_train, X_val, y_val, X_test):
    reg = xgb.XGBRegressor(**best_params)
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)
    y_predicted = reg.predict(X_test)
    
    fi_img = feature_importance(reg, "MSE")

    return y_predicted, reg, fi_img

def predict_return_mae(best_params, X_train, y_train, X_val, y_val, X_test):
    best_params.update({"objective": "reg:squarederror", "eval_metric": "mae"})

    reg = xgb.XGBRegressor(**best_params)
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=25)
    y_predicted = reg.predict(X_test)

    fi_img = feature_importance(reg, "MAE")

    return y_predicted, reg, fi_img


def linear_regression(X_train, y_train, X_test, y_test):
    merged = pd.concat([X_train, y_train], axis=1)
    merged_cleaned = merged.dropna()

    X_train_cleaned = merged_cleaned.iloc[:, :-1]
    y_train_cleaned = merged_cleaned.iloc[:, -1]

    merged = pd.concat([X_test, y_test], axis=1)
    merged_cleaned = merged.dropna()

    X_test_cleaned = merged_cleaned.iloc[:, :-1]
    y_test_cleaned = merged_cleaned.iloc[:, -1]


    #Fit the model
    model = LinearRegression()
    model.fit(X_train_cleaned, y_train_cleaned)

    predictions = model.predict(X_test_cleaned)

    # Plot actual vs predicted
    plt.figure(figsize=(50, 6))
    plt.plot(X_test_cleaned.index, y_test_cleaned, color='blue', label='Actual Data')
    plt.plot(X_test_cleaned.index, predictions, color='red', linewidth=2, label='Regression Line')
    plt.xlabel('Index')
    plt.ylabel('Target Variable')
    plt.title('Regression Line vs Actual Data')
    plt.legend()
    plt.show()

    rmse = mean_squared_error(y_test_cleaned, predictions) ** 0.5
    r2 = r2_score(y_test_cleaned, predictions)

    print(f"Root Mean Squared Error: {rmse}")
    print(f"R-squared: {r2}")