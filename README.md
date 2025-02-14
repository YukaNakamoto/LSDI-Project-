# BTW 2025 Data Science Challenge Participation TU Berlin

## Getting Started

To set up the project locally, follow these steps:

### Prerequisites
Ensure you have the following installed on your system:
- Python (>=3.8)
- Poetry (>=1.0.0)

To install Poetry, run:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```
Alternatively, on macOS, install it via Homebrew:
```sh
brew install poetry
```

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/YukaNakamoto/LSDI-Project-.git
   cd LSDI-Project-/hand_in
   ```

2. Install dependencies:
   ```sh
   poetry install
   ```
   This will create a virtual environment and install all dependencies specified in `pyproject.toml`.

3. Activate the virtual environment:
   ```sh
   poetry shell
   ```
   Now you are inside the virtual environment and can run Python scripts with the installed dependencies. (Requires Poetry to be set in $PATH)

### Verify Dependencies
To check installed dependencies, run:
```sh
poetry show
```

### Running Jupyter Notebook with Poetry
To use Jupyter Notebook with Poetry dependencies, follow these steps:

1. Install Jupyter if not already installed (Should have been installed by poetry):
   ```sh
   poetry add jupyter ipykernel
   ```

2. Add the Poetry environment as a Jupyter kernel:
   ```sh
   poetry run python -m ipykernel install --user --name=poetry-env --display-name "Python (Poetry)"
   ```

3. Start Jupyter Notebook:
   ```sh
   poetry run jupyter notebook
   ```

4. In Jupyter, select the kernel named `Python (Poetry)` to ensure the notebook runs with the Poetry dependencies.

### Deactivating the Virtual Environment
Exit the Poetry virtual environment by typing:
```sh
exit
```

### Report


## Introduction

Forecasting energy prices in Germany is a challenging task due to the complex interplay of multiple factors, including renewable energy generation, fossil fuel prices, weather conditions, and market regulations. 
As participants in the “BTW 2025 Data Science Challenge,” our goal is find relevant datasets and to build a predictive model that forecasts hourly day-ahead energy prices for Germany on February 18, 2025, using historical data. This report begins with an overview of the domain knowledge surrounding Germany’s energy market, followed by a discussion of the data sources used in modeling. Subsequent sections detail the methodology, present modeling results, and offer conclusions.

## 1. Background & Domain Knowledge

### 1.1 Energy Market and Price Dynamics

The German day ahead energy price is determined by demand und supply. At 12:00 CES the bids of customers and energy producers are collected. Subsequently, the price is determined by the merit order principle, in which the output of power plants with the lowest marginal costs is accumulated until the demand for a given time is met. The price is determined by the marginal cost of the final plant added. Because renewable energy sources generally have low marginal costs, they are dispatched preferentially.


### 1.2 Energy Mix
The energy mix refers to the combination of different energy sources used to generate electricity. It includes a mix of renewable and non-renewable sources, each contributing a different share to the total energy production.

Germany’s energy mix has undergone significant changes, with renewable accounting for approximately 50% of total electricity generation in 2023. Wind and solar power are the dominant sources among renewable. Given that the merit order principle prioritizes renewable energy, understanding the proportion of renewable versus conventional energy sources is particularly relevant.


### 1.3 Weather Impacts
Weather conditions significantly influence both the supply and demand for electricity, shaping energy prices in the process. Several key factors determine these effects:

- Wind Power
  - Wind energy generation depends on steady wind speeds. High wind speeds can result in high energy production. However, if the wind is to strong, energy production ceases. In northern Germany, where wind resources are abundant, higher wind speeds typically increase supply and lower wholesale electricity prices.
- Solar Power
  - Solar energy output depends on the amount of direct radiation. Intense sunlight, especially in southern Germany, boosts solar power generation and can also drive electricity prices down.
- Hydropower
  - Electricity generation from hydropower is influenced by precipitation levels.

- Temperature changes drive electricity demand, which in turn affects prices:
  - Cold weather increases demand for heating, pushing prices higher.
  - Hot weather raises cooling demand, also driving up prices.

### 2. Data Integration
Retrieving, understanding and preparing the data is a crucial step in any predictive modeling task. In energy price forecasting, data reliability, completeness, and structure play a fundamental role in ensuring the models can capture key trends and dependencies. Given the complexity of the energy market, multiple data sources were used, each contributing essential information. However, some critical datasets were not readily accessible—gas and oil price data were behind a paywall or not availble in the required resolution. Electricity demand data was available but consistently missing the last three days, making it unsuitbale for modeling. Addressing these challenges required careful data cleaning, merging, and feature engineering to build a robust foundation for forecasting models.


### 2.1 Data Sources & Reliabality

Ensuring the reliability of data sources is fundamental in energy price modeling. The datasets used in this analysis are sourced from reputable organizations: Energy prices are sourced from SMARD.de, a platform operated by the German Federal Network Agency, which is recommended for energy price data. Weather data is obtained from Open-Meteo, a provider of historical and forecast weather information that collaborates with national weather services. Data on the energy mix is provided by Agora Energiewende, which offers interactive tools and datasets widely used for analyzing energy mix information. (SMARD, 2025) (Agora Energiewende, 2025)


### 2.2 Data Preparation and Cleaning  

To ensure the datasets were ready for analysis, a comprehensive data cleaning process was conducted. This involved handling missing data, managing outliers, and merging various datasets.

#### 2.2.1 Handling Missing Data  

One of the primary challenges in data preparation was dealing with missing values. The approach varied depending on the dataset.  

For the energy mix and energy price data, rows containing missing entries in either column were removed to maintain data integrity. A special case arose with nuclear energy data, which was no longer reported after April 15, 2023, due to Germany’s nuclear phase-out. To ensure consistency, all missing nuclear values from that point onward were set to 0.0.  

The weather dataset was complete, so no additional imputation was necessary.  

The energy price dataset, required some None value removal but was complete otherwise.

#### 2.2.2 Outlier Management  

Given the volatility of energy prices, a nuanced approach was taken for outlier handling. Instead of outright removal, extreme values were assessed for their impact on model performance.  

For statistical tests, the Interquartile Range (IQR) method was used to identify and exclude extreme values. For model training, a manual outlier removal approach was implemented using a slider in the “Configuration” interface, allowing for precise control over which data points to retain. 

#### 2.2.3 Dataset Merging and Synchronization  

After cleaning the data, different datasets were merged to create a structured and comprehensive dataset for modeling. Weather data, energy price data, and energy mix data were aligned by date to ensure consistency across variables. This alignment was crucial for meaningful feature extraction and accurate analysis in the subsequent modeling phase.  

### 2.3 Hypothesis
We hypothesize that there is an inverse relationship between renewable energy production (including solar, wind onshore, wind offshore, and hydro) and next-day energy prices. In other words, as renewable generation increases, the day-ahead market price tends to decrease, largely due to the low marginal costs associated with renewable sources.

### 2.4 Hypothesis Testing

We explored the relationship between different factors (weather variables, energy mix) and energy prices.

#### r = -0.312
The correlation coefficient of -0.312 suggests a moderate negative correlation between the summed renewable energy production and the Energy Price.
This means that as renewable energy generation increases, energy prices tend to decrease, albeit with some variability.
The magnitude of the correlation (around 0.3) indicates a moderate linear relationship.

#### Practical Interpretation:
The negative correlation suggests that an increase in renewable energy production (solar, wind, hydro) could be associated with lower electricity prices. This aligns with expectations, as renewable energy sources typically have lower marginal costs compared to fossil fuels, which can lead to lower market prices when their availability is high.
However, the moderate strength of the correlation (r = -0.312) indicates that while the relationship exists, other factors (e.g., demand fluctuations, fossil fuel prices, regulatory policies) might also influence the price.

#### P-Value ≈ 0
The extremely small p-value suggests that the observed negative correlation is highly statistically significant.
In practical terms, this means that the probability of obtaining such a correlation by random chance under the null hypothesis (that there is no correlation) is virtually zero.
Given this result we can reject the null hypothesis and conclude that the negative relationship between renewable energy generation and prices is statistically significant.

#### Null-hypothesis 
There exist no inverse relationship between Renewable energy sources (solar, wind onshore, wind offshore) and next-day energy prices


## 3. Features
Selecting relevant features is crucial for building an effective predictive model. Given the diverse range of available data—including time-based attributes, moving averages, energy generation sources, and weather variables—our selection process focused on maximizing predictive power while avoiding redundancy. 

Since energy prices exhibit strong temporal dependencies, time-based features were incorporated to capture seasonal and weekly patterns. The importance of these features has been previously discussed in the context of price dynamics (see Section 1.1). Additionally, moving averages played a key role in smoothing short-term fluctuations and enhancing model stability, as outlined in Section 3.1. 

Beyond time-related attributes, renewables energy mix variables were selected based on their direct impact on electricity prices. Renewable energy sources, particularly solar and hydro, were included due to their well-documented influence on price formation. Weather variables, such as temperature, precipitation, and direct radiation, were chosen for their role in both energy demand and renewable generation variability. These relationships have been analyzed in Section 1.3, reinforcing their relevance in predictive modeling. 

By carefully curating our feature set, we ensured that the model balances complexity and interpretability while capturing the fundamental drivers of energy price fluctuations. This selection process aimed to improve forecasting accuracy without introducing unnecessary computational overhead.


### 3.1 Moving Averages and Their Role in Forecasting
Moving averages are essential for smoothing out short-term fluctuations in time-series data and revealing underlying trends. By averaging values over a defined window, moving averages reduce noise and enhance the visibility of long-term patterns.
For linear regression-based models, moving averages provide several benefits:
- *Noise Reduction*: Energy price data can be volatile due to external shocks, market fluctuations, and unexpected demand surges. Moving averages help smooth these fluctuations, enabling the model to focus on broader trends rather than reacting to random variations.
- *Feature Engineering*: Moving averages can serve as additional explanatory variables in regression models, capturing smoothed past behavior that informs future predictions.
Seasonality Handling: By choosing appropriate window sizes (e.g., 24-hour, 7-day, or 30-day moving averages), it is possible to better capture daily, weekly, or monthly cycles in energy prices.
- *Enhanced Interpretability*: Linear models often struggle with high-frequency variations. Incorporating moving averages helps them generalize better by reducing the emphasis on short-term noise.
Given the time-series nature of energy prices, we applied moving averages with various window sizes (e.g., hourly and daily) to analyze how different trends affect price fluctuations. This was particularly useful in identifying seasonal effects and ensuring that models could make robust predictions without overfitting to short-term anomalies.

## 3.2 Evaluation Criteria

To assess model performance, we considered the following:
1. Error Objectiv Functions
- RMSE (Root Mean Squared Error): Measures overall prediction error, with a stronger penalty for large deviations.
- MAE (Mean Absolute Error): Reflects the average magnitude of errors without emphasizing outliers as strongly as RMSE.
- MSE (Mean Squared Error): Calculates the average of the squared differences between actual and predicted values, providing a general measure of error while penalizing larger errors more than MAE.
- MAPE (Mean Absolute Percentage Error): Expresses the average absolute error as a percentage of actual values, making it useful for comparing errors across different scales or datasets.
2. Handling of Seasonality and Outliers
- The ability to capture daily, weekly, and yearly seasonality effectively.
- Robustness to price spikes and outliers due to external factors like demand surges or market shocks.
3. Ease of Interpretation
- Linear models like Regression provide direct insight into feature importance.
- Prophet allows for intuitive trend and seasonality decomposition.
- XGBoost requires careful feature engineering to extract meaningful insights.
4. Faster Iteration and Scalability
- Training time and computational efficiency.
- Ability to adapt and fine-tune hyperparameter efficiently.



## 4 Model Selection and Training

Selecting an appropriate modeling approach is a critical step in building a reliable forecasting system. Given the complexity of energy price dynamics, the choice of models must balance accuracy, interpretability, and computational efficiency. A well-defined modeling strategy typically begins with a simple yet effective baseline model before progressing to more advanced techniques. This ensures that improvements gained from sophisticated models are measured against a meaningful reference point. In this section, we outline the reasoning behind our model selection process, beginning with a baseline model and advancing to more specialized forecasting methods.

### 4.1 Choosing a Baseline Model
For our initial baseline model, we selected Linear Regression due to its simplicity, interpretability, and computational efficiency. Linear Regression provides a straightforward approach to capturing linear relationships in the data, making it an excellent reference point for evaluating more complex models. However, it struggles with capturing non-linear dependencies and seasonality, which are crucial in energy price forecasting.
We opted against using neural networks due to several reasons:
- **Computational Cost**: Training deep learning models requires significant computational resources, which were not justified given the available dataset size and forecasting horizon.
- **Interpretability**: Neural networks act as black-box models, making it difficult to extract insights about seasonal trends, price spikes, or external influencing factors.
- **Data Requirements**: Deep learning models typically require large amounts of high-quality data to generalize well. Given our dataset’s structure, traditional time-series models were more suitable.

### 4.2 XGBoost
XGBoost (Extreme Gradient Boosting) is a machine learning algorithm based on decision trees, well-suited for structured datasets like ours. It excels at capturing non-linear relationships, allowing it to model complex interactions between features and energy prices, which the Linear Regression cannot do. XGBoost also provides built-in feature importance metrics, helping identify key factors like weather, energy mix, and demand fluctuations. Furthermore, it can handle missing values. These factors make it a robust choice for real-world forecasting. Computationally efficient compared to deep learning models, XGBoost performs well for our dataset, although it requires feature engineering and hyperparameter tuning. Its ability to capture short-term fluctuations and complex feature interactions can enhance forecasting accuracy.

Gradient Boosting is an ensemble learning technique that builds a model by iteratively improving the predictions of weak learners (typically decision trees). Each new model focuses on the errors of the previous one, reducing them in a sequential manner. This allows Gradient Boosting to handle complex patterns and interactions in the data, making it effective for tasks like forecasting energy prices.

### 4.3 Prophet
Prophet, developed by Facebook, is designed for time-series forecasting, particularly in capturing seasonality, trends, and holiday effects. It automatically detects daily, weekly, and yearly seasonality, which is helpful for energy price forecasting, where such patterns dominate. Prophet is robust to missing data and outliers, improving forecast reliability. Unlike traditional models, Prophet allows trend changes over time, making it ideal for dynamic markets. It also provides clear decompositions of trend, seasonality, and holiday effects, offering interpretability that aids understanding of energy price drivers. Prophet’s combination of seasonality detection, trend flexibility, and ease of use makes it a powerful tool for forecasting energy prices.

## 6. Benchmark

The Benchmark Test in this model evaluates the performance of different forecasting algorithms (Prophet, Linear Regression, and XGBoost) to predict the energy price for February 18th.  

#### Data Preparation  
- The dataset is merged and split into training, evaluation, and test sets.  
- Training data includes all historical data up to a certain point.  
- Evaluation and test sets assess model performance on unseen data.  
- The benchmark dataset serves as a reference for comparing model predictions.  

#### Rolling Forecast Loop  
- The dataset is processed in 24-hour intervals.  
- Starting 24 hours before the target day, the loop moves in 24-hour steps, predicting the price for the target day each time.  
- The models are retrained on all available data up to each step, expanding the dataset continuously.  
- If data is missing, the forecast for that window is skipped.  

#### Model Evaluation  

- In each iteration, the three models (Prophet, Linear Regression, XGBoost) are trained and used to predict the price for one day.  
- After running all simulations, the results are aggregated.  
- The average RMSE (Root Mean Square Error) is calculated for each model.  
- The model with the lowest RMSE is selected as the best model for predicting the energy price on February 18th.  

#### Significance of the Benchmark Test  

The Benchmark Test Setup helps simulate real-world forecasting conditions by continuously training and testing models over time. By comparing RMSE values, we can objectively determine which model provides the most accurate prediction for the target date.

## 7. Result

#### Initial Model Testing  

We began our analysis by testing three different models—Linear Regression, XGBoost, and Prophet—to determine the most effective approach for forecasting energy prices.  

- To establish a baseline, we initially evaluated these models using only historical price data without any additional features.  
- Following this assessment, we introduced external features to enhance predictive accuracy.  

#### Feature Impact  

We tested different feature sets to assess their impact on model performance (models without further hyperparameters):  

1. Energy Mix Features Only  
2. Weather Features Only  
3. Combination of Energy Mix and Weather Features  

| Model            | Weather Variables | Energy Mix Variables | Combined   |
|-----------------|------------------|----------------------|----------------------|
| Prophet      | 27.64296                | 21.64449                    | 20.34853   |
| Linear Regression | 27.59678         | 31.00825             | 24.65667
| XGBoost      |  41.91715        | 83.92071             | 35.84758    | 


Both feature sets individually improved model accuracy, but the best results were achieved when combining energy mix and weather features.  

#### AutoML for Feature Selection  

To further refine the models, we applied an AutoML process to:  

- Automatically identify the most relevant variables.  
- Eliminate features that contributed little to predictive accuracy.  

This feature selection step significantly improved performance across all three models.  

#### Hyperparameter Optimization  

After finalizing the feature set, we conducted a second AutoML process to optimize hyperparameters for XGBoost and Prophet.  

#### Model Performance Evaluation  

With the optimized models in place, we compared their final performance using MAE, MSE, RMSE, and MAPE.  

- Prophet performed the worst, showing the highest RMSE and MAE among all three models.  
- Linear Regression also struggled but performed slightly better than Prophet.  
- XGBoost outperformed both models, achieving the lowest RMSE and MAE, making it the most reliable choice for our forecasting needs.  

#### Final Model Selection  

Prophet, despite its ability to model seasonality, trends, and external variables, failed to deliver accurate predictions in our scenario. Its high error rates and reliance on unavailable data made it impractical.  

Given these results, we selected XGBoost as our final model to predict the energy price for February 18th, as it consistently provided the most accurate forecasts.


#### 6. Conclusion

Our analysis revealed several key insights about energy price forecasting. One of the most surprising findings was that hourly moving averages did not improve model performance. This was due to the fact that they had to be calculated from sampled data for the 24h that were to be predicted. While moving averages are commonly used to smooth short-term fluctuations, they rely on future observations, which are unavailable in a true forecasting scenario. This limitation made them ineffective for our models. However, daily moving averages, especially those spanning over multiple days did improve the model.

Another significant result was the correlation between renewable energy generation and energy prices. As the share of renewable increased, prices tended to decrease, highlighting the impact of renewable energy sources on market dynamics. Additionally, we found that incorporating weather and energy mix features significantly improved forecasting accuracy, as reflected in the reduction of RMSE across all models. These external factors provided crucial information about supply and demand fluctuations, making them essential for accurate price predictions.

We also discovered that some months are easier to predict than others. For example, December was easier to forecast than January. Similarly, certain time periods, such as daytime, were more nighttime, likely due to more stable demand patterns. Both XGBoost and Prophet performed well in capturing trends and spikes, but they consistently exhibited slight shifts, particularly around sharp price spikes.

Among the models tested, XGBoost emerged as the best performer, achieving the lowest average RMSE in our specific scenario. While Prophet demonstrated superior accuracy in ideal conditions due to its ability to model seasonality, trends, and external regressors, its performance relied on the availability of the energy mix forecast, which is published too late to take it into account for the challenge. Without this crucial feature, XGBoost outperformed Prophet, making it the more reliable choice for our forecasting task. Given these constraints, we selected XGBoost as our final model.

For further improvements, several steps could be taken. One important enhancement would be to incorporate gas prices, as these external economic factors have a direct influence on energy markets. Adding such indicators could provide a more comprehensive view of price fluctuations. Additionally, increasing the amount of historical data available for training could further improve model performance by capturing long-term trends and rare market conditions.

Another key improvement would be further optimization through AutoML. We encountered challenges in efficiently utilizing all available CPU cores, particularly because Prophet is single-threaded, making it computationally expensive. Ensuring proper multi-threading capabilities for Prophet could significantly reduce training times. AutoML, while beneficial for feature selection and hyperparameter tuning, had a very high runtime, which limited its full potential. Addressing these computational constraints could lead to even better model performance and efficiency.

By expanding the dataset, integrating more relevant features, optimizing computational resources, and leveraging AutoML more effectively, the accuracy and robustness of our forecasts could be further enhanced, making the predictions even more reliable.