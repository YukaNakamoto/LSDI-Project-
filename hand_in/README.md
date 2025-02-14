# BTW 2025 Data Science Challenge Participation TU Berlin

## Introduction
This repository contains our solution for the BTW 2025 Data Science Challenge. The project leverages Python and Poetry for dependency management, ensuring a reproducible and consistent development environment.

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


## 1. Introduction

Forecasting energy prices in Germany is a challenging task due to the complex interplay of multiple factors, including renewable energy generation, fossil fuel prices, weather conditions, and market regulations. 
As participants in the “BTW 2025 Data Science Challenge,” our goal is find relevant datasets and to build a predictive model that forecasts hourly day-ahead energy prices for Germany on February 18, 2025, using historical data. This report begins with an overview of the domain knowledge surrounding Germany’s energy market, followed by a discussion of the data sources used in modeling. Subsequent sections detail the methodology, present modeling results, and offer conclusions.

## 2. Background & Domain Knowledge

### 2.1 Energy Market and Price Dynamics
The German day ahead energy price is determined by demand und supply. At 12:00 CES the bids of customers and energy producers are gathered.
Subsequently, the price is determined by the merit order principle, in which the output of power plants with the lowest marginal costs are accumulated until to meet demand; ultimately, the price is set by the last plant needed to satisfy demand. Because renewable energy sources generally have low marginal costs, they are dispatched preferentially. However, oversupply, along with grid limitations, can lead to negative prices (Wissenschaftliche Dienste, 2022).


## 2.2 Energy Mix
Germany’s energy mix has evolved significantly, with the share of renewables reaching about 50% of total generation in 2023 (BMWK-Energiewende, 2024). Among these, wind and solar power dominate:
- Wind Power: Wind power generation generally peaks during the winter months, partly balancing out lower power input by solar installations (Clean Energy Wire).
- Solar Power: Photovoltaic (PV) generation typically reaches its highest output during summer daytime hours, when intense sunlight drives abundant electricity production. This increased solar output during the day shifts the supply curve via the merit‐order effect, reducing wholesale electricity prices (Cludius et al., 2014).

Despite the rise in renewables, fossil fuels—including hard coal, lignite, and natural gas—still play a critical role. They contribute to price volatility, influenced by fluctuating fuel costs and carbon pricing (de Lagarde, C., & Lantz, F., 2018). Understanding both the renewable and fossil-fuel components of the energy mix is therefore crucial for accurate price forecasting.

Hypothesis
We hypothesize that there is an inverse relationship between renewable energy production (including solar, wind onshore, wind offshore, and hydro) and next-day energy prices. In other words, as renewable generation increases, the day-ahead market price tends to decrease, largely due to the low marginal costs associated with renewable sources.


## 2.3 Weather Impacts
Weather conditions significantly affect energy supply and demand, thereby influencing prices. Key factors include:
- Wind Speeds: Higher wind speeds, especially in northern Germany where wind resources are abundant, increase wind power generation and lower wholesale electricity prices by shifting the merit‐order (Sensfuß et al., 2008; Cludius et al., 2014).
- Solar Irradiance: Stronger sunlight in southern regions increases solar power output, putting downward pressure on prices.
- Temperature:
  - Low Temperatures → Increased heating demand → Higher prices
  - High Temperatures → Increased cooling demand → Higher prices
- Precipitation: Affects electricity prices by influencing hydropower generation, with price sensitivity determined by country‐specific precipitation thresholds (Mosquera‐López et al., 2024).

### 3. Data Cleaning
Understanding and preparing the data is a crucial step in any predictive modeling task. In energy price forecasting, data reliability, completeness, and structure play a fundamental role in ensuring the models can capture key trends and dependencies. Given the complexity of the energy market, multiple data sources were used, each contributing essential information. However, some critical datasets were not readily accessible—gas and oil price data were behind a paywall, preventing their inclusion, and electricity demand data was available but consistently missing the last three days, making it unreliable for modeling. Addressing these challenges required careful data cleaning, merging, and feature engineering to build a robust foundation for forecasting models.

## 3.1 Data Sources & Reliabality

Ensuring the reliability of data sources is fundamental in energy price modeling. The datasets used in this analysis are sourced from reputable organizations: Energy prices are sourced from SMARD.de, a platform operated by the German Federal Network Agency, which is recommended for energy price data. Weather data is obtained from Open-Meteo, a provider of historical and forecast weather information that collaborates with national weather services. Data on the energy mix is provided by Agora Energiewende, which offers interactive tools and datasets widely used for analyzing energy mix information. (SMARD, 2025) (Agora Energiewende, 2025)

## 3.2 Data Cleaning and Preparation
To ensure the datasets were ready for analysis, a thorough data cleaning process was carried out. The first step involved handling missing data. One notable challenge was the absence of nuclear energy data after April 15, 2023, due to Germany’s nuclear phase-out. Additionally, any missing values in other datasets were addressed by re-fetching the data from their respective sources to maintain completeness. Ensuring a fully populated dataset was crucial for maintaining consistency across all variables and enabling reliable model training.
Another key aspect was outlier detection. Given the inherent volatility of energy prices, extreme values were carefully examined rather than being removed outright. Instead of discarding outliers indiscriminately, we assessed their impact on the model’s performance. In some cases, extreme values contained meaningful signals about rare market conditions and were retained to ensure the model could adapt to such situations. Each outlier was evaluated in context, preserving those that could provide valuable insights while mitigating the effect of spurious anomalies.
After cleaning the data, we merged different datasets to establish a comprehensive dataset for modeling. Weather data, energy price data, and energy mix data were aligned by date, ensuring that all records were synchronized correctly. This step was crucial for creating a structured dataset that allowed for meaningful feature extraction and analysis in the subsequent modeling phase.


## 3.3 Correlation Analysis
We explored the relationship between different factors (weather variables, energy mix) and energy prices. Initial findings revealed an inverse correlation between renewable energy generation (solar, wind onshore, wind offshore, hydro) and energy prices, with an observed moderate negative correlation coefficient of -0.312. 

## 3.4 Data Loading
In handling missing values, we removed all rows containing missing entries in either the energy price or energy mix columns. The weather dataset was complete, so no additional steps were necessary for that. Since nuclear energy production reporting ceased in 2024, we set any missing nuclear values to 0.0 from that point onward.

For outlier removal, we employed two different approaches. When conducting statistical tests, we used the Interquartile Range (IQR) method to identify and exclude extreme values. For model training, outliers were removed manually using a slider in the “Configuration” interface, allowing fine-grained control over which data points to retain.

### 4. Features

Selecting relevant features is crucial for building an effective predictive model. Given the diverse range of available data—including time-based attributes, moving averages, energy generation sources, and weather variables—our selection process focused on maximizing predictive power while avoiding redundancy.
Since energy prices exhibit strong temporal dependencies, time-based features were incorporated to capture seasonal and weekly patterns. The importance of these features has been previously discussed in the context of price dynamics. Additionally, moving averages played a key role in smoothing short-term fluctuations and enhancing model stability.
Beyond time-related attributes, energy mix variables were selected based on their direct impact on electricity prices. Renewable energy sources, particularly solar and hydro, were included due to their well-documented influence on price formation. Weather variables, such as temperature, precipitation, and direct radiation, were chosen for their role in both energy demand and renewable generation variability. These relationships have been analyzed, reinforcing their relevance in predictive modeling.
By carefully curating our feature set, we ensured that the model balances complexity and interpretability while capturing the fundamental drivers of energy price fluctuations. This selection process aimed to improve forecasting accuracy without introducing unnecessary computational overhead.

## 4.1 Moving Averages and Their Role in Forecasting
Moving averages are essential for smoothing out short-term fluctuations in time-series data and revealing underlying trends. By averaging values over a defined window, moving averages reduce noise and enhance the visibility of long-term patterns.
For linear regression-based models, moving averages provide several benefits:
- *Noise Reduction*: Energy price data can be volatile due to external shocks, market fluctuations, and unexpected demand surges. Moving averages help smooth these fluctuations, enabling the model to focus on broader trends rather than reacting to random variations.
- *Feature Engineering*: Moving averages can serve as additional explanatory variables in regression models, capturing smoothed past behavior that informs future predictions.
Seasonality Handling: By choosing appropriate window sizes (e.g., 24-hour, 7-day, or 30-day moving averages), it is possible to better capture daily, weekly, or monthly cycles in energy prices.
- *Enhanced Interpretability*: Linear models often struggle with high-frequency variations. Incorporating moving averages helps them generalize better by reducing the emphasis on short-term noise.
Given the time-series nature of energy prices, we applied moving averages with various window sizes (e.g., hourly and daily) to analyze how different trends affect price fluctuations. This was particularly useful in identifying seasonal effects and ensuring that models could make robust predictions without overfitting to short-term anomalies.

## 4.2 Evaluation Criteria

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
- Ability to adapt and fine-tune hyperparameters efficiently.

### 5. Model Selection and Training


Selecting an appropriate modeling approach is a critical step in building a reliable forecasting system. Given the complexity of energy price dynamics, the choice of models must balance accuracy, interpretability, and computational efficiency. A well-defined modeling strategy typically begins with a simple yet effective baseline model before progressing to more advanced techniques. This ensures that improvements gained from sophisticated models are measured against a meaningful reference point. In this section, we outline the reasoning behind our model selection process, beginning with a baseline model and advancing to more specialized forecasting methods.

## 5.1 Choosing a Baseline Model
For our initial baseline model, we selected Linear Regression due to its simplicity, interpretability, and computational efficiency. Linear Regression provides a straightforward approach to capturing linear relationships in the data, making it an excellent reference point for evaluating more complex models. However, it struggles with capturing non-linear dependencies and seasonality, which are crucial in energy price forecasting.
We opted against using neural networks due to several reasons:
- **Computational Cost**: Training deep learning models requires significant computational resources, which were not justified given the available dataset size and forecasting horizon.
- **Interpretability**: Neural networks act as black-box models, making it difficult to extract insights about seasonal trends, price spikes, or external influencing factors.
- **Data Requirements**: Deep learning models typically require large amounts of high-quality data to generalize well. Given our dataset’s structure, traditional time-series models were more suitable.
## 5.2 XGBoost
XGBoost (Extreme Gradient Boosting) is a machine learning algorithm based on decision trees, well-suited for structured datasets like ours. It excels at capturing non-linear relationships, allowing it to model complex interactions between features and energy prices, which Linear Regression cannot do. XGBoost also provides built-in feature importance metrics, helping identify key factors like weather, energy mix, and demand fluctuations. It handles missing data and outliers better than traditional models, making it a robust choice for real-world forecasting. Computationally efficient compared to deep learning models, XGBoost performs well for our dataset, although it requires feature engineering and hyperparameter tuning. Its ability to capture short-term fluctuations and complex feature interactions can enhance forecasting accuracy.

Gradient Boosting is an ensemble learning technique that builds a model by iteratively improving the predictions of weak learners (typically decision trees). Each new model focuses on the errors of the previous one, reducing them in a sequential manner. This allows Gradient Boosting to handle complex patterns and interactions in the data, making it effective for tasks like forecasting energy prices.

## 5.3 Prophet
Prophet, developed by Facebook, is designed for time-series forecasting, particularly in capturing seasonality, trends, and holiday effects. It automatically detects daily, weekly, and yearly seasonality, which is helpful for energy price forecasting, where such patterns dominate. Prophet is robust to missing data and outliers, improving forecast reliability. Unlike traditional models, Prophet allows trend changes over time, making it ideal for dynamic markets. It also provides clear decompositions of trend, seasonality, and holiday effects, offering interpretability that aids understanding of energy price drivers. Prophet’s combination of seasonality detection, trend flexibility, and ease of use makes it a powerful tool for forecasting energy prices.

### 6.  Benchmarking
The Benchmark Test in this model evaluates the performance of different forecasting algorithms (Prophet, Linear Regression, and XGBoost) to predict the energy price for February 18th. To determine which model performs best, we simulate predicting the price for this specific day across a whole month, testing the models' abilities to accurately forecast the price. The Data Preparation step involves merging and splitting the dataset into training, evaluation, and test sets. The training data includes all available historical data up to a specific point, while the evaluation and test sets are used to assess model performance on unseen data. The benchmark dataset is used to simulate forecasting and serves as the reference to compare the models’ predictions. The core of the test is the Rolling Forecast Loop, which processes the dataset in 24-hour intervals. Starting 24 hours before the day to predict, the loop moves in steps of 24 hours, predicting the price for the day to predict each time. The models are retrained on all available data up to the current point, extending the dataset with each iteration. If there is any missing data, the forecast for that window is skipped. In each iteration, the three models are trained and used to predict the price for one day. After simulating predictions for all relevant windows, the results are aggregated. The average RMSE is computed for each model, and the one with the lowest RMSE will be selected as the best model for predicting the price on February 18th. The Benchmark Test Setup helps simulate real-world forecasting conditions by repeatedly training and testing the models over a period of time. By comparing the RMSE values, we can objectively assess which model provides the most accurate prediction for the specific day we are targeting — February 18th.

### 7. Results
We began our analysis by testing three different models - Linear Regression, XGBoost, and Prophet - to determine the most effective approach for forecasting energy prices. To establish a baseline, we initially evaluated these models without any additional features, relying solely on historical price data. Following this initial assessment, we introduced external features to enhance the models' predictive capabilities. We first tested the models using only energy mix features. Then we tested only weather features. While both feature sets individually improved model accuracy, the best results were achieved when combining energy mix and weather features. To further refine the models, we applied an AutoML process for feature selection. This step automatically identified the most relevant variables while eliminating those that contributed little to predictive accuracy. The inclusion of these optimized features improved the predictive performance across all three models. After finalizing the feature set, we conducted a second AutoML process to optimize hyperparameters for XGBoost and Prophet. With the optimized models in place, we compared their final performance based on MAE, MSE, RMSE, and MAPE. Linear Regression performed the worst, showing the highest RMSE and MAE. XGBoost performed better than expected and ultimately outperformed Prophet in our specific scenario. While Prophet achieved the lowest RMSE and MAE, its accuracy depended on the availability of the energy mix forecast, which is published too late for the challenge. Without access to this crucial feature, XGBoost delivered better results than Prophet. Although Prophet’s ability to model seasonality, trends, and external variables makes it the most accurate option in ideal conditions, its reliance on unavailable data made it impractical for our forecasting needs. Given these constraints, we selected XGBoost as our final model to predict the energy price for February 18th.


### 8. Conclusion
Our analysis revealed several key insights about energy price forecasting. One of the most surprising findings was that moving averages did not improve model performance due to the lack of accurate future data. While moving averages are commonly used to smooth short-term fluctuations, they rely on future observations, which are unavailable in a true forecasting scenario. This limitation made them ineffective for our models.

Another significant result was the correlation between renewable energy generation and energy prices. As the share of renewables increased, prices tended to decrease, highlighting the impact of renewable energy sources on market dynamics. Additionally, we found that incorporating weather and energy mix features significantly improved forecasting accuracy, as reflected in the reduction of RMSE across all models. These external factors provided crucial information about supply and demand fluctuations, making them essential for accurate price predictions. We also discovered that some months are easier to predict than others. For example, December was easier to forecast than January. Similarly, certain time periods, such as nighttime, were more predictable than daytime, likely due to more stable demand patterns. Both XGBoost and Prophet performed well in capturing trends and spikes, but they consistently exhibited slight shifts, particularly around sharp price spikes. Among the models tested, XGBoost emerged as the best performer, achieving the lowest average RMSE in our specific scenario. While Prophet demonstrated superior accuracy in ideal conditions due to its ability to model seasonality, trends, and external regressors, its performance relied on the availability of the energy mix forecast, which is published too late for the challenge. Without this crucial feature, XGBoost outperformed Prophet, making it the more reliable choice for our forecasting task. Given these constraints, we selected XGBoost as our final model.

For further improvements, several steps could be taken. One important enhancement would be to incorporate oil and gas prices, as these external economic factors have a direct influence on energy markets. Adding such financial indicators could provide a more comprehensive view of price fluctuations. Additionally, increasing the amount of historical data available for training could further improve model performance by capturing long-term trends and rare market conditions.

Another key improvement would be further optimization through AutoML. We encountered challenges in efficiently utilizing all available CPU cores, particularly because Prophet is single-threaded, making it computationally expensive. Ensuring proper multi-threading capabilities for Prophet could significantly reduce training times. AutoML, while beneficial for feature selection and hyperparameter tuning, had a very high runtime, which limited its full potential. Addressing these computational constraints could lead to even better model performance and efficiency. By expanding the dataset, integrating more relevant features, optimizing computational resources, and leveraging AutoML more effectively, the accuracy and robustness of our forecasts could be further enhanced, making the predictions even more reliable.
