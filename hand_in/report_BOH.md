# Project Report: Next-Day Hourly Energy Price Prediction

Our group, consisting of Elias, Yuka, Zeynep, and myself, undertook a project to predict the hourly next-day energy price for February 13, 2025. This required us to identify relevant data sources, select suitable models, and conduct multiple testing iterations, including feature selection and hyperparameter-tuning.

To efficiently divide the workload, our group split into two subgroups. Elias and Zeynep focused on building a scraper for weather data and training a Prophet model, while Yuka and I build a scraper to collect historical next-day hourly energy prices and trained an XGBoost model. Throughout the project, we ensured that all datasets were shared between both groups. 

After fine-tuning XGBoost through hyperparameter search, we successfully reduced the RMSE of our initial feature set (hours day of week, day of year, energy mix and weather variables) to around 20. At every testing iteration we saved the hyperparameter combination, the plots of our predictions and a binary dump of the model  to be able to reproduce our findings. 

A key insight emerged when we discovered that moving averages significantly improved model performance, leading both groups to incorporate this feature into their respective models. At this point the we were getting RMSE's around 7 for Prophet and 10 for XGBoost.

Towards the end of the project we discovered, that the moving averages (spanning the day of testing) where not as compatible with the projects objective as we thought since, we were lacking the price on the days of prediction to be able to calculate the moving averages. Sampling this price from previous days resulted in a negative signal to all models that worsened our result. We therefore only added those moving averages that span multiple days as a feature. 

With those changes Prophet was quite often capable to predict a curve for the energy price that was fairly similar in terms of shape to the actual energy price for that day, often  only offset in the the y-axis (Price). In our final meeting we decided on adding a constant to the predictions. With this in place, we were able to get an average RSME (add models and final RMSEs) over the prediction of 30 days in our benchmark set. From these these results we chose ... as our final model to make the final prediction.

While writing the project report my primary contributions were summarising background information on how energy prices are determined, explaining the impact of energy mix and weather factors, and formulating our hypothesis. Additionally, I conducted a statistical correlation analysis to evaluate the influence of renewable energy on prices and created the  plots for visualization of the correlation and the energy mix as well as other plots (e.g. box plots) that did not make the final cut. 

Furthermore, Elias, Zeynep, and I worked on merging the code bases for Prophet and XGBoost. We also built a benchmark comparison, evaluating the performance of XGBoost, Prophet, and a linear regression model. Additionally, I set up the final pipeline for our energy price prediction deliverable.

Overall, the workload was fairly distributed among all team members, with each of us contributing to different aspects of the project.