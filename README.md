# <div align="center"> NYC Taxi Demand Prediction Project
### Tools & Techniques used:
* [Streamlit App]()
* LightGBM Time-Series Prediction
* Optuna Hyper-parameter Tuning
* Hopsworks Feature Store
* Hopsworks Model Registry
* GitHub Actions (Automate Feature Store population)

---
# <div align="center">DESIGN OVERVIEW

This project for predicting NYC taxi demand is an end-to-end machine learning project that has been designed to predict demand at a granularity of 1 hour and a prediction horizon of 1 hour. This project leverages a range of machine learning techniques, including LightGBM, Optuna hyper-parameter tuning, scikit-learn pipelines, and feature engineering. The feature engineering includes extracting date-time features, extracting latitude and longitude data from location IDs, and extracting weekly seasonality. The raw data for the model is obtained from the [NYC.gov](http://nyc.gov/) website and is preprocessed using various techniques to ensure that it is clean and accurate.

One of the most notable components of this project is the use of the Hopsworks feature store and model registry. The feature store is a centralized repository of data features that allows data scientists to share and reuse features across different ML pipelines, improving the efficiency of the overall process. The model registry is a centralized repository for storing and versioning machine learning models, ensuring that the models can be easily tracked, managed, and updated.

Furthermore, this project utilizes Streamlit, an open-source app framework for machine learning and data science teams, to host the model. Streamlit provides an easy-to-use interface for users to interact with the model and view the predictions. It is also highly customizable, allowing the project team to create a tailored experience that meets the needs of their users.

The model architecture for the project is shown below.

![model_architecture.png](https://github.com/ani-portfolio/1_taxi_demand_forecasting/blob/5_deploy_app/docs/model_architecture.png)

This project utilizes the following tools:

- Hopsworks Feature Store
- Hopsworks Model Registry
- Streamlit App Framework
- LightGBM
- Optuna Hyper-paramater Tuning

It should be noted that due to inaccessibility to latest data from NYC.gov, a production environment was simulated by shifting historical data to current date. Therefore, the results from this project is not a true representation of NYC taxi demand for the current date, but is more a Proof of Concept of an end-to-end ML project. 

---

# <div align="center">DESIGN DOCUMENTATION

## Purpose & Objective

Customers and taxicab businesses can benefit from taxi demand predictions as they can help to optimize the allocation of taxis, reducing wait times for customers and increasing profitability for taxicab businesses. 

By accurately predicting demand, taxicab businesses can ensure that they have the right number of taxis in the right locations at the right times, improving the overall customer experience and increasing revenue. Additionally, demand predictions can also help businesses to optimize their pricing strategies, ensuring that they are charging the right amount for their services based on the level of demand.

The objective of this project is to accurately predict the demand for NYC taxi rides every hour for the next hour, for each of the different zones within the city, and serve the predictions via a web application that is updated hourly with the latest predictions.

## Requirements & Constraints

For this project, the requirements state that NYC taxi demand should be predicted with an hourly granularity and prediction horizon of 1 hour. Furthermore, it is essential that these predictions are updated and made available automatically every hour. This ensures that the predictions remain relevant and timely, allowing taxicab businesses to optimize their operations and pricing strategies accordingly.

It is also important that the predictions are served in a web application format, making them easily accessible and user-friendly. The web application should display the predictions for each of the different zones within the city, providing a comprehensive view of the demand across the city. This enables taxicab businesses to allocate their resources effectively, ensuring that they have the right number of taxis in the right locations at the right times.

**Summary of requirements & constraints:**

- NYC taxi demand should be predicted with an hourly granularity and prediction horizon of 1 hour
- Predictions should be automatically updated and made available every hour
- Predictions should be served in a web application format, displaying predictions for each of the different zones within the city

Overall, the requirements highlight the importance of accuracy, timeliness, and accessibility in providing effective predictions for NYC taxi demand. By meeting these requirements, the project aims to deliver a valuable tool for taxicab businesses to optimize their operations and maximize their profitability.

## Methodology

### Problem Statement

The goal of this project is to accurately predict the demand for NYC taxi rides every hour for the next hour, for each of the different zones within the city. This demand prediction is essential for taxicab businesses to optimize the allocation of taxis, reducing wait times for customers and increasing profitability for taxicab businesses. The project aims to deliver a valuable tool for taxicab businesses to optimize their operations and maximize their profitability by providing effective predictions for NYC taxi demand.

### Data

The data used for this project is obtained from the [NYC.gov](http://nyc.gov/) website and includes information on taxi demand for 262 unique location IDs in New York City. The data is collected for each time a taxi is called, and includes the number of taxi pickups for each location ID, as shown below. 

| tpep_pickup_datetime | PULocationID |
| --- | --- |
| 2022-01-01 00:35:40 | 142 |
| 2022-01-01 00:33:43 | 236 |
| 2022-01-01 00:53:21 | 166 |
| 2022-01-01 00:25:21 | 114 |
| 2022-01-01 00:36:48 | 68 |
| 2022-01-01 00:40:15 | 138 |
| 2022-01-01 00:20:50 | 233 |
| 2022-01-01 00:13:04 | 238 |

To obtain hourly rides for each location ID, the data was pre-processed by aggregating the number of taxi pickups for each location ID at hourly intervals. Any missing data was filled with 0 to ensure that the resulting dataset was in a time-series format, as shown below. 

| pickup_hour | rides | pickup_location_id |
| --- | --- | --- |
| 2022-01-01 00:00:00 | 11 | 4 |
| 2022-01-01 01:00:00 | 15 | 4 |
| 2022-01-01 02:00:00 | 26 | 4 |
| 2022-01-01 03:00:00 | 8 | 4 |
| 2022-01-01 04:00:00 | 9 | 4 |
| ... | ... | ... |
| 2022-01-31 19:00:00 | 0 | 176 |
| 2022-01-31 20:00:00 | 0 | 176 |
| 2022-01-31 21:00:00 | 0 | 176 |
| 2022-01-31 22:00:00 | 0 | 176 |
| 2022-01-31 23:00:00 | 0 | 176 |

A plot of the data is shown below, for a particular location ID. (Location 43)

![time_series_plot.png](https://github.com/ani-portfolio/1_taxi_demand_forecasting/blob/5_deploy_app/docs/time_series_plot.png)

A clear seasonality can be observed in the data. This seasonality can be leveraged to feature engineer. 

### Baseline Models

A baseline model is required in order to provide a point of comparison for the performance of the more advanced machine learning models. By establishing a baseline, the project team can determine whether the more complex models are providing significant improvements over a simple model. This is important in order to ensure that the additional complexity of the more advanced models is justified by the improvement in accuracy or other performance metrics. 

Additionally, a baseline model can help to identify any issues with the data or the feature engineering process, as it provides a simple and straightforward model that can be compared to the more complex models.

To establish a baseline for comparison, three simple models were created;

1. The first model was a "last observation carried forward" (LOCF) model, which simply predicts the demand for the next hour based on the demand for the current hour.
2. The second model utilized the weekly seasonality in the data and predicted the demand for the next hour based on the demand for the same hour on the same day of the week from the previous week (t-7).
3. The third model was an average of the demand for the same hour and same day over the past four weeks. These models were used to establish a baseline for comparison with the more advanced models.

### Techniques

**Gradient Boosted Decision Trees**

LightGBM was used as the primary machine learning algorithm for this project's time-series prediction. It was trained on the pre-processed dataset, with features engineered to capture date-time, location, and weekly seasonality. The algorithm was tuned with Optuna hyper-parameter tuning and used in conjunction with scikit-learn pipelines to preprocess and transform the data. Feature importance was also examined to gain insights into the most significant predictors of taxi demand.

**Bayesian Hyper-parameter Tuning**

Optuna is a Python package that provides a framework for hyper-parameter tuning. It uses a Bayesian optimization algorithm that adapts to the results of previous trials in order to find the best set of hyper-parameters for a given machine learning model. This allows for a more efficient and effective approach to hyper-parameter tuning, as it reduces the number of trials required to find the optimal hyper-parameters. In this project, Optuna was used to optimize the performance of the LightGBM algorithm, resulting in improved accuracy for the NYC taxi demand prediction model.

**Feature Engineering**

To perform feature engineering for this project, scikit-learn pipelines were used to extract and transform various features from the raw data. The feature engineering steps included:

1. Extracting date-time features: The date and time information was extracted from the pickup_hour column, and features such as the hour of the day, day of the week, and month of the year were created.
2. Extracting latitude and longitude data from location IDs: The latitude and longitude data for each location ID was obtained from the [NYC.gov](http://nyc.gov/) website and was used to create additional features related to the location of the pickup.
3. Averaging the demand for the same hour and same day over the past four weeks: Another feature was created by averaging the demand for the same hour and same day of the week over the past four weeks.

These feature engineering steps were implemented using scikit-learn pipelines, which allowed for the efficient extraction and transformation of the data. 

**Feature Importance**

Feature importance can be used to better understand the model by identifying the most significant predictors of taxi demand. This information can be used to further improve the model's accuracy by focusing on the most important features. 

### Model Architecture

The model architecture for this project involves several steps. Firstly, the raw data is obtained from the [NYC.gov](http://nyc.gov/) website and is pre-processed to obtain a time-series dataset with hourly rides for each location ID. This dataset is then saved to the Hopsworks feature store, which serves as a centralized repository of data features that can be shared and reused across different machine learning pipelines.

GitHub Actions is used to automate the process of populating the Hopsworks feature store with data. 

Next, the LightGBM algorithm is trained on the pre-processed dataset, with the Optuna package used to tune hyper-parameters and scikit-learn pipelines used to preprocess and transform the data. 

Once the model is trained and optimized, it is saved to the Hopsworks model registry, which is a centralized repository for storing and versioning machine learning models. This ensures that the models can be easily tracked, managed, and updated.

Finally, the predictions are served via a Streamlit app, which provides an easy-to-use interface for users to interact with the model and view the predictions. The app is highly customizable, allowing the project team to create a tailored experience that meets the needs of their users. 

Below is a figure of the model architecture. 

![model_architecture.png](https://github.com/ani-portfolio/1_taxi_demand_forecasting/blob/5_deploy_app/docs/model_architecture.png)


The MLOps tools used in this project include:

- Hopsworks Feature Store
- Hopsworks Model Registry
- GitHub Actions for automated Feature Store population
- Streamlit App Framework

Overall, this model architecture provides an end-to-end machine learning solution for predicting NYC taxi demand with an hourly granularity and prediction horizon of 1 hour.

### Evaluation

**Mean Absolute Error**

MAE stands for Mean Absolute Error. It is a metric used to evaluate the performance of a regression model. The metric is calculated by taking the average of the absolute differences between the predicted values and the actual values. A lower MAE indicates that the model is better at predicting the target variable. 

The formula for MAE is;

$$ \frac{1}{n}\sum_{i=1}^{n}|x_i-y_i| $$

Where n is the number of samples, $x$ is the prediction and $y$ is the actual value. 

The MAE can be used to evaluate the performance of the baseline models, and how the LightGBM model performs against simple baseline models. 

---

# <div align="center">RESULTS

Multiple iterations were attempted in order to minimize the Mean Absolute Error (MAE) score. The results are tabulated below. 

| Model | Description | MAE | Percent Difference |
| --- | --- | --- | --- |
| Baseline Model 1 | Last Observation Carried Forward | 6.1138 |  |
| Baseline Model 2 | Actual demand observed at t-7 days | 3.4420 | -43.70% |
| Baseline Model 3 | Actual demand observed at t-7 days, t-14 days, t-21 days, t-28 | 3.0108 | -12.53% |
| XGBoost Iteration 1 | Out of the box | 2.6277 | -12.72% |
| LightGBM Iteration 2 | Out of the box | 2.5524 | -2.87% |
| LightGBM Iteration 3 | With Feature Engineering | 2.5597 | 0.29% |
| LightGBM Iteration 4 | With Hyper-parameter Tuning (10 Optuna Trials) | 2.5062 | -2.09% |

Each successive iteration results in a reduction in error, except for the LightGBM Iteration 3, where the error increases as compared to the previous iteration. However, this is probably because the ‘out-of-the-box’ hyper-parameters weren’t suited for that particular iteration. 

Once the hyper-parameters were tuned in the last iteration, a further reduction in error can be observed. 10 Optuna trials were attempted for the last iteration.