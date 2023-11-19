# STORE SALES REGRESSION
---

## Description:

The purpose of this project is educational; the methods demonstrated here may not be the most appropriate for solving a specific problem, although they adhere to the standard machine learning project development workflow.

## Data:

The source data can be found [here](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data).

**NOTE** 
It's important to clarify that in this context, our focus is not on evaluating the data's suitability for a regression problem or whether it can be approached in such a manner. Rather, the primary objective of this project is to test the integration of MLflow with Hyperopt.


## About the data processing.

In this project, our approach involves aggregating all sales per store. Subsequently, this aggregated data is employed to derive time-related features, including factors such as day of the week, day of the year, and more. While these features are typically utilized for forecasting problems, in this particular case, we're employing them to enhance the quality of our feature set for simplicity and clarity of analysis.