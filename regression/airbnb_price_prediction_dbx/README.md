# Airbnb Price Prediction Project

**NOTE:**

This project has been developed for learning purposes. This may not be the best approach to solve this problem. However, it demonstrates a comprehensive series of steps that can be followed when developing projects using dbx, feature_store and mlflow.
----

This project aims to predict the price of Airbnb listings using historical data. The prediction model is built using Python, PySpark, DBX (Databricks), Feature Store, and MLflow.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview
In this project, we leverage historical data from Airbnb listings to train a machine learning model that can predict the price of new listings. The project utilizes PySpark for distributed data processing, DBX as the development environment, Feature Store for managing features, and MLflow for experiment tracking and model management.

## Installation
To set up the project, follow these steps:

1. Clone the repository: `git clone <repository-url>`
2. Install the required dependencies: `pip install -r requirements.txt`

## Usage

### Databricks settings. 
1. To setup the databricks-cli with your current databricks workspace: 

    ```
    datarbicks configure token
    Databricks Host (should begin with https://):
    Token:
    ```
    
You can create a databricks token following:    
* Go to User Settings.
* Click on Generate New Token.
* Fill the required fields.
* Save the token.

2. To setup dbx. 

``` 
dbx configure 
```
To use the project, follow these steps:
Note: You need to provide the cluster id to run the batch scripts. Example:

```
>> run.bat <my_cluster_id>
```

1. Prepare the data: 
    * Run `feature_preparation.py` job. To do this go to the `run.bat` script and uncomment the respective line: 
    ```call dbx execute --cluster-id=%1 --job=feature_preparation --no-package```

2. Train the model: Once the features have been created we can use them to train a machine learning model. You can train a model using crossvalidation or not. To do this uncomment the respective lines in the `run.bat`script.

```
call dbx execute --cluster-id=%1 --job=pyspark_pipeline --no-package
or
call dbx execute --cluster-id=%1 --job=crossvalidation_pipeline --no-package
```

3. Inference: To run the inference job 

## Data
The data used for this project consists of historical Airbnb listings. [airbnb data](http://insideairbnb.com/get-the-data/)

## Model Training
The model is trained using PySpark, a distributed computing framework designed for big data processing. Since the label (price) is a continuous real value, a regressor was employed for building the prediction model. Regression algorithms are specifically suited for predicting continuous numeric values, making them an appropriate choice for this task.

## Evaluation
To evaluate the model's performance, we use various metrics such as root mean squared error (RMSE), and R-squared.

## Contributing
We welcome contributions to this project! If you'd like to contribute, please follow these steps:
- Fork the repository
- Create a new branch
- Make your changes
- Submit a pull request

## License
This project is distributed under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute the code in this project for both commercial and non-commercial purposes. Please see the [LICENSE](LICENSE) file for more details.

## Resources:

* [dbx by Databricks Labs](https://dbx.readthedocs.io/en/latest/)
* [databricks-cli](https://learn.microsoft.com/en-us/azure/databricks/dev-tools/cli/databricks-cli)
* [mlflow](https://mlflow.org/)