# Wine Quality Prediction

## Introduction

This project is to predict the quality of wine based on the physicochemical tests. The dataset is from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality). The dataset contains 11 variables and 4898 observations. The variables are:

1. fixed acidity
2. volatile acidity
3. citric acid
4. residual sugar
5. chlorides
6. free sulfur dioxide
7. total sulfur dioxide
8. density
9. pH
10. sulphates
11. alcohol
12. quality (score between 0 and 10)

## Model

The model is built using scikit-learn pipeline. The pipeline includes the following steps:

1. replace missing values with median
2. HotEncode the categorical variables. In this case there are no categorical variables.
3. Use Random Forest to predict the quality of wine.

## Optimization

The model is optimized using Hyperopt. The hyperparameters optimized are:

1. n_estimators: number of trees in the forest
2. max_depth: maximum depth of the tree

The optimization is done using Tree of Parzen Estimators (TPE) algorithm. The optimization is done using 20 iterations.


