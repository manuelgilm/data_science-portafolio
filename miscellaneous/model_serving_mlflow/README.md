
# Serving multiple Models with MLflow

The code in this project showcases the utilization of MLflow for deploying multiple models and serving them through a single endpoint. It's important to note that the project doesn't employ real-world data. Instead, it leverages the scikit-learn datasets module to generate synthetic data. Keep in mind that this is a toy project and might not necessarily represent the best approach for deploying models in a production environment. Nonetheless, the code adheres to best practices in terms of reusability, reliability, and coding style.

It's worth mentioning that the project doesn't delve into data analysis, data exploration, or feature engineering processes. Its primary aim is to explore the capabilities of MLflow.


## Run Locally

**Clone the project**

```bash
  git clone https://github.com/manuelgilm/data_science-portafolio.git
```

**Go to the project directory**

```bash
  cd machine_learning/model_serving_mlflow
```
It is advisable to create a virtual environment before running the project code.

**Install dependencies**

```bash
  pip install build
  install_package.bat
```
The script above will install a custom package `training_package`. This custom package automatically installs packages such as scikit-learn, mlflow, pandas, etc. You might need to install `build` package before running the batch script.

**Run the training code.**

```bash
  python run.py
```

**Deploy the model**


```bash
  mlflow models serve --model-uri runs:/<run-id>/multimodel
```


**Run the inference code**

```bash
  python inference.py
```


## Support

For support, email manuelgilsitio@gmail.com or www.linkedin.com/in/manuelgilmatheus.


## Authors

- [@manuelgil](https://github.com/manuelgilm)


## Documentation.


- [Serving multiple models with MLflow](https://medium.com/@manuel-gilm/serving-multiple-models-with-mlflow-8311ba7939c7)
- [mlflow.pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html?highlight=pythonmodel#mlflow.pyfunc.PythonModel)
- [Serve multiple models to a Model Serving endpoint](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/serve-multiple-models-to-serving-endpoint)

- [MLflow for Machine Learning Development](https://youtube.com/playlist?list=PLQqR_3C2fhUUkoXAcomOxcvfPwRn90U-g&si=kppeeKQzP-ar2rHv)