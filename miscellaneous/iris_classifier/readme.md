## Model generation.

The script called `model_creation.py` is in charge of creating the machine learning model. When you run this script, it will generate a pickle file with a classifier model, this example uses the popular Iris dataset. This model classifies plant species using both sepal and petal shapes of Iris plant.

![Iris plants](iris_plants.png)

The iris plant features are:
+ Sepal length.
+ Sepal width.
+ Petal length.
+ Petal width.
All this features are in cm.

## API

![FastAPI](fastapi_logo.png)

The API was created with FastAPI, a modern web framework which allows build APIs, in this case to deploy machine learning models. The folder `app_classifier` contains all the necessary files to deploy this API on Heroku. 

## References.

+ [FastAPI website](https://fastapi.tiangolo.com/)


