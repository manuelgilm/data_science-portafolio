<h1 style="text-align: center;">Iris Classifier</h1>


![Iris](iris_classifier/eda/iris-dataset.png)

## Description

This project is a simple implementation of a classifier for the Iris dataset. The classifier is a simple decision tree model trained on the Iris dataset. The dataset is a small dataset that contains 150 samples of iris flowers. The dataset contains 4 features: sepal length, sepal width, petal length, and petal width. The target variable is the species of the iris flower. The dataset contains 3 classes of iris flowers: setosa, versicolor, and virginica. The classifier is trained on the dataset and can predict the species of an iris flower based on the 4 features.

## Installation

Provide instructions on how to install and run your project.

### Installing with Poetry.

```bash
cd data_science-portfolio/classification/iris_classifier
poetry install
```

### Running the project

**Entry points** 

- **train**: Train the model
- **predict**: Get predictions from the model


There are different ways to run the project. For training the model, you can run the following command:

```bash
poetry run train
```

To visualize the model you can run the following command:
```bash
poetry run mlflow ui
```

To get predictions from the model, you can run the following command:
```bash
poetry run predict
```

### Using Docker 

You can also use the project with Docker. To do this, you can run the following commands:

```bash
cd data_science-portfolio/classification/iris_classifier
docker build -t iris_classifier .
```

To train the model, you can run the following command:

```bash
docker run -e "MODE=TRAIN" -p5000:5000 iris_classifier
```
The command above will start the training process and the mlflow server. You can visualize the model by going to `http://localhost:5000` in your browser.

To get predictions from the model, you can run the following command:

```bash
docker run -e "MODE=SCORE" -p5000:5000 iris_classifier
```

Then you can make a POST request to `http://localhost:5000/invocations` for example:

```python
import requests
import json 

payload = {"dataframe_split":
    {"columns":[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
        ]
    ,"data":[
        [5.1,3.5,1.4,0.2]]
    }
}

headers = {
    'Content-Type': 'application/json',
}
response = requests.post('http://localhost:5000/invocations', headers=headers, data=json.dumps(payload))
print(response.json())
```
## Visualize Explorative Data Analysis

### Using Poetry

To visualize the explorative data analysis, you can run the following command:

```bash
cd data_science-portfolio/classification/iris_classifier
poetry run streamlit run iris_classifier/eda/vz_app.py
```

### Using Docker

To visualize the explorative data analysis, you can run the following command:

```bash
cd data_science-portfolio/classification/iris_classifier
docker run -e MODE=EDA -p5000:5000 iris_classifier
```

Then you can go to `http://localhost:5000` in your browser to visualize the explorative data analysis.

## Contributing

If you want to contribute to this project and make it better, your help is very welcome. Contributing is also a great way to learn more and improve your skills. You can contribute in different ways:

- Reporting a bug
- Coming up with a feature request
- Writing code
- Writing tests
- Writing documentation
- Reviewing code
- Giving feedback on the project
- Spreading the word
- Sharing the project
  
## Contact

If you need to contact me, you can reach me at:

- [manuelgilsitio@gmail.com](manuelgilsitio@gmail.com)
- [linkedin](www.linkedin.com/in/manuelgilmatheus)
  