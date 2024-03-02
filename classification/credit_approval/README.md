<h1 style="text-align: center;">Credit Approval</h1>


![Iris](credit_approval/notebooks/approved.jpeg)

## Description

This project is a credit approval model. The goal is to predict if a customer will be able to pay a loan or not. The dataset used in this project is from the UCI Machine Learning Repository. The dataset contains 15 attributes and 1 target variable. The target variable is a binary variable that indicates if the customer was able to pay the loan or not. 

## Installation

Provide instructions on how to install and run your project.

### Installing with Poetry.

```bash
cd data_science-portfolio/classification/credit_approval
poetry install
```

### Running the project

**Entry points** 

- **train**: Train the model
- **score_last_model**: Get predictions from the last model using the test set


There are different ways to run the project. For training the model, you can run the following command:

```bash
poetry run train
```

To get predictions from the model, you can run the following command:
```bash
poetry run score_last_model
```

### Using Docker

You can also run the project using Docker. To do so, you can run the following command:

```bash
cd data_science-portfolio/classification/credit_approval
docker build -t credit_approval .
```


To train the model, you can run the following command:

```bash
docker run -e "MODE=TRAIN" -p5000:5000 credit_approval
```

The command above will start the training process and the mlflow server. You can visualize the model by going to `http://localhost:5000` in your browser.

To get predictions from the model, you can run the following command:

```bash
docker run -e "MODE=SCORE" -p5000:5000 credit_approval
```

Then you can make a POST request to `http://localhost:5000/invocations` for example:

```python
import requests
import json 

payload = {"dataframe_split":
           {"columns":[
               "A15",
               "A14",
               "A13",
               "A12",
               "A11",
               "A10",
               "A9",
               "A8",
               "A7",
               "A6",
               "A5",
               "A4",
               "A3",
               "A2",
               "A1",
               ]
              ,"data":
                    [[0,202.0,"g","f",1,"t","t",1.25,"v","w","g","u",0.000,30.83,"b"]]
    }
}

headers = {
    'Content-Type': 'application/json',
}
response = requests.post('http://localhost:5000/invocations', headers=headers, data=json.dumps(payload))
print(response.json())
´´´

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
  