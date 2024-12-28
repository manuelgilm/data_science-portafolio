# Simple FastAPI application.

# Description

This application is a simple API developed using FastAPI. It is designed to test the implementation of machine learning (ML) monitoring with ML endpoints. The API provides endpoints to interact with an ML model, allowing users to make predictions and monitor the performance of the model in real-time.

# Installation

### Installing with Poetry

This assumes that you have poetry installed.
```
cd data_science-portfolio/mlops/iris_api
poetry install
```

**Running the project.**

```
poetry run fastapi dev app --port 8000
```

### Using Docker


You can also run the project using Docker. To do so, you can run the following command:

```
cd data_science-portfolio/mlops/iris_api
docker build -t iris_api .
```

then:
```
docker run -p8000:5000 iris_api
```
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

You can reach me at: 

- [manuelgilsitio@gmail.com](mailto:manuelgilsitio@gmail.com)
- [Linkedin](www.linkedin.com/in/manuelgilmatheus)