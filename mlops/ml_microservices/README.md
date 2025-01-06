# Machine Learning Models as Microservices.
This project demonstrates how to deploy machine learning models as microservices. By containerizing models and using RESTful APIs, we can easily integrate machine learning capabilities into various applications, ensuring scalability, maintainability, and ease of deployment.

## Project Architecture.

<TODO>

## How to test.
To run the code, it is necessary to install Docker. For the next steps, assume that Docker is already installed.

1. Clone the Repository.
```
git clone https://github.com/manuelgilm/data_science-portafolio.git`
```
2. Go to the respective directory.
```
cd data_science-portafolio/mlops/ml_microservices
```
3. Run Docker.
```
docker compose up --build
```
4. Go to the browser and enter `http:localhost:8000/docs` and you will see FastAPI Documentation. 

### Using API Docs.

#### Create an User:

Go to signup endpoint.
![alt text](./images/image.png)

Create a new user.
![alt text](./images/image-1.png)

Once the user is created. Go to login endpoint.
![alt text](./images/image-2.png)

Use the just created user.
![alt text](./images/image-3.png)

Copy the generated `access_token`
![alt text](./images/image-4.png)

Authorize using the UI.
![alt text](./images/image-5.png)

Now you can  use the ML services. For example, Service 1.
![alt text](./images/image-6.png)

Done! Now you have the prediction.
![alt text](./images/image-7.png)

## Contributing
If you want to contribute to this project and make it better, your help is very welcome. Contributing is also a great way to learn more and improve your skills. You can contribute in different ways:

* Reporting a bug
* Coming up with a feature request
* Writing code
* Writing tests
* Writing documentation
* Reviewing code
* Giving feedback on the project
* Spreading the word
* Sharing the project

## Contact
If you need to contact me, you can reach me at:

* [manuelgilsitio@gmail.com](manuelgilsitio@gmail.com)
* [linkedin](www.linkedin.com/in/manuelgilmatheus)
