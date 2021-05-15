from fastapi import FastAPI
from fastapi.responses import JSONResponse
from iris_classifier_object import Iris_Classifier
from pydantic import BaseModel

#create the application
app = FastAPI(
    title = "Irir Classifier API",
    version = 1.0,
    description = "Simple API to make predict class of iris plant."
)

#creating the classifier
classifier = Iris_Classifier("iris_classifier.pkl")

#Model
class Iris(BaseModel):
    sepal_length:float
    sepal_width:float
    petal_length:float
    petal_width:float

@app.post("/",tags = ["iris_classifier"])
def get_prediction(features:Iris):
    species_pred = classifier.make_prediction(features.dict())
    return JSONResponse({"species":species_pred})
