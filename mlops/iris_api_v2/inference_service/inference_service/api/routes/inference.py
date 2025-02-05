from fastapi import FastAPI 
from fastapi import HTTPException
from fastapi import status 
from fastapi.responses import JSONResponse
from inference_service.api.schemas.model_signature import InferenceData

inference_router = FastAPI()

@inference_router.post("/predict")
async def predict(inference_data: InferenceData):
    try:
        # do some inference
        features = inference_data.inputs.dict()
        # model = load_latest_model("iris")
        # prediction = model.predict(features)
        # return JSONResponse(content={"prediction": prediction})
        return JSONResponse(content={"prediction": "some prediction"})
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))