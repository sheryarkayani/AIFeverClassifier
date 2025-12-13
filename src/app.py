from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.model import predict_fever

app = FastAPI(title="Fever Classifier AI")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientData(BaseModel):
    Age: int
    Sex: int # 1 for Male, 0 for Female
    temperature: float
    wbc: float
    platelets: float
    headache: int
    joint_pain: int
    rash: int
    travel_to_hot_area: int
    mosquito_exposure: int

@app.post("/predict")
async def predict(data: PatientData):
    try:
        # Convert pydantic model to dict
        data_dict = data.dict()
        
        # Get prediction
        prediction, probabilities = predict_fever(data_dict)
        
        return {
            "prediction": prediction,
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
