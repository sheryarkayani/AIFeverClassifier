from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
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
    # Symptoms
    headache: int
    joint_pain: int
    rash: int
    vomiting: int
    fatigue: int
    chills: int
    fever_pattern: int # 0=Constant, 1=Intermittent, 2=Spikes
    # Exposure
    travel_to_hot_area: int
    mosquito_exposure: int
    sun_exposure: int
    hygiene_issue: int
    # Labs (Optional)
    wbc: Optional[float] = None
    platelets: Optional[float] = None
    haemoglobin: Optional[float] = None

def calculate_severity(prediction: str, data: dict) -> str:
    """
    Determines severity level based on diagnosis and vitals.
    """
    temp = data.get('temperature', 0)
    platelets = data.get('platelets', 250000) # Default if None
    wbc = data.get('wbc', 7000)
    
    # Defaults in case of None (handled by Pydantic but good to be safe)
    if platelets is None: platelets = 250000
    if wbc is None: wbc = 7000

    severity = "Low"

    # High Severity Rules
    if prediction == "heat-stroke":
        if temp > 40: return "High"
    
    if prediction == "dengue":
        if platelets < 50000 or temp > 40: return "High"
        if platelets < 100000: return "Medium"

    if prediction == "malaria":
        if temp > 40.5: return "High"
        if temp > 39.5: return "Medium"

    if prediction == "typhoid":
        if temp > 40: return "High"
        if data.get('vomiting') == 1 and data.get('fatigue') == 1: return "Medium"

    # General High Fever
    if temp > 40.5: return "High"
    if temp > 39.5 and severity == "Low": return "Medium"

    return severity

def get_advice(prediction: str, severity: str) -> str:
    """
    Returns actionable advice based on diagnosis and severity.
    """
    if severity == "High":
         return "URGENT: Seek immediate medical attention. Your symptoms and vitals suggest a severe condition requiring professional management."
    
    advice_map = {
        "dengue": "Stay hydrated and rest. Monitor platelet count closely. Avoid NSAIDs (like Ibuprofen/Aspirin). Consult a doctor.",
        "malaria": "Consult a doctor for antimalarial medication. Complete the full course of treatment as prescribed.",
        "typhoid": "Consult a doctor for antibiotics. detailed attention to hygiene and diet is required.",
        "viral": "Self-limiting. Rest, hydration, and antipyretics (Paracetamol) for fever control. Isolate if contagious.",
        "heat-stroke": "Move to a cool place immediately. Hydrate with water or electrolytes. Apply cool cloths to skin."
    }
    
    base_advice = advice_map.get(prediction, "Consult a healthcare provider for a detailed diagnosis.")
    
    if severity == "Medium":
        return f"Monitor Closely: {base_advice} If symptoms worsen (e.g., persistent vomiting, higher fever), visit a hospital."
    
    return f"Home Care: {base_advice}"

@app.post("/predict")
async def predict(data: PatientData):
    try:
        # Convert pydantic model to dict
        data_dict = data.dict()
        
        # Handle optional labs imputation for model
        # The model expects these values, so we impute defaults if missing just for prediction
        # (Real app might use a more sophisticated imputer trained on the data)
        model_input = data_dict.copy()
        if model_input['wbc'] is None: model_input['wbc'] = 7000.0
        if model_input['platelets'] is None: model_input['platelets'] = 250000.0
        # Removing haemoglobin if not in model
        # The training logic in data.py includes haemoglobin in DF but strict features list exclusion?
        # Let's check data.py... feature_cols does NOT include haemoglobin currently.
        # So we don't need to pass it to the model.
        
        # Get prediction
        prediction, probabilities = predict_fever(model_input)
        
        # Calculate Metadata
        severity = calculate_severity(prediction, data_dict)
        advice = get_advice(prediction, severity)
        
        return {
            "prediction": prediction,
            "probabilities": probabilities,
            "severity": severity,
            "advice": advice
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
