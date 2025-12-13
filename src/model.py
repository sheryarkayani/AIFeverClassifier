import joblib
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
# Use absolute imports if running as module, or relative ? 
# Simplest is to assume this script is run from project root as 'python -m src.model' or 'python src/model.py'
# We will use sys.path hack or relative imports if run as package
import sys

# Add project root to path if running directly
if __name__ == "__main__":
    sys.path.append(os.getcwd())

from src.data import load_and_simulate_data, get_pipeline

MODEL_PATH = "fever_model.pkl"

def train_model():
    """
    Trains the fever classification model and saves it.
    """
    print("Loading and simulating data...")
    X, y = load_and_simulate_data()
    
    print(f"Data shape: {X.shape}")
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Initializing pipeline...")
    clf = get_pipeline()
    
    print("Training model...")
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    print("Done.")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} not found. Train model first.")
    return joblib.load(MODEL_PATH)

def predict_fever(data_dict):
    """
    Predicts fever type from a dictionary of inputs.
    """
    model = load_model()
    # Convert dict to DataFrame
    df = pd.DataFrame([data_dict])
    
    # Predict
    prediction = model.predict(df)[0]
    probabilities = model.predict_proba(df)[0]
    
    # Get class labels
    classes = model.classes_
    prob_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
    
    return prediction, prob_dict

if __name__ == "__main__":
    train_model()
