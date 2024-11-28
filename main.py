import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from starlette.middleware.cors import CORSMiddleware

# Define FastAPI app
app = FastAPI()

# Allow cross-origin requests from localhost (or your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as needed for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model file and dataset
MODEL_PATH = "student-predictor.pkl"
DATASET_PATH = "student-data.csv"

# Load the initial model
def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        raise RuntimeError(f"Model file {MODEL_PATH} not found.")

# Load the RandomForest model
model = load_model()

# Define input schema
class StudentData(BaseModel):
    marital_status: int
    application_mode: int
    application_order: int
    course: int
    daytime_evening_attendance: int
    previous_qualification: int
    nacionality: int
    unemployment_rate: float
    inflation_rate: float
    gdp: float

@app.post("/predict/")
def predict_student_status(data: StudentData):
    """Predict student status using the trained model."""
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])
        # Make a prediction
        prediction = model.predict(input_data)
        status_mapping = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}
        predicted_status = status_mapping[prediction[0]]

        return {"predicted_status": predicted_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain/")
def retrain_model(file: UploadFile = File(...)):
    """Retrain the model with new data."""
    try:
        # Load new dataset
        new_data = pd.read_csv(file.file)
        if "Target" not in new_data.columns:
            raise HTTPException(status_code=400, detail="Dataset must include 'Target' column.")

        # Split data into features and target
        X = new_data.drop("Target", axis=1)
        y = new_data["Target"]

        # Train/test split
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the RandomForest model
        global model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(trainX, trainY)

        # Save the retrained model
        with open(MODEL_PATH, "wb") as file:
            pickle.dump(model, file)

        # Evaluate accuracy
        accuracy = accuracy_score(testY, model.predict(testX))
        return {"message": "Model retrained successfully", "accuracy": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/download_model/")
def download_model():
    """Download the retrained model."""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model file not found.")
    return FileResponse(MODEL_PATH, media_type="application/octet-stream", filename="student-predictor.pkl")

@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "Welcome to the Student Dropout Prediction API"}
