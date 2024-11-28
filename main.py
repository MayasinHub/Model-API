from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Initialize FastAPI app
app = FastAPI(
    title="Student Dropout Prediction API",
    description="API for predicting student dropout or success based on various features",
)

# Load the trained model
MODEL_PATH = "student-predictor.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Ensure the .pkl file is present.")

# Define input data model
class StudentData(BaseModel):
    marital_status: int = Field(..., ge=0, le=1, description="Marital status (0 for single, 1 for married)")
    application_mode: int = Field(..., ge=1, le=1000, description="Application mode (number between 1-10000)")
    application_order: int = Field(..., ge=1, le=10000, description="Application order (number between 1-10000)")
    course: int = Field(..., ge=1, le=10000, description="Course (number between 1-10000)")
    daytime_evening_attendance: int = Field(..., ge=0, le=1, description="Daytime/evening attendance (0 for daytime, 1 for evening)")
    previous_qualification: int = Field(..., ge=0, le=1, description="Previous qualification (0 for yes, 1 for no)")
    nacionality: int = Field(..., ge=0, le=1, description="Nacionality (0 for local, 1 for foreigner)")
    unemployment_rate: float = Field(..., ge=1.0, le=50.0, description="Unemployment rate (float between 1.0 and 500.0)")
    inflation_rate: float = Field(..., description="Inflation rate (any float number)")
    gdp: float = Field(..., description="GDP (any float number)")

class TrainingData(BaseModel):
    students: List[StudentData]
    labels: List[int] = Field(..., description="Target labels (0: Dropout, 1: Graduate, 2: Enrolled)")

@app.post("/predict")
async def predict(student: StudentData):
    # Convert input data to numpy array
    input_data = np.array([[
        student.marital_status,
        student.application_mode,
        student.application_order,
        student.course,
        student.daytime_evening_attendance,
        student.previous_qualification,
        student.nacionality,
        student.unemployment_rate,
        student.inflation_rate,
        student.gdp
    ]])

    # Make prediction
    predicted_class = model.predict(input_data)[0]
    class_mapping = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}

    return {
        "predicted_class": int(predicted_class),
        "class_label": class_mapping[predicted_class]
    }

@app.post("/retrain")
async def retrain(data: TrainingData):
    # Convert input data to numpy arrays
    X = np.array([
        [
            s.marital_status,
            s.application_mode,
            s.application_order,
            s.course,
            s.daytime_evening_attendance,
            s.previous_qualification,
            s.nacionality,
            s.unemployment_rate,
            s.inflation_rate,
            s.gdp
        ]
        for s in data.students
    ])
    y = np.array(data.labels)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest model
    global model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    accuracy = model.score(X_test, y_test)

    # Save the updated model
    joblib.dump(model, MODEL_PATH)

    return {
        "message": "Model retrained successfully",
        "accuracy": accuracy
    }

@app.get("/")
async def root():
    return {"message": "Hola! Welcome to the Student Dropout r Success Prediction API"}
