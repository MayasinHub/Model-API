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

# Allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model file and dataset
MODEL_PATH = "student-predictor.pkl"
DATASET_PATH = "student-data.csv"

# Global variable to hold the expected features
expected_features = []

# Load the initial model
def load_model():
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

            # Extract feature names from the trained model
            if hasattr(model, "feature_names_in_"):
                global expected_features
                expected_features = model.feature_names_in_.tolist()  # This ensures exact feature order
            else:
                raise RuntimeError("Sorry! feature names are not available in the saved model.")

            return model
    except FileNotFoundError:
        raise RuntimeError(f"Model file {MODEL_PATH} not found.")

# Load the RandomForest model
model = load_model()

# Define input schema
class StudentData(BaseModel):
    Course: int
    Daytime_evening_attendance: int
    Previous_qualification: int
    Previous_qualification_grade: int
    Admission_grade: int
    Educational_special_needs: int
    Tuition_fees_up_to_date: int
    Gender: int
    Scholarship_holder: int
    Age_at_enrollment: int
    Curricular_units_1st_sem_evaluations: int
    Curricular_units_1st_sem_approved: int
    Curricular_units_1st_sem_grade: int
    Curricular_units_1st_sem_without_evaluations: int
    Curricular_units_2nd_sem_credited: int
    Curricular_units_2nd_sem_enrolled: int
    Curricular_units_2nd_sem_evaluations: int
    Curricular_units_2nd_sem_approved: int
    Curricular_units_2nd_sem_grade: int
    Curricular_units_2nd_sem_without_evaluations: int

@app.post("/predict/")
def predict_student_status(data: StudentData):
    """Predict student status using the trained model."""
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Reorder input data to match the model's expected feature names
        input_data = input_data.reindex(columns=expected_features, fill_value=0)

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

        # Ensure "Target" column exists
        if "Target" not in new_data.columns:
            raise HTTPException(status_code=400, detail="Dataset must include 'Target' column.")

        # Align the new dataset to match the original feature names
        X = new_data.drop("Target", axis=1).reindex(columns=expected_features, fill_value=0)
        y = new_data["Target"]

        # Train/test split
        trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

        # Retrain the model
        global model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(trainX, trainY)

        # Save the retrained model
        with open(MODEL_PATH, "wb") as model_file:
            pickle.dump(model, model_file)

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
    return {"message": "Karibu! Welcome to the Student Dropout and Success Prediction API"}
