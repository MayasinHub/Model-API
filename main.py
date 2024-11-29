import os
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from starlette.middleware.cors import CORSMiddleware

# Define FastAPI app
app = FastAPI()

# Allow cross-origin requests from localhost (or your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust for your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model and encoder files
MODEL_PATH = "student-predictor.pkl"
ENCODER_PATH = "label-encoder.pkl"

# Load the model and label encoder
def load_model_and_encoder():
    try:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
        with open(ENCODER_PATH, "rb") as encoder_file:
            encoder = pickle.load(encoder_file)
        return model, encoder
    except FileNotFoundError:
        raise RuntimeError(f"Model or encoder file not found at {MODEL_PATH} or {ENCODER_PATH}.")

model, label_encoder = load_model_and_encoder()

# Define input schema
class StudentData(BaseModel):
    Course: int
    Daytime_evening_attendance: int
    Previous_qualification: int
    Previous_qualification_grade: float
    Admission_grade: float
    Educational_special_needs: int
    Tuition_fees_up_to_date: int
    Gender: int
    Scholarship_holder: int
    Age_at_enrollment: int
    Curricular_units_1st_sem_credited: int
    Curricular_units_1st_sem_enrolled: int
    Curricular_units_1st_sem_evaluations: int
    Curricular_units_1st_sem_approved: int
    Curricular_units_1st_sem_grade: float
    Curricular_units_1st_sem_without_evaluations: int
    Curricular_units_2nd_sem_credited: int
    Curricular_units_2nd_sem_enrolled: int
    Curricular_units_2nd_sem_evaluations: int
    Curricular_units_2nd_sem_approved: int
    Curricular_units_2nd_sem_grade: float
    Curricular_units_2nd_sem_without_evaluations: int

@app.post("/predict/")
def predict_student_status(data: StudentData):
    """Predict student status using the trained model."""
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([data.dict()])
        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make prediction
        prediction = model.predict(input_data)
        predicted_status = label_encoder.inverse_transform(prediction)

        return {"predicted_status": predicted_status[0]}
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

        # Separate features and target
        X = new_data.drop("Target", axis=1)
        y = new_data["Target"]

        # Encode target labels
        global label_encoder
        label_encoder = pickle.load(open(ENCODER_PATH, "rb"))  # Reuse existing encoder
        y_encoded = label_encoder.transform(y)

        # Align features with the model
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

        # Split data into train and test sets
        trainX, testX, trainY, testY = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Retrain the model
        global model
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
        model.fit(trainX, trainY)

        # Evaluate the model
        accuracy = accuracy_score(testY, model.predict(testX))
        report = classification_report(testY, model.predict(testX), target_names=label_encoder.classes_)

        # Save the retrained model
        with open(MODEL_PATH, "wb") as model_file:
            pickle.dump(model, model_file)

        return {"message": "Model retrained successfully", "accuracy": accuracy, "classification_report": report}
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
    return {"message": "Welcome to the Student Dropout and success Prediction API"}
