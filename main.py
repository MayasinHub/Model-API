# import libraries
import os
import pickle
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.responses import FileResponse
from pydantic import BaseModel, create_model, ValidationError
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from threading import Lock

# Define FastAPI app
app = FastAPI(
    title="Student Dropout Prediction API",
    description="An API to predict student outcomes and retrain the model.",
    version="1.1.0",
)

# Paths to model and encoder files (use environment variables)
MODEL_PATH = os.getenv("MODEL_PATH", "student-predictor.pkl")
ENCODER_PATH = os.getenv("ENCODER_PATH", "label-encoder.pkl")
TRAIN_CSV_PATH = os.getenv("TRAIN_CSV_PATH", "X_train_standardized.csv")

# Thread lock for model retraining
model_lock = Lock()

# Load the model and label encoder
def load_model_and_encoder():
    try:
        with open(MODEL_PATH, "rb") as model_file:
            loaded_model = pickle.load(model_file)
        with open(ENCODER_PATH, "rb") as encoder_file:
            loaded_encoder = pickle.load(encoder_file)
        return loaded_model, loaded_encoder
    except FileNotFoundError:
        raise RuntimeError(f"Model or encoder file not found at {MODEL_PATH} or {ENCODER_PATH}.")

model, label_encoder = load_model_and_encoder()

# Load standardized training data to get feature names
try:
    standardized_train_data = pd.read_csv(TRAIN_CSV_PATH)
    feature_columns = standardized_train_data.columns.tolist()
except FileNotFoundError:
    raise RuntimeError(f"Standardized train CSV file not found at {TRAIN_CSV_PATH}.")

# Dynamically create Pydantic model for the input data
StudentData = create_model(
    "StudentData",
    **{col: (float, ...) for col in feature_columns}  # All fields are required and of type float
)

@app.post("/predict/", summary="Predict Student Status", tags=["Prediction"])
def predict_student_status(data: List[StudentData]):
    """
    Predict student status using the trained model.
    Accepts a list of records for batch predictions.
    """
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([item.dict() for item in data])
        input_data = input_data.reindex(columns=feature_columns, fill_value=0)

        # Validate input data ranges (example range validation)
        if not (input_data.ge(0).all().all() and input_data.le(10000).all().all()):
            raise HTTPException(status_code=400, detail="Input features must be between 0 and 1.")

        # Make predictions
        predictions = model.predict(input_data)
        predicted_statuses = label_encoder.inverse_transform(predictions)

        return {"predictions": predicted_statuses.tolist()}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/retrain/", summary="Retrain any student model with Target feature", tags=["Retraining"])
def retrain_model(file: UploadFile = File(...)):
    """
    Retrain the model with a new dataset.
    The uploaded file must be a CSV containing a 'Target' column.
    """
    global model, label_encoder
    if model_lock.locked():
        raise HTTPException(status_code=503, detail="Wait! Model retraining is already in progress.")
    with model_lock:
        try:
            # Validate file type
            if not file.filename.endswith(".csv"):
                raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

            # Load new dataset
            new_data = pd.read_csv(file.file)
            if "Target" not in new_data.columns:
                raise HTTPException(status_code=400, detail="Dataset must include 'Target' column.")

            # Validate 'Target' values
            target_values = set(new_data["Target"].unique())
            valid_values = set(label_encoder.classes_)
            if not target_values.issubset(valid_values):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid 'Target' values. Allowed values: {valid_values}",
                )

            # Separate features and target
            X = new_data.drop("Target", axis=1)
            y = new_data["Target"]

            # Align features with the model
            X = X.reindex(columns=feature_columns, fill_value=0)

            # Encode target labels
            y_encoded = label_encoder.transform(y)

            # Stratified split for balance
            trainX, testX, trainY, testY = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

            # Retrain the model
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
            model.fit(trainX, trainY)

            # Evaluate the model using cross-validation
            skf = StratifiedKFold(n_splits=5)
            cv_accuracies = []
            for train_idx, val_idx in skf.split(X, y_encoded):
                model.fit(X.iloc[train_idx], y_encoded[train_idx])
                cv_accuracies.append(accuracy_score(y_encoded[val_idx], model.predict(X.iloc[val_idx])))

            avg_cv_accuracy = sum(cv_accuracies) / len(cv_accuracies)
            report = classification_report(testY, model.predict(testX), target_names=label_encoder.classes_, output_dict=True)

            # Save the retrained model with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_model_path = f"student-predictor_{timestamp}.pkl"
            with open(new_model_path, "wb") as model_file:
                pickle.dump(model, model_file)

            return {
                "message": "Yayy! Model retrained successfully",
                "cv_accuracy": avg_cv_accuracy,
                "classification_report": report,
                "model_path": new_model_path,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

@app.get("/download_model/", summary="Download Model as a Pickle file", tags=["Download Retrained Model"])
def download_model(model_version: str = None):
    """
    Download a specific version of the retrained model.
    Defaults to the most recent model.
    """
    model_path = model_version if model_version else MODEL_PATH
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Sorry! Model file not found.")
    return FileResponse(model_path, media_type="application/octet-stream", filename=os.path.basename(model_path))

@app.get("/", summary="Root Endpoint", tags=["Utility"])
def root():
    """Root endpoint to check the API status."""
    return {"KARIBU! Welcome to the Student Dropout and Success Prediction API"}
