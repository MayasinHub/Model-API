from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Initialize FastAPI app
app = FastAPI(
    title="Student Dropout Prediction API",
    description="API for predicting student dropout or success based on various features",
)

# Load the trained model
try:
    model = joblib.load("student-predictor.pkl")
except FileNotFoundError:
    raise RuntimeError("Model file not found. Ensure 'model.pkl' is in the correct directory.")

# Define input data model
class StudentData(BaseModel):
    marital_status: int = Field(..., ge=0, le=1, description="Marital status (0 for single, 1 for married)")
    application_mode: int = Field(..., ge=1, le=1000, description="Application mode (number between 1-1000)")
    application_order: int = Field(..., ge=1, le=100, description="Application order (number between 1-100)")
    course: int = Field(..., ge=1, le=10000, description="Course (number between 1-10000)")
    daytime_evening_attendance: int = Field(..., ge=0, le=1, description="Daytime/evening attendance (0 for daytime, 1 for evening)")
    previous_qualification: int = Field(..., ge=0, le=1, description="Previous qualification (0 for yes, 1 for no)")
    nacionality: int = Field(..., ge=0, le=1, description="Nacionality (0 for local, 1 for foreigner)")
    unemployment_rate: float = Field(..., ge=1.0, le=50.0, description="Unemployment rate (float between 1.0 and 50.0)")
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
    
    # Preprocess the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    predicted_class = int(np.argmax(prediction, axis=-1)[0])
    class_mapping = {0: "Dropout", 1: "Graduate", 2: "Enrolled"}
    
    return {
        "predicted_class": predicted_class,
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

    # Scale the features
    global scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define and compile the model
    global model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(10,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: Dropout, Graduate, Enrolled
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

    # Save the updated model and scaler
    model.save('models/student_success_model.pkl')

    return {
        "message": "Model retrained successfully",
        "accuracy": accuracy,
        "loss": loss
    }

@app.get("/")
async def root():
    return {"message": "Welcome to the Student Dropout or success Prediction API"}
