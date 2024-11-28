from locust import HttpUser, TaskSet, task, between

class StudentDropoutPredictionTaskSet(TaskSet):
    @task
    def predict(self):
        # Sample data for the API (all integers as required)
        student_data = {
            "Marital_status": 0,  # e.g., 0: Single, 1: Married, etc.
            "Course": 1,          # e.g., 1: Engineering, 2: Science, etc.
            "Attendance": 80,     # Attendance percentage as an integer
            "Nacionality": 1,     # e.g., 1: Somali, 2: Rwandan, etc.
            "Unemployment_rate": 5,  # Rounded unemployment rate
            "Inflation_rate": 2,     # Rounded inflation rate
            "GDP": 30000            # GDP in integers (e.g., scaled down)
        }
        # Make a POST request to the predict endpoint
        response = self.client.post("/predict", json=student_data)

        # Optional: Print the response for debugging during testing
        print(response.text)

class StudentDropoutPredictionUser(HttpUser):
    tasks = [StudentDropoutPredictionTaskSet]
    wait_time = between(1, 5)
