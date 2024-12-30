from flask import Flask, render_template, request
import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

# Path to the dataset (make sure the CSV file is in the same directory as the app)
CSV_FILE = "dia.csv"

# Initialize the database
def init_db():
    conn = sqlite3.connect("user_inputs.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age REAL,
            hypertension INTEGER,
            heart_disease INTEGER,
            bmi REAL,
            HbA1c_level REAL,
            blood_glucose_level REAL,
            prediction TEXT
        )
    """)
    conn.commit()
    conn.close()

# Validate input fields
def validate_input(input_data):
    required_fields = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level"]
    for field in required_fields:
        if field not in input_data or not input_data[field]:
            return False
    return True

# Train model once (you can optimize this to load the pre-trained model)
def train_model():
    # Ensure the CSV file exists
    if not os.path.exists(CSV_FILE):
        raise FileNotFoundError(f"The file {CSV_FILE} is missing.")
    
    # Load the data
    data = pd.read_csv(CSV_FILE, header=None)
    diabete = data.values

    # Features and labels
    x = diabete[:, :6]  # All columns except the last (features)
    y = diabete[:, 6]   # Last column (target)

    # Initialize and train the model
    model = LogisticRegression()
    model.fit(x, y)
    return model

# Load pre-trained model
model = train_model()

@app.route('/', methods=["POST", "GET"])
def page():
    conn = sqlite3.connect("user_inputs.db")
    cursor = conn.cursor()

    if request.method == "POST":
        input_data = {
            "age": float(request.form.get("age")),
            "hypertension": int(request.form.get("hypertension")),
            "heart_disease": int(request.form.get("heart_disease")),
            "bmi": float(request.form.get("bmi")),
            "HbA1c_level": float(request.form.get("HbA1c_level")),
            "blood_glucose_level": float(request.form.get("blood_glucose_level"))
        }

        if validate_input(input_data):
            try:
                # Prediction using pre-trained model
                prediction = model.predict([[ 
                    input_data["age"], 
                    input_data["hypertension"], 
                    input_data["heart_disease"], 
                    input_data["bmi"], 
                    input_data["HbA1c_level"], 
                    input_data["blood_glucose_level"]
                ]])
                result = str(prediction[0])

                # Save the input and prediction in the database
                cursor.execute("""
                    INSERT INTO user_data (age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level, prediction)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (input_data["age"], input_data["hypertension"], input_data["heart_disease"],
                      input_data["bmi"], input_data["HbA1c_level"], input_data["blood_glucose_level"], result))
                conn.commit()
                
            except Exception as e:
                result = "Error occurred during prediction: {str(e)}"
            
            # Fetch all records from the database
            cursor.execute("SELECT * FROM user_data")
            records = cursor.fetchall()
            conn.close()

            return render_template("index.html", data=result, records=records)
        else:
            error_message = "Please provide values for all input fields."
            return render_template("index.html", error_message=error_message)

    # For GET request, fetch all data from database
    cursor.execute("SELECT * FROM user_data")
    records = cursor.fetchall()
    conn.close()

    return render_template("index.html", error_message="", records=records)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
