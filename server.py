from flask import Flask, render_template, request
import pandas as pd
import sqlite3
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the diabetes dataset from CSV file stored in github
url = "https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/diabetes1.csv"
diab = pd.read_csv(url)

# Preprocessing the data
X = diab[diab.columns[:8]]  # Features
y = diab['Outcome']  # Target variable

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Create a database and table to store the prediction results
def init_db():
    conn = sqlite3.connect('diabetes_predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pregnancies REAL,
                    glucose REAL,
                    blood_pressure REAL,
                    skin_thickness REAL,
                    insulin REAL,
                    bmi REAL,
                    diabetes_pedigree_function REAL,
                    age REAL,
                    prediction TEXT)''')
    conn.commit()
    conn.close()

# Initialize the database when the app starts
init_db()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    result = None  # Initialize result variable
    records = []

    if request.method == "POST":
        try:
            # Get form data
            pregnancies = float(request.form["pregnancies"])
            glucose = float(request.form["glucose"])
            blood_pressure = float(request.form["blood_pressure"])
            skin_thickness = float(request.form["skin_thickness"])
            insulin = float(request.form["insulin"])
            bmi = float(request.form["bmi"])
            diabetes_pedigree_function = float(request.form["diabetes_pedigree_function"])
            age = float(request.form["age"])

            # Prepare the input data for prediction
            test_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
            
            # Make prediction using the trained model
            prediction = model.predict(test_data)[0]

            # Map the prediction to a more readable result
            result = "Diabetic" if prediction == 1 else "Not Diabetic"

            # Save the input data and prediction to the database
            conn = sqlite3.connect('diabetes_predictions.db')
            c = conn.cursor()
            c.execute('''INSERT INTO predictions (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, prediction)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                         (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, result))
            conn.commit()
            conn.close()

        except Exception as e:
            return render_template("index.html", error_message="Please provide valid values for all input fields.")

    # Fetch all records from the database, ordered by ID in descending order (latest record first)
    conn = sqlite3.connect('diabetes_predictions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY id DESC")
    records = c.fetchall()
    conn.close()

    # Ensure to pass the result variable here
    return render_template("index.html", prediction=result if result else None, records=records)


if __name__ == "__main__":
    app.run(debug=True)
