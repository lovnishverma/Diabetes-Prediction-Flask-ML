from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the diabetes dataset from CSV file
dataset = "diabetes1.csv"
diab = pd.read_csv(dataset)

# Preprocessing the data
X = diab[diab.columns[:8]]  # Features
y = diab['Outcome']  # Target variable i.e Labels

# X: Contains the features (independent variables) extracted from the first 8 columns of the dataset.
# y: Contains the target variable (dependent variable) from the 'Outcome' column.

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    result = None

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

        except Exception as e:
            return render_template("index.html", error_message="Please provide valid values for all input fields.")

    # Ensure to pass the result variable here
    return render_template("index.html", prediction=result if result else None)

if __name__ == "__main__":
    app.run(debug=True)
