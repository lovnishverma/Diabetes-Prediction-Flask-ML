from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load the diabetes dataset from CSV file
url = "https://raw.githubusercontent.com/sarwansingh/Python/master/ClassExamples/data/diabetes1.csv"
df = pd.read_csv(url)

# Preprocess the data
X = df.drop("Outcome", axis=1)  # Features
y = df["Outcome"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    records = []
    record_id = 1  # Initialize record ID counter

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
            input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
            input_data_scaled = scaler.transform(input_data)

            # Make prediction
            prediction = model.predict(input_data_scaled)[0]

            # Map the prediction to a more readable result
            result = "Diabetic" if prediction == 1 else "Not Diabetic"

            # Save the input data and prediction along with a unique ID
            records.append([record_id, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, result])

            # Increment record_id for the next entry
            record_id += 1

        except Exception as e:
            return render_template("index.html", error_message="Please provide valid values for all input fields.", records=records)

    return render_template("index.html", prediction=result if prediction else None, records=records)

if __name__ == "__main__":
    app.run(debug=True)
