# index.py
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# -----------------------------------
# 1. Generate synthetic hospital data
# -----------------------------------
np.random.seed(42)
n = 500

data = pd.DataFrame({
    "age": np.random.randint(18, 90, n),
    "sex": np.random.choice(["M", "F"], n),
    "insurance": np.random.choice(["Medicare", "Private", "None"], n),
    "days_in_hospital": np.random.randint(1, 15, n),
    "num_procedures": np.random.randint(0, 5, n),
    "has_chronic_disease": np.random.choice([0, 1], n),
    "readmitted": np.random.choice([0, 1], n, p=[0.7, 0.3])
})

# Split features and target
X = data.drop("readmitted", axis=1)
y = data["readmitted"]

# --------------------------
# 2. Preprocessing pipeline
# --------------------------
numeric_features = ["age", "days_in_hospital", "num_procedures"]
categorical_features = ["sex", "insurance", "has_chronic_disease"]

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# -----------------------
# 3. Model and Training
# -----------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)

pipe = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipe.fit(X_train, y_train)

# -----------------------
# 4. Evaluate the model
# -----------------------
y_pred = pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Confusion Matrix:\n", cm)
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# -----------------------
# 5. Flask API
# -----------------------
app = Flask(__name__)

@app.route('/')
def home():
    return "🏥 Patient Readmission Risk Model API is running."

@app.route('/predict', methods=['POST'])
def predict():
    """
Expect JSON like this:
{
    "patients": [
        {
            "age": 72,
            "sex": "M",
            "insurance": "Medicare",
            "days_in_hospital": 4,
            "num_procedures": 2,
            "has_chronic_disease": 1
        }
    ]
}
    """
    try:
        data = request.get_json()
        patients = data.get('patients')
        if not patients:
            return jsonify({"error": "Missing 'patients' field"}), 400

        X_input = pd.DataFrame(patients)
        probs = pipe.predict_proba(X_input)[:, 1]
        preds = pipe.predict(X_input)

        results = []
        for p, pr, pdx in zip(patients, probs.tolist(), preds.tolist()):
            results.append({
                "input": p,
                "readmit_probability": round(float(pr), 3),
                "readmit_prediction": int(pdx)
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
