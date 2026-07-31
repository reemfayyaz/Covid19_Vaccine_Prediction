"""
COVID-19 Vaccine Acceptance Predictor
--------------------------------------
Flask web app that serves predictions from a pre-trained
scikit-learn RandomForestClassifier (Covid_19_model.pkl).

Run locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load the trained model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Covid_19_model.pkl")
model = joblib.load(MODEL_PATH)

# Feature order the model expects (from model.feature_names_in_)
FEATURE_ORDER = [
    "Age",
    "Gender",
    "State",
    "Chronic_Disease",
    "Previous_COVID",
    "Trust_in_Vaccine",
    "Fear_of_Side_Effects",
    "Vaccine_Type_Preference",
]

# ---------------------------------------------------------------------------
# IMPORTANT: category encodings
# ---------------------------------------------------------------------------
# The model was trained on numeric-encoded categorical columns. Update the
# mappings below so they EXACTLY match the encoding used at training time
# (e.g. the LabelEncoder / manual mapping you used in your notebook).
# ---------------------------------------------------------------------------
GENDER_MAP = {
    "Male": 0,
    "Female": 1,
    "Other": 2,
}

YES_NO_MAP = {
    "No": 0,
    "Yes": 1,
}

VACCINE_TYPE_MAP = {
    "No Preference": 0,
    "Covishield": 1,
    "Covaxin": 2,
    "Pfizer": 3,
    "Moderna": 4,
}

# Update this list with the actual states used in your training data
STATE_MAP = {
    "State A": 0,
    "State B": 1,
    "State C": 2,
    "State D": 3,
}

TRUST_FEAR_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        genders=GENDER_MAP.keys(),
        yes_no=YES_NO_MAP.keys(),
        vaccine_types=VACCINE_TYPE_MAP.keys(),
        states=STATE_MAP.keys(),
        levels=TRUST_FEAR_MAP.keys(),
        prediction=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        gender = GENDER_MAP[request.form["gender"]]
        state = STATE_MAP[request.form["state"]]
        chronic_disease = YES_NO_MAP[request.form["chronic_disease"]]
        previous_covid = YES_NO_MAP[request.form["previous_covid"]]
        trust_in_vaccine = TRUST_FEAR_MAP[request.form["trust_in_vaccine"]]
        fear_of_side_effects = TRUST_FEAR_MAP[request.form["fear_of_side_effects"]]
        vaccine_type_preference = VACCINE_TYPE_MAP[request.form["vaccine_type_preference"]]

        features = np.array(
            [[
                age,
                gender,
                state,
                chronic_disease,
                previous_covid,
                trust_in_vaccine,
                fear_of_side_effects,
                vaccine_type_preference,
            ]]
        )

        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

        result = {
            "label": "Likely to Accept Vaccine" if pred == 1 else "Likely to Be Hesitant",
            "confidence": round(float(max(proba)) * 100, 2),
        }

        return render_template(
            "index.html",
            genders=GENDER_MAP.keys(),
            yes_no=YES_NO_MAP.keys(),
            vaccine_types=VACCINE_TYPE_MAP.keys(),
            states=STATE_MAP.keys(),
            levels=TRUST_FEAR_MAP.keys(),
            prediction=result,
        )
    except Exception as e:
        return render_template(
            "index.html",
            genders=GENDER_MAP.keys(),
            yes_no=YES_NO_MAP.keys(),
            vaccine_types=VACCINE_TYPE_MAP.keys(),
            states=STATE_MAP.keys(),
            levels=TRUST_FEAR_MAP.keys(),
            prediction=None,
            error=str(e),
        )


if __name__ == "__main__":
    app.run(debug=True)
