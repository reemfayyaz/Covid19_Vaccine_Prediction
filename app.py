from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="COVID-19 Vaccine Prediction",
    page_icon="💉",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "Covid_19_model.pkl",
    BASE_DIR / "Covid_19_model (4)(1).pkl",
]

ENCODER_CANDIDATES = [
    BASE_DIR / "feature_label_encoders.pkl",
    BASE_DIR / "feature_label_encoders (1)(1).pkl",
]


@st.cache_resource
def load_pickle(candidates: list[Path]):
    """Load the first available pickle file from the candidate paths."""
    for path in candidates:
        if path.exists():
            return joblib.load(path)

    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"Could not find any of these files: {expected}")


try:
    model = load_pickle(MODEL_CANDIDATES)
    encoders = load_pickle(ENCODER_CANDIDATES)
except Exception as error:
    st.error(f"Unable to load the model files: {error}")
    st.info("Place the model and encoder .pkl files in the same folder as app.py.")
    st.stop()


st.title("💉 COVID-19 Vaccine Prediction")
st.write("Enter the information below and click **Predict**.")

with st.form("prediction_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)

    gender = st.selectbox("Gender", encoders["Gender"].classes_.tolist())
    state = st.selectbox("State", encoders["State"].classes_.tolist())
    chronic_disease = st.selectbox(
        "Chronic Disease", encoders["Chronic_Disease"].classes_.tolist()
    )
    previous_covid = st.selectbox(
        "Previous COVID-19 Infection", encoders["Previous_COVID"].classes_.tolist()
    )

    trust_in_vaccine = st.slider(
        "Trust in Vaccine", min_value=1, max_value=5, value=3,
        help="1 = Very low trust, 5 = Very high trust",
    )
    fear_of_side_effects = st.slider(
        "Fear of Side Effects", min_value=1, max_value=5, value=3,
        help="1 = Very low fear, 5 = Very high fear",
    )

    vaccine_preference = st.selectbox(
        "Vaccine Type Preference",
        encoders["Vaccine_Type_Preference"].classes_.tolist(),
    )

    submitted = st.form_submit_button("Predict", use_container_width=True)


if submitted:
    input_data = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [encoders["Gender"].transform([gender])[0]],
            "State": [encoders["State"].transform([state])[0]],
            "Chronic_Disease": [
                encoders["Chronic_Disease"].transform([chronic_disease])[0]
            ],
            "Previous_COVID": [
                encoders["Previous_COVID"].transform([previous_covid])[0]
            ],
            "Trust_in_Vaccine": [trust_in_vaccine],
            "Fear_of_Side_Effects": [fear_of_side_effects],
            "Vaccine_Type_Preference": [
                encoders["Vaccine_Type_Preference"].transform(
                    [vaccine_preference]
                )[0]
            ],
        }
    )

    try:
        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")
        prediction_text = str(prediction).strip().lower()

        if prediction_text in {"1", "yes", "true", "willing"}:
            st.success("The person is likely to take the COVID-19 vaccine.")
        elif prediction_text in {"0", "no", "false", "not willing"}:
            st.warning("The person may not be willing to take the COVID-19 vaccine.")
        else:
            st.success(f"Predicted class: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            predicted_index = list(model.classes_).index(prediction)
            confidence = probabilities[predicted_index] * 100
            st.metric("Prediction Confidence", f"{confidence:.2f}%")

    except Exception as error:
        st.error(f"Prediction failed: {error}")
