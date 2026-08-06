import os
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Vaccine Prediction",
    page_icon="💉",
    layout="centered",
)


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
MODEL_CANDIDATES = [
    "model.pkl",
    "covid_vaccine_model.pkl",
    "57264280-cae8-4b99-97eb-118af040605c.pkl",
    "ad2cad04-c50d-46c5-9528-39582d359a39.pkl",
]


@st.cache_resource
def load_model():
    """Load the first available model file."""
    for filename in MODEL_CANDIDATES:
        if Path(filename).exists():
            return joblib.load(filename), filename

    raise FileNotFoundError(
        "Model file was not found. Rename your model to 'model.pkl' "
        "and place it in the same folder as app.py."
    )


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------
st.title("💉 COVID-19 Vaccine Prediction")
st.write(
    "Enter the information below to predict whether a person is likely "
    "to choose **Yes** or **No**."
)

try:
    model, loaded_filename = load_model()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

with st.sidebar:
    st.header("Model Information")
    st.success("Model loaded successfully")
    st.caption(f"Loaded file: {loaded_filename}")
    st.info(
        "State and vaccine preference are entered as numeric category codes "
        "because their original label mappings were not included with the model."
    )

with st.form("prediction_form"):
    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=30,
            step=1,
        )

        gender = st.selectbox(
            "Gender code",
            options=[0, 1],
            format_func=lambda value: f"Code {value}",
        )

        state = st.selectbox(
            "State code",
            options=[0, 1, 2, 3, 4],
            format_func=lambda value: f"Code {value}",
        )

        chronic_disease = st.selectbox(
            "Chronic disease",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

    with col2:
        previous_covid = st.selectbox(
            "Previous COVID-19 infection",
            options=[0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No",
        )

        trust_in_vaccine = st.slider(
            "Trust in vaccine",
            min_value=1,
            max_value=10,
            value=5,
        )

        fear_of_side_effects = st.slider(
            "Fear of side effects",
            min_value=1,
            max_value=10,
            value=5,
        )

        vaccine_preference = st.selectbox(
            "Vaccine type preference code",
            options=[0, 1, 2, 3, 4],
            format_func=lambda value: f"Code {value}",
        )

    submitted = st.form_submit_button(
        "Predict",
        use_container_width=True,
        type="primary",
    )


if submitted:
    input_data = pd.DataFrame(
        [
            {
                "Age": age,
                "Gender": gender,
                "State": state,
                "Chronic_Disease": chronic_disease,
                "Previous_COVID": previous_covid,
                "Trust_in_Vaccine": trust_in_vaccine,
                "Fear_of_Side_Effects": fear_of_side_effects,
                "Vaccine_Type_Preference": vaccine_preference,
            }
        ]
    )

    try:
        prediction = model.predict(input_data)[0]

        st.divider()
        st.subheader("Prediction Result")

        if str(prediction).strip().lower() == "yes":
            st.success("Prediction: Yes")
        else:
            st.warning(f"Prediction: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            probability_table = pd.DataFrame(
                {
                    "Result": classes,
                    "Probability": probabilities,
                }
            )
            probability_table["Probability"] = probability_table[
                "Probability"
            ].map(lambda value: f"{value:.2%}")

            st.write("Prediction probabilities")
            st.dataframe(
                probability_table,
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("View entered information"):
            st.dataframe(input_data, hide_index=True, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.divider()
st.caption(
    "This application is for educational purposes and should not be used "
    "as medical advice."
)
