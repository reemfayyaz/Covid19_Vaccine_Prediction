"""Professional Streamlit app for COVID-19 vaccine prediction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Vaccine Prediction",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "COVID-19 Vaccine Prediction"
APP_DESCRIPTION = (
    "Enter the information below to estimate the model's predicted outcome."
)

MODEL_CANDIDATES = (
    "Covid_19_model.pkl",
    "covid_19_model.pkl",
    "model.pkl",
    "2741ed34-c744-48c5-a99a-63f426661847.pkl",
    "ef4fa14a-e9c4-4482-a0b4-11c22db3dc43.pkl",
)

FEATURE_COLUMNS = [
    "Age",
    "Gender",
    "State",
    "Chronic_Disease",
    "Previous_COVID",
    "Trust_in_Vaccine",
    "Fear_of_Side_Effects",
    "Vaccine_Type_Preference",
]


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero-card {
            padding: 1.6rem 1.8rem;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 16px;
            margin-bottom: 1.25rem;
            background: rgba(128, 128, 128, 0.05);
        }
        .hero-card h1 {
            margin: 0 0 0.4rem 0;
            font-size: 2rem;
        }
        .hero-card p {
            margin: 0;
            opacity: 0.8;
        }
        div[data-testid="stForm"] {
            border-radius: 16px;
            padding: 1.25rem;
        }
        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            font-weight: 600;
            min-height: 3rem;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model utilities
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> Any:
    """Load and cache a serialized machine-learning model."""
    return joblib.load(model_path)


def find_local_model() -> Path | None:
    """Return the first supported model file found in the app directory."""
    app_directory = Path(__file__).resolve().parent

    for filename in MODEL_CANDIDATES:
        candidate = app_directory / filename
        if candidate.is_file():
            return candidate

    return None


def save_uploaded_model(uploaded_file: Any) -> str:
    """Save an uploaded model to a temporary file and return its path."""
    suffix = Path(uploaded_file.name).suffix or ".pkl"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def validate_model(model: Any) -> None:
    """Raise a helpful error if the loaded object cannot make predictions."""
    if not hasattr(model, "predict"):
        raise TypeError("The uploaded file does not contain a valid prediction model.")

    expected_count = getattr(model, "n_features_in_", None)
    if expected_count is not None and expected_count != len(FEATURE_COLUMNS):
        raise ValueError(
            f"This model expects {expected_count} features, but the app provides "
            f"{len(FEATURE_COLUMNS)} features."
        )


def build_input_dataframe(values: dict[str, int]) -> pd.DataFrame:
    """Create a model-ready DataFrame using the training feature order."""
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def format_prediction(prediction: Any) -> tuple[str, str]:
    """Return a display label and message type for the prediction."""
    normalized = str(prediction).strip().lower()

    positive_values = {"yes", "1", "true", "positive", "vaccinated"}
    negative_values = {"no", "0", "false", "negative", "not vaccinated"}

    if normalized in positive_values:
        return "Yes", "success"
    if normalized in negative_values:
        return "No", "warning"

    return str(prediction), "info"


def show_probability_results(model: Any, input_data: pd.DataFrame) -> None:
    """Display class probabilities when the model supports predict_proba."""
    if not hasattr(model, "predict_proba"):
        return

    probabilities = model.predict_proba(input_data)[0]
    classes = getattr(model, "classes_", range(len(probabilities)))

    probability_data = pd.DataFrame(
        {
            "Outcome": [str(value) for value in classes],
            "Probability": probabilities,
        }
    ).sort_values("Probability", ascending=False)

    most_likely = probability_data.iloc[0]
    st.metric(
        label="Highest confidence",
        value=f"{most_likely['Probability']:.1%}",
        delta=f"Outcome: {most_likely['Outcome']}",
        delta_color="off",
    )

    display_table = probability_data.copy()
    display_table["Probability"] = display_table["Probability"].map(
        lambda value: f"{value:.2%}"
    )
    st.dataframe(display_table, hide_index=True, use_container_width=True)


# -----------------------------------------------------------------------------
# Header and sidebar
# -----------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-card">
        <h1>💉 {APP_TITLE}</h1>
        <p>{APP_DESCRIPTION}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model Settings")
    local_model_path = find_local_model()
    uploaded_model = None

    if local_model_path:
        st.success(f"Model found: {local_model_path.name}")
        selected_model_path = str(local_model_path)
    else:
        st.warning("No model file was found in the app folder.")
        uploaded_model = st.file_uploader(
            "Upload a trained model",
            type=["pkl", "joblib"],
            help="Upload the same model used with these eight input features.",
        )
        selected_model_path = None

        if uploaded_model is not None:
            selected_model_path = save_uploaded_model(uploaded_model)
            st.success(f"Uploaded: {uploaded_model.name}")

    st.divider()
    st.subheader("About")
    st.caption(
        "This app sends the entered values to a trained machine-learning model "
        "and displays its predicted class and confidence, when available."
    )


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------
if not selected_model_path:
    st.info(
        "Place the trained model in the same folder as app.py, or upload it "
        "from the sidebar to begin."
    )
    st.stop()

try:
    with st.spinner("Loading prediction model..."):
        model = load_model(selected_model_path)
        validate_model(model)
except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.stop()

input_column, information_column = st.columns([2, 1], gap="large")

with input_column:
    st.subheader("Patient Information")

    with st.form("prediction_form", clear_on_submit=False):
        first_row_left, first_row_right = st.columns(2)

        with first_row_left:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
            )
            gender_label = st.selectbox("Gender", ["Female", "Male"])
            state = st.selectbox(
                "State Code",
                options=[0, 1, 2, 3, 4],
                help="Select the same state code definition used during training.",
            )
            chronic_label = st.selectbox("Chronic Disease", ["No", "Yes"])

        with first_row_right:
            previous_covid_label = st.selectbox(
                "Previous COVID-19 Infection",
                ["No", "Yes"],
            )
            trust_in_vaccine = st.slider(
                "Trust in Vaccine",
                min_value=1,
                max_value=10,
                value=5,
                help="1 means very low trust and 10 means very high trust.",
            )
            fear_of_side_effects = st.slider(
                "Fear of Side Effects",
                min_value=1,
                max_value=10,
                value=5,
                help="1 means very low fear and 10 means very high fear.",
            )
            vaccine_type_preference = st.selectbox(
                "Vaccine Preference Code",
                options=[0, 1, 2, 3, 4],
                help="Select the same vaccine code definition used during training.",
            )

        submitted = st.form_submit_button(
            "Generate Prediction",
            type="primary",
            use_container_width=True,
        )

with information_column:
    st.subheader("Input Guide")
    st.info(
        "The numeric state and vaccine codes must match the coding used in the "
        "training dataset. Incorrect code mappings can produce unreliable results."
    )
    st.write("**Model input features:**")
    st.write(" · ".join(FEATURE_COLUMNS))

if submitted:
    values = {
        "Age": int(age),
        "Gender": 0 if gender_label == "Female" else 1,
        "State": int(state),
        "Chronic_Disease": 0 if chronic_label == "No" else 1,
        "Previous_COVID": 0 if previous_covid_label == "No" else 1,
        "Trust_in_Vaccine": int(trust_in_vaccine),
        "Fear_of_Side_Effects": int(fear_of_side_effects),
        "Vaccine_Type_Preference": int(vaccine_type_preference),
    }

    input_data = build_input_dataframe(values)

    try:
        prediction = model.predict(input_data)[0]
        prediction_label, message_type = format_prediction(prediction)

        st.divider()
        st.subheader("Prediction Result")

        message = f"Predicted outcome: **{prediction_label}**"
        if message_type == "success":
            st.success(message)
        elif message_type == "warning":
            st.warning(message)
        else:
            st.info(message)

        result_column, data_column = st.columns(2, gap="large")

        with result_column:
            st.markdown("#### Prediction Confidence")
            show_probability_results(model, input_data)

        with data_column:
            st.markdown("#### Submitted Data")
            display_data = pd.DataFrame(
                {
                    "Field": [
                        "Age",
                        "Gender",
                        "State Code",
                        "Chronic Disease",
                        "Previous COVID-19",
                        "Trust in Vaccine",
                        "Fear of Side Effects",
                        "Vaccine Preference Code",
                    ],
                    "Value": [
                        age,
                        gender_label,
                        state,
                        chronic_label,
                        previous_covid_label,
                        trust_in_vaccine,
                        fear_of_side_effects,
                        vaccine_type_preference,
                    ],
                }
            )
            st.dataframe(display_data, hide_index=True, use_container_width=True)

    except Exception as error:
        st.error(
            "Prediction failed. Confirm that the model was trained with the same "
            "feature names, order, data types, and category codes used by this app."
        )
        st.exception(error)

st.divider()
st.caption(
    "Disclaimer: This application is for educational and demonstration purposes "
    "only. It does not provide medical advice, diagnosis, or treatment."
)
