import os
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
    "Covid_19_model.pkl",
    "covid_19_model.pkl",
    "model.pkl",
    "2741ed34-c744-48c5-a99a-63f426661847.pkl",
    "ef4fa14a-e9c4-4482-a0b4-11c22db3dc43.pkl",
]


@st.cache_resource
def load_model(model_path):
    """Load and cache the trained machine-learning model."""
    return joblib.load(model_path)


def find_model():
    """Return the first model file found in the application folder."""
    for filename in MODEL_CANDIDATES:
        if os.path.exists(filename):
            return filename
    return None


# ---------------------------------------------------------
# User interface
# ---------------------------------------------------------
st.title("💉 COVID-19 Vaccine Prediction")
st.write(
    "Enter the person's information below to predict the model's "
    "Yes/No outcome."
)

model_path = find_model()

if model_path is None:
    st.warning("Model file not found in the application folder.")
    uploaded_model = st.file_uploader(
        "Upload your trained .pkl model",
        type=["pkl"],
    )

    if uploaded_model is not None:
        temporary_path = "uploaded_model.pkl"
        with open(temporary_path, "wb") as file:
            file.write(uploaded_model.getbuffer())
        model_path = temporary_path

if model_path is not None:
    try:
        model = load_model(model_path)

        with st.form("prediction_form"):
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                step=1,
            )

            gender_label = st.selectbox(
                "Gender",
                ["Female", "Male"],
            )
            gender = 0 if gender_label == "Female" else 1

            state = st.selectbox(
                "State Code",
                options=[0, 1, 2, 3, 4],
                help="Use the same state code used when training the model.",
            )

            chronic_label = st.selectbox(
                "Chronic Disease",
                ["No", "Yes"],
            )
            chronic_disease = 0 if chronic_label == "No" else 1

            previous_covid_label = st.selectbox(
                "Previous COVID-19 Infection",
                ["No", "Yes"],
            )
            previous_covid = 0 if previous_covid_label == "No" else 1

            trust_in_vaccine = st.slider(
                "Trust in Vaccine",
                min_value=1,
                max_value=10,
                value=5,
            )

            fear_of_side_effects = st.slider(
                "Fear of Side Effects",
                min_value=1,
                max_value=10,
                value=5,
            )

            vaccine_type_preference = st.selectbox(
                "Vaccine Type Preference Code",
                options=[0, 1, 2, 3, 4],
                help="Use the same vaccine code used when training the model.",
            )

            submitted = st.form_submit_button(
                "Predict",
                use_container_width=True,
            )

        if submitted:
            input_data = pd.DataFrame(
                [[
                    age,
                    gender,
                    state,
                    chronic_disease,
                    previous_covid,
                    trust_in_vaccine,
                    fear_of_side_effects,
                    vaccine_type_preference,
                ]],
                columns=[
                    "Age",
                    "Gender",
                    "State",
                    "Chronic_Disease",
                    "Previous_COVID",
                    "Trust_in_Vaccine",
                    "Fear_of_Side_Effects",
                    "Vaccine_Type_Preference",
                ],
            )

            prediction = model.predict(input_data)[0]

            if str(prediction).lower() == "yes":
                st.success(f"Prediction: {prediction}")
            else:
                st.info(f"Prediction: {prediction}")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                probability_table = pd.DataFrame({
                    "Result": model.classes_,
                    "Probability": probabilities,
                })
                probability_table["Probability"] = (
                    probability_table["Probability"] * 100
                ).round(2).astype(str) + "%"
                st.subheader("Prediction Probability")
                st.dataframe(
                    probability_table,
                    hide_index=True,
                    use_container_width=True,
                )

    except Exception as error:
        st.error(f"Unable to load or use the model: {error}")

st.divider()
st.caption(
    "This application provides a machine-learning prediction and is not "
    "medical advice."
)
