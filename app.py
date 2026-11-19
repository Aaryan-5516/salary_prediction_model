
import streamlit as st
import pandas as pd
import joblib

# Load trained model and dataset
model = joblib.load("models/salary_model.joblib")
data = pd.read_csv("Salary_Data.csv")

st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="centered"
)

st.title("💼 Employee Salary Prediction")

st.write(
    "Enter employee details to estimate their salary."
)

st.divider()

# Input fields
age = st.number_input(
    "Age",
    min_value=18,
    max_value=65,
    value=25
)

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=45.0,
    value=2.0,
    step=0.5
)

gender_options = sorted(
    data["Gender"].dropna().astype(str).unique()
)

education_options = sorted(
    data["Education Level"].dropna().astype(str).unique()
)

job_options = sorted(
    data["Job Title"].dropna().astype(str).unique()
)

gender = st.selectbox("Gender", gender_options)

education = st.selectbox(
    "Education Level", education_options
)

job = st.selectbox("Job Title", job_options)

st.divider()

# Prediction button
if st.button("Predict Salary", type="primary"):

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job,
        "Years of Experience": experience
    }])

    prediction = model.predict(input_data)[0]

    st.success("Salary Prediction Completed!")

    st.metric(
        label="Estimated Salary",
        value=f"{prediction:,.2f}"
    )

    st.info(
        "This is an estimated value based on the "
        "training dataset, not a guaranteed salary."
    )

st.divider()

st.subheader("Salary Dataset Overview")

st.write("Total employee records:", len(data))

st.write("Average salary:", round(data["Salary"].mean(), 2))

st.subheader("Salary Distribution")

st.bar_chart(data["Salary"].dropna())
