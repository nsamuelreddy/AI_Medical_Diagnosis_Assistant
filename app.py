import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    layout="wide"
)

st.title("AI Medical Diagnosis Assistant")
st.caption("Symptom-based disease prediction using machine learning")

@st.cache_resource
def load_files():
    model = joblib.load("disease_model_v2.pkl")
    symptoms = list(joblib.load("symptoms_v2.pkl"))
    return model, symptoms

model, symptoms = load_files()

def load_severity():
    file = "Symptom-severity.csv"
    if not os.path.exists(file):
        return {}
    df = pd.read_csv(file)
    if "Symptom" not in df.columns or "weight" not in df.columns:
        return {}
    return dict(zip(df["Symptom"].astype(str).str.strip(), df["weight"]))

def load_specialists():
    return {
        "heart": "Cardiologist",
        "skin": "Dermatologist",
        "lung": "Pulmonologist",
        "respiratory": "Pulmonologist",
        "stomach": "Gastroenterologist",
        "liver": "Hepatologist",
        "kidney": "Nephrologist",
        "urinary": "Urologist",
        "brain": "Neurologist",
        "mental": "Psychiatrist",
        "joint": "Orthopedic Specialist",
        "eye": "Ophthalmologist",
        "ear": "ENT Specialist"
    }

severity_map = load_severity()
specialist_map = load_specialists()

st.sidebar.header("Patient Information")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 1, 120, 20)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])

st.subheader("Enter Symptoms")

selected = st.multiselect(
    "Select your symptoms",
    sorted(symptoms)
)

text = st.text_input(
    "Or enter symptoms separated by commas",
    placeholder="fever, headache, cough"
)

if text:
    words = [x.strip().lower() for x in text.split(",")]
    for symptom in symptoms:
        if symptom.lower() in words and symptom not in selected:
            selected.append(symptom)

if selected:
    st.write("Selected symptoms:")
    st.write(", ".join(selected))

if st.button("Predict Disease", type="primary"):
    if not selected:
        st.warning("Please select at least one symptom.")
        st.stop()

    data = {symptom: 0 for symptom in symptoms}

    for symptom in selected:
        data[symptom] = 1

    input_data = pd.DataFrame([data])[symptoms]

    probabilities = model.predict_proba(input_data)[0]
    diseases = model.classes_

    results = sorted(
        zip(diseases, probabilities),
        key=lambda x: x[1],
        reverse=True
    )

    top_disease = results[0][0]
    top_probability = results[0][1]

    st.divider()
    st.subheader("Diagnosis Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Patient", name if name else "Not provided")

    with col2:
        st.metric("Age", age)

    with col3:
        st.metric("Symptoms", len(selected))

    st.subheader("Most Likely Diagnosis")
    st.success(
        f"{top_disease.title()} — {top_probability * 100:.2f}% probability"
    )

    st.subheader("Top 5 Possible Diagnoses")

    for rank, (disease, probability) in enumerate(results[:5], 1):
        st.write(
            f"{rank}. {disease.title()} — {probability * 100:.2f}%"
        )
        st.progress(float(probability))

    severity_score = 0

    for symptom in selected:
        severity_score += severity_map.get(symptom, 0)

    st.subheader("Symptom Severity")

    if severity_score == 0:
        st.info("Severity data is not available for the selected symptoms.")
    elif severity_score <= 5:
        st.success(f"Severity score: {severity_score} — Low")
    elif severity_score <= 10:
        st.warning(f"Severity score: {severity_score} — Moderate")
    else:
        st.error(f"Severity score: {severity_score} — High")

    st.subheader("Suggested Specialist")

    disease_text = top_disease.lower()
    specialist = "General Physician"

    for keyword, doctor in specialist_map.items():
        if keyword in disease_text:
            specialist = doctor
            break

    st.info(specialist)

    st.caption(
        "Educational project only. This prediction is not a medical diagnosis "
        "and should not replace professional medical advice."
    )
