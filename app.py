# placement_prediction.py
import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    try:
        with open('placement_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("`placement_model.pkl` not found! Run the notebook first.")
        st.stop()

pkg = load_model()
model          = pkg['model']
scaler         = pkg['scaler']
label_encoders = pkg['label_encoders']
feature_names  = pkg['feature_names']
numeric_cols   = pkg['numeric_cols']
categorical_cols = pkg['categorical_cols']

# ---------------- UI ----------------
st.set_page_config(page_title="Placement Predictor", page_icon="Graduation Cap", layout="wide")
st.title("Campus Placement Predictor")
st.markdown("Enter your academic & skill details to predict placement chances.")

# Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Academics")
    cgpa = st.slider("CGPA", 0.0, 10.0, 7.5, 0.1)
    ssc  = st.slider("SSC Marks (%)", 0, 100, 75)
    hsc  = st.slider("HSC Marks (%)", 0, 100, 75)

with col2:
    st.subheader("Skills & Experience")
    internships = st.number_input("Internships", 0, 5, 1)
    projects    = st.number_input("Projects", 0, 10, 2)
    workshops   = st.number_input("Workshops/Certifications", 0, 10, 2)
    soft_skills = st.slider("Soft Skills Rating", 1.0, 5.0, 3.0, 0.1)

with col3:
    st.subheader("Other Factors")
    aptitude = st.slider("Aptitude Score", 0, 100, 75)
    extra    = st.selectbox("Extracurricular Activities", ["Yes", "No"])
    training = st.selectbox("Placement Training", ["Yes", "No"])

# ---------------- Predict ----------------
if st.button("Predict Placement", use_container_width=True):
    # Build input
    input_data = {
        'CGPA': cgpa,
        'Internships': internships,
        'Projects': projects,
        'Workshops/Certifications': workshops,
        'AptitudeTestScore': aptitude,
        'SoftSkillsRating': soft_skills,
        'ExtracurricularActivities': extra,
        'PlacementTraining': training,
        'SSC_Marks': ssc,
        'HSC_Marks': hsc
    }
    df = pd.DataFrame([input_data])

    # Encode categorical
    for col in categorical_cols:
        df[col] = label_encoders[col].transform(df[col])

    # Scale numeric
    if scaler:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Reorder columns to match training
    df = df[feature_names]

    # Predict
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1] * 100

    # Results
    st.markdown("---")
    if pred == 1:
        st.success("**HIGH CHANCE OF PLACEMENT!**")
    else:
        st.warning("**NOT PLACED! NEEDS IMPROVEMENT**")
    st.metric("Placement Probability", f"{prob:.1f}%")

    # Bar chart
    fig = go.Figure([
        go.Bar(x=['Not Placed', 'Placed'], y=[100-prob, prob],
               marker_color=['#ff6b6b', '#51cf66'])
    ])
    fig.update_layout(title="Placement Probability", height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Improvement Tips
    st.subheader("Improvement Tips")
    tips = []
    if cgpa < 7.5: tips.append("Raise CGPA to 7.5+")
    if internships < 1: tips.append("Complete at least 1 internship")
    if aptitude < 70: tips.append("Practice aptitude tests")
    if soft_skills < 4.0: tips.append("Improve communication & soft skills")
    if extra == "No": tips.append("Join clubs or extracurriculars")
    if training == "No": tips.append("Attend placement training")

    for tip in tips:
        st.info(tip)