import streamlit as st
import numpy as np
import joblib
import os
import xgboost as xgb

st.set_page_config(page_title="Career Path Predictor", page_icon="🎯", layout="centered")

# Load model and encoders
MODEL_PATH = "xgb_career_model.json"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"

# Check for required files
if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Required model files not found. Please upload 'xgb_career_model.json', 'label_encoder.pkl', and 'scaler.pkl'.")
    st.stop()

# Load model and preprocessing objects
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

label_encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# App title and intro
st.title("🎓 AI Career Prediction Tool")
st.markdown("Rate your skills and personality traits to get personalized career suggestions.")
st.markdown("---")

# Form input
with st.form("career_form"):
    st.subheader("📝 Rate Yourself")
    col1, col2 = st.columns(2)

    with col1:
        comp_arch = st.slider("💻 Computer Architecture", 0.0, 5.0, 3.0)
        prog_skills = st.slider("👨‍💻 Programming Skills", 0.0, 5.0, 3.0)
        proj_mgmt = st.slider("📋 Project Management", 0.0, 5.0, 3.0)
        comm_skills = st.slider("🗣️ Communication Skills", 0.0, 5.0, 3.0)
        openness = st.slider("🔍 Openness", 0.0, 1.0, 0.5)
        conscientiousness = st.slider("✅ Conscientiousness", 0.0, 1.0, 0.5)
        extraversion = st.slider("🎉 Extraversion", 0.0, 1.0, 0.5)

    with col2:
        agreeableness = st.slider("🤝 Agreeableness", 0.0, 1.0, 0.5)
        emotional_range = st.slider("🌪️ Emotional Range", 0.0, 1.0, 0.5)
        conversation = st.slider("💬 Conversation", 0.0, 1.0, 0.5)
        openness_change = st.slider("🔄 Openness to Change", 0.0, 1.0, 0.5)
        hedonism = st.slider("🎈 Hedonism", 0.0, 1.0, 0.5)
        self_enhancement = st.slider("🚀 Self-enhancement", 0.0, 1.0, 0.5)
        self_transcendence = st.slider("🧘 Self-transcendence", 0.0, 1.0, 0.5)

    submitted = st.form_submit_button("🔍 Predict My Career")

# Prediction
if submitted:
    input_array = np.array([[comp_arch, prog_skills, proj_mgmt, comm_skills,
                             openness, conscientiousness, extraversion, agreeableness,
                             emotional_range, conversation, openness_change, hedonism,
                             self_enhancement, self_transcendence]])

    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0]
    predicted_career = label_encoder.inverse_transform([pred])[0]

    st.success(f"🎯 Suggested Career: **{predicted_career}**")

    # Show top 3 matches if probabilities available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        top_indices = np.argsort(proba)[::-1][:3]
        st.subheader("🔝 Top 3 Career Matches:")
        for i in top_indices:
            role = label_encoder.inverse_transform([i])[0]
            st.write(f"- **{role}** — {proba[i]*100:.2f}%")

    st.info("You can adjust your sliders to explore different career scenarios!")
