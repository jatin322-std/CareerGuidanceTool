import streamlit as st
import numpy as np
import joblib
import os
import xgboost as xgb
import plotly.express as px
import pandas as pd
import random

st.set_page_config(page_title="Career Path Predictor", page_icon="🎯", layout="centered")

# Load model and encoders
MODEL_PATH = "xgb_career_model.json"
ENCODER_PATH = "label_encoder.pkl"
SCALER_PATH = "scaler.pkl"


if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Required model files not found. Please upload 'xgb_career_model.json', 'label_encoder.pkl', and 'scaler.pkl'.")
    st.stop()

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# Sidebar definitions and quote
with st.sidebar:
    st.title("ℹ️ Personality Traits Explained")
    traits = {
        "Openness": "Willingness to try new experiences and be creative.",
        "Conscientiousness": "Being organized, responsible, and hardworking.",
        "Extraversion": "Sociability, assertiveness, and high energy.",
        "Agreeableness": "Being friendly, compassionate, and cooperative.",
        "Emotional Range": "Tendency to experience negative emotions.",
        "Conversation": "Comfort level in engaging and continuing conversations.",
        "Openness to Change": "How adaptable and flexible you are.",
        "Hedonism": "Desire for pleasure, fun, and enjoyment.",
        "Self-enhancement": "Drive to improve status or success.",
        "Self-transcendence": "Concern for others and the environment."
    }
    for key, val in traits.items():
        st.markdown(f"**{key}**: {val}")

    st.markdown("---")
    st.markdown("🧠 **Quote of the Day:**")
    st.info(random.choice([
        "🚀 'Choose a job you love, and you will never have to work a day in your life.'",
        "💡 'AI is not replacing you. A person using AI will.'",
        "🎯 'The future belongs to those who learn more skills and combine them creatively.'"
    ]))

# App title and intro
st.title("🎓 AI Career Prediction Tool")
st.markdown("Rate your skills and personality traits to get personalized career suggestions.")
st.markdown("---")

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

if submitted:
    input_array = np.array([[comp_arch, prog_skills, proj_mgmt, comm_skills,
                             openness, conscientiousness, extraversion, agreeableness,
                             emotional_range, conversation, openness_change, hedonism,
                             self_enhancement, self_transcendence]])

    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)[0]
    predicted_career = label_encoder.inverse_transform([pred])[0]

    st.success(f"🎯 Suggested Career: **{predicted_career}**")

    # Top 3 predictions
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_scaled)[0]
        top_indices = np.argsort(proba)[::-1][:3]
        st.subheader("🔝 Top 3 Career Matches:")
        for i in top_indices:
            role = label_encoder.inverse_transform([i])[0]
            st.write(f"- **{role}** — {proba[i]*100:.2f}%")

    # Radar chart of profile
    st.subheader("📊 Your Profile Overview")
    radar_df = pd.DataFrame({
        'Trait': ['Computer Architecture', 'Programming Skills', 'Project Mgmt', 'Communication',
                  'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness',
                  'Emotional Range', 'Conversation', 'Openness to Change', 'Hedonism',
                  'Self-enhancement', 'Self-transcendence'],
        'Value': input_array[0]
    })
    fig = px.line_polar(radar_df, r='Value', theta='Trait', line_close=True, title="Skill & Personality Radar", range_r=[0, 5])
    st.plotly_chart(fig, use_container_width=True)

    # Market demand (mock value)
    st.subheader("📈 Market Demand")
    st.progress(int(proba[pred]*100))

    # Roadmap
    st.subheader("🗺️ Career Roadmap")
    st.markdown(f"""
    To pursue **{predicted_career}**, you can consider:
    - 📚 Taking certification courses (Coursera, edX, Udemy)
    - 🛠️ Practicing real-world projects
    - 🤝 Networking with professionals on LinkedIn
    - 🧠 Improving key traits like **Programming** or **Communication**
    """)

    # External resources
    st.subheader("🔗 Learn & Apply")
    st.write("[📘 Coursera](https://www.coursera.org)")
    st.write("[💼 LinkedIn Jobs](https://www.linkedin.com/jobs)")
    st.write("[🌐 Indeed](https://www.indeed.com)")

    st.info("Try adjusting your sliders to simulate different profiles and explore career possibilities!")

