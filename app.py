import streamlit as st
import requests
import os

st.set_page_config(page_title="Virality Predictor", page_icon="📈", layout="centered")

st.title("📈 YouTube Virality Predictor")
st.markdown("Enter your video metadata below to predict the expected 30-day view count.")

# Input Form
with st.form("prediction_form"):
    channel_id = st.text_input("Channel ID (e.g., UC_x5XG1OV2P6uZZ5FSM9Ttw)", value="")
    title = st.text_input("Video Title", value="I Survived 100 Days in Hardcore Minecraft!")
    description = st.text_area("Video Description", value="In this video, we build an iron farm...")
    tags_input = st.text_input("Tags (comma separated)", value="minecraft, hardcore, survival")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        duration_seconds = st.number_input("Duration (Seconds)", min_value=1, value=1200)
    with col2:
        publish_hour = st.slider("Publish Hour", min_value=0, max_value=23, value=15)
    with col3:
        is_weekend = st.checkbox("Publishing on Weekend?")
        is_weekend_int = 1 if is_weekend else 0
        
    submitted = st.form_submit_button("Predict Views")

# API Connection
if submitted:
    if not channel_id:
        st.error("Please enter a Channel ID.")
    else:
        # Parse tags into a list
        tags_list = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        
        # Prepare payload for FastAPI
        payload = {
            "channel_id": channel_id,
            "title": title,
            "description": description,
            "tags": tags_list,
            "duration_seconds": duration_seconds,
            "publish_hour": publish_hour,
            "is_weekend": is_weekend_int
        }
        
        with st.spinner("Analyzing metadata and calculating trajectory..."):
            try:
                # Use environment variable for Docker, fallback to localhost for local testing
                api_url = os.getenv("API_URL", "http://127.0.0.1:8000/predict")
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    views = data["predicted_views_30_days"]
                    source = data["baseline_source"]
                    
                    st.success("Prediction Complete!")
                    st.metric(label="Predicted Views (at 30 days)", value=f"{views:,}")
                    
                    if source == "global_fallback":
                        st.warning("⚠️ Channel history not found. Used global average baseline.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Failed to connect to the API. Is your FastAPI server running on port 8000?")