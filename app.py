import os
import re
import requests
import streamlit as st
from datetime import datetime

# Grab the API key from your environment
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

st.set_page_config(page_title="Virality Predictor", page_icon="📈", layout="centered")

# --- HELPER FUNCTIONS ---
def extract_video_id(url):
    """Extracts the 11-character video ID from a YouTube URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def parse_yt_duration(duration_str):
    """Converts YouTube's PT1H2M10S format into total seconds."""
    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
    if not match:
        return 0
    hours, minutes, seconds = match.groups()
    return (int(hours or 0) * 3600) + (int(minutes or 0) * 60) + int(seconds or 0)

def fetch_youtube_data(url):
    """Hits the YouTube Data API to get video metadata."""
    if not YOUTUBE_API_KEY:
        st.error("YouTube API Key not found in environment variables.")
        return
        
    vid_id = extract_video_id(url)
    if not vid_id:
        st.error("Invalid YouTube URL.")
        return

    api_url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,contentDetails&id={vid_id}&key={YOUTUBE_API_KEY}"
    res = requests.get(api_url).json()
    
    if not res.get("items"):
        st.error("Video not found or is private.")
        return
        
    data = res["items"][0]
    snippet = data["snippet"]
    
    # Save the fetched data into Streamlit's session state so the form auto-fills
    st.session_state['channel_id'] = snippet.get("channelId", "")
    st.session_state['title'] = snippet.get("title", "")
    st.session_state['description'] = snippet.get("description", "")
    
    tags = snippet.get("tags", [])
    st.session_state['tags'] = ", ".join(tags)
    
    duration_str = data["contentDetails"].get("duration", "PT0S")
    st.session_state['duration'] = parse_yt_duration(duration_str)
    
    # Extract publish hour and weekend status
    pub_date = datetime.strptime(snippet["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
    st.session_state['publish_hour'] = pub_date.hour
    st.session_state['is_weekend'] = True if pub_date.weekday() >= 5 else False

# --- UI LAYOUT ---
st.title("📈 YouTube Virality Predictor")
st.markdown("Predict a video's 30-day view count. Paste an existing URL to auto-fill, or test a brand new idea manually!")

# URL Fetcher Section
st.markdown("### 1. Auto-Fill from Existing Video")
col_url, col_btn = st.columns([3, 1])
with col_url:
    url_input = st.text_input("Paste YouTube URL (Optional)", label_visibility="collapsed", placeholder="https://www.youtube.com/watch?v=...")
with col_btn:
    if st.button("Fetch Data", use_container_width=True):
        if url_input:
            fetch_youtube_data(url_input)

st.divider()

# Input Form Section
st.markdown("### 2. Video Metadata")
with st.form("prediction_form"):
    channel_id = st.text_input("Channel ID", value=st.session_state.get('channel_id', ''))
    title = st.text_input("Video Title", value=st.session_state.get('title', ''))
    description = st.text_area("Video Description", value=st.session_state.get('description', ''))
    tags_input = st.text_input("Tags (comma separated)", value=st.session_state.get('tags', ''))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        duration_seconds = st.number_input("Duration (Seconds)", min_value=1, value=st.session_state.get('duration', 1200))
    with col2:
        publish_hour = st.slider("Publish Hour (UTC)", min_value=0, max_value=23, value=st.session_state.get('publish_hour', 15))
    with col3:
        is_weekend = st.checkbox("Publishing on Weekend?", value=st.session_state.get('is_weekend', False))
        is_weekend_int = 1 if is_weekend else 0
        
    submitted = st.form_submit_button("Predict Views", type="primary")

# --- API CONNECTION ---
if submitted:
    if not channel_id:
        st.error("Please enter a Channel ID.")
    else:
        tags_list = [tag.strip() for tag in tags_input.split(",") if tag.strip()]
        
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
                api_url = os.getenv("API_URL", "http://api:8000/predict")
                response = requests.post(api_url, json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    views = data["predicted_views_30_days"]
                    source = data["baseline_source"]
                    
                    st.success("Prediction Complete!")
                    st.metric(label="Predicted Views (at 30 days)", value=f"{views:,}")
                    
                    if source == "global_fallback":
                        st.warning("⚠️ Channel history not found in database. Using global average baseline.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error("Failed to connect to the prediction API.")