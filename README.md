# 📈 YouTube Virality Predictor: End-to-End ML Architecture

An end-to-end Machine Learning pipeline that predicts the expected 30-day view count of a YouTube video based on its metadata and historical channel momentum. 

This project encompasses the entire ML lifecycle: automated data ingestion via the YouTube Data API, leakage-safe feature engineering, hyperparameter tuning with Optuna, and deployment via a Dockerized FastAPI & Streamlit microservice architecture.

## 🏗️ System Architecture

* **Data Ingestion:** Python scraper querying the YouTube Data API v3, applying upsert logic to a PostgreSQL database.
* **Feature Engineering & ML:** Pandas, Scikit-Learn, LightGBM, Optuna, and SHAP.
* **Backend API:** FastAPI (served via Uvicorn) with Pydantic data validation.
* **Frontend:** Streamlit for an interactive, web-based prediction UI.
* **Deployment:** Fully containerized using Docker & Docker Compose.

---

## 🧠 Key Engineering Highlights

### 1. Leakage-Safe Feature Engineering
Predicting future views requires knowing a channel's historical baseline performance. However, calculating a standard average across a channel's entire dataset introduces **data leakage** (the model sees future data during training). 
* **The Solution:** Implemented a strict chronologically expanding window. The `channel_avg_log_views` feature is calculated iteratively, ensuring that a video published on Day 10 is evaluated *only* against the average views of videos published from Day 1 to Day 9. 

### 2. Model Tuning & Explainability
* **LightGBM + Optuna:** Tuned hyperparameters (`num_leaves`, `max_depth`, `min_child_samples`, and regularization) using 5-fold cross-validation on a dataset of ~5,400 videos. This prevented the gradient boosted trees from overfitting to massive viral outliers.
* **SHAP (SHapley Additive exPlanations):** Utilized SHAP tree explainers to map feature importance. The analysis proved that historical channel momentum is the primary driver of views, with tag volume and publish timing acting as secondary modifiers.
* **Performance:** The final Optuna-tuned LightGBM model achieved an **R² of 0.7585**, proving that ~76% of a video's success can be predicted purely through metadata and baseline momentum, prior to thumbnail or content analysis.

### 3. Handling the "Cold Start" Problem
The FastAPI backend dynamically handles inference for both known and unknown entities:
* **Known Channels:** Looks up the channel's specific historical baseline via a serialized JSON artifact.
* **Unknown Channels:** If a brand-new channel ID is queried, the API safely defaults to a "global fallback baseline" (the mathematical average of all training data), allowing the LightGBM model to make a prediction based purely on text length, tags, and timing optimization without crashing.

---

## 🚀 How to Run the Project (Docker)

The entire architecture is containerized and requires zero local environment configuration to run. 

### Prerequisites
* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.

### Quick Start
1. **Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/virality-predictor.git](https://github.com/YOUR_USERNAME/virality-predictor.git)
cd virality-predictor
```

2. **Boot the architecture:**
```bash
docker-compose up --build -d

```


3. **Access the Services:**
* **Streamlit Frontend:** [http://localhost:8501](http://localhost:8501?utm_source=gemini)
* **FastAPI Interactive Docs (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs?utm_source=gemini)
* **pgAdmin (Database GUI):** [http://localhost:5050](http://localhost:5050?utm_source=gemini)


4. **Shut down the system:**
```bash
docker-compose down

```



---

## 📂 Repository Structure

```text
virality-predictor/
├── api/
│   └── main.py                   # FastAPI server and prediction logic
├── models/
│   ├── lightgbm_model.pkl        # Serialized LightGBM regression model
│   └── inference_metadata.json   # Channel baselines and feature parameters
├── src/
│   ├── data/
│   │   ├── collector.py          # YouTube API scraper and data ingestion
│   │   └── db.py                 # PostgreSQL connection pool and schemas
├── notebooks/
│   └── 01_feature_engineering_and_baselines.ipynb  # Training & evaluation pipeline
├── app.py                        # Streamlit frontend application
├── docker-compose.yml            # Multi-container orchestration
├── Dockerfile                    # Production image blueprint
└── requirements.txt              # Strict dependency lockfile

```

---

## 🔮 Future Work (v2 Roadmap)

* **Live On-Demand Fetching:** Upgrade the FastAPI `/predict` endpoint to ping the YouTube Data API when an unseen Channel ID is provided, dynamically fetching their last 10 videos to calculate a real-time historical baseline instead of relying on the global fallback.
* **Thumbnail Vision Integration:** Extract RGB histograms, text-to-image ratios, and facial recognition features from thumbnails via OpenCV to improve the model's ability to predict viral breakouts.

```
