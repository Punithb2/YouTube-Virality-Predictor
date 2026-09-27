import joblib
import json
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

# Import your custom modules
from src.features.build_features import prep_training_data
import os
from dotenv import load_dotenv

def optimize_hyperparameters(X_train, y_train):
    """Runs Optuna to find the best LightGBM parameters."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            "objective": "regression", "metric": "mae", "verbosity": -1, "random_state": 42,
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        mae_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_va, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            mae_scores.append(mean_absolute_error(y_val, model.predict(X_va)))
            
        return np.mean(mae_scores)

    print("Tuning hyperparameters with Optuna (50 trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    return study.best_params

def main():
    print("1. Fetching data from PostgreSQL...")
    
    # Load environment variables (fallback to local Docker credentials)
    load_dotenv()
    DB_USER = os.getenv("DB_USER", "admin")
    DB_PASS = os.getenv("DB_PASS", "password123")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5433")
    DB_NAME = os.getenv("DB_NAME", "youtube_ml")
    
    # Create the raw database URL string
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
    # Hand the string directly to Pandas! 
    # Pandas will automatically create and manage the engine under the hood.
    raw_df = pd.read_sql("SELECT * FROM videos", con=db_url)
        
    print(f"Loaded {len(raw_df)} videos.")
    
    print("2. Engineering features...")
    X, y, df, feature_names = prep_training_data(raw_df)
    
    print("3. Optimizing model...")
    best_params = optimize_hyperparameters(X, y)
    
    print("4. Training Final Model...")
    best_lgb = lgb.LGBMRegressor(**best_params, random_state=42, verbosity=-1)
    best_lgb.fit(X, y)
    
    preds = best_lgb.predict(X)
    print(f"Final Model MAE (Log Views): {mean_absolute_error(y, preds):.4f}")
    print(f"Final Model R2:  {r2_score(y, preds):.4f}")
    
    print("5. Saving model and inference artifacts...")
    joblib.dump(best_lgb, "models/lightgbm_model.pkl")
    
    latest_stats = df.groupby('channel_id').last()['channel_avg_log_views'].to_dict()
    metadata = {
        "features": feature_names,
        "global_avg_log_views": float(df['log_views'].mean()),
        "fixed_inference_days": 30,
        "fixed_log_age": float(np.log1p(30)),
        "channel_stats": latest_stats
    }
    
    with open("models/inference_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Pipeline Complete! API is ready to serve predictions.")

if __name__ == "__main__":
    main()