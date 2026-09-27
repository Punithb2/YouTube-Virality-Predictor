import pandas as pd
import numpy as np

def prep_training_data(raw_df):
    """Transforms raw PostgreSQL data into ML-ready features and targets."""
    df = raw_df.copy()
    
    # Ensure dates are datetime objects and explicitly set to UTC
    df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
    
    # 1. Target Variable
    df['log_views'] = np.log1p(df['view_count'])
    
    # 2. Leakage-Safe Channel Baseline (Chronological Expanding Mean)
    df = df.sort_values(['channel_id', 'published_at'])
    df['channel_avg_log_views'] = df.groupby('channel_id')['log_views'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    # 3. Metadata Features
    df['title_length'] = df['title'].fillna('').str.len()
    df['description_length'] = df['description'].fillna('').str.len()
    
    # Safely count tags (handles both lists and strings)
    df['tag_count'] = df['tags'].apply(
        lambda x: len(x) if isinstance(x, list) else (len(x.split(',')) if isinstance(x, str) and x else 0)
    )
    
    # Temporal Features
    df['publish_hour'] = df['published_at'].dt.hour
    df['is_weekend'] = df['published_at'].dt.weekday.apply(lambda x: 1 if x >= 5 else 0)
    
    # Calculate video age in days (using current time as proxy for scrape time)
    now_utc = pd.Timestamp.now(tz='UTC')
    df['age_days'] = (now_utc - df['published_at']).dt.total_seconds() / 86400
    df['log_age'] = np.log1p(df['age_days'])
    
    # Drop rows that don't have a historical baseline (the first video of every channel)
    df = df.dropna(subset=['channel_avg_log_views'])
    
    # 4. Final Feature Selection
    features = [
        'channel_avg_log_views', 'tag_count', 'duration_seconds', 
        'title_length', 'log_age', 'description_length', 
        'publish_hour', 'is_weekend'
    ]
    
    X = df[features]
    y = df['log_views']
    
    return X, y, df, features