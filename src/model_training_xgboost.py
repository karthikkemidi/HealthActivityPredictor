import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

print("="*70)
print("HEALTH ACTIVITY PREDICTOR - XGBOOST TRAINING")
print("="*70)

# ---------------------------------------------------------
# 1. LOAD & PREPROCESS DATA
# ---------------------------------------------------------
print("\n[1/5] Loading dataset...")
df = pd.read_csv('data/health_fitness_dataset.csv')
print(f"   ✓ Loaded {df.shape[0]:,} rows, {df.shape[1]} columns")

# Convert Date and Sort (CRITICAL for time-series diff)
print("\n[2/5] Preprocessing...")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['participant_id', 'date']).reset_index(drop=True)
print("   ✓ Date converted and sorted")

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
print("\n[3/5] Feature engineering...")

# Calculate the 'Growth' (Difference in Fitness Level from previous day)
df['fitness_growth'] = df.groupby('participant_id')['fitness_level'].diff()
df['fitness_growth'] = df['fitness_growth'].fillna(df['fitness_level'])
print("   ✓ Created fitness_growth target")

# Calculate activity metrics growth
df['steps_growth'] = df.groupby('participant_id')['daily_steps'].diff().fillna(0)
df['hr_change'] = df.groupby('participant_id')['avg_heart_rate'].diff().fillna(0)
df['sleep_change'] = df.groupby('participant_id')['hours_sleep'].diff().fillna(0)

# Encode 'Intensity' (Ordinal Encoding)
intensity_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df['intensity_encoded'] = df['intensity'].map(intensity_mapping)
print("   ✓ Encoded intensity levels")

# Encode gender
gender_mapping = {'M': 1, 'F': 2, 'Other': 3}
df['gender_encoded'] = df['gender'].map(gender_mapping)

# Create day of week feature
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

print("   ✓ Created additional features")

# ---------------------------------------------------------
# 3. DEFINE FEATURES AND TARGETS
# ---------------------------------------------------------
# Features for training (no data leakage)
features = [
    'duration_minutes', 'calories_burned', 'intensity_encoded',
    'avg_heart_rate', 'hours_sleep', 'daily_steps', 
    'stress_level', 'bmi', 'resting_heart_rate',
    'age', 'weight_kg', 'height_cm', 'gender_encoded',
    'day_of_week', 'is_weekend', 'hydration_level'
]

# We'll train 3 separate models for different predictions
targets = {
    'fitness_growth': 'Fitness Growth',
    'daily_steps': 'Daily Steps',
    'hours_sleep': 'Sleep Hours'
}

# ---------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------
print("\n[4/5] Training XGBoost models...")

models = {}
results = {}

for target_name, target_label in targets.items():
    print(f"\n{'─'*70}")
    print(f"Training model for: {target_label}")
    print(f"{'─'*70}")
    
    # Prepare data
    X = df[features].fillna(0)
    y = df[target_name]
    
    # Remove any infinite or NaN values
    mask = ~(np.isinf(y) | y.isna())
    X = X[mask]
    y = y[mask]
    
    # Train/Val/Test split: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"   Training:   {X_train.shape[0]:6,} samples")
    print(f"   Validation: {X_val.shape[0]:6,} samples")
    print(f"   Test:       {X_test.shape[0]:6,} samples")
    
    # Train XGBoost
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        early_stopping_rounds=10,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    
    print("\n   🚀 Training...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Avoid division by zero in MAPE
    mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-10))) * 100
    
    print(f"\n   📊 RESULTS:")
    print(f"   ├─ R² Score:  {r2:8.4f}")
    print(f"   ├─ MAE:       {mae:8.4f}")
    print(f"   ├─ RMSE:      {rmse:8.4f}")
    print(f"   └─ MAPE:      {mape:7.2f}%")
    
    # Save model
    os.makedirs('models/saved_models', exist_ok=True)
    model_path = f'models/saved_models/xgboost_{target_name}.pkl'
    joblib.dump(model, model_path)
    print(f"\n   ✓ Saved to: {model_path}")
    
    # Store results
    models[target_name] = model
    results[target_name] = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'feature_importance': pd.Series(
            model.feature_importances_, 
            index=features
        ).sort_values(ascending=False)
    }

# ---------------------------------------------------------
# 5. SAVE FEATURE NAMES
# ---------------------------------------------------------
print("\n[5/5] Saving metadata...")
joblib.dump(features, 'models/saved_models/feature_names.pkl')
print("   ✓ Feature names saved")

# ---------------------------------------------------------
# 6. FINAL SUMMARY
# ---------------------------------------------------------
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

for target_name, target_label in targets.items():
    r = results[target_name]
    print(f"\n📊 {target_label.upper()}:")
    print(f"   R² Score: {r['r2']:.4f}")
    print(f"   MAE:      {r['mae']:.4f}")
    print(f"   RMSE:     {r['rmse']:.4f}")
    print(f"   MAPE:     {r['mape']:.2f}%")
    
    print(f"\n   Top 5 Features:")
    for i, (feat, imp) in enumerate(r['feature_importance'].head(5).items(), 1):
        print(f"   {i}. {feat:25s} {imp:.4f}")

print("\n" + "="*70)
print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
print("="*70)
print("\n📁 Saved models:")
for target_name in targets.keys():
    print(f"   - models/saved_models/xgboost_{target_name}.pkl")

print("\n🚀 Ready to launch Streamlit app!")
