import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class HealthDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_and_preprocess(self, filepath='data/health_fitness_dataset.csv'):
        """Load and preprocess the dataset"""
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE")
        print("="*70)
        
        print("\n[1/5] Loading data...")
        df = pd.read_csv(filepath)
        print(f"   ✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Convert date to datetime
        print("\n[2/5] Processing date column...")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            print("   ✓ Date column converted to datetime")
        
        # Sort by participant and date
        print("\n[3/5] Sorting data...")
        df = df.sort_values(['participant_id', 'date']).reset_index(drop=True)
        print("   ✓ Data sorted by participant_id and date")
        
        # Handle missing values
        print("\n[4/5] Handling missing values...")
        if 'health_condition' in df.columns:
            df['health_condition'].fillna('None', inplace=True)
            print("   ✓ Filled missing health_condition values with 'None'")
        
        # Create time-based features
        print("\n[5/5] Creating time-based features...")
        if 'date' in df.columns:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day_of_year'] = df['date'].dt.dayofyear
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            print("   ✓ Created: day_of_week, month, day_of_year, is_weekend")
        
        # Encode categorical variables
        print("\n[6/5] Encoding categorical variables...")
        categorical_cols = ['gender', 'activity_type', 'intensity', 
                          'health_condition', 'smoking_status']
        
        encoded_count = 0
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                encoded_count += 1
                print(f"   ✓ Encoded: {col}")
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Final shape: {df.shape}")
        print(f"   Categorical columns encoded: {encoded_count}")
        
        return df
    
    def create_time_series_sequences(self, df, participant_id, 
                                     lookback=7, target_col='daily_steps'):
        """Create sequences for time series prediction"""
        participant_data = df[df['participant_id'] == participant_id].copy()
        participant_data = participant_data.sort_values('date')
        
        # Select features
        feature_cols = [
            'daily_steps', 'hours_sleep', 'avg_heart_rate', 
            'stress_level', 'calories_burned', 'hydration_level',
            'resting_heart_rate', 'bmi', 'duration_minutes',
            'day_of_week', 'activity_type_encoded', 'intensity_encoded'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in participant_data.columns]
        
        if target_col not in available_cols:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        data = participant_data[available_cols].values
        
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            target_idx = available_cols.index(target_col)
            y.append(data[i, target_idx])
        
        return np.array(X), np.array(y), available_cols
    
    def prepare_training_data(self, df, lookback=7, target_col='daily_steps', 
                            n_participants=500):
        """Prepare data for all participants"""
        print("\n" + "="*70)
        print(f"PREPARING TRAINING SEQUENCES FOR: {target_col}")
        print("="*70)
        
        X_all, y_all = [], []
        
        unique_participants = df['participant_id'].unique()[:n_participants]
        successful = 0
        failed = 0
        
        print(f"\nProcessing {len(unique_participants)} participants...")
        print(f"Lookback window: {lookback} days")
        print(f"Target column: {target_col}\n")
        
        for idx, pid in enumerate(unique_participants):
            try:
                X, y, features = self.create_time_series_sequences(
                    df, pid, lookback, target_col
                )
                if len(X) > 0:
                    X_all.extend(X)
                    y_all.extend(y)
                    successful += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"   Processed {idx + 1}/{len(unique_participants)} participants... "
                          f"({successful} successful, {failed} failed)")
                    
            except Exception as e:
                failed += 1
                continue
        
        X_all = np.array(X_all)
        y_all = np.array(y_all)
        
        self.feature_columns = features
        
        print(f"\n✅ Sequence creation complete!")
        print(f"   Successful participants: {successful}")
        print(f"   Failed participants: {failed}")
        print(f"   Total sequences created: {len(X_all):,}")
        print(f"   X shape: {X_all.shape}")
        print(f"   y shape: {y_all.shape}")
        print(f"   Features used: {len(features)}")
        
        return X_all, y_all, features
    
    def save_preprocessor(self, path='models/preprocessor.pkl'):
        """Save the preprocessor"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"\n✅ Preprocessor saved to: {path}")
    
    @staticmethod
    def load_preprocessor(path='models/preprocessor.pkl'):
        """Load a saved preprocessor"""
        return joblib.load(path)

if __name__ == "__main__":
    print("="*70)
    print("HEALTH ACTIVITY PREDICTOR - PREPROCESSING")
    print("="*70)
    
    preprocessor = HealthDataPreprocessor()
    df = preprocessor.load_and_preprocess()
    
    # Prepare sequences for daily steps prediction
    X, y, features = preprocessor.prepare_training_data(
        df, lookback=7, target_col='daily_steps', n_participants=200
    )
    
    print(f"\n📊 Feature columns ({len(features)}):")
    for i, feat in enumerate(features, 1):
        print(f"   {i:2d}. {feat}")
    
    # Save preprocessor
    preprocessor.save_preprocessor()
    
    print("\n✅ Preprocessing pipeline completed successfully!")
