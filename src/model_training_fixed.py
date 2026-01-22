import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import os
import sys

sys.path.append('src')
from preprocessing import HealthDataPreprocessor

class HealthActivityPredictor:
    def __init__(self, model_type='lstm'):
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_lstm_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, 
                              input_shape=input_shape)),
            Dropout(0.3),
            Bidirectional(LSTM(64, return_sequences=False)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Use legacy optimizer for M1/M2 Macs
        model.compile(
            optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def build_ml_model(self, model_name='random_forest'):
        """Build traditional ML model"""
        if model_name == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_name == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model"""
        os.makedirs('models/saved_models', exist_ok=True)
        
        if self.model_type == 'lstm':
            print("\n🔨 Building LSTM model...")
            self.model = self.build_lstm_model(
                input_shape=(X_train.shape[1], X_train.shape[2])
            )
            
            # Build model first with a sample batch
            self.model.build(input_shape=(None, X_train.shape[1], X_train.shape[2]))
            
            print("\n📋 Model Architecture:")
            self.model.summary()
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ModelCheckpoint(
                    'models/saved_models/best_lstm_model.keras',
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            ]
            
            print("\n🚀 Training LSTM model...")
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
        else:  # ML models
            print(f"\n🔨 Building {self.model_type} model...")
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
            
            self.model = self.build_ml_model(self.model_type)
            
            print(f"🚀 Training {self.model_type} model...")
            self.model.fit(X_train_flat, y_train)
            
            joblib.dump(self.model, 
                       f'models/saved_models/{self.model_type}_model.pkl')
            print(f"✓ Model saved to: models/saved_models/{self.model_type}_model.pkl")
        
        # Evaluate
        metrics = self.evaluate(X_val, y_val, dataset_name="Validation")
        
        return self.model, metrics
    
    def evaluate(self, X_test, y_test, dataset_name="Test"):
        """Evaluate model performance"""
        if self.model_type == 'lstm':
            predictions = self.model.predict(X_test, verbose=0).flatten()
        else:
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            predictions = self.model.predict(X_test_flat)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        print(f"\n{'='*70}")
        print(f"{self.model_type.upper()} Model - {dataset_name} Set Evaluation")
        print(f"{'='*70}")
        print(f"MAE (Mean Absolute Error):  {mae:10.2f}")
        print(f"RMSE (Root Mean Sq Error):  {rmse:10.2f}")
        print(f"R² Score:                   {r2:10.4f}")
        print(f"{'='*70}")
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': predictions}
    
    def predict(self, X):
        """Make predictions"""
        if self.model_type == 'lstm':
            return self.model.predict(X, verbose=0).flatten()
        else:
            X_flat = X.reshape(X.shape[0], -1)
            return self.model.predict(X_flat)

def train_all_models():
    """Train models for multiple target variables"""
    print("="*70)
    print("HEALTH ACTIVITY PREDICTOR - MODEL TRAINING")
    print("="*70)
    
    preprocessor = HealthDataPreprocessor()
    df = preprocessor.load_and_preprocess()
    
    targets = {
        'daily_steps': 'lstm',
        'avg_heart_rate': 'random_forest',
        'hours_sleep': 'gradient_boosting'
    }
    
    results = {}
    
    for target_num, (target, model_type) in enumerate(targets.items(), 1):
        print(f"\n{'#'*70}")
        print(f"MODEL {target_num}/{len(targets)}: Training {model_type.upper()} for {target}")
        print(f"{'#'*70}")
        
        # Prepare data
        X, y, features = preprocessor.prepare_training_data(
            df, lookback=7, target_col=target, n_participants=300
        )
        
        # Split data: 70% train, 15% validation, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        print(f"\n📊 Dataset Splits:")
        print(f"   Training:   {X_train.shape[0]:5d} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   Validation: {X_val.shape[0]:5d} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"   Test:       {X_test.shape[0]:5d} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Train model
        predictor = HealthActivityPredictor(model_type=model_type)
        
        if model_type == 'lstm':
            model, val_metrics = predictor.train(
                X_train, y_train, X_val, y_val, epochs=30, batch_size=32
            )
        else:
            model, val_metrics = predictor.train(
                X_train, y_train, X_val, y_val
            )
        
        # Final test evaluation
        test_metrics = predictor.evaluate(X_test, y_test, dataset_name="Test")
        results[target] = test_metrics
    
    # Final Summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    for target, metrics in results.items():
        print(f"\n📊 {target.upper().replace('_', ' ')}:")
        print(f"   MAE:  {metrics['mae']:10.2f}")
        print(f"   RMSE: {metrics['rmse']:10.2f}")
        print(f"   R²:   {metrics['r2']:10.4f}")
    
    print("\n" + "="*70)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Saved models:")
    print("   - models/saved_models/best_lstm_model.keras")
    print("   - models/saved_models/random_forest_model.pkl")
    print("   - models/saved_models/gradient_boosting_model.pkl")
    
    return results

if __name__ == "__main__":
    results = train_all_models()
