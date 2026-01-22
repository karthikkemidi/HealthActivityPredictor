import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from preprocessing import HealthDataPreprocessor

class HealthPredictor:
    def __init__(self):
        self.models = {}
        self.preprocessor = None
        
    def load_models(self):
        """Load all trained models"""
        try:
            self.models['steps_lstm'] = tf.keras.models.load_model(
                'models/saved_models/best_lstm_model.keras'
            )
            self.models['heart_rate_rf'] = joblib.load(
                'models/saved_models/random_forest_model.pkl'
            )
            self.models['sleep_gb'] = joblib.load(
                'models/saved_models/gradient_boosting_model.pkl'
            )
            self.preprocessor = HealthDataPreprocessor.load_preprocessor(
                'models/preprocessor.pkl'
            )
            print("✓ All models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict_next_day(self, input_sequence):
        """Predict next day metrics"""
        predictions = {}
        
        # Predict steps using LSTM
        steps_pred = self.models['steps_lstm'].predict(
            input_sequence, verbose=0
        )[0][0]
        predictions['daily_steps'] = steps_pred
        
        # Predict heart rate using Random Forest
        input_flat = input_sequence.reshape(1, -1)
        hr_pred = self.models['heart_rate_rf'].predict(input_flat)[0]
        predictions['avg_heart_rate'] = hr_pred
        
        # Predict sleep using Gradient Boosting
        sleep_pred = self.models['sleep_gb'].predict(input_flat)[0]
        predictions['hours_sleep'] = sleep_pred
        
        return predictions

if __name__ == "__main__":
    predictor = HealthPredictor()
    if predictor.load_models():
        print("✅ Prediction system ready!")
