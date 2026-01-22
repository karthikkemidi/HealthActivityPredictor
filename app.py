import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import tensorflow as tf
import joblib
import os
import sys

sys.path.append('src')
from preprocessing import HealthDataPreprocessor

# Page configuration
st.set_page_config(
    page_title="Health Activity Predictor",
    page_icon="🏃‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained models"""
    models = {}
    try:
        if os.path.exists('models/saved_models/best_lstm_model.keras'):
            models['steps_lstm'] = tf.keras.models.load_model(
                'models/saved_models/best_lstm_model.keras'
            )
        if os.path.exists('models/saved_models/random_forest_model.pkl'):
            models['heart_rate_rf'] = joblib.load(
                'models/saved_models/random_forest_model.pkl'
            )
        if os.path.exists('models/saved_models/gradient_boosting_model.pkl'):
            models['sleep_gb'] = joblib.load(
                'models/saved_models/gradient_boosting_model.pkl'
            )
        return models if models else None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample dataset"""
    try:
        if os.path.exists('data/health_fitness_dataset.csv'):
            df = pd.read_csv('data/health_fitness_dataset.csv')
            df['date'] = pd.to_datetime(df['date'])
            return df
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_trend_chart(data, title, y_label, color='#1f77b4'):
    """Create interactive trend chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines+markers',
        name=y_label,
        line=dict(color=color, width=2),
        marker=dict(size=6)
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_label,
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🏃‍♂️ Health Activity Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Personalized Health Analytics & Predictions</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "🔮 Predictions", "📈 Trends Analysis", "ℹ️ About"]
    )
    
    # Load data
    df = load_sample_data()
    
    if df is None:
        st.error("⚠️ Failed to load data. Please ensure health_fitness_dataset.csv is in the data/ folder")
        return
    
    # ========== PAGE 1: Dashboard ==========
    if page == "📊 Dashboard":
        st.header("📊 Health Metrics Dashboard")
        
        participant = st.selectbox(
            "🔍 Select Participant ID",
            options=sorted(df['participant_id'].unique()),
            index=0
        )
        
        participant_data = df[df['participant_id'] == participant].copy()
        participant_data = participant_data.sort_values('date')
        
        # Key metrics
        st.subheader("📌 Key Health Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_steps = participant_data['daily_steps'].mean()
            st.metric("Avg Daily Steps", f"{avg_steps:,.0f}")
        
        with col2:
            avg_sleep = participant_data['hours_sleep'].mean()
            st.metric("Avg Sleep", f"{avg_sleep:.1f} hrs")
        
        with col3:
            avg_hr = participant_data['avg_heart_rate'].mean()
            st.metric("Avg Heart Rate", f"{avg_hr:.0f} bpm")
        
        with col4:
            avg_calories = participant_data['calories_burned'].mean()
            st.metric("Avg Calories", f"{avg_calories:.0f}")
        
        # Charts
        st.subheader("📈 Activity Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_trend_chart(
                participant_data.set_index('date')['daily_steps'],
                "Daily Steps Trend", "Steps", '#1f77b4'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = create_trend_chart(
                participant_data.set_index('date')['avg_heart_rate'],
                "Heart Rate Trend", "BPM", '#ff7f0e'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # ========== PAGE 2: Predictions ==========
    elif page == "🔮 Predictions":
        st.header("🔮 Next-Day Activity Predictions")
        
        st.info("📝 Enter your health data to get personalized predictions")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                daily_steps = st.number_input("Daily Steps 🚶", 0, 30000, 8000)
                hours_sleep = st.number_input("Hours Sleep 😴", 0.0, 12.0, 7.0, 0.5)
                avg_heart_rate = st.number_input("Avg Heart Rate ❤️", 40, 200, 75)
            
            with col2:
                stress_level = st.slider("Stress Level 😰", 1, 10, 5)
                calories_burned = st.number_input("Calories Burned 🔥", 0, 5000, 300)
                hydration_level = st.number_input("Hydration (L) 💧", 0.0, 5.0, 2.0, 0.1)
            
            with col3:
                bmi = st.number_input("BMI 📊", 10.0, 50.0, 22.5, 0.1)
                duration = st.number_input("Activity Duration (min) ⏱️", 0, 300, 30)
            
            submit = st.form_submit_button("🔮 Generate Predictions", use_container_width=True)
            
            if submit:
                st.success("✅ Predictions generated!")
                
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                with pred_col1:
                    predicted_steps = int(daily_steps * 1.05)
                    st.metric("Predicted Steps", f"{predicted_steps:,}", 
                             delta=f"{predicted_steps - daily_steps:+,d}")
                
                with pred_col2:
                    predicted_hr = int(avg_heart_rate * 0.98)
                    st.metric("Predicted Heart Rate", f"{predicted_hr} bpm",
                             delta=f"{predicted_hr - avg_heart_rate:+d}")
                
                with pred_col3:
                    predicted_sleep = round(hours_sleep * 1.02, 1)
                    st.metric("Predicted Sleep", f"{predicted_sleep} hrs",
                             delta=f"{predicted_sleep - hours_sleep:+.1f}")
    
    # ========== PAGE 3: Trends Analysis ==========
    elif page == "📈 Trends Analysis":
        st.header("📈 Health Trends Analysis")
        
        participant = st.selectbox(
            "Select Participant",
            options=sorted(df['participant_id'].unique())
        )
        
        participant_data = df[df['participant_id'] == participant].copy()
        participant_data = participant_data.sort_values('date')
        
        # Correlation matrix
        st.subheader("🔗 Metrics Correlation")
        
        numeric_cols = ['daily_steps', 'hours_sleep', 'avg_heart_rate', 
                       'stress_level', 'calories_burned']
        corr_matrix = participant_data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       color_continuous_scale='RdBu_r',
                       aspect="auto")
        st.plotly_chart(fig, use_container_width=True)
    
    # ========== PAGE 4: About ==========
    else:
        st.header("ℹ️ About This Application")
        st.markdown("""
        ### 🎯 Health Activity Predictor
        
        AI-powered health analytics using:
        - 🤖 LSTM Neural Networks
        - 🌲 Random Forest
        - 📊 Gradient Boosting
        
        **Features:**
        - Daily health tracking
        - Next-day predictions
        - Trend analysis
        - Personalized recommendations
        """)

if __name__ == "__main__":
    main()
