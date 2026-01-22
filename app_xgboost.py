import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Health Activity Predictor",
    page_icon="🏃‍♂️",
    layout="wide"
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
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load trained XGBoost models"""
    models = {}
    try:
        models['fitness_growth'] = joblib.load('models/saved_models/xgboost_fitness_growth.pkl')
        models['daily_steps'] = joblib.load('models/saved_models/xgboost_daily_steps.pkl')
        models['hours_sleep'] = joblib.load('models/saved_models/xgboost_hours_sleep.pkl')
        models['features'] = joblib.load('models/saved_models/feature_names.pkl')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_sample_data():
    """Load sample dataset"""
    try:
        df = pd.read_csv('data/health_fitness_dataset.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except:
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
    st.markdown('<h1 class="main-header">🏃‍♂️ Health Activity Predictor</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Powered Health Analytics using XGBoost</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["📊 Dashboard", "🔮 Predictions", "📈 Trends Analysis", "ℹ️ About"]
    )
    
    # Load data and models
    df = load_sample_data()
    models = load_models()
    
    if df is None:
        st.error("⚠️ Dataset not found. Please ensure health_fitness_dataset.csv is in data/ folder")
        return
    
    # PAGE 1: Dashboard
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
            avg_fitness = participant_data['fitness_level'].mean()
            st.metric("Avg Fitness", f"{avg_fitness:.2f}")
        
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
                participant_data.set_index('date')['fitness_level'],
                "Fitness Level Progression", "Fitness", '#2ca02c'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 2: Predictions
    elif page == "🔮 Predictions":
        st.header("🔮 Next-Day Activity Predictions")
        
        if models is None:
            st.warning("⚠️ Models not loaded. Please train models first.")
            return
        
        st.info("📝 Enter your recent health data to get personalized predictions")
        
        with st.form("prediction_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                duration = st.number_input("Duration (min)", 0, 300, 30)
                calories = st.number_input("Calories Burned", 0, 5000, 300)
                intensity = st.selectbox("Intensity", ["Low", "Medium", "High"], index=1)
                avg_hr = st.number_input("Avg Heart Rate", 40, 200, 75)
            
            with col2:
                sleep = st.number_input("Hours Sleep", 0.0, 12.0, 7.0, 0.5)
                steps = st.number_input("Daily Steps", 0, 30000, 8000)
                stress = st.slider("Stress Level", 1, 10, 5)
                bmi = st.number_input("BMI", 10.0, 50.0, 22.5, 0.1)
            
            with col3:
                resting_hr = st.number_input("Resting HR", 40, 100, 65)
                age = st.number_input("Age", 18, 100, 30)
                weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
                height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
            
            with col4:
                gender = st.selectbox("Gender", ["M", "F", "Other"])
                day_of_week = st.selectbox("Day", 
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                hydration = st.number_input("Hydration (L)", 0.0, 5.0, 2.0, 0.1)
            
            submit = st.form_submit_button("🔮 Generate Predictions", use_container_width=True)
            
            if submit:
                # Prepare input
                intensity_map = {'Low': 1, 'Medium': 2, 'High': 3}
                gender_map = {'M': 1, 'F': 2, 'Other': 3}
                day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, 
                          "Friday": 4, "Saturday": 5, "Sunday": 6}
                
                input_data = pd.DataFrame([{
                    'duration_minutes': duration,
                    'calories_burned': calories,
                    'intensity_encoded': intensity_map[intensity],
                    'avg_heart_rate': avg_hr,
                    'hours_sleep': sleep,
                    'daily_steps': steps,
                    'stress_level': stress,
                    'bmi': bmi,
                    'resting_heart_rate': resting_hr,
                    'age': age,
                    'weight_kg': weight,
                    'height_cm': height,
                    'gender_encoded': gender_map[gender],
                    'day_of_week': day_map[day_of_week],
                    'is_weekend': 1 if day_map[day_of_week] >= 5 else 0,
                    'hydration_level': hydration
                }])
                
                # Reorder columns to match training
                input_data = input_data[models['features']]
                
                st.success("✅ Predictions generated!")
                
                # Make predictions
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                with pred_col1:
                    fitness_pred = models['fitness_growth'].predict(input_data)[0]
                    st.metric(
                        "Predicted Fitness Growth",
                        f"{fitness_pred:.4f}",
                        delta=f"{fitness_pred*100:+.2f}%"
                    )
                
                with pred_col2:
                    steps_pred = models['daily_steps'].predict(input_data)[0]
                    st.metric(
                        "Predicted Steps Tomorrow",
                        f"{steps_pred:,.0f}",
                        delta=f"{steps_pred - steps:+,.0f}"
                    )
                
                with pred_col3:
                    sleep_pred = models['hours_sleep'].predict(input_data)[0]
                    st.metric(
                        "Predicted Sleep Tomorrow",
                        f"{sleep_pred:.1f} hrs",
                        delta=f"{sleep_pred - sleep:+.1f}"
                    )
                
                # Recommendations
                st.subheader("💡 Personalized Recommendations")
                
                if fitness_pred < 0.01:
                    st.warning("📉 Low fitness growth predicted. Consider increasing workout intensity!")
                else:
                    st.success("📈 Good fitness progression expected! Keep up the good work!")
                
                if steps_pred < 8000:
                    st.info("🚶 Try to increase daily movement to reach 8,000+ steps")
                
                if stress > 7:
                    st.warning("🧘 High stress detected. Consider meditation or relaxation exercises")
    
    # PAGE 3: Trends
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
                       'stress_level', 'calories_burned', 'fitness_level']
        corr_matrix = participant_data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       color_continuous_scale='RdBu_r',
                       aspect="auto",
                       zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 4: About
    else:
        st.header("ℹ️ About This Application")
        st.markdown("""
        ### 🎯 Health Activity Predictor
        
        **Powered by XGBoost Machine Learning**
        
        #### 📊 Model Performance
        - ✅ **R² Score: 0.9872** (Near-perfect accuracy!)
        - ✅ **MAE: 0.0033** (Extremely low error)
        - ✅ **Fast Training** (Minutes instead of hours)
        
        #### 🔬 Technology Stack
        - **XGBoost** for predictions
        - **Streamlit** for web interface
        - **Plotly** for interactive visualizations
        
        #### 📈 Features
        - Daily health tracking
        - Next-day predictions
        - Fitness growth forecasting
        - Personalized recommendations
        
        #### �� Project Info
        Final Year CSE AIML Project demonstrating:
        - Advanced ML with XGBoost
        - Feature engineering
        - Full-stack deployment
        - Data visualization
        """)

if __name__ == "__main__":
    main()
