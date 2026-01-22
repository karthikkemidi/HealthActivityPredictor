import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
import os
import sys

sys.path.append('src')
from src.auth import AuthManager, require_auth
from src.database import UserDatabase

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
    }
    .login-container {
        max-width: 400px;
        margin: 50px auto;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database
@st.cache_resource
def init_database():
    db = UserDatabase()
    return db

@st.cache_resource
def load_models():
    """Load trained XGBoost models"""
    models = {}
    try:
        if os.path.exists('models/saved_models/xgboost_fitness_growth.pkl'):
            models['fitness_growth'] = joblib.load('models/saved_models/xgboost_fitness_growth.pkl')
        if os.path.exists('models/saved_models/xgboost_daily_steps.pkl'):
            models['daily_steps'] = joblib.load('models/saved_models/xgboost_daily_steps.pkl')
        if os.path.exists('models/saved_models/xgboost_hours_sleep.pkl'):
            models['hours_sleep'] = joblib.load('models/saved_models/xgboost_hours_sleep.pkl')
        if os.path.exists('models/saved_models/feature_names.pkl'):
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

def show_login_page(db):
    """Display login page"""
    st.markdown('<h1 class="main-header">🏃‍♂️ Health Activity Predictor</h1>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            
            with st.form("login_form"):
                username = st.text_input("Username or Email", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("🔓 Login", use_container_width=True)
                
                if submit:
                    if username and password:
                        user = db.authenticate_user(username, password)
                        
                        if user:
                            # Create JWT token
                            token = AuthManager.create_token(
                                user['user_id'],
                                user['participant_id'],
                                user['username']
                            )
                            
                            # Store in session
                            st.session_state['auth_token'] = token
                            st.session_state['user'] = user
                            
                            st.success(f"✅ Welcome back, {user['full_name'] or user['username']}!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
                    else:
                        st.warning("⚠️ Please fill in all fields")
            
            st.info("💡 **Demo Accounts:**\n- Username: `user1` | Password: `health1`\n- Username: `user2` | Password: `health2`")
        
        with tab2:
            st.subheader("Create New Account")
            
            with st.form("register_form"):
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_email = st.text_input("Email", placeholder="your.email@example.com")
                new_password = st.text_input("Password", type="password", placeholder="Create a password")
                new_password_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                participant_id = st.number_input("Participant ID", min_value=1, max_value=3000, value=1, 
                                               help="Your unique participant ID from the study")
                full_name = st.text_input("Full Name (Optional)", placeholder="Your full name")
                
                register_submit = st.form_submit_button("📝 Register", use_container_width=True)
                
                if register_submit:
                    if not all([new_username, new_email, new_password, new_password_confirm]):
                        st.warning("⚠️ Please fill in all required fields")
                    elif new_password != new_password_confirm:
                        st.error("❌ Passwords do not match")
                    elif len(new_password) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        success, message = db.register_user(
                            new_username, new_email, new_password, 
                            participant_id, full_name
                        )
                        
                        if success:
                            st.success(f"✅ {message} Please login now!")
                        else:
                            st.error(f"❌ {message}")
        
        st.markdown('</div>', unsafe_allow_html=True)

@require_auth
def show_main_app(db):
    """Display main application (requires authentication)"""
    user = AuthManager.get_current_user()
    participant_id = user['participant_id']
    
    # Sidebar
    with st.sidebar:
        st.title("🎯 Navigation")
        
        # User info
        st.markdown("---")
        st.markdown(f"**👤 Logged in as:**")
        st.markdown(f"**{user['username']}**")
        st.markdown(f"📊 Participant ID: {participant_id}")
        
        if st.button("🚪 Logout", use_container_width=True):
            AuthManager.logout()
            st.rerun()
        
        st.markdown("---")
        
        page = st.radio(
            "Select Page",
            ["📊 My Dashboard", "🔮 Predictions", "📈 Trends Analysis", "👤 Profile", "ℹ️ About"]
        )
    
    # Load data
    df = load_sample_data()
    models = load_models()
    
    if df is None:
        st.error("⚠️ Failed to load data")
        return
    
    # Filter data for current user only
    user_data = df[df['participant_id'] == participant_id].copy()
    user_data = user_data.sort_values('date')
    
    # Header
    st.markdown(f'<h1 class="main-header">🏃‍♂️ Health Activity Predictor</h1>', 
                unsafe_allow_html=True)
    
    # PAGE 1: Dashboard
    if page == "📊 My Dashboard":
        st.header(f"📊 Your Health Dashboard")
        
        if len(user_data) == 0:
            st.warning("⚠️ No health data found for your account")
            return
        
        # Key metrics
        st.subheader("📌 Your Key Health Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_steps = user_data['daily_steps'].mean()
            st.metric("Avg Daily Steps", f"{avg_steps:,.0f}")
        
        with col2:
            avg_sleep = user_data['hours_sleep'].mean()
            st.metric("Avg Sleep", f"{avg_sleep:.1f} hrs")
        
        with col3:
            avg_fitness = user_data['fitness_level'].mean()
            st.metric("Avg Fitness", f"{avg_fitness:.2f}")
        
        with col4:
            total_days = len(user_data)
            st.metric("Total Days Tracked", f"{total_days}")
        
        # Charts
        st.subheader("📈 Your Activity Trends")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = create_trend_chart(
                user_data.set_index('date')['daily_steps'],
                "Your Daily Steps Trend", "Steps", '#1f77b4'
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            fig = create_trend_chart(
                user_data.set_index('date')['fitness_level'],
                "Your Fitness Level Progression", "Fitness", '#2ca02c'
            )
            st.plotly_chart(fig, width="stretch")
        
        # Recent activity
        st.subheader("📅 Recent Activity")
        recent_data = user_data.tail(7)[['date', 'daily_steps', 'hours_sleep', 'calories_burned', 'fitness_level']]
        st.dataframe(recent_data, use_container_width=True, hide_index=True)
    
    # PAGE 2: Predictions
    elif page == "🔮 Predictions":
        st.header("🔮 Personalized Predictions")
        
        if models is None or 'features' not in models:
            st.warning("⚠️ Models not loaded. Please train models first.")
            st.code("python src/model_training_xgboost.py")
            return
        
        st.info("📝 Enter your recent health data to get AI-powered predictions")
        
        # Get user's latest data as defaults
        if len(user_data) > 0:
            latest = user_data.iloc[-1]
        else:
            latest = None
        
        with st.form("prediction_form"):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                duration = st.number_input("Duration (min)", 0, 300, 
                                          int(latest['duration_minutes']) if latest is not None else 30)
                calories = st.number_input("Calories Burned", 0, 5000, 
                                          int(latest['calories_burned']) if latest is not None else 300)
                intensity = st.selectbox("Intensity", ["Low", "Medium", "High"], index=1)
                avg_hr = st.number_input("Avg Heart Rate", 40, 200, 
                                        int(latest['avg_heart_rate']) if latest is not None else 75)
            
            with col2:
                sleep = st.number_input("Hours Sleep", 0.0, 12.0, 
                                       float(latest['hours_sleep']) if latest is not None else 7.0, 0.5)
                steps = st.number_input("Daily Steps", 0, 30000, 
                                       int(latest['daily_steps']) if latest is not None else 8000)
                stress = st.slider("Stress Level", 1, 10, 
                                  int(latest['stress_level']) if latest is not None else 5)
                bmi = st.number_input("BMI", 10.0, 50.0, 
                                     float(latest['bmi']) if latest is not None else 22.5, 0.1)
            
            with col3:
                resting_hr = st.number_input("Resting HR", 40, 100, 
                                            int(latest['resting_heart_rate']) if latest is not None else 65)
                age = st.number_input("Age", 18, 100, 
                                     int(latest['age']) if latest is not None else 30)
                weight = st.number_input("Weight (kg)", 30.0, 200.0, 
                                        float(latest['weight_kg']) if latest is not None else 70.0)
                height = st.number_input("Height (cm)", 100.0, 250.0, 
                                        float(latest['height_cm']) if latest is not None else 170.0)
            
            with col4:
                gender = st.selectbox("Gender", ["M", "F", "Other"], 
                                     index=0 if latest is None else ["M", "F", "Other"].index(latest['gender']))
                day_of_week = st.selectbox("Day", 
                    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
                hydration = st.number_input("Hydration (L)", 0.0, 5.0, 
                                           float(latest['hydration_level']) if latest is not None else 2.0, 0.1)
            
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
                
                # Reorder columns
                input_data = input_data[models['features']]
                
                st.success("✅ Predictions generated!")
                
                # Make predictions
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                if 'fitness_growth' in models:
                    with pred_col1:
                        fitness_pred = models['fitness_growth'].predict(input_data)[0]
                        st.metric(
                            "Predicted Fitness Growth",
                            f"{fitness_pred:.4f}",
                            delta=f"{fitness_pred*100:+.2f}%"
                        )
                
                if 'daily_steps' in models:
                    with pred_col2:
                        steps_pred = models['daily_steps'].predict(input_data)[0]
                        st.metric(
                            "Predicted Steps Tomorrow",
                            f"{steps_pred:,.0f}",
                            delta=f"{steps_pred - steps:+,.0f}"
                        )
                
                if 'hours_sleep' in models:
                    with pred_col3:
                        sleep_pred = models['hours_sleep'].predict(input_data)[0]
                        st.metric(
                            "Predicted Sleep Tomorrow",
                            f"{sleep_pred:.1f} hrs",
                            delta=f"{sleep_pred - sleep:+.1f}"
                        )
    
    # PAGE 3: Trends
    elif page == "📈 Trends Analysis":
        st.header("📈 Your Health Trends Analysis")
        
        if len(user_data) < 7:
            st.warning("⚠️ Need at least 7 days of data for trend analysis")
            return
        
        # Correlation matrix
        st.subheader("🔗 Your Metrics Correlation")
        
        numeric_cols = ['daily_steps', 'hours_sleep', 'avg_heart_rate', 
                       'stress_level', 'calories_burned', 'fitness_level']
        corr_matrix = user_data[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       color_continuous_scale='RdBu_r',
                       aspect="auto",
                       zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly patterns
        st.subheader("📅 Your Weekly Activity Patterns")
        
        user_data['day_name'] = user_data['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        col1, col2 = st.columns(2)
        
        with col1:
            weekly_steps = user_data.groupby('day_name')['daily_steps'].mean().reindex(day_order)
            fig = px.bar(
                x=weekly_steps.index,
                y=weekly_steps.values,
                title="Your Average Steps by Day of Week",
                labels={'x': 'Day', 'y': 'Average Steps'},
                color=weekly_steps.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            weekly_fitness = user_data.groupby('day_name')['fitness_level'].mean().reindex(day_order)
            fig = px.bar(
                x=weekly_fitness.index,
                y=weekly_fitness.values,
                title="Your Average Fitness by Day of Week",
                labels={'x': 'Day', 'y': 'Average Fitness'},
                color=weekly_fitness.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # PAGE 4: Profile
    elif page == "👤 Profile":
        st.header("👤 Your Profile")
        
        user_info = db.get_user_by_id(user['user_id'])
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Account Information")
            st.info(f"""
            **Username:** {user_info['username']}  
            **Email:** {user_info['email']}  
            **Participant ID:** {user_info['participant_id']}  
            **Full Name:** {user_info['full_name'] or 'Not set'}  
            **Member Since:** {user_info['created_at'][:10]}  
            **Last Login:** {user_info['last_login'][:19] if user_info['last_login'] else 'N/A'}
            """)
        
        with col2:
            st.markdown("### Update Password")
            
            with st.form("password_form"):
                new_pass = st.text_input("New Password", type="password")
                confirm_pass = st.text_input("Confirm Password", type="password")
                update_btn = st.form_submit_button("🔒 Update Password")
                
                if update_btn:
                    if new_pass != confirm_pass:
                        st.error("❌ Passwords do not match")
                    elif len(new_pass) < 6:
                        st.error("❌ Password must be at least 6 characters")
                    else:
                        if db.update_password(user['user_id'], new_pass):
                            st.success("✅ Password updated successfully!")
                        else:
                            st.error("❌ Failed to update password")
    
    # PAGE 5: About
    else:
        st.header("ℹ️ About This Application")
        st.markdown("""
        ### 🎯 Health Activity Predictor
        
        **Personalized AI-Powered Health Analytics with Secure Authentication**
        
        #### 🔐 Security Features
        - ✅ **JWT Authentication** - Secure token-based login
        - ✅ **Password Hashing** - bcrypt encryption
        - ✅ **User Isolation** - Each user sees only their own data
        - ✅ **Session Management** - 24-hour token expiration
        
        #### 📊 Model Performance
        - ✅ **R² Score: 0.9872** (Near-perfect accuracy!)
        - ✅ **MAE: 0.0033** (Extremely low error)
        - ✅ **XGBoost Algorithm** - Fast and accurate predictions
        
        #### 🎓 Project Info
        Final Year CSE AIML Project demonstrating:
        - Advanced ML with XGBoost
        - JWT-based authentication
        - Multi-user system architecture
        - Full-stack deployment
        - Secure data handling
        
        #### 📁 System Statistics
        - **Total Users:** 3,000 participants
        - **Your Data:** Private and secure
        - **Predictions:** Personalized for you
        """)

def main():
    """Main application entry point"""
    db = init_database()
    
    # Check if users already exist FIRST
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        conn.close()
        
        # Only create if database is empty
        if user_count == 0:
            st.info("🔄 Creating default users (first run - takes ~30 seconds)...")
            df = load_sample_data()
            if df is not None:
                created = db.create_default_users(df)
                st.success(f"✅ Created {created} users! Please refresh the page.")
                st.stop()  # Stop and ask user to refresh
        
        # Mark as initialized
        if 'users_initialized' not in st.session_state:
            st.session_state['users_initialized'] = True
            
    except Exception as e:
        st.error(f"⚠️ Database setup error: {e}")
        st.info("You can still register manually using the Register tab.")
    
    # Route based on authentication
    if not AuthManager.is_authenticated():
        show_login_page(db)
    else:
        show_main_app(db)

if __name__ == "__main__":
    main()

