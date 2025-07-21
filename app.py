import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Load data directly from compressed files
@st.cache_resource
def load_data():
    """Load data from compressed files"""
    try:
        # Load health data from compressed file
        health_data = pd.read_csv('data/health_fitness_dataset_compressed.csv.bz2', compression='bz2')
        
        # Load food data
        food_data = pd.read_csv('data/FOOD-DATA-GROUP1.csv')
        
        # Load sleep data
        sleep_data = pd.read_csv('data/Sleep_health_and_lifestyle_dataset.csv')
        
        # Create simple models and scalers for demonstration
        # In a real application, you would load pre-trained models
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        
        # Create label encoders
        label_encoders = {}
        for col in ['gender', 'activity_type', 'intensity', 'health_condition', 'smoking_status']:
            if col in health_data.columns:
                le = LabelEncoder()
                health_data[col + '_encoded'] = le.fit_transform(health_data[col].fillna('Unknown'))
                label_encoders[col] = le
        
        # Create simple scalers
        numeric_cols = ['age', 'height_cm', 'weight_kg', 'duration_minutes', 'calories_burned', 
                       'avg_heart_rate', 'hours_sleep', 'stress_level', 'daily_steps']
        scaler = StandardScaler()
        health_data[numeric_cols] = scaler.fit_transform(health_data[numeric_cols].fillna(0))
        
        # Add nutrition score to food data if not present
        if 'nutrition_score' not in food_data.columns:
            food_data['nutrition_score'] = (
                food_data['Protein'] * 0.3 + 
                food_data['Dietary Fiber'] * 0.2 + 
                food_data['Vitamin C'] * 0.1 + 
                food_data['Calcium'] * 0.1 + 
                food_data['Iron'] * 0.1
            )
        
        return {
            'health_data': health_data,
            'food_data': food_data,
            'sleep_data': sleep_data,
            'label_encoders': label_encoders,
            'scaler': scaler
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data()

if data is None:
    st.error("Failed to load data. Please ensure all data files are present.")
    st.stop()

health_data = data['health_data']
food_data = data['food_data']
sleep_data = data['sleep_data']
label_encoders = data['label_encoders']
scaler = data['scaler']

# Page configuration
st.set_page_config(
    page_title="Precision Medicine - Diet & Exercise Recommendations",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card, .recommendation-card {
        background: rgba(30, 32, 38, 0.7); /* subtle dark overlay */
        border: 1px solid #333a;
        border-radius: 0.75rem;
        box-shadow: 0 2px 8px 0 rgba(0,0,0,0.10);
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover, .recommendation-card:hover {
        box-shadow: 0 4px 16px 0 rgba(0,0,0,0.18);
        border-color: #1f77b4;
    }
    .thin-separator {
        border: none;
        border-top: 1px solid #444;
        margin: 1.5rem 0 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def calculate_bmi(weight_kg, height_cm):
    height_m = height_cm / 100
    return weight_kg / (height_m ** 2)

def get_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"

def get_diet_recommendations(user_data):
    """Get diet recommendations based on user profile"""
    bmi_category = user_data.get('bmi_category')
    
    # Filter foods based on user profile
    if bmi_category == 'Underweight':
        filtered_foods = food_data[
            (food_data['Caloric Value'] > 200) & 
            (food_data['Protein'] > 15) & 
            (food_data['Fat'] > 10)
        ].sort_values('nutrition_score', ascending=False)
    elif bmi_category == 'Obese':
        filtered_foods = food_data[
            (food_data['Caloric Value'] < 150) & 
            (food_data['Dietary Fiber'] > 3) & 
            (food_data['Fat'] < 10)
        ].sort_values('nutrition_score', ascending=False)
    else:
        filtered_foods = food_data[
            (food_data['Caloric Value'].between(100, 300)) & 
            (food_data['Protein'] > 10) & 
            (food_data['Dietary Fiber'] > 2)
        ].sort_values('nutrition_score', ascending=False)
    
    recommendations = []
    for _, food in filtered_foods.head(10).iterrows():
        recommendations.append({
            'food': food['food'],
            'calories': food['Caloric Value'],
            'protein': food['Protein'],
            'carbs': food['Carbohydrates'],
            'fat': food['Fat'],
            'fiber': food['Dietary Fiber'],
            'nutrition_score': food['nutrition_score']
        })
    return recommendations

def get_exercise_recommendations(user_data):
    """Get exercise recommendations based on user profile"""
    bmi_category = user_data.get('bmi_category')
    
    # Get activity types from health data
    available_activities = health_data['activity_type'].unique()
    
    if bmi_category == 'Underweight':
        preferred_activities = ['Weight Training', 'Strength Training', 'Yoga']
    elif bmi_category == 'Obese':
        preferred_activities = ['Walking', 'Swimming', 'Cycling']
    else:
        preferred_activities = available_activities
    
    # Filter activities that exist in our data
    filtered_activities = [act for act in preferred_activities if act in available_activities]
    
    if not filtered_activities:
        filtered_activities = available_activities[:5]
    
    # Get average metrics for each activity
    activity_stats = health_data[health_data['activity_type'].isin(filtered_activities)].groupby('activity_type').agg({
        'calories_burned': 'mean',
        'avg_heart_rate': 'mean',
        'duration_minutes': 'mean'
    }).reset_index()
    
    recommendations = []
    for _, activity in activity_stats.head(5).iterrows():
        recommendations.append({
            'activity': activity['activity_type'],
            'calories_burned': activity['calories_burned'],
            'heart_rate': activity['avg_heart_rate'],
            'duration': activity['duration_minutes'],
            'effectiveness': activity['calories_burned'] / activity['duration_minutes'] if activity['duration_minutes'] > 0 else 0
        })
    
    return recommendations

def assess_health_risk(user_data):
    """Assess health risk based on user data"""
    # Simple risk calculation
    risk_score = (
        (user_data.get('bmi', 24) - 22) ** 2 * 0.3 +
        (7 - user_data.get('sleep_hours', 7)) ** 2 * 0.2 +
        user_data.get('stress_level', 5) * 0.2 +
        (10000 - user_data.get('daily_steps', 8000)) * 0.0001 +
        (2.5 - user_data.get('hydration_level', 2.5)) ** 2 * 0.1
    )
    
    # Determine risk level
    if risk_score > 50:
        risk_level = "High"
    elif risk_score > 20:
        risk_level = "Medium"
    else:
        risk_level = "Low"
    
    recommendations = []
    if risk_level in ['High']:
        recommendations.extend([
            "Consider consulting a healthcare provider for a comprehensive health assessment",
            "Focus on stress management techniques like meditation or yoga",
            "Aim for 7-9 hours of quality sleep per night",
            "Increase daily physical activity gradually",
            "Monitor blood pressure regularly"
        ])
    elif risk_level == 'Medium':
        recommendations.extend([
            "Maintain current healthy habits",
            "Consider adding more physical activity to your routine",
            "Focus on stress reduction techniques",
            "Ensure adequate sleep and hydration"
        ])
    else:
        recommendations.extend([
            "Continue maintaining your healthy lifestyle",
            "Regular check-ups are still important",
            "Consider preventive health measures"
        ])
        
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'recommendations': recommendations
    }

def main():
    # Sidebar for user input
    st.sidebar.header("📋 Personal Information")
    with st.sidebar.form("user_info"):
        st.subheader("Basic Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["M", "F"])
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        st.subheader("Health Information")
        activity_level = st.selectbox("Activity Level", ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"])
        health_condition = st.selectbox("Health Condition", ["None", "Diabetes", "Hypertension", "Heart Disease", "Obesity", "Asthma", "Other"])
        fitness_goal = st.selectbox("Fitness Goal", ["Weight Loss", "Weight Gain", "Maintenance", "Muscle Building", "General Health"])
        stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
        sleep_hours = st.slider("Hours of Sleep", min_value=4, max_value=12, value=7)
        daily_steps = st.number_input("Daily Steps", min_value=0, max_value=30000, value=8000)
        hydration_level = st.number_input("Hydration Level (L)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
        resting_heart_rate = st.number_input("Resting Heart Rate", min_value=30, max_value=200, value=70)
        blood_pressure_systolic = st.number_input("Blood Pressure Systolic", min_value=80, max_value=200, value=120)
        blood_pressure_diastolic = st.number_input("Blood Pressure Diastolic", min_value=40, max_value=130, value=80)
        submitted = st.form_submit_button("Get Recommendations")

    if submitted:
        bmi = calculate_bmi(weight, height)
        bmi_category = get_bmi_category(bmi)
        user_data = {
            'age': age,
            'gender': gender,
            'height': height,
            'weight': weight,
            'bmi': bmi,
            'bmi_category': bmi_category,
            'activity_level': activity_level,
            'health_condition': health_condition,
            'fitness_goal': fitness_goal,
            'stress_level': stress_level,
            'sleep_hours': sleep_hours,
            'daily_steps': daily_steps,
            'hydration_level': hydration_level,
            'resting_heart_rate': resting_heart_rate,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
        }
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Health Dashboard", "🏃‍♂️ Exercise Recommendations", "🍎 Diet Recommendations", "⚠️ Health Risk Assessment"])
        
        with tab1:
            st.markdown('<h3 class="sub-header">Your Health Profile</h3>', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("BMI", f"{user_data['bmi']:.1f}", f"{user_data['bmi_category']}")
            with col2:
                st.metric("Age", f"{user_data['age']} years")
            with col3:
                st.metric("Weight", f"{user_data['weight']} kg")
            with col4:
                st.metric("Height", f"{user_data['height']} cm")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="thin-separator">', unsafe_allow_html=True)
                st.subheader("BMI Analysis")
                if bmi < 18.5:
                    st.warning("You are underweight. Consider increasing caloric intake and strength training.")
                elif bmi < 25:
                    st.success("Your BMI is in the healthy range. Maintain your current lifestyle.")
                elif bmi < 30:
                    st.warning("You are overweight. Consider diet and exercise modifications.")
                else:
                    st.error("You are in the obese category. Consult a healthcare provider for guidance.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="thin-separator">', unsafe_allow_html=True)
                st.subheader("Sleep Analysis")
                if sleep_hours < 6:
                    st.warning("Insufficient sleep. Aim for 7-9 hours for optimal health.")
                elif sleep_hours <= 9:
                    st.success("Good sleep duration. Keep it up!")
                else:
                    st.info("Adequate sleep duration.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=bmi,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "BMI"},
                gauge={
                    'axis': {'range': [None, 40]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 18.5], 'color': "lightgray"},
                        {'range': [18.5, 25], 'color': "lightgreen"},
                        {'range': [25, 30], 'color': "yellow"},
                        {'range': [30, 40], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown('<h3 class="sub-header">Exercise Recommendations</h3>', unsafe_allow_html=True)
            exercise_recs = get_exercise_recommendations(user_data)
            if exercise_recs:
                for i, rec in enumerate(exercise_recs[:5]):
                    if i > 0:
                        st.markdown('<hr class="thin-separator">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Activity", rec['activity'])
                    with col2:
                        st.metric("Calories Burned", f"{rec['calories_burned']:.1f} kcal/min")
            
            st.markdown('<div class="thin-separator">', unsafe_allow_html=True)
            st.subheader("Exercise Tips")
            if user_data['bmi'] < 18.5:
                st.write("• Focus on strength training to build muscle mass")
                st.write("• Include compound exercises like squats and deadlifts")
                st.write("• Aim for 3-4 strength training sessions per week")
            elif user_data['bmi'] > 30:
                st.write("• Start with low-impact cardio like walking or swimming")
                st.write("• Gradually increase intensity and duration")
                st.write("• Include strength training for muscle preservation")
            else:
                st.write("• Mix cardio and strength training")
                st.write("• Aim for 150 minutes of moderate activity per week")
                st.write("• Include flexibility and balance exercises")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<h3 class="sub-header">Diet Recommendations</h3>', unsafe_allow_html=True)
            diet_recs = get_diet_recommendations(user_data)
            st.markdown('<div class="thin-separator">', unsafe_allow_html=True)
            st.subheader("Recommended Foods")
            if diet_recs:
                for i, rec in enumerate(diet_recs[:8]):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**{rec['food'].title()}**")
                    with col2:
                        st.write(f"Calories: {rec['calories']:.0f}")
                    with col3:
                        st.write(f"Protein: {rec['protein']:.1f}g")
                    with col4:
                        st.write(f"Fiber: {rec['fiber']:.1f}g")
                    st.divider()
            
            st.markdown('<div class="thin-separator">', unsafe_allow_html=True)
            st.subheader("Nutrition Tips")
            if user_data['bmi'] < 18.5:
                st.write("• Increase caloric intake with nutrient-dense foods")
                st.write("• Include healthy fats like nuts and avocados")
                st.write("• Eat frequent meals throughout the day")
            elif user_data['bmi'] > 30:
                st.write("• Focus on high-fiber, low-calorie foods")
                st.write("• Increase protein intake for satiety")
                st.write("• Limit processed foods and added sugars")
            else:
                st.write("• Maintain a balanced diet with all food groups")
                st.write("• Include plenty of fruits and vegetables")
                st.write("• Stay hydrated throughout the day")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab4:
            st.markdown('<h3 class="sub-header">Health Risk Assessment</h3>', unsafe_allow_html=True)
            risk_result = assess_health_risk(user_data)
            st.metric("Risk Level", risk_result['risk_level'])
            st.metric("Risk Score", f"{risk_result['risk_score']:.2f}")
            
            st.subheader("Recommendations")
            for rec in risk_result['recommendations']:
                st.write(f"- {rec}")
    else:
        st.markdown("""
        <div style='text-align: center; padding: 2rem; background-color: #1B2631; border-radius: 1rem; margin: 2rem 0;'>
            <h3>Welcome to Precision Medicine!</h3>
            <p>Please fill out your personal information in the sidebar and click \"Get Recommendations\" to receive personalized diet and exercise recommendations based on your health profile.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Available Data")
            st.write(f"• Health records: {len(health_data)}")
            st.write(f"• Food items: {len(food_data)}")
            st.write(f"• Sleep records: {len(sleep_data)}")
        with col2:
            st.subheader("🎯 Features")
            st.write("• Personalized diet recommendations")
            st.write("• Exercise suggestions based on health profile")
            st.write("• BMI analysis and health insights")
            st.write("• Progress tracking capabilities")

if __name__ == "__main__":
    main()