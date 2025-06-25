import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class PrecisionMedicineModelBuilder:
    def __init__(self):
        self.food_data = None
        self.health_data = None
        self.diet_model = None
        self.exercise_model = None
        self.health_risk_model = None
        self.diet_scaler = StandardScaler()
        self.exercise_scaler = StandardScaler()
        self.health_risk_scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.label_encoders = {}
        
    def load_data(self):
        """Load and preprocess the datasets"""
        print("Loading datasets...")
        
        # Load food data
        self.food_data = pd.read_csv('data/FOOD-DATA-GROUP1.csv')
        print(f"Food data loaded: {len(self.food_data)} records")
        
        # Load health data
        self.health_data = pd.read_csv('data/health_fitness_dataset.csv')
        print(f"Health data loaded: {len(self.health_data)} records")
        
        # Clean and preprocess data
        self._preprocess_food_data()
        self._preprocess_health_data()
        
    def _preprocess_food_data(self):
        """Preprocess food data"""
        # Remove rows with missing values in key columns
        key_columns = ['Caloric Value', 'Protein', 'Carbohydrates', 'Fat', 'Dietary Fiber']
        self.food_data = self.food_data.dropna(subset=key_columns)
        
        # Fill remaining missing values with 0 for nutritional values
        nutritional_columns = ['Vitamin A', 'Vitamin C', 'Calcium', 'Iron', 'Nutrition Density']
        self.food_data[nutritional_columns] = self.food_data[nutritional_columns].fillna(0)
        
        # Create nutritional categories
        self.food_data['calorie_category'] = pd.cut(
            self.food_data['Caloric Value'], 
            bins=[0, 100, 200, 300, 500, 1000], 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        self.food_data['protein_category'] = pd.cut(
            self.food_data['Protein'], 
            bins=[0, 5, 10, 15, 25, 50], 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        self.food_data['fiber_category'] = pd.cut(
            self.food_data['Dietary Fiber'], 
            bins=[0, 1, 3, 5, 10, 20], 
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Create nutrition score
        self.food_data['nutrition_score'] = (
            self.food_data['Protein'] * 0.3 +
            self.food_data['Dietary Fiber'] * 0.2 +
            (self.food_data['Vitamin A'] + self.food_data['Vitamin C']) * 0.1 +
            (self.food_data['Calcium'] + self.food_data['Iron']) * 0.1 +
            (1000 - self.food_data['Caloric Value']) * 0.0003  # Lower calories = higher score
        )
        
    def _preprocess_health_data(self):
        """Preprocess health data"""
        # Remove rows with missing values in key columns
        key_columns = ['age', 'gender', 'height_cm', 'weight_kg', 'bmi', 'activity_type']
        self.health_data = self.health_data.dropna(subset=key_columns)
        
        # Fill missing values for other columns
        numeric_columns = ['hours_sleep', 'stress_level', 'daily_steps', 'hydration_level', 
                          'resting_heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic']
        self.health_data[numeric_columns] = self.health_data[numeric_columns].fillna(
            self.health_data[numeric_columns].mean()
        )
        
        # Create BMI categories
        self.health_data['bmi_category'] = pd.cut(
            self.health_data['bmi'], 
            bins=[0, 18.5, 25, 30, 50], 
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        
        # Create activity effectiveness score
        self.health_data['activity_effectiveness'] = (
            self.health_data['calories_burned'] * 0.4 +
            (220 - self.health_data['avg_heart_rate']) * 0.3 +  # Lower HR = better
            self.health_data['duration_minutes'] * 0.3
        )
        
        # Create health risk score
        self.health_data['health_risk_score'] = (
            (self.health_data['bmi'] - 22) ** 2 * 0.3 +  # BMI deviation from ideal
            (10 - self.health_data['hours_sleep']) ** 2 * 0.2 +  # Sleep deviation
            self.health_data['stress_level'] * 0.2 +  # Stress level
            (10000 - self.health_data['daily_steps']) * 0.0001 +  # Step count deviation
            (2.5 - self.health_data['hydration_level']) ** 2 * 0.1  # Hydration deviation
        )
        
    def build_diet_model(self):
        """Build diet recommendation model"""
        print("Building diet recommendation model...")
        
        # Prepare features for diet model
        diet_features = [
            'Caloric Value', 'Fat', 'Carbohydrates', 'Protein', 'Dietary Fiber',
            'Vitamin A', 'Vitamin C', 'Calcium', 'Iron', 'Nutrition Density'
        ]
        
        # Create target variable (nutrition score)
        X = self.food_data[diet_features]
        y = self.food_data['nutrition_score']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.diet_scaler.fit_transform(X_train)
        X_test_scaled = self.diet_scaler.transform(X_test)
        
        # Train model
        self.diet_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.diet_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.diet_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Diet model MSE: {mse:.4f}")
        
        return self.diet_model
        
    def build_exercise_model(self):
        """Build exercise recommendation model"""
        print("Building exercise recommendation model...")
        exercise_features = [
            'age', 'gender', 'height_cm', 'weight_kg', 'bmi', 'hours_sleep',
            'stress_level', 'daily_steps', 'hydration_level', 'resting_heart_rate'
        ]
        X = self.health_data[exercise_features]
        y = self.health_data['activity_effectiveness']
        # Sample a subset for faster training
        if len(X) > 20000:
            X = X.sample(n=20000, random_state=42)
            y = y.loc[X.index]
        le_gender = LabelEncoder()
        X['gender'] = le_gender.fit_transform(X['gender'])
        self.label_encoders['gender'] = le_gender
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.exercise_scaler.fit_transform(X_train)
        X_test_scaled = self.exercise_scaler.transform(X_test)
        self.exercise_model = RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)
        self.exercise_model.fit(X_train_scaled, y_train)
        y_pred = self.exercise_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Exercise model MSE: {mse:.4f}")
        return self.exercise_model
        
    def build_health_risk_model(self):
        """Build health risk assessment model"""
        print("Building health risk assessment model...")
        
        # Prepare features for health risk model
        health_features = [
            'age', 'gender', 'height_cm', 'weight_kg', 'bmi', 'hours_sleep',
            'stress_level', 'daily_steps', 'hydration_level', 'resting_heart_rate',
            'blood_pressure_systolic', 'blood_pressure_diastolic'
        ]
        
        # Create target variable (health risk category)
        risk_categories = pd.cut(
            self.health_data['health_risk_score'],
            bins=[0, 10, 20, 30, 100],
            labels=['Low', 'Medium', 'High', 'Very High']
        )
        
        X = self.health_data[health_features]
        y = risk_categories
        
        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        # Encode categorical variables
        le_gender = LabelEncoder()
        X['gender'] = le_gender.fit_transform(X['gender'])
        self.label_encoders['gender_risk'] = le_gender
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.health_risk_scaler.fit_transform(X_train)
        X_test_scaled = self.health_risk_scaler.transform(X_test)
        
        # Train model
        self.health_risk_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.health_risk_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.health_risk_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Health risk model accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return self.health_risk_model
        
    def get_diet_recommendations(self, user_profile):
        """Get personalized diet recommendations"""
        # Prepare user features
        user_features = np.array([
            user_profile.get('calories_needed', 2000),
            user_profile.get('fat_preference', 50),
            user_profile.get('carbs_preference', 200),
            user_profile.get('protein_needed', 80),
            user_profile.get('fiber_needed', 25),
            user_profile.get('vitamin_a_needed', 100),
            user_profile.get('vitamin_c_needed', 100),
            user_profile.get('calcium_needed', 1000),
            user_profile.get('iron_needed', 18),
            user_profile.get('nutrition_density_preference', 50)
        ]).reshape(1, -1)
        
        # Scale features
        user_features_scaled = self.diet_scaler.transform(user_features)
        
        # Predict nutrition score
        predicted_score = self.diet_model.predict(user_features_scaled)[0]
        
        # Filter foods based on user profile
        if user_profile.get('bmi_category') == 'Underweight':
            filtered_foods = self.food_data[
                (self.food_data['Caloric Value'] > 200) &
                (self.food_data['Protein'] > 15) &
                (self.food_data['Fat'] > 10)
            ].sort_values('nutrition_score', ascending=False)
        elif user_profile.get('bmi_category') == 'Obese':
            filtered_foods = self.food_data[
                (self.food_data['Caloric Value'] < 150) &
                (self.food_data['Dietary Fiber'] > 3) &
                (self.food_data['Fat'] < 10)
            ].sort_values('nutrition_score', ascending=False)
        else:
            filtered_foods = self.food_data[
                (self.food_data['Caloric Value'].between(100, 300)) &
                (self.food_data['Protein'] > 10) &
                (self.food_data['Dietary Fiber'] > 2)
            ].sort_values('nutrition_score', ascending=False)
        
        # Return top recommendations
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
        
    def get_exercise_recommendations(self, user_profile):
        """Get personalized exercise recommendations"""
        # Prepare user features
        user_features = np.array([
            user_profile.get('age', 30),
            self.label_encoders['gender'].transform([user_profile.get('gender', 'M')])[0],
            user_profile.get('height', 170),
            user_profile.get('weight', 70),
            user_profile.get('bmi', 24),
            user_profile.get('sleep_hours', 7),
            user_profile.get('stress_level', 5),
            user_profile.get('daily_steps', 8000),
            user_profile.get('hydration_level', 2.5),
            user_profile.get('resting_heart_rate', 70)
        ]).reshape(1, -1)
        
        # Scale features
        user_features_scaled = self.exercise_scaler.transform(user_features)
        
        # Predict activity effectiveness
        predicted_effectiveness = self.exercise_model.predict(user_features_scaled)[0]
        
        # Filter activities based on user profile
        if user_profile.get('bmi_category') == 'Underweight':
            filtered_activities = self.health_data[
                (self.health_data['activity_type'].isin(['Weight Training', 'Strength Training', 'Yoga']))
            ].groupby('activity_type').agg({
                'calories_burned': 'mean',
                'avg_heart_rate': 'mean',
                'duration_minutes': 'mean',
                'activity_effectiveness': 'mean'
            }).sort_values('activity_effectiveness', ascending=False)
        elif user_profile.get('bmi_category') == 'Obese':
            filtered_activities = self.health_data[
                (self.health_data['activity_type'].isin(['Walking', 'Swimming', 'Cycling']))
            ].groupby('activity_type').agg({
                'calories_burned': 'mean',
                'avg_heart_rate': 'mean',
                'duration_minutes': 'mean',
                'activity_effectiveness': 'mean'
            }).sort_values('activity_effectiveness', ascending=False)
        else:
            filtered_activities = self.health_data.groupby('activity_type').agg({
                'calories_burned': 'mean',
                'avg_heart_rate': 'mean',
                'duration_minutes': 'mean',
                'activity_effectiveness': 'mean'
            }).sort_values('activity_effectiveness', ascending=False)
        
        # Return top recommendations
        print('DEBUG: filtered_activities columns:', filtered_activities.columns)
        # Handle possible MultiIndex columns (if any)
        effectiveness_col = 'activity_effectiveness'
        if effectiveness_col not in filtered_activities.columns:
            # Try to find the correct column if it's a MultiIndex
            for col in filtered_activities.columns:
                if isinstance(col, tuple) and 'activity_effectiveness' in col:
                    effectiveness_col = col
                    break
        recommendations = []
        for activity, data in filtered_activities.head(5).iterrows():
            recommendations.append({
                'activity': activity,
                'calories_burned': data['calories_burned'],
                'heart_rate': data['avg_heart_rate'],
                'duration': data['duration_minutes'],
                'effectiveness': data[effectiveness_col]
            })
        
        return recommendations
        
    def assess_health_risk(self, user_profile):
        """Assess health risk and provide recommendations"""
        # Prepare user features
        user_features = np.array([
            user_profile.get('age', 30),
            self.label_encoders['gender_risk'].transform([user_profile.get('gender', 'M')])[0],
            user_profile.get('height', 170),
            user_profile.get('weight', 70),
            user_profile.get('bmi', 24),
            user_profile.get('sleep_hours', 7),
            user_profile.get('stress_level', 5),
            user_profile.get('daily_steps', 8000),
            user_profile.get('hydration_level', 2.5),
            user_profile.get('resting_heart_rate', 70),
            user_profile.get('blood_pressure_systolic', 120),
            user_profile.get('blood_pressure_diastolic', 80)
        ]).reshape(1, -1)
        
        # Scale features
        user_features_scaled = self.health_risk_scaler.transform(user_features)
        
        # Predict health risk
        risk_prediction = self.health_risk_model.predict(user_features_scaled)[0]
        risk_probabilities = self.health_risk_model.predict_proba(user_features_scaled)[0]
        
        # Calculate risk score
        risk_score = (
            (user_profile.get('bmi', 24) - 22) ** 2 * 0.3 +
            (7 - user_profile.get('sleep_hours', 7)) ** 2 * 0.2 +
            user_profile.get('stress_level', 5) * 0.2 +
            (10000 - user_profile.get('daily_steps', 8000)) * 0.0001 +
            (2.5 - user_profile.get('hydration_level', 2.5)) ** 2 * 0.1
        )
        
        # Generate recommendations based on risk level
        recommendations = []
        if risk_prediction in ['High', 'Very High']:
            recommendations.extend([
                "Consider consulting a healthcare provider for a comprehensive health assessment",
                "Focus on stress management techniques like meditation or yoga",
                "Aim for 7-9 hours of quality sleep per night",
                "Increase daily physical activity gradually",
                "Monitor blood pressure regularly"
            ])
        elif risk_prediction == 'Medium':
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
            'risk_level': risk_prediction,
            'risk_score': risk_score,
            'risk_probabilities': dict(zip(self.health_risk_model.classes_, risk_probabilities)),
            'recommendations': recommendations
        }
        
    def save_models(self):
        """Save all models to pickle files"""
        print("Saving models to pickle files...")
        
        models = {
            'diet_model': self.diet_model,
            'exercise_model': self.exercise_model,
            'health_risk_model': self.health_risk_model,
            'diet_scaler': self.diet_scaler,
            'exercise_scaler': self.exercise_scaler,
            'health_risk_scaler': self.health_risk_scaler,
            'label_encoders': self.label_encoders,
            'food_data': self.food_data,
            'health_data': self.health_data
        }
        
        with open('models/precision_medicine_models.pkl', 'wb') as f:
            pickle.dump(models, f)
        
        print("Models saved successfully!")
        
    def load_models(self):
        """Load models from pickle files"""
        print("Loading models from pickle files...")
        
        with open('models/precision_medicine_models.pkl', 'rb') as f:
            models = pickle.load(f)
        
        self.diet_model = models['diet_model']
        self.exercise_model = models['exercise_model']
        self.health_risk_model = models['health_risk_model']
        self.diet_scaler = models['diet_scaler']
        self.exercise_scaler = models['exercise_scaler']
        self.health_risk_scaler = models['health_risk_scaler']
        self.label_encoders = models['label_encoders']
        self.food_data = models['food_data']
        self.health_data = models['health_data']
        
        print("Models loaded successfully!")

def main():
    """Main function to build and save models"""
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize model builder
    builder = PrecisionMedicineModelBuilder()
    
    # Load data
    builder.load_data()
    
    # Build models
    builder.build_diet_model()
    builder.build_exercise_model()
    builder.build_health_risk_model()
    
    # Save models
    builder.save_models()
    
    print("All models built and saved successfully!")

if __name__ == "__main__":
    main() 