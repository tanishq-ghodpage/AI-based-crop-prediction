import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import re
from flask import Flask, json, request, render_template, jsonify

# Configuration
MODEL_PATH = 'crop_yield_model.pkl'
FEATURE_NAMES_PATH = 'feature_names.pkl'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'crop_yield.csv')
STATE_DATA_PATH = 'static/data/state_data.json'

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')

# ------------------- Data Preparation & Model Training ------------------- #
def normalize_state_name(state_name):
    """Normalize state names to match GeoJSON format"""
    return re.sub(
        r'\s+', ' ',  # Replace multiple spaces with single space
        state_name.strip().title()  # Proper capitalization
    )

def validate_dataset(data_path):
    """Ensure dataset exists and has required columns"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    required_columns = {'Crop', 'Season', 'State', 'Area', 
                       'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Yield'}
    df = pd.read_csv(data_path)
    
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Dataset missing required columns: {missing}")
    
    return df

def train_model(data_path=DATA_PATH):
    """Enhanced model training with better error handling"""
    try:
        df = validate_dataset(data_path)
        print(f"Data loaded successfully: {df.shape[0]} records")
        
        # Normalize state names
        df['State'] = df['State'].apply(lambda x: normalize_state_name(x))
        
        # Feature engineering
        X = df[['Crop', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
        y = df['Yield']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Preprocessing pipeline
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Crop', 'Season', 'State']),
            ('num', StandardScaler(), ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])
        ])
        
        # Model pipeline
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=150,
                max_depth=12,
                min_samples_split=6,
                random_state=42,
                n_jobs=-1
            ))
        ])
        
        # Training
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model trained - MSE: {mse:.4f}, R²: {r2:.4f}")
        
        # Save artifacts
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
            
        # Extract feature names after encoding
        cat_features = model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
        num_features = X.select_dtypes(include=np.number).columns.tolist()
        feature_names = np.concatenate([cat_features, num_features]).tolist()
        
        with open(FEATURE_NAMES_PATH, 'wb') as f:
            pickle.dump(feature_names, f)
            
        return model, feature_names
        
    except Exception as e:
        print(f"Model training failed: {str(e)}")
        return None, None

def get_model():
    """Improved model loading with validation"""
    try:
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(FEATURE_NAMES_PATH, 'rb') as f:
                feature_names = pickle.load(f)
                
            # Basic model validation
            if not hasattr(model, 'predict') or not feature_names:
                raise ValueError("Invalid model or feature names")
                
            return model, feature_names
        return train_model()
    except Exception as e:
        print(f"Model loading error: {str(e)}")
        return train_model()

# ------------------- Data Initialization ------------------- #
def generate_state_data():
    """Generate normalized state data"""
    try:
        df = pd.read_csv(DATA_PATH)
        df['State'] = df['State'].apply(lambda x: normalize_state_name(x))
        
        state_data = {}
        for state in df['State'].unique():
            state_df = df[df['State'] == state]
            state_data[state] = {
                'avg_yield': round(state_df['Yield'].mean(), 2),
                'top_crop': state_df.groupby('Crop')['Yield'].mean().idxmax(),
                'avg_rainfall': round(state_df['Annual_Rainfall'].mean(), 2),
                'crop_count': state_df['Crop'].nunique()
            }
        
        with open(STATE_DATA_PATH, 'w') as f:
            json.dump(state_data, f)
            
        return True
    
    except Exception as e:
        print(f"State data generation failed: {str(e)}")
        return False

# ------------------- Flask Routes ------------------- #
@app.route('/')
def home():
    """Enhanced home route with proper error handling"""
    try:
        df = validate_dataset(DATA_PATH)
        
        return render_template('index.html',
            crops=sorted(df['Crop'].unique()),
            seasons=sorted(df['Season'].unique()),
            states=sorted(df['State'].unique())
        )
        
    except Exception as e:
        return render_template('index.html',
            error_message=f"Data initialization error: {str(e)}"
        ), 500

@app.route('/get_state_data/<state>')
def get_state_data(state):
    """Improved state data endpoint with normalization"""
    try:
        state = normalize_state_name(state)
        df = validate_dataset(DATA_PATH)
        state_df = df[df['State'] == state]
        
        if state_df.empty:
            return jsonify({
                'success': False,
                'message': f'No data found for state: {state}'
            }), 404
            
        return jsonify({
            'success': True,
            'state': state,
            'crops': sorted(state_df['Crop'].unique()),
            'seasons': sorted(state_df['Season'].unique()),
            'avg_rainfall': round(state_df['Annual_Rainfall'].mean(), 2),
            'avg_yield': round(state_df['Yield'].mean(), 2),
            'top_crop': state_df.groupby('Crop')['Yield'].mean().idxmax(),
            'historic_data': state_df.groupby('Crop_Year')['Yield'].mean().to_dict()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error processing request: {str(e)}'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Enhanced prediction endpoint with feature importance"""
    try:
        # Validate input data
        required_fields = ['crop', 'season', 'state', 'area', 
                          'rainfall', 'fertilizer', 'pesticide']
        data = {field: request.form.get(field) for field in required_fields}
        
        if any(not v for v in data.values()):
            raise ValueError("All form fields are required")
            
        # Convert numeric fields
        numeric_fields = ['area', 'rainfall', 'fertilizer', 'pesticide']
        for field in numeric_fields:
            data[field] = float(data[field])
            
        # Create input dataframe
        input_df = pd.DataFrame([{
            'Crop': data['crop'],
            'Season': data['season'],
            'State': normalize_state_name(data['state']),
            'Area': data['area'],
            'Annual_Rainfall': data['rainfall'],
            'Fertilizer': data['fertilizer'],
            'Pesticide': data['pesticide']
        }])
        
        # Get prediction
        model, feature_names = get_model()
        prediction = model.predict(input_df)[0]
        
        # Get feature importance
        # In the predict route, modify feature importance aggregation:
        importance = {}
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            importances = model.named_steps['regressor'].feature_importances_
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            
            # Improved feature name parsing
            for name, score in zip(feature_names, importances):
                if '_' in name:
                    base_feature = name.split('_', 1)[0]  # Only split on first underscore
                else:
                    base_feature = name
                    
                # Clean up feature names
                base_feature = base_feature.replace('num__', '').replace('cat__', '')
                importance[base_feature] = importance.get(base_feature, 0) + score

            # Normalize and filter
            total = sum(importance.values())
            importance = {k: round((v/total)*100, 1) 
                        for k, v in importance.items() 
                        if (v/total)*100 > 1}
            total = sum(importance.values())
            if total != 100:
                remainder = 100 - total
                if importance:
                    max_key = max(importance, key=importance.get)
                    importance[max_key] += remainder# Filter small values
                
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'feature_importance': importance,
            'message': f'Predicted yield: {prediction:.2f} tonnes/hectare'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Prediction error: {str(e)}'
        }), 400

# ------------------- Main Execution ------------------- #
if __name__ == '__main__':
    # Initialize data before starting the app
    print("Initializing application data...")
    try:
        generate_state_data()
        model, feature_names = get_model()
    except Exception as e:
        print(f"Initialization error: {e}")
    
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)