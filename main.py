import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")

# Initialize FastAPI app
app = FastAPI(title="House Price Prediction API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and scalers
models = {}
scaler = None
label_encoder = None
feature_columns = None
model_metrics = {}

# Pydantic models
class HouseFeatures(BaseModel):
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: int
    view: int
    condition: int
    sqft_above: float
    sqft_basement: float
    yr_built: int
    yr_renovated: int
    city: str

class PredictionResponse(BaseModel):
    ensemble_prediction: float
    individual_predictions: Dict[str, float]
    confidence: str

class MetricsResponse(BaseModel):
    model_metrics: Dict[str, Dict[str, float]]
    error_distributions: Dict[str, List[float]]

def load_and_preprocess_data():
    """Load and preprocess the housing data."""
    print("Loading data...")
    df = pd.read_csv("data/data.csv")
    
    # Initialize label encoder for city
    global label_encoder
    label_encoder = LabelEncoder()
    df['city_encoded'] = label_encoder.fit_transform(df['city'])
    
    # Select numeric columns and city_encoded
    req_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                   'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
                   'yr_built', 'yr_renovated', 'city_encoded', 'price']
    
    df_clean = df[req_columns].copy()
    
    # Create derived features
    df_clean['Total_Rooms'] = df_clean['bedrooms'] + df_clean['bathrooms']
    df_clean['Built_sqft'] = df_clean['sqft_living'] + df_clean['sqft_above'] + df_clean['sqft_basement']
    df_clean['Age'] = 2026 - df_clean['yr_built']
    df_clean['sqft_per_room'] = df_clean['Built_sqft'] / (df_clean['Total_Rooms'] + 0.1)  # Avoid division by zero
    df_clean['renovated'] = (df_clean['yr_renovated'] > 0).astype(int)
    
    # Handle any remaining NaN values
    df_clean = df_clean.fillna(0)
    
    print(f"Data loaded: {len(df_clean)} records")
    return df_clean

def train_models():
    """Train all models and save them."""
    global models, scaler, feature_columns, model_metrics
    
    print("Training models...")
    df = load_and_preprocess_data()
    
    # Prepare features and target
    X = df.drop('price', axis=1)
    y = df['price']
    feature_columns = X.columns.tolist()
    
    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Define models
    model_configs = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        'XGBoost': XGBRegressor(n_estimators=100, random_state=42, max_depth=6),
        'KNN': KNeighborsRegressor(n_neighbors=5)
    }
    
    # Train each model
    error_distributions = {}
    
    for name, model in model_configs.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Store model and metrics
        models[name] = model
        model_metrics[name] = {
            'train_mse': float(train_mse),
            'test_mse': float(test_mse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'train_rmse': float(np.sqrt(train_mse)),
            'test_rmse': float(np.sqrt(test_mse))
        }
        
        # Calculate error distribution (residuals)
        errors = y_test - y_pred_test
        error_distributions[name] = errors.tolist()[:100]  # Limit to 100 samples for visualization
        
        print(f"{name} - Test R2: {test_r2:.4f}, Test RMSE: {np.sqrt(test_mse):.2f}")
    
    # Save models and scalers
    os.makedirs('models', exist_ok=True)
    joblib.dump(models, 'models/trained_models.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(feature_columns, 'models/feature_columns.pkl')
    joblib.dump(model_metrics, 'models/model_metrics.pkl')
    joblib.dump(error_distributions, 'models/error_distributions.pkl')
    
    print("Models trained and saved successfully!")

def load_models():
    """Load pre-trained models."""
    global models, scaler, label_encoder, feature_columns, model_metrics
    
    try:
        models = joblib.load('models/trained_models.pkl')
        scaler = joblib.load('models/scaler.pkl')
        label_encoder = joblib.load('models/label_encoder.pkl')
        feature_columns = joblib.load('models/feature_columns.pkl')
        model_metrics = joblib.load('models/model_metrics.pkl')
        print("Models loaded successfully!")
        return True
    except FileNotFoundError:
        print("No pre-trained models found. Training new models...")
        return False

def preprocess_input(house_data: HouseFeatures) -> np.ndarray:
    """Preprocess input data for prediction."""
    # Encode city
    try:
        city_encoded = label_encoder.transform([house_data.city])[0]
    except ValueError:
        # If city not in training data, use most common city encoding
        city_encoded = 0
    
    # Create feature dictionary
    features = {
        'bedrooms': house_data.bedrooms,
        'bathrooms': house_data.bathrooms,
        'sqft_living': house_data.sqft_living,
        'sqft_lot': house_data.sqft_lot,
        'floors': house_data.floors,
        'waterfront': house_data.waterfront,
        'view': house_data.view,
        'condition': house_data.condition,
        'sqft_above': house_data.sqft_above,
        'sqft_basement': house_data.sqft_basement,
        'yr_built': house_data.yr_built,
        'yr_renovated': house_data.yr_renovated,
        'city_encoded': city_encoded,
        'Total_Rooms': house_data.bedrooms + house_data.bathrooms,
        'Built_sqft': house_data.sqft_living + house_data.sqft_above + house_data.sqft_basement,
        'Age': 2026 - house_data.yr_built,
        'sqft_per_room': (house_data.sqft_living + house_data.sqft_above + house_data.sqft_basement) / 
                         (house_data.bedrooms + house_data.bathrooms + 0.1),
        'renovated': 1 if house_data.yr_renovated > 0 else 0
    }
    
    # Create DataFrame with correct column order
    df = pd.DataFrame([features])
    df = df[feature_columns]
    
    # Scale features
    scaled_features = scaler.transform(df)
    
    return scaled_features

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load or train models on startup."""
    if not load_models():
        train_models()

@app.get("/")
async def read_root():
    """Serve the frontend."""
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "models_loaded": len(models) > 0}

@app.post("/predict", response_model=PredictionResponse)
async def predict_price(house: HouseFeatures):
    """Predict house price using ensemble of models."""
    try:
        # Preprocess input
        features = preprocess_input(house)
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            pred = model.predict(features)[0]
            predictions[name] = float(pred)
        
        # Calculate ensemble prediction (average)
        ensemble_pred = np.mean(list(predictions.values()))
        
        # Calculate confidence based on prediction variance
        pred_std = np.std(list(predictions.values()))
        if pred_std < 50000:
            confidence = "High"
        elif pred_std < 100000:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return PredictionResponse(
            ensemble_prediction=float(ensemble_pred),
            individual_predictions=predictions,
            confidence=confidence
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get model performance metrics and error distributions."""
    try:
        error_distributions = joblib.load('models/error_distributions.pkl')
        
        return MetricsResponse(
            model_metrics=model_metrics,
            error_distributions=error_distributions
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)