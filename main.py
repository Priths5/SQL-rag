from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, LargeBinary, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError
import pandas as pd
import numpy as np
import io
import os
import joblib
import time
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import logging
from pydantic import BaseModel

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration with retry logic
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "password")
MYSQL_HOST = os.getenv("MYSQL_HOST", "mysql")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "maintenance_data")
DB_RETRY_ATTEMPTS = int(os.getenv("DB_RETRY_ATTEMPTS", "10"))
DB_RETRY_DELAY = int(os.getenv("DB_RETRY_DELAY", "5"))

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"

def create_engine_with_retry():
    """Create database engine with retry logic"""
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            logger.info(f"Attempting to connect to database (attempt {attempt + 1}/{DB_RETRY_ATTEMPTS})")
            engine = create_engine(
                DATABASE_URL, 
                echo=True,
                pool_pre_ping=True,  # Enables automatic reconnection
                pool_recycle=3600,   # Recycle connections every hour
                connect_args={
                    "connect_timeout": 60,
                    "read_timeout": 60,
                    "write_timeout": 60,
                }
            )
            
            # Test the connection
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            
            logger.info("Successfully connected to database")
            return engine
            
        except OperationalError as e:
            logger.warning(f"Database connection attempt {attempt + 1} failed: {e}")
            if attempt < DB_RETRY_ATTEMPTS - 1:
                logger.info(f"Retrying in {DB_RETRY_DELAY} seconds...")
                time.sleep(DB_RETRY_DELAY)
            else:
                logger.error("All database connection attempts failed")
                raise
        except Exception as e:
            logger.error(f"Unexpected error connecting to database: {e}")
            raise

def create_tables_with_retry():
    """Create database tables with retry logic"""
    for attempt in range(DB_RETRY_ATTEMPTS):
        try:
            logger.info(f"Attempting to create tables (attempt {attempt + 1}/{DB_RETRY_ATTEMPTS})")
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
            return True
            
        except OperationalError as e:
            logger.warning(f"Table creation attempt {attempt + 1} failed: {e}")
            if attempt < DB_RETRY_ATTEMPTS - 1:
                logger.info(f"Retrying in {DB_RETRY_DELAY} seconds...")
                time.sleep(DB_RETRY_DELAY)
            else:
                logger.error("All table creation attempts failed")
                return False
        except Exception as e:
            logger.error(f"Unexpected error creating tables: {e}")
            return False

# SQLAlchemy setup
Base = declarative_base()

# Initialize database connection with retry
try:
    engine = create_engine_with_retry()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    sys.exit(1)

# Pydantic models for API requests/responses
class TrainingRequest(BaseModel):
    model_type: str
    test_size: float = 0.2
    hyperparameters: Dict[str, Any] = {}
    model_name: Optional[str] = None
    description: Optional[str] = None

class PredictionRequest(BaseModel):
    model_id: int
    data: List[Dict[str, float]]

class ModelResponse(BaseModel):
    id: int
    name: str
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    created_at: str
    description: Optional[str] = None

# Database Models
class SensorData(Base):
    __tablename__ = "sensor_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False)
    temperature = Column(Float, nullable=False)  # Temperature (°C)
    vibration = Column(Float, nullable=False)    # Vibration (mm/s)
    pressure = Column(Float, nullable=False)     # Pressure (Pa)
    rpm = Column(Float, nullable=False)          # RPM
    maintenance_required = Column(Integer, nullable=False)  # 0 or 1
    temp_change = Column(Float, nullable=False)  # Temperature change
    vib_change = Column(Float, nullable=False)   # Vibration change
    upload_datetime = Column(DateTime, default=datetime.utcnow)

class UploadLog(Base):
    __tablename__ = "upload_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    upload_datetime = Column(DateTime, default=datetime.utcnow)
    records_count = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False)
    error_message = Column(Text, nullable=True)

class MLModel(Base):
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=False)
    model_data = Column(LargeBinary(length=16777216), nullable=False)  # 16MB LONGBLOB
    scaler_data = Column(LargeBinary(length=16777216), nullable=False)  # 16MB LONGBLOB
    feature_columns = Column(Text, nullable=False)  # JSON string of feature column names
    accuracy = Column(Float, nullable=False)
    precision = Column(Float, nullable=False)
    recall = Column(Float, nullable=False)
    f1_score = Column(Float, nullable=False)
    hyperparameters = Column(Text, nullable=False)  # JSON string
    training_data_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)

class PredictionResult(Base):
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    input_data = Column(Text, nullable=False)  # JSON string of input features
    predicted_maintenance_required = Column(Integer, nullable=False)  # 0 or 1
    prediction_probability = Column(Text, nullable=True)  # JSON string of class probabilities
    created_at = Column(DateTime, default=datetime.utcnow)

class TrainingHistory(Base):
    __tablename__ = "training_history"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, nullable=False)
    training_started = Column(DateTime, nullable=False)
    training_completed = Column(DateTime, nullable=False)
    training_duration_seconds = Column(Float, nullable=False)
    dataset_size = Column(Integer, nullable=False)
    test_size = Column(Float, nullable=False)
    confusion_matrix = Column(Text, nullable=False)  # JSON string
    classification_report = Column(Text, nullable=False)
    feature_importance = Column(Text, nullable=True)  # JSON string
    status = Column(String(50), nullable=False)
    error_message = Column(Text, nullable=True)

# Create tables function
def create_tables():
    """Create tables using retry logic"""
    return create_tables_with_retry()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ML Trainer Class
class MaintenancePredictionTrainer:
    def __init__(self):
        self.models_config = {
            'random_forest': {
                'name': 'Random Forest',
                'class': RandomForestClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                }
            },
            'svm': {
                'name': 'Support Vector Machine',
                'class': SVC,
                'default_params': {
                    'C': 1.0,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                    'probability': True,  # Enable probability predictions
                    'random_state': 42
                }
            },
            'gradient_boosting': {
                'name': 'Gradient Boosting',
                'class': GradientBoostingClassifier,
                'default_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'name': 'Logistic Regression',
                'class': LogisticRegression,
                'default_params': {
                    'C': 1.0,
                    'solver': 'liblinear',
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
            'neural_network': {
                'name': 'Neural Network (MLP)',
                'class': MLPClassifier,
                'default_params': {
                    'hidden_layer_sizes': (100,),
                    'activation': 'relu',
                    'solver': 'adam',
                    'alpha': 0.0001,
                    'learning_rate': 'constant',
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }

    def load_data_from_db(self, db: Session):
        """Load sensor data from database"""
        try:
            # Query all sensor data
            sensor_data = db.query(SensorData).all()
            
            if not sensor_data:
                raise ValueError("No sensor data found in database")
            
            # Convert to DataFrame
            data_list = []
            for record in sensor_data:
                data_list.append({
                    'temperature': record.temperature,
                    'vibration': record.vibration,
                    'pressure': record.pressure,
                    'rpm': record.rpm,
                    'temp_change': record.temp_change,
                    'vib_change': record.vib_change,
                    'maintenance_required': record.maintenance_required
                })
            
            df = pd.DataFrame(data_list)
            logger.info(f"Loaded {len(df)} records from database")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from database: {e}")
            raise

    def preprocess_data(self, df):
        """Preprocess the data for training"""
        # Separate features and target
        feature_columns = ['temperature', 'vibration', 'pressure', 'rpm', 'temp_change', 'vib_change']
        X = df[feature_columns]
        y = df['maintenance_required']
        
        # Check for missing values
        if X.isnull().any().any():
            X = X.fillna(X.mean())
        
        return X, y, feature_columns

    def train_model(self, db: Session, model_type: str, hyperparameters: Dict, 
                   test_size: float = 0.2, model_name: str = None, description: str = None):
        """Train a maintenance prediction model"""
        
        if model_type not in self.models_config:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        training_start = datetime.utcnow()
        
        try:
            # Load data
            df = self.load_data_from_db(db)
            X, y, feature_columns = self.preprocess_data(df)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Prepare model parameters
            model_config = self.models_config[model_type]
            params = model_config['default_params'].copy()
            params.update(hyperparameters)
            
            # Train model
            model = model_config['class'](**params)
            model.fit(X_train_scaled, y_train)
            
            training_end = datetime.utcnow()
            training_duration = (training_end - training_start).total_seconds()
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Generate reports
            cm = confusion_matrix(y_test, y_pred)
            class_report = classification_report(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(feature_columns, model.feature_importances_.tolist()))
            
            # Serialize model and scaler using BytesIO
            model_buffer = io.BytesIO()
            scaler_buffer = io.BytesIO()
            joblib.dump(model, model_buffer)
            joblib.dump(scaler, scaler_buffer)
            model_data = model_buffer.getvalue()
            scaler_data = scaler_buffer.getvalue()
            
            # Generate model name if not provided
            if not model_name:
                model_name = f"{model_config['name']}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Save to database
            ml_model = MLModel(
                name=model_name,
                model_type=model_type,
                model_data=model_data,
                scaler_data=scaler_data,
                feature_columns=str(feature_columns),
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                hyperparameters=str(params),
                training_data_count=len(df),
                description=description
            )
            
            db.add(ml_model)
            db.flush()  # Get the ID
            
            # Save training history
            training_history = TrainingHistory(
                model_id=ml_model.id,
                training_started=training_start,
                training_completed=training_end,
                training_duration_seconds=training_duration,
                dataset_size=len(df),
                test_size=test_size,
                confusion_matrix=str(cm.tolist()),
                classification_report=class_report,
                feature_importance=str(feature_importance) if feature_importance else None,
                status="completed"
            )
            
            db.add(training_history)
            db.commit()
            
            return {
                'model_id': ml_model.id,
                'model_name': model_name,
                'model_type': model_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'training_duration': training_duration,
                'confusion_matrix': cm.tolist(),
                'classification_report': class_report,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            # Rollback the session first
            db.rollback()
            
            # Log error in training history
            training_history = TrainingHistory(
                model_id=0,  # Will be updated if model was created
                training_started=training_start,
                training_completed=datetime.utcnow(),
                training_duration_seconds=0,
                dataset_size=0,
                test_size=test_size,
                confusion_matrix="[]",
                classification_report="",
                status="failed",
                error_message=str(e)
            )
            db.add(training_history)
            db.commit()
            raise

    def predict(self, db: Session, model_id: int, input_data: List[Dict]):
        """Make predictions using a trained model"""
        try:
            # Load model from database
            ml_model = db.query(MLModel).filter(MLModel.id == model_id).first()
            if not ml_model:
                raise ValueError(f"Model with ID {model_id} not found")
            
            # Deserialize model and scaler using BytesIO
            model = joblib.load(io.BytesIO(ml_model.model_data))
            scaler = joblib.load(io.BytesIO(ml_model.scaler_data))
            
            # Get feature columns
            feature_columns = eval(ml_model.feature_columns)
            
            # Prepare input data
            df_input = pd.DataFrame(input_data)
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in df_input.columns:
                    raise ValueError(f"Missing feature: {col}")
            
            # Select and order features correctly
            X_input = df_input[feature_columns]
            
            # Scale input data
            X_input_scaled = scaler.transform(X_input)
            
            # Make predictions
            predictions = model.predict(X_input_scaled)
            
            # Get prediction probabilities if available
            probabilities = None
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_input_scaled)
            
            # Save prediction results
            results = []
            for i, pred in enumerate(predictions):
                prob_dict = None
                if probabilities is not None:
                    # Get unique classes and their probabilities
                    classes = model.classes_
                    prob_dict = {int(cls): float(prob) for cls, prob in zip(classes, probabilities[i])}
                
                prediction_result = PredictionResult(
                    model_id=model_id,
                    input_data=str(input_data[i]),
                    predicted_maintenance_required=int(pred),
                    prediction_probability=str(prob_dict) if prob_dict else None
                )
                db.add(prediction_result)
                
                results.append({
                    'input': input_data[i],
                    'predicted_maintenance_required': int(pred),
                    'maintenance_needed': bool(pred),  # Convert to boolean for clarity
                    'probabilities': prob_dict
                })
            
            db.commit()
            return results
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

# FastAPI app
app = FastAPI(
    title="Industrial Maintenance Prediction API",
    description="API for uploading sensor data, training ML models, and predicting maintenance requirements",
    version="2.0.0"
)

trainer = MaintenancePredictionTrainer()

@app.on_event("startup")
async def startup_event():
    """Startup event with better error handling"""
    logger.info("Starting application...")
    
    # Try to create tables
    success = create_tables()
    if not success:
        logger.error("Failed to create database tables. Application may not work properly.")
    
    logger.info("Application startup completed")

@app.get("/")
async def root():
    return {"message": "Industrial Maintenance Prediction API is running"}

@app.get("/health")
async def health_check():
    """Enhanced health check with database status"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        # Check table existence
        table_count = 0
        try:
            db = SessionLocal()
            tables_result = db.execute(text("SHOW TABLES")).fetchall()
            table_count = len(tables_result)
            db.close()
        except:
            pass
        
        return {
            "status": "healthy",
            "database": "connected",
            "tables_created": table_count > 0,
            "table_count": table_count,
            "database_url_host": MYSQL_HOST,
            "database_name": MYSQL_DATABASE
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "database_url_host": MYSQL_HOST,
            "database_name": MYSQL_DATABASE
        }

@app.get("/health/db")
async def database_health_check():
    """Check database connectivity"""
    try:
        db = SessionLocal()
        result = db.execute(text("SELECT 1 as health_check")).fetchone()
        db.close()
        
        return {
            "status": "healthy",
            "database": "connected",
            "test_query_result": result[0] if result else None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e)
        }

# CSV upload endpoint for new data format
@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    upload_log = UploadLog(
        filename=file.filename,
        records_count=0,
        status="processing"
    )
    
    try:
        content = await file.read()
        csv_data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        
        required_columns = [
            'Timestamp', 'Temperature (°C)', 'Vibration (mm/s)', 
            'Pressure (Pa)', 'RPM', 'Maintenance Required', 
            'Temp_Change', 'Vib_Change'
        ]
        
        missing_columns = [col for col in required_columns if col not in csv_data.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            upload_log.status = "failed"
            upload_log.error_message = error_msg
            db.add(upload_log)
            db.commit()
            raise HTTPException(status_code=400, detail=error_msg)
        
        csv_data['Timestamp'] = pd.to_datetime(csv_data['Timestamp'])
        
        records_inserted = 0
        for _, row in csv_data.iterrows():
            sensor_record = SensorData(
                timestamp=row['Timestamp'],
                temperature=float(row['Temperature (°C)']),
                vibration=float(row['Vibration (mm/s)']),
                pressure=float(row['Pressure (Pa)']),
                rpm=float(row['RPM']),
                maintenance_required=int(row['Maintenance Required']),
                temp_change=float(row['Temp_Change']),
                vib_change=float(row['Vib_Change'])
            )
            db.add(sensor_record)
            records_inserted += 1
        
        upload_log.records_count = records_inserted
        upload_log.status = "completed"
        db.add(upload_log)
        db.commit()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "CSV uploaded successfully",
                "filename": file.filename,
                "records_inserted": records_inserted,
                "upload_time": datetime.utcnow().isoformat()
            }
        )
    
    except Exception as e:
        error_msg = f"Error processing CSV: {str(e)}"
        upload_log.status = "failed"
        upload_log.error_message = error_msg
        db.add(upload_log)
        db.commit()
        raise HTTPException(status_code=500, detail=error_msg)

# ML endpoints
@app.get("/ml/models/")
async def get_available_models():
    """Get available ML model types and their parameters"""
    return {
        "available_models": {
            key: {
                "name": config["name"],
                "default_parameters": config["default_params"]
            }
            for key, config in trainer.models_config.items()
        }
    }

@app.post("/ml/train/")
async def train_model(request: TrainingRequest, db: Session = Depends(get_db)):
    """Train a new ML model on the sensor data"""
    try:
        result = trainer.train_model(
            db=db,
            model_type=request.model_type,
            hyperparameters=request.hyperparameters,
            test_size=request.test_size,
            model_name=request.model_name,
            description=request.description
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Model trained successfully",
                "result": result
            }
        )
    
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/models/trained/", response_model=List[ModelResponse])
async def get_trained_models(db: Session = Depends(get_db)):
    """Get list of all trained models"""
    try:
        models = db.query(MLModel).filter(MLModel.is_active == True).order_by(MLModel.created_at.desc()).all()
        
        return [
            ModelResponse(
                id=model.id,
                name=model.name,
                model_type=model.model_type,
                accuracy=model.accuracy,
                precision=model.precision,
                recall=model.recall,
                f1_score=model.f1_score,
                created_at=model.created_at.isoformat(),
                description=model.description
            )
            for model in models
        ]
    
    except Exception as e:
        logger.error(f"Error retrieving models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/models/{model_id}/")
async def get_model_details(model_id: int, db: Session = Depends(get_db)):
    """Get detailed information about a specific model"""
    try:
        model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get training history
        training_history = db.query(TrainingHistory).filter(TrainingHistory.model_id == model_id).first()
        
        return {
            "model": {
                "id": model.id,
                "name": model.name,
                "model_type": model.model_type,
                "accuracy": model.accuracy,
                "precision": model.precision,
                "recall": model.recall,
                "f1_score": model.f1_score,
                "created_at": model.created_at.isoformat(),
                "description": model.description,
                "hyperparameters": eval(model.hyperparameters),
                "feature_columns": eval(model.feature_columns),
                "training_data_count": model.training_data_count
            },
            "training_history": {
                "training_duration_seconds": training_history.training_duration_seconds if training_history else None,
                "confusion_matrix": eval(training_history.confusion_matrix) if training_history else None,
                "classification_report": training_history.classification_report if training_history else None,
                "feature_importance": eval(training_history.feature_importance) if training_history and training_history.feature_importance else None
            }
        }
    
    except Exception as e:
        logger.error(f"Error retrieving model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict/")
async def make_prediction(request: PredictionRequest, db: Session = Depends(get_db)):
    """Make maintenance predictions using a trained model"""
    try:
        results = trainer.predict(db, request.model_id, request.data)
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Predictions made successfully",
                "model_id": request.model_id,
                "predictions": results
            }
        )
    
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml/predict-single/")
async def predict_single(
    model_id: int,
    temperature: float,
    vibration: float,
    pressure: float,
    rpm: float,
    temp_change: float,
    vib_change: float,
    db: Session = Depends(get_db)
):
    """Make a single maintenance prediction"""
    try:
        input_data = [{
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "rpm": rpm,
            "temp_change": temp_change,
            "vib_change": vib_change
        }]
        
        results = trainer.predict(db, model_id, input_data)
        
        if results:
            result = results[0]
            return {
                "model_id": model_id,
                "input": result["input"],
                "predicted_maintenance_required": result["predicted_maintenance_required"],
                "maintenance_needed": result["maintenance_needed"],
                "confidence": result["probabilities"][result["predicted_maintenance_required"]] if result["probabilities"] else None,
                "all_probabilities": result["probabilities"]
            }
        else:
            raise HTTPException(status_code=500, detail="No prediction result returned")
    
    except Exception as e:
        logger.error(f"Error making single prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ml/predictions/")
async def get_prediction_history(
    model_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get prediction history"""
    try:
        query = db.query(PredictionResult)
        
        if model_id:
            query = query.filter(PredictionResult.model_id == model_id)
        
        predictions = query.order_by(PredictionResult.created_at.desc()).offset(offset).limit(limit).all()
        
        return {
            "predictions": [
                {
                    "id": pred.id,
                    "model_id": pred.model_id,
                    "input_data": eval(pred.input_data),
                    "predicted_maintenance_required": pred.predicted_maintenance_required,
                    "maintenance_needed": bool(pred.predicted_maintenance_required),
                    "prediction_probability": eval(pred.prediction_probability) if pred.prediction_probability else None,
                    "created_at": pred.created_at.isoformat()
                }
                for pred in predictions
            ],
            "total_records": len(predictions),
            "offset": offset,
            "limit": limit
        }
    
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/ml/models/{model_id}/")
async def delete_model(model_id: int, db: Session = Depends(get_db)):
    """Soft delete a model (mark as inactive)"""
    try:
        model = db.query(MLModel).filter(MLModel.id == model_id).first()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model.is_active = False
        db.commit()
        
        return {"message": f"Model {model_id} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Data retrieval endpoints
@app.get("/sensor-data/")
async def get_sensor_data(
    limit: int = 100, 
    offset: int = 0,
    maintenance_required: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Retrieve sensor data with optional filtering"""
    query = db.query(SensorData)
    
    if maintenance_required is not None:
        query = query.filter(SensorData.maintenance_required == maintenance_required)
    
    data = query.offset(offset).limit(limit).all()
    
    return {
        "data": [
            {
                "id": record.id,
                "timestamp": record.timestamp.isoformat(),
                "temperature": record.temperature,
                "vibration": record.vibration,
                "pressure": record.pressure,
                "rpm": record.rpm,
                "maintenance_required": record.maintenance_required,
                "maintenance_needed": bool(record.maintenance_required),
                "temp_change": record.temp_change,
                "vib_change": record.vib_change,
                "upload_datetime": record.upload_datetime.isoformat()
            }
            for record in data
        ],
        "total_records": len(data),
        "offset": offset,
        "limit": limit
    }

@app.get("/upload-logs/")
async def get_upload_logs(db: Session = Depends(get_db)):
    """Get upload history"""
    logs = db.query(UploadLog).order_by(UploadLog.upload_datetime.desc()).limit(50).all()
    
    return {
        "logs": [
            {
                "id": log.id,
                "filename": log.filename,
                "upload_datetime": log.upload_datetime.isoformat(),
                "records_count": log.records_count,
                "status": log.status,
                "error_message": log.error_message
            }
            for log in logs
        ]
    }

# Data Management & Cleanup Endpoints
@app.get("/database-stats/")
async def get_database_stats(db: Session = Depends(get_db)):
    """Get comprehensive statistics about data in the database"""
    try:
        stats = {
            "sensor_data": {
                "total_records": db.query(SensorData).count(),
                "maintenance_distribution": {}
            },
            "ml_models": {
                "total_models": db.query(MLModel).filter(MLModel.is_active == True).count(),
                "inactive_models": db.query(MLModel).filter(MLModel.is_active == False).count()
            },
            "predictions": {
                "total_predictions": db.query(PredictionResult).count()
            },
            "uploads": {
                "total_uploads": db.query(UploadLog).count(),
                "successful_uploads": db.query(UploadLog).filter(UploadLog.status == "completed").count(),
                "failed_uploads": db.query(UploadLog).filter(UploadLog.status == "failed").count()
            },
            "training_history": {
                "total_trainings": db.query(TrainingHistory).count(),
                "successful_trainings": db.query(TrainingHistory).filter(TrainingHistory.status == "completed").count(),
                "failed_trainings": db.query(TrainingHistory).filter(TrainingHistory.status == "failed").count()
            }
        }
        
        # Get maintenance requirement distribution
        maintenance_counts = db.execute(
            text("SELECT maintenance_required, COUNT(*) as count FROM sensor_data GROUP BY maintenance_required ORDER BY maintenance_required")
        ).fetchall()
        
        for maintenance_req, count in maintenance_counts:
            label = "maintenance_needed" if maintenance_req == 1 else "no_maintenance"
            stats["sensor_data"]["maintenance_distribution"][label] = count
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting database stats: {str(e)}")

@app.delete("/clear-everything/")
async def clear_everything(confirm: str = Query(..., description="Type 'YES_DELETE_EVERYTHING' to confirm")):
    """Clear ALL data from database - sensor data, models, predictions, everything"""
    if confirm != "YES_DELETE_EVERYTHING":
        raise HTTPException(
            status_code=400, 
            detail="Confirmation required. Add ?confirm=YES_DELETE_EVERYTHING to the URL"
        )
    
    try:
        db = SessionLocal()
        
        # Count all records
        sensor_count = db.query(SensorData).count()
        upload_count = db.query(UploadLog).count()
        models_count = db.query(MLModel).count()
        predictions_count = db.query(PredictionResult).count()
        training_count = db.query(TrainingHistory).count()
        
        # Delete everything in correct order (foreign keys)
        db.query(PredictionResult).delete()
        db.query(TrainingHistory).delete()
        db.query(MLModel).delete()
        db.query(SensorData).delete()
        db.query(UploadLog).delete()
        db.commit()
        db.close()
        
        return {
            "message": "EVERYTHING CLEARED! Database is now completely empty.",
            "deleted_records": {
                "sensor_data": sensor_count,
                "upload_logs": upload_count,
                "ml_models": models_count,
                "predictions": predictions_count,
                "training_history": training_count,
                "total_records_deleted": sensor_count + upload_count + models_count + predictions_count + training_count
            },
            "warning": "All data has been permanently deleted!"
        }
    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)