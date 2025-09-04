"""
Coral Reef Health Guardian - FastAPI Web Application
Main application entry point for the coral health monitoring system
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Optional
import uuid
from datetime import datetime

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import aiofiles
import numpy as np
from PIL import Image
import io

# Import our modules
from ml.coral_classifier import CoralHealthClassifier
from ml.data_analysis import CoralDataAnalyzer, load_coral_dataset_info
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Coral Reef Health Guardian",
    description="AI-powered coral reef monitoring system for classifying coral health status",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables
classifier = None
analyzer = CoralDataAnalyzer()
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Create a demo model for testing
class DemoCoralClassifier:
    """Demo classifier for testing purposes when no trained model is available"""
    
    def __init__(self):
        self.class_names = ['Dead', 'Healthy', 'Unhealthy']
        logger.info("Using demo classifier (no trained model loaded)")
    
    def predict(self, image_path):
        """Simulate prediction for demo purposes"""
        # Simulate random but realistic predictions
        import random
        
        # Weighted random prediction (more healthy corals in demo)
        weights = [0.2, 0.5, 0.3]  # Dead, Healthy, Unhealthy
        predicted_class = np.random.choice(self.class_names, p=weights)
        confidence = random.uniform(0.7, 0.95)
        
        # Create realistic probabilities
        base_probs = np.random.dirichlet([1, 1, 1])  # Random probabilities that sum to 1
        # Boost the predicted class probability
        predicted_idx = self.class_names.index(predicted_class)
        base_probs[predicted_idx] = confidence
        base_probs = base_probs / np.sum(base_probs)  # Normalize
        
        result = {
            'prediction': predicted_class,
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, base_probs)
            },
            'health_score': float(base_probs[1] * 100 + base_probs[2] * 50)  # Healthy=100%, Unhealthy=50%, Dead=0%
        }
        
        return result

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    global classifier
    
    logger.info("Starting Coral Reef Health Guardian...")
    
    # Try to load a trained model, fallback to demo classifier
    model_path = "models/coral_classifier.h5"
    
    try:
        if os.path.exists(model_path):
            classifier = CoralHealthClassifier(model_path)
            logger.info(f"Loaded trained model from {model_path}")
        else:
            classifier = DemoCoralClassifier()
            logger.info("No trained model found, using demo classifier")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        classifier = DemoCoralClassifier()
    
    logger.info("Application startup complete!")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse) 
async def dashboard(request: Request):
    """Dashboard page"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/analysis", response_class=HTMLResponse)
async def analysis(request: Request):
    """Analysis page"""
    return templates.TemplateResponse("analysis.html", {"request": request})

@app.post("/api/predict")
async def predict_coral_health(file: UploadFile = File(...)):
    """
    Predict coral health from uploaded image
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results with confidence scores
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = upload_dir / unique_filename
        
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        # Validate image
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Make prediction
        result = classifier.predict(str(file_path))
        
        # Add metadata
        result.update({
            'filename': file.filename,
            'file_size': len(content),
            'timestamp': datetime.now().isoformat(),
            'model_type': 'demo' if isinstance(classifier, DemoCoralClassifier) else 'trained'
        })
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Could not remove temporary file {file_path}: {e}")
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict coral health for multiple images
    
    Args:
        files: List of uploaded image files
        
    Returns:
        List of prediction results
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            # Process each file similar to single prediction
            if not file.content_type.startswith('image/'):
                results.append({
                    'filename': file.filename,
                    'error': 'File must be an image',
                    'prediction': None
                })
                continue
            
            # Save and process file
            file_extension = os.path.splitext(file.filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = upload_dir / unique_filename
            
            async with aiofiles.open(file_path, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            # Make prediction
            result = classifier.predict(str(file_path))
            result['filename'] = file.filename
            result['model_type'] = 'demo' if isinstance(classifier, DemoCoralClassifier) else 'trained'
            
            results.append(result)
            
            # Clean up
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {file_path}: {e}")
                
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                'filename': file.filename,
                'error': str(e),
                'prediction': None
            })
    
    return JSONResponse(content={'results': results, 'total_processed': len(results)})

@app.get("/api/dataset/summary")
async def get_dataset_summary():
    """Get dataset summary statistics"""
    try:
        # Load coral dataset summary
        if os.path.exists("coral_dataset_summary.csv"):
            data_summary = load_coral_dataset_info("coral_dataset_summary.csv")
        else:
            # Use default demo data
            data_summary = pd.DataFrame({
                'Class': ['Healthy', 'Unhealthy', 'Dead'],
                'Count': [661, 508, 430],
                'Percentage': [41.3, 31.8, 26.9],
                'Description': [
                    'Vibrant, colorful coral with good health',
                    'Bleached or stressed coral, whitish/pale',
                    'Dead coral, darker/brownish, no living tissue'
                ]
            })
        
        # Analyze distribution
        dist_analysis = analyzer.analyze_dataset_distribution(data_summary)
        
        return JSONResponse(content=dist_analysis)
        
    except Exception as e:
        logger.error(f"Error getting dataset summary: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve dataset summary")

@app.get("/api/analysis/color")
async def get_color_analysis():
    """Get color analysis for coral classes"""
    try:
        # Mock color analysis data for demo
        color_stats = {
            'Healthy': {
                'rgb_mean': [103.3, 136.1, 106.4],
                'rgb_std': [25.2, 30.8, 28.1],
                'brightness_mean': 115.3,
                'brightness_std': 30.5,
                'contrast_mean': 36.3,
                'contrast_std': 10.7,
                'sample_count': 100
            },
            'Unhealthy': {
                'rgb_mean': [145.8, 142.3, 138.9],
                'rgb_std': [22.1, 25.4, 24.8],
                'brightness_mean': 142.1,
                'brightness_std': 28.3,
                'contrast_mean': 32.1,
                'contrast_std': 9.2,
                'sample_count': 100
            },
            'Dead': {
                'rgb_mean': [85.2, 95.8, 88.1],
                'rgb_std': [18.7, 21.3, 19.8],
                'brightness_mean': 89.7,
                'brightness_std': 22.1,
                'contrast_mean': 28.4,
                'contrast_std': 8.9,
                'sample_count': 100
            }
        }
        
        return JSONResponse(content=color_stats)
        
    except Exception as e:
        logger.error(f"Error getting color analysis: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve color analysis")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": classifier is not None,
        "model_type": 'demo' if isinstance(classifier, DemoCoralClassifier) else 'trained'
    }

@app.get("/api/stats")
async def get_system_stats():
    """Get system statistics"""
    return {
        "total_predictions": 0,  # Would track in a real database
        "model_accuracy": 94.2,  # Example metric
        "classes_supported": ['Healthy', 'Unhealthy', 'Dead'],
        "max_file_size": "10MB",
        "supported_formats": ["JPEG", "PNG", "JPG"],
        "last_updated": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    logger.error(f"Server error: {exc}")
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)

if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )