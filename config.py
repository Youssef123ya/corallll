"""
Configuration settings for Coral Reef Health Guardian
Centralized configuration management for the application
"""

import os
from pathlib import Path
from typing import Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    app_name: str = "Coral Reef Health Guardian"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    reload: bool = Field(default=False, env="RELOAD")
    
    # API settings
    api_prefix: str = "/api"
    docs_url: str = "/api/docs"
    redoc_url: str = "/api/redoc"
    
    # File upload settings
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    allowed_extensions: list = [".jpg", ".jpeg", ".png", ".gif"]
    upload_dir: Path = PROJECT_ROOT / "uploads"
    
    # Model settings
    model_path: str = Field(default=str(PROJECT_ROOT / "models" / "coral_classifier.h5"), env="MODEL_PATH")
    backup_model_path: str = Field(default=str(PROJECT_ROOT / "models" / "backup_model.h5"), env="BACKUP_MODEL_PATH")
    model_input_size: tuple = (224, 224)
    class_names: list = ["Dead", "Healthy", "Unhealthy"]
    
    # Data settings
    data_dir: Path = PROJECT_ROOT / "data"
    sample_data_dir: Path = data_dir / "sample_images"
    processed_data_dir: Path = data_dir / "processed"
    
    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Path = PROJECT_ROOT / "logs" / "app.log"
    
    # Database settings (optional)
    database_url: str = Field(default="sqlite:///coral_health.db", env="DATABASE_URL")
    
    # Security settings
    secret_key: str = Field(default="coral-reef-secret-key-change-in-production", env="SECRET_KEY")
    allowed_origins: list = Field(default=["*"], env="ALLOWED_ORIGINS")
    
    # Performance settings
    prediction_cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    max_concurrent_predictions: int = Field(default=10, env="MAX_CONCURRENT_PREDICTIONS")
    
    # Analytics settings
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    analytics_db: str = Field(default="analytics.db", env="ANALYTICS_DB")
    
    # Email settings (for notifications)
    smtp_server: str = Field(default="", env="SMTP_SERVER")
    smtp_port: int = Field(default=587, env="SMTP_PORT")
    smtp_username: str = Field(default="", env="SMTP_USERNAME")
    smtp_password: str = Field(default="", env="SMTP_PASSWORD")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()

# Model configuration
MODEL_CONFIG = {
    "architecture": "EfficientNetB0",
    "input_shape": (224, 224, 3),
    "num_classes": 3,
    "class_names": settings.class_names,
    "preprocessing": {
        "rescale": 1./255,
        "rotation_range": 20,
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "horizontal_flip": True,
        "zoom_range": 0.1,
        "brightness_range": [0.8, 1.2]
    },
    "training": {
        "batch_size": 32,
        "epochs": 50,
        "learning_rate": 0.001,
        "patience": 10,
        "min_lr": 1e-7
    }
}

# Dataset configuration
DATASET_CONFIG = {
    "source": "esahit/coral-health-classification",
    "total_images": 1599,
    "image_format": "64x64 RGB PNG",
    "classes": {
        "Healthy": {"count": 661, "percentage": 41.3},
        "Unhealthy": {"count": 508, "percentage": 31.8},
        "Dead": {"count": 430, "percentage": 26.9}
    },
    "source_datasets": ["BU", "BHD", "EILAT"],
    "balance_ratio": 1.54,
    "coefficient_variation": 0.180,
    "balance_status": "Well-balanced"
}

# Color analysis configuration
COLOR_ANALYSIS_CONFIG = {
    "sample_size": 100,
    "features": [
        "brightness", "contrast", "rgb_mean", "rgb_std",
        "saturation", "hue", "texture"
    ],
    "mock_data": {
        "Healthy": {
            "rgb_mean": [103.3, 136.1, 106.4],
            "rgb_std": [25.2, 30.8, 28.1],
            "brightness_mean": 115.3,
            "brightness_std": 30.5,
            "contrast_mean": 36.3,
            "contrast_std": 10.7,
            "sample_count": 100
        },
        "Unhealthy": {
            "rgb_mean": [145.8, 142.3, 138.9],
            "rgb_std": [22.1, 25.4, 24.8],
            "brightness_mean": 142.1,
            "brightness_std": 28.3,
            "contrast_mean": 32.1,
            "contrast_std": 9.2,
            "sample_count": 100
        },
        "Dead": {
            "rgb_mean": [85.2, 95.8, 88.1],
            "rgb_std": [18.7, 21.3, 19.8],
            "brightness_mean": 89.7,
            "brightness_std": 22.1,
            "contrast_mean": 28.4,
            "contrast_std": 8.9,
            "sample_count": 100
        }
    }
}

# Performance metrics configuration
PERFORMANCE_CONFIG = {
    "target_metrics": {
        "accuracy": 0.94,
        "precision": 0.93,
        "recall": 0.94,
        "f1_score": 0.93
    },
    "confusion_matrix": [
        [132, 4, 2],    # Healthy
        [3, 101, 4],    # Unhealthy
        [1, 2, 86]      # Dead
    ],
    "processing_time": {
        "target": 100,  # milliseconds
        "average": 85,
        "max_acceptable": 500
    }
}

# API configuration
API_CONFIG = {
    "rate_limiting": {
        "predictions_per_minute": 60,
        "uploads_per_hour": 100
    },
    "response_formats": ["json", "xml"],
    "supported_image_formats": ["JPEG", "PNG", "JPG", "GIF"],
    "max_batch_size": 10,
    "timeout_seconds": 30
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s"
        }
    },
    "handlers": {
        "default": {
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": "DEBUG",
            "formatter": "detailed",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(settings.log_file),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": settings.log_level,
            "propagate": False
        }
    }
}

# Security configuration
SECURITY_CONFIG = {
    "cors": {
        "allow_origins": settings.allowed_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"]
    },
    "file_validation": {
        "max_file_size": settings.max_file_size,
        "allowed_extensions": settings.allowed_extensions,
        "scan_for_malware": False,  # Enable in production
        "virus_scan_timeout": 10
    }
}

# Deployment configuration
DEPLOYMENT_CONFIG = {
    "production": {
        "debug": False,
        "reload": False,
        "workers": 4,
        "log_level": "warning"
    },
    "staging": {
        "debug": True,
        "reload": True,
        "workers": 2,
        "log_level": "info"
    },
    "development": {
        "debug": True,
        "reload": True,
        "workers": 1,
        "log_level": "debug"
    }
}

def get_environment() -> str:
    """Get current environment"""
    return os.getenv("ENVIRONMENT", "development").lower()

def get_deployment_config() -> Dict[str, Any]:
    """Get configuration for current environment"""
    env = get_environment()
    return DEPLOYMENT_CONFIG.get(env, DEPLOYMENT_CONFIG["development"])

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.upload_dir,
        settings.data_dir,
        settings.sample_data_dir,
        settings.processed_data_dir,
        PROJECT_ROOT / "models",
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "plots"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories on import
create_directories()

# Export commonly used settings
__all__ = [
    "settings",
    "MODEL_CONFIG",
    "DATASET_CONFIG", 
    "COLOR_ANALYSIS_CONFIG",
    "PERFORMANCE_CONFIG",
    "API_CONFIG",
    "LOGGING_CONFIG",
    "SECURITY_CONFIG",
    "DEPLOYMENT_CONFIG",
    "PROJECT_ROOT",
    "get_environment",
    "get_deployment_config",
    "create_directories"
]