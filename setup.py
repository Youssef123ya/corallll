#!/usr/bin/env python3
"""
Setup script for Coral Reef Health Guardian
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoralSetup:
    """Setup and configuration manager for Coral Reef Health Guardian"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.requirements_file = self.project_root / "requirements.txt"
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.info("Checking Python version...")
        
        if sys.version_info < (3, 8):
            logger.error("Python 3.8 or higher is required")
            sys.exit(1)
        
        logger.info(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} - OK")
    
    def create_directories(self):
        """Create necessary project directories"""
        logger.info("Creating project directories...")
        
        directories = [
            "logs",
            "uploads", 
            "models",
            "data/raw",
            "data/processed",
            "data/sample_images",
            "results",
            "plots",
            "notebooks",
            "tests/unit",
            "tests/integration"
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            
            # Create .gitkeep files for empty directories
            gitkeep_file = dir_path / ".gitkeep"
            if not gitkeep_file.exists():
                gitkeep_file.touch()
    
    def install_dependencies(self, dev_mode=False):
        """Install Python dependencies"""
        logger.info("Installing Python dependencies...")
        
        if not self.requirements_file.exists():
            logger.error(f"Requirements file not found: {self.requirements_file}")
            sys.exit(1)
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)], check=True)
            
            if dev_mode:
                # Install development dependencies
                dev_packages = [
                    "pytest-cov",
                    "black", 
                    "flake8",
                    "mypy",
                    "jupyter",
                    "ipykernel"
                ]
                
                for package in dev_packages:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
            
            logger.info("Dependencies installed successfully")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            sys.exit(1)
    
    def create_config_files(self):
        """Create configuration files"""
        logger.info("Creating configuration files...")
        
        # Create .env file if it doesn't exist
        env_file = self.project_root / ".env"
        if not env_file.exists():
            env_content = """# Coral Reef Health Guardian Configuration
DEBUG=false
LOG_LEVEL=info
HOST=0.0.0.0
PORT=8000
MAX_FILE_SIZE=10485760
SECRET_KEY=coral-reef-secret-key-change-in-production
DATABASE_URL=sqlite:///coral_health.db

# Model settings
MODEL_PATH=models/coral_classifier.h5

# Optional: Email settings
SMTP_SERVER=
SMTP_PORT=587
SMTP_USERNAME=
SMTP_PASSWORD=

# Optional: Analytics
ENABLE_ANALYTICS=true
"""
            env_file.write_text(env_content)
            logger.info("Created .env configuration file")
        
        # Create gitignore if it doesn't exist
        gitignore_file = self.project_root / ".gitignore"
        if not gitignore_file.exists():
            gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
uploads/*
!uploads/.gitkeep
logs/*
!logs/.gitkeep
models/*.h5
results/*
!results/.gitkeep
plots/*
!plots/.gitkeep

# Environment variables
.env
.env.local

# Database
*.db
*.sqlite

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/
.coverage
htmlcov/

# mypy
.mypy_cache/
"""
            gitignore_file.write_text(gitignore_content)
            logger.info("Created .gitignore file")
    
    def setup_git_repository(self):
        """Initialize git repository if not already done"""
        logger.info("Setting up git repository...")
        
        try:
            # Check if git is available
            subprocess.run(["git", "--version"], check=True, capture_output=True)
            
            # Initialize git repo if .git doesn't exist
            git_dir = self.project_root / ".git"
            if not git_dir.exists():
                subprocess.run(["git", "init"], cwd=self.project_root, check=True)
                logger.info("Initialized git repository")
            
            # Set up initial commit
            subprocess.run(["git", "add", ".gitignore", "README.md"], 
                          cwd=self.project_root, check=False)
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("Git not available or failed to initialize repository")
    
    def create_sample_data(self):
        """Create sample data and configuration files"""
        logger.info("Creating sample data...")
        
        # Create sample coral dataset summary
        sample_data_dir = self.project_root / "data" / "sample_images"
        
        # Create sample CSV with coral data
        csv_content = """Class,Count,Percentage,Description
Healthy,661,41.3,"Vibrant, colorful coral with good health"
Unhealthy,508,31.8,"Bleached or stressed coral, whitish/pale"
Dead,430,26.9,"Dead coral, darker/brownish, no living tissue"
Total,1599,100.0,Complete dataset summary"""
        
        csv_file = self.project_root / "coral_dataset_summary.csv"
        if not csv_file.exists():
            csv_file.write_text(csv_content)
            logger.info("Created sample dataset CSV")
        
        # Create sample training configuration
        train_config = {
            "model": {
                "architecture": "EfficientNetB0",
                "input_shape": [224, 224, 3],
                "num_classes": 3,
                "learning_rate": 0.001
            },
            "training": {
                "batch_size": 32,
                "epochs": 50,
                "validation_split": 0.2,
                "early_stopping": True,
                "patience": 10
            },
            "data_augmentation": {
                "rotation_range": 20,
                "width_shift_range": 0.1,
                "height_shift_range": 0.1,
                "horizontal_flip": True,
                "zoom_range": 0.1,
                "brightness_range": [0.8, 1.2]
            }
        }
        
        config_file = self.project_root / "config" / "training_config.json"
        config_file.parent.mkdir(exist_ok=True)
        
        if not config_file.exists():
            with open(config_file, 'w') as f:
                json.dump(train_config, f, indent=2)
            logger.info("Created training configuration")
    
    def setup_pm2_ecosystem(self):
        """Setup PM2 ecosystem for production deployment"""
        logger.info("Setting up PM2 ecosystem...")
        
        ecosystem_file = self.project_root / "ecosystem.config.js"
        if ecosystem_file.exists():
            logger.info("PM2 ecosystem configuration already exists")
        else:
            logger.warning("PM2 ecosystem configuration not found")
    
    def create_startup_scripts(self):
        """Create startup scripts for different environments"""
        logger.info("Creating startup scripts...")
        
        # Development startup script
        dev_script_content = """#!/bin/bash
# Development startup script for Coral Reef Health Guardian

echo "Starting Coral Reef Health Guardian (Development Mode)..."

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DEBUG=true
export LOG_LEVEL=debug

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
pip install -r requirements.txt

# Start the application
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level debug
"""
        
        dev_script = self.project_root / "start_dev.sh"
        dev_script.write_text(dev_script_content)
        dev_script.chmod(0o755)
        
        # Production startup script  
        prod_script_content = """#!/bin/bash
# Production startup script for Coral Reef Health Guardian

echo "Starting Coral Reef Health Guardian (Production Mode)..."

# Set environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export DEBUG=false
export LOG_LEVEL=info

# Activate virtual environment
source venv/bin/activate

# Start with PM2
pm2 start ecosystem.config.js --env production

echo "Application started with PM2"
echo "View logs with: pm2 logs coral-reef-guardian"
echo "Stop with: pm2 stop coral-reef-guardian"
"""
        
        prod_script = self.project_root / "start_prod.sh" 
        prod_script.write_text(prod_script_content)
        prod_script.chmod(0o755)
        
        logger.info("Created startup scripts")
    
    def run_initial_tests(self):
        """Run initial tests to verify setup"""
        logger.info("Running initial setup verification...")
        
        try:
            # Test imports
            sys.path.insert(0, str(self.project_root))
            
            # Test basic imports
            import config
            from src.ml.coral_classifier import CoralHealthClassifier
            from src.ml.data_analysis import CoralDataAnalyzer
            from src.utils.helpers import validate_image_file
            
            logger.info("✓ All imports successful")
            
            # Test configuration
            if hasattr(config, 'settings'):
                logger.info("✓ Configuration loaded successfully")
            
            # Test model initialization (without loading weights)
            classifier = CoralHealthClassifier()
            logger.info("✓ Classifier initialization successful")
            
            # Test analyzer
            analyzer = CoralDataAnalyzer()
            logger.info("✓ Data analyzer initialization successful")
            
            logger.info("Setup verification completed successfully!")
            
        except ImportError as e:
            logger.error(f"Import error during verification: {e}")
        except Exception as e:
            logger.error(f"Verification error: {e}")
    
    def display_next_steps(self):
        """Display next steps for the user"""
        logger.info("Setup completed! Next steps:")
        
        print(f"""
{'='*60}
🪸 Coral Reef Health Guardian - Setup Complete! 🪸
{'='*60}

Next Steps:

1. 📊 PREPARE DATA (Optional):
   - Place coral images in data/raw/ directory
   - Organize by class: data/raw/healthy/, data/raw/unhealthy/, data/raw/dead/

2. 🤖 TRAIN MODEL (Optional):
   cd {self.project_root}
   python src/ml/train_model.py --data_dir data/raw --epochs 50

3. 🚀 START APPLICATION:

   Development Mode:
   ./start_dev.sh
   
   OR manually:
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload

   Production Mode:
   ./start_prod.sh

4. 🌐 ACCESS APPLICATION:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs
   - Dashboard: http://localhost:8000/dashboard
   - Analysis: http://localhost:8000/analysis

5. 🛠️  OPTIONAL CONFIGURATIONS:
   - Edit .env file for custom settings
   - Modify config.py for advanced configuration
   - Set up email notifications (SMTP settings in .env)

6. 📈 MONITORING:
   - PM2 Dashboard: pm2 monit
   - Application Logs: tail -f logs/app.log
   - System Health: http://localhost:8000/api/health

{'='*60}
Happy coral monitoring! 🌊🪸
{'='*60}
""")


def main():
    """Main setup function"""
    
    parser = __import__('argparse').ArgumentParser(description='Setup Coral Reef Health Guardian')
    parser.add_argument('--dev', action='store_true', help='Setup for development (installs dev dependencies)')
    parser.add_argument('--no-git', action='store_true', help='Skip git repository setup')
    parser.add_argument('--no-test', action='store_true', help='Skip initial verification tests')
    
    args = parser.parse_args()
    
    setup = CoralSetup()
    
    try:
        logger.info("🪸 Starting Coral Reef Health Guardian setup...")
        
        # Run setup steps
        setup.check_python_version()
        setup.create_directories()
        setup.create_config_files()
        setup.install_dependencies(dev_mode=args.dev)
        
        if not args.no_git:
            setup.setup_git_repository()
        
        setup.create_sample_data()
        setup.setup_pm2_ecosystem()
        setup.create_startup_scripts()
        
        if not args.no_test:
            setup.run_initial_tests()
        
        setup.display_next_steps()
        
        logger.info("🎉 Setup completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()