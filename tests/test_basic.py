"""
Basic tests for Coral Reef Health Guardian
Testing core functionality and imports
"""

import pytest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all core modules can be imported"""
    
    # Test configuration import
    try:
        import config
        assert hasattr(config, 'settings')
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")
    
    # Test ML modules
    try:
        from src.ml.coral_classifier import CoralHealthClassifier
        from src.ml.data_analysis import CoralDataAnalyzer
    except ImportError as e:
        pytest.fail(f"Failed to import ML modules: {e}")
    
    # Test utility modules
    try:
        from src.utils.helpers import validate_image_file, format_file_size
    except ImportError as e:
        pytest.fail(f"Failed to import utility modules: {e}")

def test_classifier_initialization():
    """Test coral classifier can be initialized"""
    from src.ml.coral_classifier import CoralHealthClassifier
    
    classifier = CoralHealthClassifier()
    assert classifier is not None
    assert classifier.class_names == ['Dead', 'Healthy', 'Unhealthy']
    assert classifier.img_size == (224, 224)
    assert classifier.num_classes == 3

def test_data_analyzer_initialization():
    """Test data analyzer can be initialized"""
    from src.ml.data_analysis import CoralDataAnalyzer
    
    analyzer = CoralDataAnalyzer()
    assert analyzer is not None
    assert analyzer.class_names == ['Dead', 'Healthy', 'Unhealthy']

def test_config_loading():
    """Test configuration loading"""
    import config
    
    settings = config.settings
    assert settings.app_name == "Coral Reef Health Guardian"
    assert settings.port == 8000
    assert len(settings.class_names) == 3

def test_helper_functions():
    """Test utility helper functions"""
    from src.utils.helpers import format_file_size, calculate_health_score
    
    # Test file size formatting
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1048576) == "1.0 MB"
    
    # Test health score calculation
    probabilities = {'Healthy': 0.7, 'Unhealthy': 0.2, 'Dead': 0.1}
    health_score = calculate_health_score(probabilities)
    assert 0 <= health_score <= 100
    assert health_score == 80.0  # 0.7*100 + 0.2*50 + 0.1*0

def test_project_structure():
    """Test that required project directories exist"""
    
    required_dirs = [
        'src/ml',
        'src/utils', 
        'templates',
        'static/css',
        'config',
        'logs',
        'models',
        'data'
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        assert full_path.exists(), f"Required directory missing: {dir_path}"

def test_required_files():
    """Test that required files exist"""
    
    required_files = [
        'app.py',
        'requirements.txt',
        'README.md',
        'config.py',
        'ecosystem.config.js'
    ]
    
    for file_path in required_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"Required file missing: {file_path}"

if __name__ == "__main__":
    # Run tests directly
    test_imports()
    test_classifier_initialization()
    test_data_analyzer_initialization()
    test_config_loading()
    test_helper_functions()
    test_project_structure()
    test_required_files()
    
    print("✅ All basic tests passed!")