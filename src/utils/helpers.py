"""
Utility functions and helpers for the Coral Reef Health Guardian application
Common functions used across different modules
"""

import os
import hashlib
import mimetypes
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

def validate_image_file(file_content: bytes, filename: str, max_size: int = 10 * 1024 * 1024) -> Dict[str, Union[bool, str]]:
    """
    Validate uploaded image file
    
    Args:
        file_content: File content as bytes
        filename: Original filename
        max_size: Maximum file size in bytes
        
    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'error': None,
        'mime_type': None,
        'size': len(file_content)
    }
    
    try:
        # Check file size
        if len(file_content) > max_size:
            result['error'] = f"File size ({len(file_content)} bytes) exceeds maximum ({max_size} bytes)"
            return result
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type or not mime_type.startswith('image/'):
            result['error'] = "File must be an image"
            return result
        
        result['mime_type'] = mime_type
        
        # Validate image by trying to open it
        try:
            image = Image.open(io.BytesIO(file_content))
            image.verify()
            result['valid'] = True
        except Exception as e:
            result['error'] = f"Invalid image file: {str(e)}"
            
    except Exception as e:
        result['error'] = f"File validation error: {str(e)}"
    
    return result

def resize_image(image: Image.Image, target_size: Tuple[int, int], maintain_aspect: bool = True) -> Image.Image:
    """
    Resize image to target size
    
    Args:
        image: PIL Image object
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if maintain_aspect:
        image.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and paste resized image
        new_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # Calculate position to center the image
        x = (target_size[0] - image.width) // 2
        y = (target_size[1] - image.height) // 2
        
        new_image.paste(image, (x, y))
        return new_image
    else:
        return image.resize(target_size, Image.Resampling.LANCZOS)

def calculate_file_hash(file_content: bytes) -> str:
    """
    Calculate SHA-256 hash of file content
    
    Args:
        file_content: File content as bytes
        
    Returns:
        Hexadecimal hash string
    """
    return hashlib.sha256(file_content).hexdigest()

def generate_unique_filename(original_filename: str, prefix: str = "") -> str:
    """
    Generate unique filename with timestamp
    
    Args:
        original_filename: Original filename
        prefix: Optional prefix
        
    Returns:
        Unique filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    name, ext = os.path.splitext(original_filename)
    
    if prefix:
        return f"{prefix}_{timestamp}_{name}{ext}"
    else:
        return f"{timestamp}_{name}{ext}"

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL Image to base64 string
    
    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Base64 encoded image string
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    return f"data:image/{format.lower()};base64,{img_base64}"

def base64_to_image(base64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image
    
    Args:
        base64_string: Base64 encoded image string
        
    Returns:
        PIL Image object
    """
    # Remove data URL prefix if present
    if base64_string.startswith('data:image'):
        base64_string = base64_string.split(',')[1]
    
    img_bytes = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(img_bytes))

def calculate_image_stats(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic statistics for an image
    
    Args:
        image: Image array (H, W, C)
        
    Returns:
        Dictionary with image statistics
    """
    if len(image.shape) == 3:
        # Color image
        stats = {
            'mean_brightness': float(np.mean(image)),
            'std_brightness': float(np.std(image)),
            'mean_r': float(np.mean(image[:, :, 0])),
            'mean_g': float(np.mean(image[:, :, 1])),
            'mean_b': float(np.mean(image[:, :, 2])),
            'std_r': float(np.std(image[:, :, 0])),
            'std_g': float(np.std(image[:, :, 1])),
            'std_b': float(np.std(image[:, :, 2]))
        }
        
        # Calculate contrast (standard deviation of grayscale)
        gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
        stats['contrast'] = float(np.std(gray))
        
    else:
        # Grayscale image
        stats = {
            'mean_brightness': float(np.mean(image)),
            'std_brightness': float(np.std(image)),
            'contrast': float(np.std(image))
        }
    
    return stats

def create_response_dict(success: bool = True, 
                        data: Optional[Dict] = None, 
                        message: str = "", 
                        error: Optional[str] = None) -> Dict:
    """
    Create standardized API response dictionary
    
    Args:
        success: Whether operation was successful
        data: Response data
        message: Success message
        error: Error message if failed
        
    Returns:
        Standardized response dictionary
    """
    response = {
        'success': success,
        'timestamp': datetime.now().isoformat(),
        'data': data or {},
        'message': message
    }
    
    if error:
        response['error'] = error
        
    return response

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove or replace dangerous characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Ensure filename is not empty
    if not filename:
        filename = 'unnamed_file'
    
    return filename

def log_prediction_metrics(prediction_data: Dict, processing_time: float):
    """
    Log prediction metrics for monitoring
    
    Args:
        prediction_data: Prediction results
        processing_time: Time taken for prediction in seconds
    """
    logger.info(f"Prediction completed - "
               f"Class: {prediction_data.get('prediction', 'Unknown')}, "
               f"Confidence: {prediction_data.get('confidence', 0):.3f}, "
               f"Processing time: {processing_time:.3f}s")

def calculate_health_score(class_probabilities: Dict[str, float]) -> float:
    """
    Calculate overall coral health score from class probabilities
    
    Args:
        class_probabilities: Dictionary with class probabilities
        
    Returns:
        Health score between 0-100
    """
    # Define health weights for each class
    health_weights = {
        'Healthy': 1.0,
        'Unhealthy': 0.5,
        'Dead': 0.0
    }
    
    weighted_score = 0.0
    for class_name, probability in class_probabilities.items():
        weight = health_weights.get(class_name, 0.5)
        weighted_score += probability * weight
    
    return weighted_score * 100

def get_class_color_mapping() -> Dict[str, str]:
    """
    Get color mapping for different coral health classes
    
    Returns:
        Dictionary mapping class names to colors
    """
    return {
        'Healthy': '#28a745',    # Green
        'Unhealthy': '#ffc107',  # Yellow/Orange
        'Dead': '#dc3545'        # Red
    }

def validate_prediction_threshold(confidence: float, min_threshold: float = 0.5) -> bool:
    """
    Validate if prediction confidence meets minimum threshold
    
    Args:
        confidence: Prediction confidence score
        min_threshold: Minimum required confidence
        
    Returns:
        Whether confidence meets threshold
    """
    return confidence >= min_threshold

def format_confidence_percentage(confidence: float, decimal_places: int = 1) -> str:
    """
    Format confidence as percentage string
    
    Args:
        confidence: Confidence value (0-1)
        decimal_places: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    percentage = confidence * 100
    return f"{percentage:.{decimal_places}f}%"

def extract_image_metadata(image: Image.Image) -> Dict[str, Union[str, int, float]]:
    """
    Extract metadata from PIL Image
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image metadata
    """
    metadata = {
        'format': image.format or 'Unknown',
        'mode': image.mode,
        'size': image.size,
        'width': image.width,
        'height': image.height,
        'has_transparency': image.mode in ('RGBA', 'LA') or 'transparency' in image.info
    }
    
    # Add EXIF data if available
    if hasattr(image, '_getexif') and image._getexif():
        metadata['has_exif'] = True
    else:
        metadata['has_exif'] = False
    
    return metadata

def create_error_response(error_message: str, status_code: int = 400) -> Dict:
    """
    Create error response dictionary
    
    Args:
        error_message: Error description
        status_code: HTTP status code
        
    Returns:
        Error response dictionary
    """
    return {
        'success': False,
        'error': error_message,
        'status_code': status_code,
        'timestamp': datetime.now().isoformat()
    }

# Export commonly used functions
__all__ = [
    'validate_image_file',
    'resize_image', 
    'calculate_file_hash',
    'generate_unique_filename',
    'format_file_size',
    'image_to_base64',
    'base64_to_image',
    'calculate_image_stats',
    'create_response_dict',
    'sanitize_filename',
    'log_prediction_metrics',
    'calculate_health_score',
    'get_class_color_mapping',
    'validate_prediction_threshold',
    'format_confidence_percentage',
    'extract_image_metadata',
    'create_error_response'
]