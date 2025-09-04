"""
Coral Health Classification Model
Advanced CNN-based classifier for coral reef health assessment
"""

import os
import numpy as np
# TensorFlow imports - will be imported only when needed
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers, Model
# from tensorflow.keras.applications import EfficientNetB0
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import logging
from typing import Tuple, Dict, List, Optional
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoralHealthClassifier:
    """
    Advanced Coral Health Classification System
    
    Uses deep convolutional neural networks to classify coral health
    into three categories: Healthy, Unhealthy, Dead
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize the classifier"""
        self.model = None
        self.class_names = ['Dead', 'Healthy', 'Unhealthy']
        self.img_size = (224, 224)  # Increased from 64x64 for better accuracy
        self.num_classes = 3
        
        # Load pre-trained model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def build_model(self):
        """
        Build advanced CNN model with transfer learning
        
        Returns:
            Compiled Keras model
        """
        try:
            # Import TensorFlow only when needed
            import tensorflow as tf
            from tensorflow.keras.applications import EfficientNetB0
            from tensorflow.keras import layers, Model
            
            # Create base model using EfficientNetB0
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        except ImportError:
            raise ImportError("TensorFlow is required for model building. Install with: pip install tensorflow")
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        inputs = keras.Input(shape=(*self.img_size, 3))
        
        # Data augmentation layers
        data_augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.1),
            layers.RandomContrast(0.1),
        ])
        
        # Apply data augmentation
        x = data_augmentation(inputs)
        
        # Rescaling
        x = layers.Rescaling(1./255)(x)
        
        # Base model
        x = base_model(x, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Add dropout for regularization
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        model = Model(inputs, outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def create_fine_tuned_model(self, base_model):
        """
        Create fine-tuned version of the model
        
        Args:
            base_model: Pre-trained base model
            
        Returns:
            Fine-tuned model
        """
        # Unfreeze the top layers of the base model
        base_model.trainable = True
        
        # Fine-tune from this layer onwards
        fine_tune_at = 100
        
        # Freeze all the layers before fine_tune_at
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Use a lower learning rate for fine-tuning
        base_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001/10),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return base_model
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            if isinstance(image_path, str):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Handle PIL Image or numpy array
                image = np.array(image_path)
            
            # Resize image
            image = cv2.resize(image, self.img_size)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Could not preprocess image: {e}")
    
    def predict(self, image_path: str) -> Dict[str, any]:
        """
        Predict coral health for a single image
        
        Args:
            image_path: Path to the coral image
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please train or load a model first.")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get class probabilities
        class_probabilities = predictions[0]
        
        # Get predicted class
        predicted_class_idx = np.argmax(class_probabilities)
        predicted_class = self.class_names[predicted_class_idx]
        confidence = float(class_probabilities[predicted_class_idx])
        
        # Prepare results
        result = {
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, class_probabilities)
            },
            'health_score': self._calculate_health_score(class_probabilities)
        }
        
        return result
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict[str, any]]:
        """
        Predict coral health for multiple images
        
        Args:
            image_paths: List of paths to coral images
            
        Returns:
            List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image_path'] = image_path
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e),
                    'prediction': None,
                    'confidence': 0.0
                })
        
        return results
    
    def _calculate_health_score(self, probabilities: np.ndarray) -> float:
        """
        Calculate overall coral health score (0-100)
        
        Args:
            probabilities: Class probabilities array
            
        Returns:
            Health score between 0-100
        """
        # Weight classes: Healthy=1.0, Unhealthy=0.5, Dead=0.0
        weights = np.array([0.0, 1.0, 0.5])  # [Dead, Healthy, Unhealthy]
        health_score = np.sum(probabilities * weights) * 100
        return float(health_score)
    
    def train(self, 
              train_generator,
              validation_generator,
              epochs: int = 50,
              save_path: str = None):
        """
        Train the coral health classification model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of training epochs
            save_path: Path to save the trained model
            
        Returns:
            Training history
        """
        # Build model if not exists
        if self.model is None:
            self.model = self.build_model()
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if save_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Train model
        logger.info("Starting model training...")
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed!")
        
        return history
    
    def fine_tune(self,
                  train_generator,
                  validation_generator,
                  epochs: int = 20,
                  save_path: str = None):
        """
        Fine-tune the pre-trained model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs: Number of fine-tuning epochs
            save_path: Path to save the fine-tuned model
            
        Returns:
            Fine-tuning history
        """
        if self.model is None:
            raise ValueError("No model to fine-tune. Please train or load a model first.")
        
        # Create fine-tuned model
        self.model = self.create_fine_tuned_model(self.model)
        
        # Setup callbacks for fine-tuning
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-8,
                verbose=1
            )
        ]
        
        if save_path:
            callbacks.append(
                ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Fine-tune model
        logger.info("Starting model fine-tuning...")
        history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Fine-tuning completed!")
        
        return history
    
    def evaluate(self, test_generator) -> Dict[str, any]:
        """
        Evaluate model performance
        
        Args:
            test_generator: Test data generator
            
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("No model to evaluate. Please train or load a model first.")
        
        # Get predictions
        predictions = self.model.predict(test_generator, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Get true labels
        true_classes = test_generator.classes
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        
        # Classification report
        class_report = classification_report(
            true_classes, 
            predicted_classes,
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        
        # Model evaluation
        eval_results = self.model.evaluate(test_generator, verbose=1)
        
        results = {
            'accuracy': accuracy,
            'loss': eval_results[0],
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'model_metrics': {
                'accuracy': eval_results[1],
                'precision': eval_results[2] if len(eval_results) > 2 else None,
                'recall': eval_results[3] if len(eval_results) > 3 else None
            }
        }
        
        return results
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Please train a model first.")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load a pre-trained model
        
        Args:
            filepath: Path to the saved model
        """
        try:
            self.model = keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ValueError(f"Could not load model from {filepath}: {e}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary
        
        Returns:
            Model summary string
        """
        if self.model is None:
            return "No model loaded"
        
        import io
        import contextlib
        
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.model.summary()
        
        return f.getvalue()


def create_data_generator(data_dir: str, 
                         batch_size: int = 32,
                         img_size: Tuple[int, int] = (224, 224),
                         validation_split: float = 0.2,
                         subset: str = 'training'):
    """
    Create data generator for training/validation
    
    Args:
        data_dir: Directory containing coral images
        batch_size: Batch size for training
        img_size: Target image size
        validation_split: Fraction of data to use for validation
        subset: 'training' or 'validation'
        
    Returns:
        Data generator
    """
    if subset == 'training':
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            brightness_range=[0.8, 1.2],
            validation_split=validation_split
        )
    else:
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
    
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset=subset,
        shuffle=True if subset == 'training' else False
    )
    
    return generator


if __name__ == "__main__":
    # Example usage
    classifier = CoralHealthClassifier()
    
    # Build and display model summary
    model = classifier.build_model()
    print(classifier.get_model_summary())