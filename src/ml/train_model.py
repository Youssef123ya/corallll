"""
Training Script for Coral Health Classification Model
Comprehensive training pipeline with data loading, augmentation, and model evaluation
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from coral_classifier import CoralHealthClassifier, create_data_generator
from data_analysis import CoralDataAnalyzer

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../../logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CoralModelTrainer:
    """
    Comprehensive training pipeline for coral health classification
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trainer with configuration
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.classifier = CoralHealthClassifier()
        self.analyzer = CoralDataAnalyzer()
        
        # Create output directories
        self.model_dir = Path(config['model_dir'])
        self.model_dir.mkdir(exist_ok=True)
        
        self.results_dir = Path(config['results_dir']) 
        self.results_dir.mkdir(exist_ok=True)
        
        self.plots_dir = Path(config['plots_dir'])
        self.plots_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized trainer with config: {config}")
    
    def prepare_data(self) -> Tuple[tf.keras.utils.Sequence, tf.keras.utils.Sequence, tf.keras.utils.Sequence]:
        """
        Prepare training, validation, and test data
        
        Returns:
            Tuple of (train_generator, val_generator, test_generator)
        """
        logger.info("Preparing data generators...")
        
        data_dir = self.config['data_dir']
        batch_size = self.config['batch_size']
        img_size = tuple(self.config['img_size'])
        
        # Create training generator with augmentation
        train_gen = create_data_generator(
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            validation_split=0.2,
            subset='training'
        )
        
        # Create validation generator
        val_gen = create_data_generator(
            data_dir=data_dir,
            batch_size=batch_size,
            img_size=img_size,
            validation_split=0.2,
            subset='validation'
        )
        
        # For test set, use a separate directory if available, otherwise split validation
        test_dir = self.config.get('test_dir', data_dir)
        test_gen = create_data_generator(
            data_dir=test_dir,
            batch_size=batch_size,
            img_size=img_size,
            validation_split=0.1,
            subset='validation'  # Use small validation split as test
        )
        
        # Log dataset info
        logger.info(f"Training samples: {train_gen.samples}")
        logger.info(f"Validation samples: {val_gen.samples}")
        logger.info(f"Test samples: {test_gen.samples}")
        logger.info(f"Class indices: {train_gen.class_indices}")
        
        return train_gen, val_gen, test_gen
    
    def train_model(self, 
                   train_gen: tf.keras.utils.Sequence,
                   val_gen: tf.keras.utils.Sequence) -> tf.keras.callbacks.History:
        """
        Train the coral health classification model
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            
        Returns:
            Training history
        """
        logger.info("Starting model training...")
        
        # Build model
        model = self.classifier.build_model()
        
        # Setup callbacks
        callbacks = self._create_callbacks()
        
        # Train model
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=self.config['epochs'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Update classifier with trained model
        self.classifier.model = model
        
        logger.info("Model training completed!")
        return history
    
    def fine_tune_model(self,
                       train_gen: tf.keras.utils.Sequence,
                       val_gen: tf.keras.utils.Sequence) -> tf.keras.callbacks.History:
        """
        Fine-tune the pre-trained model
        
        Args:
            train_gen: Training data generator
            val_gen: Validation data generator
            
        Returns:
            Fine-tuning history
        """
        if not self.config.get('fine_tune', False):
            logger.info("Fine-tuning disabled in config")
            return None
        
        logger.info("Starting model fine-tuning...")
        
        # Fine-tune model
        history = self.classifier.fine_tune(
            train_gen,
            val_gen,
            epochs=self.config['fine_tune_epochs'],
            save_path=str(self.model_dir / 'coral_classifier_fine_tuned.h5')
        )
        
        logger.info("Model fine-tuning completed!")
        return history
    
    def evaluate_model(self, test_gen: tf.keras.utils.Sequence) -> Dict:
        """
        Evaluate trained model on test set
        
        Args:
            test_gen: Test data generator
            
        Returns:
            Evaluation results dictionary
        """
        logger.info("Evaluating model on test set...")
        
        # Get model evaluation
        results = self.classifier.evaluate(test_gen)
        
        # Log results
        logger.info(f"Test Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Test Loss: {results['loss']:.4f}")
        
        # Print classification report
        if 'classification_report' in results:
            logger.info("Classification Report:")
            for class_name, metrics in results['classification_report'].items():
                if isinstance(metrics, dict):
                    logger.info(f"{class_name}: Precision={metrics.get('precision', 0):.3f}, "
                              f"Recall={metrics.get('recall', 0):.3f}, "
                              f"F1-Score={metrics.get('f1-score', 0):.3f}")
        
        return results
    
    def _create_callbacks(self) -> list:
        """Create training callbacks"""
        callbacks = []
        
        # Early stopping
        if self.config.get('early_stopping', True):
            callbacks.append(
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('patience', 10),
                    restore_best_weights=True,
                    verbose=1
                )
            )
        
        # Learning rate reduction
        if self.config.get('reduce_lr', True):
            callbacks.append(
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.get('lr_factor', 0.2),
                    patience=self.config.get('lr_patience', 5),
                    min_lr=1e-7,
                    verbose=1
                )
            )
        
        # Model checkpointing
        checkpoint_path = str(self.model_dir / 'best_coral_classifier.h5')
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        )
        
        return callbacks
    
    def save_results(self, 
                    history: tf.keras.callbacks.History,
                    evaluation: Dict,
                    fine_tune_history: Optional[tf.keras.callbacks.History] = None):
        """
        Save training results and generate reports
        
        Args:
            history: Training history
            evaluation: Evaluation results
            fine_tune_history: Fine-tuning history (optional)
        """
        logger.info("Saving training results...")
        
        # Save training history
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            history_dict = {}
            for key, values in history.history.items():
                history_dict[key] = [float(v) for v in values]
            json.dump(history_dict, f, indent=2)
        
        # Save evaluation results
        eval_path = self.results_dir / 'evaluation_results.json'
        with open(eval_path, 'w') as f:
            # Handle numpy arrays in evaluation results
            eval_dict = {}
            for key, value in evaluation.items():
                if isinstance(value, np.ndarray):
                    eval_dict[key] = value.tolist()
                else:
                    eval_dict[key] = value
            json.dump(eval_dict, f, indent=2)
        
        # Generate and save plots
        self._create_training_plots(history, fine_tune_history)
        self._create_evaluation_plots(evaluation)
        
        # Generate summary report
        self._generate_training_report(history, evaluation)
        
        logger.info(f"Results saved to {self.results_dir}")
    
    def _create_training_plots(self, 
                              history: tf.keras.callbacks.History,
                              fine_tune_history: Optional[tf.keras.callbacks.History] = None):
        """Create and save training plots"""
        
        # Training history plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss plot
        axes[0, 0].plot(history.history['loss'], label='Training Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy plot
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate plot (if available)
        if 'lr' in history.history:
            axes[1, 0].plot(history.history['lr'], label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Fine-tuning history (if available)
        if fine_tune_history:
            axes[1, 1].plot(fine_tune_history.history['loss'], label='Fine-tune Loss')
            axes[1, 1].plot(fine_tune_history.history['val_loss'], label='Fine-tune Val Loss')
            axes[1, 1].set_title('Fine-tuning Loss')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_evaluation_plots(self, evaluation: Dict):
        """Create and save evaluation plots"""
        
        # Confusion matrix plot
        if 'confusion_matrix' in evaluation:
            plt.figure(figsize=(10, 8))
            
            cm = np.array(evaluation['confusion_matrix'])
            class_names = self.classifier.class_names
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                square=True,
                cbar_kws={'shrink': 0.8}
            )
            
            plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Classification metrics plot
        if 'classification_report' in evaluation:
            report = evaluation['classification_report']
            
            # Extract metrics for each class
            classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
            metrics = ['precision', 'recall', 'f1-score']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(classes))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [report[cls][metric] for cls in classes if isinstance(report[cls], dict)]
                ax.bar(x + i * width, values, width, label=metric.title())
            
            ax.set_xlabel('Classes')
            ax.set_ylabel('Score')
            ax.set_title('Classification Metrics by Class')
            ax.set_xticks(x + width)
            ax.set_xticklabels(classes)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'classification_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_training_report(self, 
                                 history: tf.keras.callbacks.History,
                                 evaluation: Dict):
        """Generate comprehensive training report"""
        
        report_path = self.results_dir / 'training_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"""# Coral Health Classification Model Training Report

## Training Configuration
- **Model Architecture**: EfficientNetB0 with custom head
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Epochs**: {self.config['epochs']}
- **Batch Size**: {self.config['batch_size']}
- **Image Size**: {self.config['img_size']}
- **Learning Rate**: {self.config.get('learning_rate', 'Default')}

## Dataset Information
- **Data Directory**: {self.config['data_dir']}
- **Classes**: {', '.join(self.classifier.class_names)}
- **Data Augmentation**: Enabled

## Training Results

### Final Metrics
- **Training Accuracy**: {history.history['accuracy'][-1]:.4f}
- **Validation Accuracy**: {history.history['val_accuracy'][-1]:.4f}
- **Training Loss**: {history.history['loss'][-1]:.4f}
- **Validation Loss**: {history.history['val_loss'][-1]:.4f}

### Test Set Evaluation
- **Test Accuracy**: {evaluation['accuracy']:.4f}
- **Test Loss**: {evaluation['loss']:.4f}

### Model Performance by Class
""")
            
            if 'classification_report' in evaluation:
                report = evaluation['classification_report']
                f.write("\n| Class | Precision | Recall | F1-Score | Support |\n")
                f.write("|-------|-----------|--------|----------|----------|\n")
                
                for class_name in self.classifier.class_names:
                    if class_name in report and isinstance(report[class_name], dict):
                        metrics = report[class_name]
                        f.write(f"| {class_name} | {metrics['precision']:.3f} | "
                               f"{metrics['recall']:.3f} | {metrics['f1-score']:.3f} | "
                               f"{metrics['support']} |\n")
                
                # Add overall metrics
                if 'weighted avg' in report:
                    metrics = report['weighted avg']
                    f.write(f"| **Weighted Avg** | {metrics['precision']:.3f} | "
                           f"{metrics['recall']:.3f} | {metrics['f1-score']:.3f} | "
                           f"{metrics['support']} |\n")
            
            f.write(f"""
## Model Architecture
```
{self.classifier.get_model_summary()}
```

## Files Generated
- `best_coral_classifier.h5` - Best model checkpoint
- `training_history.json` - Training metrics history
- `evaluation_results.json` - Test set evaluation results
- `training_history.png` - Training curves visualization
- `confusion_matrix.png` - Confusion matrix heatmap
- `classification_metrics.png` - Per-class metrics chart

## Recommendations

### Model Performance
- {'✓ Excellent' if evaluation['accuracy'] > 0.9 else '⚠ Good' if evaluation['accuracy'] > 0.8 else '✗ Needs Improvement'} test accuracy ({evaluation['accuracy']:.1%})
- {'✓ Well converged' if abs(history.history['accuracy'][-1] - history.history['val_accuracy'][-1]) < 0.05 else '⚠ Some overfitting detected'}

### Next Steps
1. {'Deploy model for production use' if evaluation['accuracy'] > 0.9 else 'Consider additional training or data augmentation'}
2. Monitor performance on new coral images
3. Collect additional training data for underperforming classes
4. Consider ensemble methods for improved accuracy

---
*Report generated automatically by Coral Health Classification training pipeline*
""")
        
        logger.info(f"Training report saved to {report_path}")


def create_training_config(args) -> Dict:
    """Create training configuration from arguments"""
    
    config = {
        # Data configuration
        'data_dir': args.data_dir,
        'test_dir': args.test_dir,
        
        # Model configuration
        'img_size': [args.img_height, args.img_width],
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        
        # Training configuration
        'fine_tune': args.fine_tune,
        'fine_tune_epochs': args.fine_tune_epochs,
        'early_stopping': args.early_stopping,
        'patience': args.patience,
        'reduce_lr': args.reduce_lr,
        'lr_factor': args.lr_factor,
        'lr_patience': args.lr_patience,
        
        # Output directories
        'model_dir': args.model_dir,
        'results_dir': args.results_dir,
        'plots_dir': args.plots_dir
    }
    
    return config


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description='Train Coral Health Classification Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing training data')
    parser.add_argument('--test_dir', type=str, default=None,
                       help='Directory containing test data (optional)')
    
    # Model arguments
    parser.add_argument('--img_height', type=int, default=224,
                       help='Input image height')
    parser.add_argument('--img_width', type=int, default=224,
                       help='Input image width')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    
    # Training arguments
    parser.add_argument('--fine_tune', action='store_true',
                       help='Enable fine-tuning after initial training')
    parser.add_argument('--fine_tune_epochs', type=int, default=20,
                       help='Number of fine-tuning epochs')
    parser.add_argument('--early_stopping', action='store_true', default=True,
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--reduce_lr', action='store_true', default=True,
                       help='Enable learning rate reduction')
    parser.add_argument('--lr_factor', type=float, default=0.2,
                       help='Learning rate reduction factor')
    parser.add_argument('--lr_patience', type=int, default=5,
                       help='Learning rate reduction patience')
    
    # Output arguments
    parser.add_argument('--model_dir', type=str, default='../../models',
                       help='Directory to save trained models')
    parser.add_argument('--results_dir', type=str, default='../../results',
                       help='Directory to save training results')
    parser.add_argument('--plots_dir', type=str, default='../../plots',
                       help='Directory to save training plots')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_training_config(args)
    
    # Initialize trainer
    trainer = CoralModelTrainer(config)
    
    try:
        # Prepare data
        train_gen, val_gen, test_gen = trainer.prepare_data()
        
        # Train model
        history = trainer.train_model(train_gen, val_gen)
        
        # Fine-tune model (if enabled)
        fine_tune_history = trainer.fine_tune_model(train_gen, val_gen)
        
        # Evaluate model
        evaluation = trainer.evaluate_model(test_gen)
        
        # Save results
        trainer.save_results(history, evaluation, fine_tune_history)
        
        # Save final model
        final_model_path = trainer.model_dir / 'coral_classifier_final.h5'
        trainer.classifier.save_model(str(final_model_path))
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Final model saved to: {final_model_path}")
        logger.info(f"Test accuracy: {evaluation['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()