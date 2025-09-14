"""
Machine learning model training pipeline for the Resume Classification NLP System.

This module provides comprehensive ML model training, evaluation, and selection
functionality for resume classification using multiple algorithms.
"""

import sys
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("Required ML libraries not installed. Please run: pip install scikit-learn matplotlib seaborn")
    sys.exit(1)

from config import ML_MODELS_CONFIG, CV_CONFIG, PERFORMANCE_REQUIREMENTS, MODELS_DIR, CATEGORY_MAPPING
from src.logger_setup import get_logger, PerformanceLogger

# Initialize logger
logger = get_logger(__name__)

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class ModelTrainingError(Exception):
    """Custom exception for model training errors."""
    pass


class ModelPerformance:
    """Class to store and manage model performance metrics."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1_score = 0.0
        self.confusion_matrix = None
        self.classification_report = ""
        self.cross_val_scores = []
        self.training_time = 0.0
        self.prediction_time = 0.0
        self.roc_auc = 0.0
        self.feature_importance = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to dictionary."""
        return {
            'model_name': self.model_name,
            'accuracy': float(self.accuracy),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'confusion_matrix': self.confusion_matrix.tolist() if self.confusion_matrix is not None else None,
            'classification_report': self.classification_report,
            'cross_val_scores': [float(score) for score in self.cross_val_scores],
            'cross_val_mean': float(np.mean(self.cross_val_scores)) if len(self.cross_val_scores) > 0 else 0.0,
            'cross_val_std': float(np.std(self.cross_val_scores)) if len(self.cross_val_scores) > 0 else 0.0,
            'training_time': float(self.training_time),
            'prediction_time': float(self.prediction_time),
            'roc_auc': float(self.roc_auc),
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }
    
    def __str__(self) -> str:
        """String representation of performance metrics."""
        return f"""
Model: {self.model_name}
Accuracy: {self.accuracy:.4f}
Precision: {self.precision:.4f}
Recall: {self.recall:.4f}
F1-Score: {self.f1_score:.4f}
Cross-Val Mean: {np.mean(self.cross_val_scores):.4f} ± {np.std(self.cross_val_scores):.4f}
Training Time: {self.training_time:.3f}s
Prediction Time: {self.prediction_time:.3f}s
ROC-AUC: {self.roc_auc:.4f}
"""


class MLModelTrainer:
    """
    Machine learning model trainer with support for multiple algorithms.
    
    This class handles training, evaluation, and selection of the best performing
    classification model for resume categorization.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ML model trainer.
        
        Args:
            config: Optional configuration dictionary for ML models
        """
        self.config = config or ML_MODELS_CONFIG.copy()
        self.models = {}
        self.trained_models = {}
        self.performance_metrics = {}
        self.best_model = None
        self.best_model_name = ""
        self.best_performance = None
        
        logger.info(f"Initialized ML trainer with {len(self.config)} model configurations")
    
    def create_models(self) -> Dict[str, Any]:
        """
        Create ML model instances with configured parameters.
        
        Returns:
            Dictionary of model instances
        """
        try:
            models = {}
            
            # K-Nearest Neighbors
            if 'knn' in self.config:
                models['knn'] = KNeighborsClassifier(**self.config['knn'])
                logger.debug("Created KNN model")
            
            # Logistic Regression
            if 'logistic_regression' in self.config:
                models['logistic_regression'] = LogisticRegression(**self.config['logistic_regression'])
                logger.debug("Created Logistic Regression model")
            
            # Random Forest
            if 'random_forest' in self.config:
                models['random_forest'] = RandomForestClassifier(**self.config['random_forest'])
                logger.debug("Created Random Forest model")
            
            # Support Vector Machine
            if 'svm' in self.config:
                models['svm'] = SVC(**self.config['svm'], probability=True)  # Enable probability for ROC-AUC
                logger.debug("Created SVM model")
            
            # Naive Bayes
            if 'naive_bayes' in self.config:
                models['naive_bayes'] = MultinomialNB(**self.config['naive_bayes'])
                logger.debug("Created Naive Bayes model")
            
            # OneVsRest wrapper for multiclass (optional enhancement)
            if len(models) > 0:
                # Add OneVsRest version of best performing base model
                base_model_name = 'logistic_regression' if 'logistic_regression' in models else list(models.keys())[0]
                models[f'ovr_{base_model_name}'] = OneVsRestClassifier(models[base_model_name])
                logger.debug(f"Created OneVsRest wrapper for {base_model_name}")
            
            self.models = models
            logger.info(f"Created {len(models)} ML models: {list(models.keys())}")
            
            return models
            
        except Exception as e:
            logger.error(f"Failed to create models: {e}")
            raise ModelTrainingError(f"Model creation failed: {e}")
    
    def train_single_model(
        self, 
        model_name: str, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray
    ) -> Tuple[Any, float]:
        """
        Train a single ML model.
        
        Args:
            model_name: Name of the model
            model: Model instance to train
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Tuple of (trained_model, training_time)
        """
        logger.debug(f"Training {model_name} model...")
        
        with PerformanceLogger(logger, f"Training {model_name}"):
            try:
                import time
                start_time = time.time()
                
                # Train the model
                trained_model = model.fit(X_train, y_train)
                
                training_time = time.time() - start_time
                
                logger.info(f"Successfully trained {model_name} in {training_time:.3f}s")
                return trained_model, training_time
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
                raise ModelTrainingError(f"Training failed for {model_name}: {e}")
    
    def evaluate_model(
        self, 
        model_name: str, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray,
        training_time: float
    ) -> ModelPerformance:
        """
        Evaluate a trained model comprehensively.
        
        Args:
            model_name: Name of the model
            model: Trained model instance
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            training_time: Time taken to train the model
            
        Returns:
            ModelPerformance object with all metrics
        """
        logger.debug(f"Evaluating {model_name} model...")
        
        try:
            performance = ModelPerformance(model_name)
            performance.training_time = training_time
            
            # Make predictions
            import time
            start_time = time.time()
            y_pred = model.predict(X_test)
            performance.prediction_time = time.time() - start_time
            
            # Basic metrics
            performance.accuracy = accuracy_score(y_test, y_pred)
            performance.precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            performance.recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            performance.f1_score = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            performance.confusion_matrix = confusion_matrix(y_test, y_pred)
            
            # Classification report
            performance.classification_report = classification_report(y_test, y_pred, zero_division=0)
            
            # Cross-validation scores
            cv_config = CV_CONFIG
            cv = StratifiedKFold(
                n_splits=cv_config.get('cv_folds', 5), 
                shuffle=True, 
                random_state=cv_config.get('random_state', 42)
            )
            
            try:
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv, 
                    scoring=cv_config.get('scoring', 'accuracy'),
                    n_jobs=-1
                )
                performance.cross_val_scores = cv_scores
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model_name}: {e}")
                performance.cross_val_scores = []
            
            # ROC-AUC for multiclass (if possible)
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    n_classes = len(np.unique(y_test))
                    
                    if n_classes > 2:
                        # Multiclass ROC-AUC
                        y_test_binarized = label_binarize(y_test, classes=np.unique(y_train))
                        if y_test_binarized.shape[1] == y_proba.shape[1]:
                            performance.roc_auc = roc_auc_score(y_test_binarized, y_proba, average='weighted', multi_class='ovr')
                    else:
                        # Binary classification
                        performance.roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            except Exception as e:
                logger.warning(f"ROC-AUC calculation failed for {model_name}: {e}")
                performance.roc_auc = 0.0
            
            # Feature importance (if available)
            try:
                if hasattr(model, 'feature_importances_'):
                    performance.feature_importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # For linear models, use coefficient magnitudes
                    coef = model.coef_
                    if coef.ndim > 1:
                        performance.feature_importance = np.mean(np.abs(coef), axis=0)
                    else:
                        performance.feature_importance = np.abs(coef)
            except Exception as e:
                logger.warning(f"Feature importance extraction failed for {model_name}: {e}")
            
            logger.info(f"Model evaluation completed for {model_name}:")
            logger.info(f"  - Accuracy: {performance.accuracy:.4f}")
            logger.info(f"  - F1-Score: {performance.f1_score:.4f}")
            logger.info(f"  - Cross-Val: {np.mean(performance.cross_val_scores):.4f} ± {np.std(performance.cross_val_scores):.4f}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Model evaluation failed for {model_name}: {e}")
            raise ModelTrainingError(f"Evaluation failed for {model_name}: {e}")
    
    def train_multiple_models(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray
    ) -> Dict[str, ModelPerformance]:
        """
        Train and evaluate multiple ML models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of model performances
        """
        if not self.models:
            self.create_models()
        
        logger.info(f"Training {len(self.models)} models...")
        
        with PerformanceLogger(logger, f"Training all {len(self.models)} models"):
            try:
                performances = {}
                
                for model_name, model in self.models.items():
                    try:
                        # Train model
                        trained_model, training_time = self.train_single_model(
                            model_name, model, X_train, y_train
                        )
                        
                        # Store trained model
                        self.trained_models[model_name] = trained_model
                        
                        # Evaluate model
                        performance = self.evaluate_model(
                            model_name, trained_model, X_train, y_train, X_test, y_test, training_time
                        )
                        
                        performances[model_name] = performance
                        self.performance_metrics[model_name] = performance
                        
                    except Exception as e:
                        logger.error(f"Failed to train/evaluate {model_name}: {e}")
                        continue
                
                logger.info(f"Successfully trained {len(performances)} models")
                return performances
                
            except Exception as e:
                logger.error(f"Multiple model training failed: {e}")
                raise ModelTrainingError(f"Multiple model training failed: {e}")
    
    def select_best_model(
        self, 
        performances: Optional[Dict[str, ModelPerformance]] = None,
        selection_criteria: str = 'f1_score'
    ) -> Tuple[str, Any, ModelPerformance]:
        """
        Select the best performing model based on specified criteria.
        
        Args:
            performances: Optional performance metrics (uses self.performance_metrics if None)
            selection_criteria: Metric to use for selection ('accuracy', 'f1_score', 'precision', 'recall')
            
        Returns:
            Tuple of (best_model_name, best_model, best_performance)
        """
        performances = performances or self.performance_metrics
        
        if not performances:
            raise ModelTrainingError("No model performances available for selection")
        
        logger.info(f"Selecting best model based on {selection_criteria}...")
        
        try:
            best_score = -1
            best_model_name = ""
            best_performance = None
            
            # Compare models
            for model_name, performance in performances.items():
                score = getattr(performance, selection_criteria, 0)
                
                # For cross-validation scores, use mean
                if selection_criteria == 'cross_val_scores':
                    score = np.mean(performance.cross_val_scores) if performance.cross_val_scores else 0
                
                logger.debug(f"{model_name} {selection_criteria}: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_model_name = model_name
                    best_performance = performance
            
            if not best_model_name:
                raise ModelTrainingError("No valid model found for selection")
            
            # Get the trained model
            best_model = self.trained_models.get(best_model_name)
            if best_model is None:
                raise ModelTrainingError(f"Trained model not found for {best_model_name}")
            
            # Store best model info
            self.best_model = best_model
            self.best_model_name = best_model_name
            self.best_performance = best_performance
            
            logger.info(f"Best model selected: {best_model_name}")
            logger.info(f"Best {selection_criteria}: {best_score:.4f}")
            
            # Check if performance meets requirements
            min_accuracy = PERFORMANCE_REQUIREMENTS.get('min_accuracy', 0.9)
            if best_performance.accuracy < min_accuracy:
                logger.warning(f"Best model accuracy {best_performance.accuracy:.4f} below requirement {min_accuracy:.4f}")
            
            return best_model_name, best_model, best_performance
            
        except Exception as e:
            logger.error(f"Model selection failed: {e}")
            raise ModelTrainingError(f"Model selection failed: {e}")
    
    def save_model(
        self, 
        model: Any, 
        model_name: str, 
        performance: ModelPerformance,
        path: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Save trained model and performance metrics.
        
        Args:
            model: Trained model to save
            model_name: Name of the model
            performance: Performance metrics
            path: Optional base path (uses MODELS_DIR if None)
            
        Returns:
            Dictionary of saved file paths
        """
        base_path = path or MODELS_DIR
        base_path = Path(base_path)
        base_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        try:
            # Save model
            model_path = base_path / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            saved_paths['model'] = model_path
            
            # Save performance metrics
            performance_path = base_path / f"{model_name}_performance.json"
            with open(performance_path, 'w') as f:
                json.dump(performance.to_dict(), f, indent=2)
            saved_paths['performance'] = performance_path
            
            # Save best model separately if this is the best
            if model_name == self.best_model_name:
                best_model_path = base_path / "best_model.pkl"
                joblib.dump(model, best_model_path)
                saved_paths['best_model'] = best_model_path
                
                best_performance_path = base_path / "model_performance.json"
                with open(best_performance_path, 'w') as f:
                    json.dump(performance.to_dict(), f, indent=2)
                saved_paths['best_performance'] = best_performance_path
            
            logger.info(f"Model {model_name} saved to {base_path}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Failed to save model {model_name}: {e}")
            raise ModelTrainingError(f"Model saving failed: {e}")
    
    def load_model(self, path: Union[str, Path]) -> Tuple[Any, Dict[str, Any]]:
        """
        Load trained model and performance metrics.
        
        Args:
            path: Path to model file or directory
            
        Returns:
            Tuple of (model, performance_dict)
        """
        path = Path(path)
        
        try:
            if path.is_file():
                # Load single model file
                model = joblib.load(path)
                performance_path = path.parent / f"{path.stem}_performance.json"
                
                if performance_path.exists():
                    with open(performance_path, 'r') as f:
                        performance_dict = json.load(f)
                else:
                    performance_dict = {}
                
            else:
                # Load best model from directory
                model_path = path / "best_model.pkl"
                performance_path = path / "model_performance.json"
                
                if not model_path.exists():
                    raise ModelTrainingError(f"Model file not found: {model_path}")
                
                model = joblib.load(model_path)
                
                if performance_path.exists():
                    with open(performance_path, 'r') as f:
                        performance_dict = json.load(f)
                else:
                    performance_dict = {}
            
            logger.info(f"Model loaded from {path}")
            return model, performance_dict
            
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            raise ModelTrainingError(f"Model loading failed: {e}")
    
    def generate_performance_report(
        self, 
        performances: Optional[Dict[str, ModelPerformance]] = None
    ) -> str:
        """
        Generate comprehensive performance report for all models.
        
        Args:
            performances: Optional performance metrics (uses self.performance_metrics if None)
            
        Returns:
            Formatted performance report string
        """
        performances = performances or self.performance_metrics
        
        if not performances:
            return "No model performances available for report generation."
        
        report_lines = [
            "=" * 80,
            "RESUME CLASSIFICATION MODEL PERFORMANCE REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Models Evaluated: {len(performances)}",
            ""
        ]
        
        # Summary table
        report_lines.extend([
            "PERFORMANCE SUMMARY",
            "-" * 40,
            f"{'Model':<20} {'Accuracy':<10} {'F1-Score':<10} {'Precision':<10} {'Recall':<10}"
        ])
        
        for model_name, perf in performances.items():
            report_lines.append(
                f"{model_name:<20} {perf.accuracy:<10.4f} {perf.f1_score:<10.4f} "
                f"{perf.precision:<10.4f} {perf.recall:<10.4f}"
            )
        
        report_lines.append("")
        
        # Best model details
        if self.best_model_name and self.best_performance:
            report_lines.extend([
                "BEST MODEL DETAILS",
                "-" * 40,
                str(self.best_performance),
                ""
            ])
        
        # Individual model details
        report_lines.append("DETAILED RESULTS")
        report_lines.append("-" * 40)
        
        for model_name, perf in performances.items():
            report_lines.extend([
                f"\n{model_name.upper()}:",
                str(perf)
            ])
        
        return "\n".join(report_lines)


def create_training_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[MLModelTrainer, str, Any, ModelPerformance]:
    """
    Create complete model training pipeline.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        config: Optional model configuration
        
    Returns:
        Tuple of (trainer, best_model_name, best_model, best_performance)
    """
    logger.info("Starting complete model training pipeline...")
    
    with PerformanceLogger(logger, "Complete model training pipeline"):
        try:
            # Create trainer
            trainer = MLModelTrainer(config)
            
            # Train all models
            performances = trainer.train_multiple_models(X_train, y_train, X_test, y_test)
            
            # Select best model
            best_model_name, best_model, best_performance = trainer.select_best_model(performances)
            
            # Save best model
            trainer.save_model(best_model, best_model_name, best_performance)
            
            # Generate report
            report = trainer.generate_performance_report(performances)
            logger.info("Training pipeline completed successfully")
            
            return trainer, best_model_name, best_model, best_performance
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise ModelTrainingError(f"Training pipeline failed: {e}")


if __name__ == "__main__":
    # Test the model training functionality
    print("Testing ML model training pipeline...")
    
    # Create sample data for testing
    from sklearn.datasets import make_classification
    
    # Generate synthetic classification data
    X, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_informative=50,
        n_redundant=10,
        n_classes=5,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    try:
        # Test model trainer
        print("\n--- Testing ML Model Trainer ---")
        
        trainer = MLModelTrainer()
        print(f"Created trainer with {len(trainer.config)} model configs")
        
        # Create models
        models = trainer.create_models()
        print(f"Created models: {list(models.keys())}")
        
        # Train models
        performances = trainer.train_multiple_models(X_train, y_train, X_test, y_test)
        print(f"Trained {len(performances)} models successfully")
        
        # Select best model
        best_name, best_model, best_perf = trainer.select_best_model()
        print(f"Best model: {best_name}")
        print(f"Best accuracy: {best_perf.accuracy:.4f}")
        
        # Test complete pipeline
        print("\n--- Testing Complete Pipeline ---")
        
        pipeline_trainer, pipeline_best_name, pipeline_best_model, pipeline_best_perf = create_training_pipeline(
            X_train, y_train, X_test, y_test
        )
        
        print(f"Pipeline best model: {pipeline_best_name}")
        print(f"Pipeline best accuracy: {pipeline_best_perf.accuracy:.4f}")
        
        # Generate report
        print("\n--- Performance Report ---")
        report = trainer.generate_performance_report()
        print(report[:500] + "..." if len(report) > 500 else report)
        
        print("\n✅ ML model training test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()