#!/usr/bin/env python3
"""
Complete training pipeline using the real resume dataset.

This script loads the resume dataset, preprocesses the data, trains multiple ML models,
and saves the best performing model for use in the web application.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any
import logging
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config import MODELS_DIR, PERFORMANCE_REQUIREMENTS
from src.logger_setup import get_logger
from src.data_loader import load_and_prepare_data
from src.feature_engineering import TFIDFFeatureExtractor
from src.model_training import MLModelTrainer, create_training_pipeline

# Initialize logger
logger = get_logger(__name__)


def main():
    """Main training pipeline execution."""
    print("🚀 Starting Resume Classification Model Training with Real Data")
    print("=" * 70)
    
    try:
        # Step 1: Load and prepare data
        print("\n📊 Step 1: Loading and Preparing Data")
        print("-" * 40)
        
        start_time = time.time()
        
        X_train_text, X_test_text, y_train, y_test, label_encoder, category_mapping = load_and_prepare_data()
        
        data_load_time = time.time() - start_time
        
        print(f"✅ Data loaded successfully in {data_load_time:.2f}s")
        print(f"   Training samples: {len(X_train_text)}")
        print(f"   Test samples: {len(X_test_text)}")
        print(f"   Categories: {len(category_mapping)}")
        print(f"   Category distribution:")
        
        # Show category distribution
        unique, counts = np.unique(y_train, return_counts=True)
        for label_id, count in zip(unique, counts):
            category_name = category_mapping.get(label_id, f"Unknown_{label_id}")
            print(f"     {category_name}: {count} samples")
        
        # Step 2: Feature extraction
        print("\n🔧 Step 2: Feature Extraction (TF-IDF)")
        print("-" * 40)
        
        start_time = time.time()
        
        # Initialize TF-IDF feature extractor
        feature_extractor = TFIDFFeatureExtractor()
        
        # Fit on training data and transform both train and test
        print("   Fitting TF-IDF vectorizer on training data...")
        feature_extractor.fit_vectorizer(X_train_text.tolist())
        X_train_features = feature_extractor.transform_texts(X_train_text.tolist())
        
        print("   Transforming test data...")
        X_test_features = feature_extractor.transform_texts(X_test_text.tolist())
        
        feature_time = time.time() - start_time
        
        print(f"✅ Feature extraction completed in {feature_time:.2f}s")
        print(f"   Feature matrix shape: {X_train_features.shape}")
        print(f"   Vocabulary size: {feature_extractor.get_vocabulary_size()}")
        print(f"   Feature density: {X_train_features.nnz / (X_train_features.shape[0] * X_train_features.shape[1]):.4f}")
        
        # Save TF-IDF vectorizer
        tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
        feature_extractor.save_vectorizer(tfidf_path)
        print(f"   TF-IDF vectorizer saved to {tfidf_path}")
        
        # Step 3: Model training
        print("\n🤖 Step 3: Training Machine Learning Models")
        print("-" * 40)
        
        start_time = time.time()
        
        # Convert sparse matrices to dense for compatibility
        print("   Converting sparse matrices to dense arrays...")
        X_train_dense = X_train_features.toarray()
        X_test_dense = X_test_features.toarray()
        
        # Train multiple models
        trainer, best_model_name, best_model, best_performance = create_training_pipeline(
            X_train_dense, y_train, X_test_dense, y_test
        )
        
        training_time = time.time() - start_time
        
        print(f"✅ Model training completed in {training_time:.2f}s")
        print(f"   Best model: {best_model_name}")
        print(f"   Best accuracy: {best_performance.accuracy:.4f}")
        print(f"   Best F1-score: {best_performance.f1_score:.4f}")
        
        # Step 4: Save additional components
        print("\n💾 Step 4: Saving Model Components")
        print("-" * 40)
        
        # Save label encoder
        import joblib
        label_encoder_path = MODELS_DIR / "label_encoder.pkl"
        joblib.dump(label_encoder, label_encoder_path)
        print(f"   Label encoder saved to {label_encoder_path}")
        
        # Save category mapping
        import json
        category_mapping_path = MODELS_DIR / "category_mapping.pkl"
        with open(category_mapping_path, 'wb') as f:
            import pickle
            pickle.dump(category_mapping, f)
        print(f"   Category mapping saved to {category_mapping_path}")
        
        # Save training metadata
        metadata = {
            'training_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_size': len(X_train_text) + len(X_test_text),
            'training_samples': len(X_train_text),
            'test_samples': len(X_test_text),
            'num_categories': len(category_mapping),
            'categories': list(category_mapping.values()),
            'feature_count': X_train_features.shape[1],
            'best_model': best_model_name,
            'best_accuracy': float(best_performance.accuracy),
            'best_f1_score': float(best_performance.f1_score),
            'total_training_time': data_load_time + feature_time + training_time,
            'meets_requirements': best_performance.accuracy >= PERFORMANCE_REQUIREMENTS.get('min_accuracy', 0.9)
        }
        
        metadata_path = MODELS_DIR / "training_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"   Training metadata saved to {metadata_path}")
        
        # Step 5: Performance evaluation
        print("\n📈 Step 5: Performance Evaluation")
        print("-" * 40)
        
        # Generate detailed performance report
        report = trainer.generate_performance_report()
        
        # Save performance report
        report_path = MODELS_DIR / "training_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"   Detailed report saved to {report_path}")
        
        # Check performance requirements
        min_accuracy = PERFORMANCE_REQUIREMENTS.get('min_accuracy', 0.9)
        max_processing_time = PERFORMANCE_REQUIREMENTS.get('max_processing_time_per_resume', 5.0)
        
        print(f"\n🎯 Performance Requirements Check:")
        print(f"   Required accuracy: ≥{min_accuracy:.1%}")
        print(f"   Achieved accuracy: {best_performance.accuracy:.1%} {'✅' if best_performance.accuracy >= min_accuracy else '❌'}")
        print(f"   Required processing time: ≤{max_processing_time}s per resume")
        print(f"   Achieved processing time: {best_performance.prediction_time:.3f}s {'✅' if best_performance.prediction_time <= max_processing_time else '❌'}")
        
        # Final summary
        print("\n🎉 Training Pipeline Completed Successfully!")
        print("=" * 70)
        print(f"📊 Dataset: {len(X_train_text) + len(X_test_text)} resumes, {len(category_mapping)} categories")
        print(f"🏆 Best Model: {best_model_name}")
        print(f"📈 Performance: {best_performance.accuracy:.1%} accuracy, {best_performance.f1_score:.3f} F1-score")
        print(f"⏱️  Total Time: {data_load_time + feature_time + training_time:.1f}s")
        print(f"💾 Models saved to: {MODELS_DIR}")
        
        if best_performance.accuracy >= min_accuracy:
            print("\n✅ Model meets performance requirements and is ready for production!")
            print("   You can now run the web application: python run_web_app.py")
        else:
            print(f"\n⚠️  Model accuracy ({best_performance.accuracy:.1%}) is below requirement ({min_accuracy:.1%})")
            print("   Consider collecting more training data or tuning hyperparameters")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training pipeline failed: {e}")
        logger.error(f"Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)