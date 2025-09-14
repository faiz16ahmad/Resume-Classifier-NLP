"""
Classification service for the Resume Classification NLP System.

This module provides real-time classification of resumes using trained models,
with confidence scoring, batch processing, and result management capabilities.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

try:
    import joblib
    from sklearn.base import BaseEstimator
except ImportError:
    print("Required ML libraries not installed. Please run: pip install scikit-learn")
    sys.exit(1)

from config import CATEGORY_MAPPING, MODELS_DIR, PERFORMANCE_REQUIREMENTS
from src.logger_setup import get_logger, PerformanceLogger
from src.data_preprocessing import clean_text
from src.feature_engineering import TFIDFFeatureExtractor, CategoryLabelEncoder

# Initialize logger
logger = get_logger(__name__)


class ClassificationError(Exception):
    """Custom exception for classification errors."""
    pass


@dataclass
class ClassificationResult:
    """Data class for storing classification results."""
    
    resume_filename: str
    predicted_category: str
    confidence_score: float
    all_probabilities: Dict[str, float]
    processing_timestamp: datetime
    processing_time: float
    raw_text_length: int
    cleaned_text_length: int
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'resume_filename': self.resume_filename,
            'predicted_category': self.predicted_category,
            'confidence_score': float(self.confidence_score),
            'all_probabilities': {k: float(v) for k, v in self.all_probabilities.items()},
            'processing_timestamp': self.processing_timestamp.isoformat(),
            'processing_time': float(self.processing_time),
            'raw_text_length': int(self.raw_text_length),
            'cleaned_text_length': int(self.cleaned_text_length),
            'error_message': self.error_message
        }
    
    def __str__(self) -> str:
        """String representation of classification result."""
        return f"""
Classification Result for: {self.resume_filename}
Predicted Category: {self.predicted_category}
Confidence Score: {self.confidence_score:.4f}
Processing Time: {self.processing_time:.3f}s
Text Length: {self.raw_text_length} -> {self.cleaned_text_length} chars
Timestamp: {self.processing_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""


class ResumeClassifier:
    """
    Resume classification service with model loading and inference capabilities.
    
    This class handles the complete classification pipeline from text input
    to category prediction with confidence scores.
    """
    
    def __init__(self, models_dir: Optional[Path] = None):
        """
        Initialize the resume classifier.
        
        Args:
            models_dir: Optional path to models directory (uses MODELS_DIR if None)
        """
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.model = None
        self.feature_extractor = None
        self.label_encoder = None
        self.category_mapping = None
        self.is_loaded = False
        
        logger.info(f"Initialized ResumeClassifier with models directory: {self.models_dir}")
    
    def load_models(self) -> None:
        """
        Load trained models and preprocessing components.
        
        Raises:
            ClassificationError: If model loading fails
        """
        logger.info("Loading trained models and components...")
        
        with PerformanceLogger(logger, "Model loading"):
            try:
                # Load main classification model
                model_path = self.models_dir / "best_model.pkl"
                if not model_path.exists():
                    raise ClassificationError(f"Model file not found: {model_path}")
                
                self.model = joblib.load(model_path)
                logger.debug(f"Loaded classification model from {model_path}")
                
                # Load TF-IDF feature extractor
                tfidf_path = self.models_dir / "tfidf_vectorizer.pkl"
                if not tfidf_path.exists():
                    raise ClassificationError(f"TF-IDF vectorizer not found: {tfidf_path}")
                
                self.feature_extractor = TFIDFFeatureExtractor()
                self.feature_extractor.load_vectorizer(tfidf_path)
                logger.debug(f"Loaded TF-IDF vectorizer from {tfidf_path}")
                
                # Load label encoder
                label_encoder_path = self.models_dir / "label_encoder.pkl"
                if label_encoder_path.exists():
                    self.label_encoder = joblib.load(label_encoder_path)
                    logger.debug(f"Loaded label encoder from {label_encoder_path}")
                else:
                    # Create default label encoder if not found
                    self.label_encoder = CategoryLabelEncoder()
                    self.label_encoder.fit_categories()
                    logger.debug("Created default label encoder")
                
                # Load category mapping
                category_mapping_path = self.models_dir / "category_mapping.pkl"
                if category_mapping_path.exists():
                    import pickle
                    with open(category_mapping_path, 'rb') as f:
                        self.category_mapping = pickle.load(f)
                else:
                    self.category_mapping = CATEGORY_MAPPING.copy()
                
                logger.debug(f"Loaded category mapping with {len(self.category_mapping)} categories")
                
                self.is_loaded = True
                logger.info("All models and components loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load models: {e}")
                raise ClassificationError(f"Model loading failed: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for classification.
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned and preprocessed text
            
        Raises:
            ClassificationError: If preprocessing fails
        """
        try:
            # Use the text preprocessing pipeline
            cleaned_text = clean_text(
                text,
                remove_urls_flag=True,
                remove_emails_phones_flag=True,
                remove_special_chars_flag=True,
                convert_to_lowercase=True,
                remove_stop_words_flag=True,
                expand_contractions_flag=True
            )
            
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            raise ClassificationError(f"Text preprocessing failed: {e}")
    
    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract TF-IDF features from preprocessed text.
        
        Args:
            text: Preprocessed text
            
        Returns:
            Feature vector as numpy array
            
        Raises:
            ClassificationError: If feature extraction fails
        """
        if not self.feature_extractor or not self.feature_extractor.is_fitted:
            raise ClassificationError("TF-IDF feature extractor not loaded or fitted")
        
        try:
            # Transform text to feature vector
            feature_matrix = self.feature_extractor.transform_texts([text])
            
            # Convert sparse matrix to dense array
            feature_vector = feature_matrix.toarray()[0]
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise ClassificationError(f"Feature extraction failed: {e}")
    
    def predict_category(self, feature_vector: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict category from feature vector.
        
        Args:
            feature_vector: TF-IDF feature vector
            
        Returns:
            Tuple of (predicted_category, confidence_score, all_probabilities)
            
        Raises:
            ClassificationError: If prediction fails
        """
        if not self.model:
            raise ClassificationError("Classification model not loaded")
        
        try:
            # Reshape for single prediction
            features = feature_vector.reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict(features)[0]
            
            # Get prediction probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                
                # Map probabilities to category names
                if self.label_encoder and hasattr(self.label_encoder, 'get_categories'):
                    categories = self.label_encoder.get_categories()
                else:
                    categories = list(self.category_mapping.values())
                
                all_probabilities = {}
                for i, prob in enumerate(probabilities):
                    if i < len(categories):
                        all_probabilities[categories[i]] = float(prob)
                
                # Get confidence score (max probability)
                confidence_score = float(np.max(probabilities))
                
                # Get predicted category name
                if self.label_encoder and hasattr(self.label_encoder, 'inverse_transform_labels'):
                    predicted_category = self.label_encoder.inverse_transform_labels([prediction])[0]
                else:
                    predicted_category = self.category_mapping.get(prediction, f"Category_{prediction}")
                
            else:
                # Fallback for models without probability prediction
                if self.label_encoder and hasattr(self.label_encoder, 'inverse_transform_labels'):
                    predicted_category = self.label_encoder.inverse_transform_labels([prediction])[0]
                else:
                    predicted_category = self.category_mapping.get(prediction, f"Category_{prediction}")
                
                confidence_score = 1.0  # Default confidence
                all_probabilities = {predicted_category: 1.0}
            
            return predicted_category, confidence_score, all_probabilities
            
        except Exception as e:
            logger.error(f"Category prediction failed: {e}")
            raise ClassificationError(f"Category prediction failed: {e}")
    
    def classify_resume(
        self, 
        text: str, 
        filename: str = "unknown_resume"
    ) -> ClassificationResult:
        """
        Classify a single resume text.
        
        Args:
            text: Raw resume text
            filename: Resume filename for identification
            
        Returns:
            ClassificationResult object with prediction details
            
        Raises:
            ClassificationError: If classification fails
        """
        if not self.is_loaded:
            self.load_models()
        
        start_time = time.time()
        processing_timestamp = datetime.now()
        
        logger.debug(f"Classifying resume: {filename}")
        
        try:
            # Store original text length
            raw_text_length = len(text)
            
            # Preprocess text
            cleaned_text = self.preprocess_text(text)
            cleaned_text_length = len(cleaned_text)
            
            # Extract features
            feature_vector = self.extract_features(cleaned_text)
            
            # Predict category
            predicted_category, confidence_score, all_probabilities = self.predict_category(feature_vector)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create result object
            result = ClassificationResult(
                resume_filename=filename,
                predicted_category=predicted_category,
                confidence_score=confidence_score,
                all_probabilities=all_probabilities,
                processing_timestamp=processing_timestamp,
                processing_time=processing_time,
                raw_text_length=raw_text_length,
                cleaned_text_length=cleaned_text_length
            )
            
            logger.info(f"Successfully classified {filename}: {predicted_category} ({confidence_score:.3f})")
            
            # Check performance requirements
            max_processing_time = PERFORMANCE_REQUIREMENTS.get('max_processing_time_per_resume', 5.0)
            if processing_time > max_processing_time:
                logger.warning(f"Processing time {processing_time:.3f}s exceeds requirement {max_processing_time}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Classification failed for {filename}: {e}")
            
            # Return error result
            return ClassificationResult(
                resume_filename=filename,
                predicted_category="ERROR",
                confidence_score=0.0,
                all_probabilities={},
                processing_timestamp=processing_timestamp,
                processing_time=processing_time,
                raw_text_length=len(text) if text else 0,
                cleaned_text_length=0,
                error_message=str(e)
            )
    
    def batch_classify(
        self, 
        texts: List[str], 
        filenames: Optional[List[str]] = None,
        progress_callback: Optional[callable] = None
    ) -> List[ClassificationResult]:
        """
        Classify multiple resume texts in batch.
        
        Args:
            texts: List of raw resume texts
            filenames: Optional list of filenames (generates if None)
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of ClassificationResult objects
        """
        if not self.is_loaded:
            self.load_models()
        
        if filenames is None:
            filenames = [f"resume_{i+1}.txt" for i in range(len(texts))]
        
        if len(texts) != len(filenames):
            raise ClassificationError("Number of texts and filenames must match")
        
        logger.info(f"Starting batch classification of {len(texts)} resumes")
        
        results = []
        successful = 0
        failed = 0
        
        with PerformanceLogger(logger, f"Batch classification of {len(texts)} resumes"):
            for i, (text, filename) in enumerate(zip(texts, filenames)):
                try:
                    result = self.classify_resume(text, filename)
                    results.append(result)
                    
                    if result.error_message is None:
                        successful += 1
                    else:
                        failed += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        try:
                            progress_callback(i + 1, len(texts), filename, result)
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")
                
                except Exception as e:
                    logger.error(f"Batch classification error for {filename}: {e}")
                    failed += 1
                    
                    # Create error result
                    error_result = ClassificationResult(
                        resume_filename=filename,
                        predicted_category="ERROR",
                        confidence_score=0.0,
                        all_probabilities={},
                        processing_timestamp=datetime.now(),
                        processing_time=0.0,
                        raw_text_length=len(text) if text else 0,
                        cleaned_text_length=0,
                        error_message=str(e)
                    )
                    results.append(error_result)
        
        # Log batch results
        error_rate = failed / len(texts) if len(texts) > 0 else 0
        logger.info(f"Batch classification completed: {successful} successful, {failed} failed")
        
        max_error_rate = PERFORMANCE_REQUIREMENTS.get('max_error_rate', 0.05)
        if error_rate > max_error_rate:
            logger.warning(f"Error rate {error_rate:.2%} exceeds threshold {max_error_rate:.2%}")
        
        return results
    
    def get_category_statistics(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """
        Generate statistics from classification results.
        
        Args:
            results: List of classification results
            
        Returns:
            Dictionary with classification statistics
        """
        if not results:
            return {}
        
        try:
            # Filter successful results
            successful_results = [r for r in results if r.error_message is None]
            
            # Category distribution
            category_counts = {}
            confidence_scores = []
            processing_times = []
            
            for result in successful_results:
                category = result.predicted_category
                category_counts[category] = category_counts.get(category, 0) + 1
                confidence_scores.append(result.confidence_score)
                processing_times.append(result.processing_time)
            
            # Calculate statistics
            stats = {
                'total_resumes': len(results),
                'successful_classifications': len(successful_results),
                'failed_classifications': len(results) - len(successful_results),
                'error_rate': (len(results) - len(successful_results)) / len(results),
                'category_distribution': category_counts,
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'min_confidence': np.min(confidence_scores) if confidence_scores else 0.0,
                'max_confidence': np.max(confidence_scores) if confidence_scores else 0.0,
                'average_processing_time': np.mean(processing_times) if processing_times else 0.0,
                'total_processing_time': np.sum(processing_times) if processing_times else 0.0,
                'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
                'unique_categories': len(category_counts)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {'error': str(e)}


def create_classification_service(models_dir: Optional[Path] = None) -> ResumeClassifier:
    """
    Create and initialize a classification service.
    
    Args:
        models_dir: Optional path to models directory
        
    Returns:
        Initialized ResumeClassifier instance
    """
    try:
        classifier = ResumeClassifier(models_dir)
        classifier.load_models()
        
        logger.info("Classification service created and initialized successfully")
        return classifier
        
    except Exception as e:
        logger.error(f"Failed to create classification service: {e}")
        raise ClassificationError(f"Service creation failed: {e}")


if __name__ == "__main__":
    # Test the classification service
    print("Testing Resume Classification Service...")
    
    # Sample resume texts for testing
    sample_resumes = [
        {
            'filename': 'python_developer.txt',
            'text': """
            John Doe - Senior Python Developer
            
            EXPERIENCE:
            5+ years of experience in Python development, machine learning, and data science.
            Proficient in Django, Flask, TensorFlow, PyTorch, pandas, NumPy, and scikit-learn.
            Built REST APIs and web applications using Python frameworks.
            Experience with data analysis, statistical modeling, and ML algorithms.
            
            SKILLS:
            Programming: Python, SQL, JavaScript
            ML/AI: TensorFlow, PyTorch, scikit-learn, Keras
            Web: Django, Flask, FastAPI
            Databases: PostgreSQL, MongoDB
            """
        },
        {
            'filename': 'java_developer.txt',
            'text': """
            Jane Smith - Java Software Engineer
            
            PROFESSIONAL SUMMARY:
            Experienced Java developer with 6+ years in enterprise application development.
            Strong expertise in Spring Framework, Spring Boot, Hibernate, and microservices.
            Built scalable web applications and RESTful services using Java technologies.
            
            TECHNICAL SKILLS:
            Languages: Java, SQL, JavaScript
            Frameworks: Spring, Spring Boot, Hibernate
            Tools: Maven, Gradle, Jenkins, Docker
            Databases: MySQL, Oracle, PostgreSQL
            """
        },
        {
            'filename': 'data_scientist.txt',
            'text': """
            Alex Johnson - Data Scientist
            
            SUMMARY:
            PhD in Statistics with 4+ years of experience in data science and analytics.
            Expert in machine learning, statistical modeling, and big data processing.
            Proficient in Python, R, SQL, and various ML libraries and frameworks.
            
            EXPERTISE:
            Languages: Python, R, SQL, Scala
            ML/Stats: scikit-learn, TensorFlow, PyTorch, pandas, NumPy
            Big Data: Spark, Hadoop, Kafka
            Visualization: Matplotlib, Seaborn, Plotly, Tableau
            """
        }
    ]
    
    try:
        # Test classification service creation
        print("\\n--- Testing Classification Service Creation ---")
        
        # Check if models exist
        models_dir = Path("models")
        if not models_dir.exists() or not (models_dir / "best_model.pkl").exists():
            print("⚠️  No trained models found. Please run model training first.")
            print("   This test will demonstrate the service structure without actual classification.")
            
            # Create mock classifier for structure testing
            classifier = ResumeClassifier(models_dir)
            print(f"✅ Created classifier instance (models not loaded)")
            
        else:
            # Test with actual models
            classifier = create_classification_service()
            print(f"✅ Classification service created and loaded successfully")
            
            # Test single classification
            print("\\n--- Testing Single Resume Classification ---")
            
            for resume in sample_resumes:
                result = classifier.classify_resume(resume['text'], resume['filename'])
                print(f"\\n📄 {resume['filename']}:")
                print(f"   Predicted: {result.predicted_category}")
                print(f"   Confidence: {result.confidence_score:.3f}")
                print(f"   Processing Time: {result.processing_time:.3f}s")
                
                if result.error_message:
                    print(f"   ❌ Error: {result.error_message}")
            
            # Test batch classification
            print("\\n--- Testing Batch Classification ---")
            
            texts = [r['text'] for r in sample_resumes]
            filenames = [r['filename'] for r in sample_resumes]
            
            def progress_callback(current, total, filename, result):
                print(f"   Progress: {current}/{total} - {filename} -> {result.predicted_category}")
            
            batch_results = classifier.batch_classify(texts, filenames, progress_callback)
            
            # Test statistics
            print("\\n--- Classification Statistics ---")
            
            stats = classifier.get_category_statistics(batch_results)
            print(f"   Total Resumes: {stats.get('total_resumes', 0)}")
            print(f"   Successful: {stats.get('successful_classifications', 0)}")
            print(f"   Average Confidence: {stats.get('average_confidence', 0):.3f}")
            print(f"   Average Processing Time: {stats.get('average_processing_time', 0):.3f}s")
            print(f"   Category Distribution: {stats.get('category_distribution', {})}")
        
        print("\\n✅ Classification service test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()