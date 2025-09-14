"""
Feature engineering module for the Resume Classification NLP System.

This module provides TF-IDF vectorization and feature extraction functionality
for converting preprocessed text data into numerical features suitable for
machine learning algorithms.
"""

import sys
import pickle
import joblib
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import logging
import numpy as np
import scipy.sparse

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
except ImportError:
    print("Required ML libraries not installed. Please run: pip install scikit-learn pandas")
    sys.exit(1)

from config import TFIDF_CONFIG, MODELS_DIR, CATEGORY_MAPPING, PERFORMANCE_REQUIREMENTS
from src.logger_setup import get_logger, PerformanceLogger

# Initialize logger
logger = get_logger(__name__)


class FeatureEngineeringError(Exception):
    """Custom exception for feature engineering errors."""
    pass


class TFIDFFeatureExtractor:
    """
    TF-IDF feature extraction class with persistence and validation capabilities.
    
    This class handles the creation, training, and persistence of TF-IDF vectorizers
    for converting text data into numerical features suitable for ML models.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TF-IDF feature extractor.
        
        Args:
            config: Optional configuration dictionary for TF-IDF parameters
        """
        self.config = config or TFIDF_CONFIG.copy()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
        
        logger.info(f"Initialized TF-IDF extractor with config: {self.config}")
    
    def create_tfidf_vectorizer(self) -> TfidfVectorizer:
        """
        Create TF-IDF vectorizer with optimal parameters.
        
        Returns:
            Configured TfidfVectorizer instance
        """
        try:
            vectorizer = TfidfVectorizer(
                max_features=self.config.get("max_features", 5000),
                ngram_range=self.config.get("ngram_range", (1, 2)),
                min_df=self.config.get("min_df", 2),
                max_df=self.config.get("max_df", 0.95),
                stop_words=self.config.get("stop_words", "english"),
                lowercase=self.config.get("lowercase", True),
                strip_accents=self.config.get("strip_accents", "unicode"),
                token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b',  # Only words starting with letters
                sublinear_tf=True,  # Apply sublinear tf scaling
                use_idf=True,  # Enable inverse document frequency reweighting
                smooth_idf=True,  # Smooth idf weights
                norm='l2'  # Apply L2 normalization
            )
            
            logger.info("TF-IDF vectorizer created successfully")
            return vectorizer
            
        except Exception as e:
            logger.error(f"Failed to create TF-IDF vectorizer: {e}")
            raise FeatureEngineeringError(f"Vectorizer creation failed: {e}")
    
    def fit_vectorizer(self, texts: List[str]) -> TfidfVectorizer:
        """
        Train TF-IDF vectorizer on text corpus.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            Fitted TfidfVectorizer instance
            
        Raises:
            FeatureEngineeringError: If fitting fails
        """
        if not texts or not isinstance(texts, list):
            raise FeatureEngineeringError("Invalid input: texts must be a non-empty list")
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text and isinstance(text, str) and len(text.strip()) > 0]
        
        if not valid_texts:
            raise FeatureEngineeringError("No valid texts found for training")
        
        if len(valid_texts) < 2:
            raise FeatureEngineeringError("At least 2 valid texts required for TF-IDF training")
        
        with PerformanceLogger(logger, f"TF-IDF fitting on {len(valid_texts)} texts"):
            try:
                # Create vectorizer if not exists
                if self.vectorizer is None:
                    self.vectorizer = self.create_tfidf_vectorizer()
                
                # Fit the vectorizer
                self.vectorizer.fit(valid_texts)
                
                # Store feature names
                self.feature_names = self.vectorizer.get_feature_names_out().tolist()
                self.is_fitted = True
                
                # Log statistics
                vocab_size = len(self.vectorizer.vocabulary_)
                feature_count = len(self.feature_names)
                
                logger.info(f"TF-IDF vectorizer fitted successfully:")
                logger.info(f"  - Vocabulary size: {vocab_size}")
                logger.info(f"  - Feature count: {feature_count}")
                logger.info(f"  - Training documents: {len(valid_texts)}")
                
                return self.vectorizer
                
            except Exception as e:
                logger.error(f"TF-IDF fitting failed: {e}")
                raise FeatureEngineeringError(f"Vectorizer fitting failed: {e}")
    
    def transform_texts(
        self, 
        texts: List[str], 
        vectorizer: Optional[TfidfVectorizer] = None
    ) -> scipy.sparse.csr_matrix:
        """
        Transform texts to TF-IDF feature vectors.
        
        Args:
            texts: List of preprocessed text documents
            vectorizer: Optional pre-fitted vectorizer (uses self.vectorizer if None)
            
        Returns:
            Sparse matrix of TF-IDF features
            
        Raises:
            FeatureEngineeringError: If transformation fails
        """
        if not texts or not isinstance(texts, list):
            raise FeatureEngineeringError("Invalid input: texts must be a non-empty list")
        
        # Use provided vectorizer or instance vectorizer
        vec = vectorizer or self.vectorizer
        
        if vec is None:
            raise FeatureEngineeringError("No fitted vectorizer available")
        
        # Check if vectorizer is fitted
        if not hasattr(vec, 'vocabulary_') or not vec.vocabulary_:
            raise FeatureEngineeringError("Vectorizer is not fitted")
        
        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []
        
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and len(text.strip()) > 0:
                valid_texts.append(text)
                valid_indices.append(i)
        
        if not valid_texts:
            raise FeatureEngineeringError("No valid texts found for transformation")
        
        with PerformanceLogger(logger, f"TF-IDF transformation of {len(valid_texts)} texts"):
            try:
                # Transform texts
                feature_matrix = vec.transform(valid_texts)
                
                # If some texts were invalid, create full matrix with zeros for invalid texts
                if len(valid_texts) < len(texts):
                    full_matrix = scipy.sparse.csr_matrix(
                        (len(texts), feature_matrix.shape[1]), 
                        dtype=feature_matrix.dtype
                    )
                    full_matrix[valid_indices] = feature_matrix
                    feature_matrix = full_matrix
                
                logger.info(f"TF-IDF transformation completed:")
                logger.info(f"  - Input texts: {len(texts)}")
                logger.info(f"  - Valid texts: {len(valid_texts)}")
                logger.info(f"  - Feature matrix shape: {feature_matrix.shape}")
                logger.info(f"  - Matrix sparsity: {1 - feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1]):.3f}")
                
                return feature_matrix
                
            except Exception as e:
                logger.error(f"TF-IDF transformation failed: {e}")
                raise FeatureEngineeringError(f"Text transformation failed: {e}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from fitted vectorizer.
        
        Returns:
            List of feature names
            
        Raises:
            FeatureEngineeringError: If vectorizer is not fitted
        """
        if not self.is_fitted or self.feature_names is None:
            raise FeatureEngineeringError("Vectorizer is not fitted")
        
        return self.feature_names.copy()
    
    def get_vocabulary_size(self) -> int:
        """
        Get vocabulary size from fitted vectorizer.
        
        Returns:
            Size of vocabulary
        """
        if not self.is_fitted or self.vectorizer is None:
            return 0
        
        return len(self.vectorizer.vocabulary_)
    
    def save_vectorizer(self, path: Union[str, Path]) -> None:
        """
        Save fitted vectorizer to disk.
        
        Args:
            path: Path to save the vectorizer
            
        Raises:
            FeatureEngineeringError: If saving fails
        """
        if not self.is_fitted or self.vectorizer is None:
            raise FeatureEngineeringError("No fitted vectorizer to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save vectorizer and metadata
            save_data = {
                'vectorizer': self.vectorizer,
                'feature_names': self.feature_names,
                'config': self.config,
                'is_fitted': self.is_fitted,
                'vocabulary_size': self.get_vocabulary_size()
            }
            
            # Use joblib for better sklearn object serialization
            joblib.dump(save_data, path)
            
            logger.info(f"TF-IDF vectorizer saved to: {path}")
            
        except Exception as e:
            logger.error(f"Failed to save vectorizer: {e}")
            raise FeatureEngineeringError(f"Vectorizer saving failed: {e}")
    
    def load_vectorizer(self, path: Union[str, Path]) -> TfidfVectorizer:
        """
        Load fitted vectorizer from disk.
        
        Args:
            path: Path to load the vectorizer from
            
        Returns:
            Loaded TfidfVectorizer instance
            
        Raises:
            FeatureEngineeringError: If loading fails
        """
        path = Path(path)
        
        if not path.exists():
            raise FeatureEngineeringError(f"Vectorizer file not found: {path}")
        
        try:
            # Load vectorizer and metadata
            save_data = joblib.load(path)
            
            self.vectorizer = save_data['vectorizer']
            self.feature_names = save_data['feature_names']
            self.config = save_data.get('config', self.config)
            self.is_fitted = save_data.get('is_fitted', True)
            
            logger.info(f"TF-IDF vectorizer loaded from: {path}")
            logger.info(f"  - Vocabulary size: {save_data.get('vocabulary_size', 'unknown')}")
            logger.info(f"  - Feature count: {len(self.feature_names) if self.feature_names else 'unknown'}")
            
            return self.vectorizer
            
        except Exception as e:
            logger.error(f"Failed to load vectorizer: {e}")
            raise FeatureEngineeringError(f"Vectorizer loading failed: {e}")


class CategoryLabelEncoder:
    """
    Label encoder for job categories with persistence capabilities.
    """
    
    def __init__(self):
        """Initialize label encoder."""
        from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
        self.encoder = SklearnLabelEncoder()
        self.category_mapping = CATEGORY_MAPPING.copy()
        self.is_fitted = False
        
    def fit_categories(self, categories: Optional[List[str]] = None) -> 'CategoryLabelEncoder':
        """
        Fit label encoder on job categories.
        
        Args:
            categories: Optional list of categories (uses CATEGORY_MAPPING if None)
            
        Returns:
            Self for method chaining
        """
        if categories is None:
            categories = list(self.category_mapping.values())
        
        try:
            self.encoder.fit(categories)
            self.is_fitted = True
            
            logger.info(f"Label encoder fitted on {len(categories)} categories")
            return self
            
        except Exception as e:
            logger.error(f"Label encoder fitting failed: {e}")
            raise FeatureEngineeringError(f"Label encoding failed: {e}")
    
    def transform_categories(self, categories: List[str]) -> np.ndarray:
        """
        Transform category names to numeric labels.
        
        Args:
            categories: List of category names
            
        Returns:
            Array of numeric labels
        """
        if not self.is_fitted:
            raise FeatureEngineeringError("Label encoder is not fitted")
        
        try:
            return self.encoder.transform(categories)
        except Exception as e:
            logger.error(f"Label transformation failed: {e}")
            raise FeatureEngineeringError(f"Label transformation failed: {e}")
    
    def inverse_transform_labels(self, labels: np.ndarray) -> List[str]:
        """
        Transform numeric labels back to category names.
        
        Args:
            labels: Array of numeric labels
            
        Returns:
            List of category names
        """
        if not self.is_fitted:
            raise FeatureEngineeringError("Label encoder is not fitted")
        
        try:
            return self.encoder.inverse_transform(labels).tolist()
        except Exception as e:
            logger.error(f"Label inverse transformation failed: {e}")
            raise FeatureEngineeringError(f"Label inverse transformation failed: {e}")
    
    def get_categories(self) -> List[str]:
        """
        Get all available categories.
        
        Returns:
            List of category names
        """
        if not self.is_fitted:
            return list(self.category_mapping.values())
        
        return self.encoder.classes_.tolist()


def create_feature_pipeline(
    texts: List[str],
    categories: List[str],
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[scipy.sparse.csr_matrix, scipy.sparse.csr_matrix, np.ndarray, np.ndarray, TFIDFFeatureExtractor, CategoryLabelEncoder]:
    """
    Create complete feature engineering pipeline.
    
    Args:
        texts: List of preprocessed text documents
        categories: List of corresponding category labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_extractor, label_encoder)
    """
    if len(texts) != len(categories):
        raise FeatureEngineeringError("Number of texts and categories must match")
    
    with PerformanceLogger(logger, "Complete feature engineering pipeline"):
        try:
            from sklearn.model_selection import train_test_split
            
            # Split data (use stratify only if each class has at least 2 samples)
            from collections import Counter
            category_counts = Counter(categories)
            min_count = min(category_counts.values())
            
            if min_count >= 2:
                texts_train, texts_test, cats_train, cats_test = train_test_split(
                    texts, categories, test_size=test_size, random_state=random_state, stratify=categories
                )
            else:
                texts_train, texts_test, cats_train, cats_test = train_test_split(
                    texts, categories, test_size=test_size, random_state=random_state
                )
            
            # Create and fit TF-IDF extractor
            feature_extractor = TFIDFFeatureExtractor()
            feature_extractor.fit_vectorizer(texts_train)
            
            # Transform texts to features
            X_train = feature_extractor.transform_texts(texts_train)
            X_test = feature_extractor.transform_texts(texts_test)
            
            # Create and fit label encoder
            label_encoder = CategoryLabelEncoder()
            label_encoder.fit_categories()
            
            # Transform categories to labels
            y_train = label_encoder.transform_categories(cats_train)
            y_test = label_encoder.transform_categories(cats_test)
            
            logger.info("Feature engineering pipeline completed successfully:")
            logger.info(f"  - Training samples: {X_train.shape[0]}")
            logger.info(f"  - Test samples: {X_test.shape[0]}")
            logger.info(f"  - Features: {X_train.shape[1]}")
            logger.info(f"  - Categories: {len(label_encoder.get_categories())}")
            
            return X_train, X_test, y_train, y_test, feature_extractor, label_encoder
            
        except Exception as e:
            logger.error(f"Feature engineering pipeline failed: {e}")
            raise FeatureEngineeringError(f"Pipeline creation failed: {e}")


def analyze_feature_importance(
    feature_matrix: scipy.sparse.csr_matrix,
    labels: np.ndarray,
    feature_names: List[str],
    top_k: int = 20
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analyze feature importance using chi-square test.
    
    Args:
        feature_matrix: TF-IDF feature matrix
        labels: Category labels
        feature_names: List of feature names
        top_k: Number of top features to return per category
        
    Returns:
        Dictionary mapping category names to top features with scores
    """
    try:
        # Perform chi-square feature selection
        selector = SelectKBest(score_func=chi2, k='all')
        selector.fit(feature_matrix, labels)
        
        # Get feature scores
        feature_scores = selector.scores_
        
        # Create feature importance mapping
        feature_importance = {}
        
        # Get unique categories
        unique_labels = np.unique(labels)
        label_encoder = CategoryLabelEncoder()
        label_encoder.fit_categories()
        
        for label in unique_labels:
            category_name = label_encoder.inverse_transform_labels([label])[0]
            
            # Get top features for this category
            label_mask = (labels == label)
            if np.sum(label_mask) > 0:
                # Calculate mean TF-IDF scores for this category
                category_scores = np.array(feature_matrix[label_mask].mean(axis=0)).flatten()
                
                # Combine with chi-square scores
                combined_scores = category_scores * feature_scores
                
                # Get top k features
                top_indices = np.argsort(combined_scores)[-top_k:][::-1]
                top_features = [(feature_names[i], combined_scores[i]) for i in top_indices]
                
                feature_importance[category_name] = top_features
        
        logger.info(f"Feature importance analysis completed for {len(unique_labels)} categories")
        return feature_importance
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        return {}


def save_feature_engineering_artifacts(
    feature_extractor: TFIDFFeatureExtractor,
    label_encoder: CategoryLabelEncoder,
    base_path: Optional[Path] = None
) -> Dict[str, Path]:
    """
    Save all feature engineering artifacts to disk.
    
    Args:
        feature_extractor: Fitted TF-IDF feature extractor
        label_encoder: Fitted label encoder
        base_path: Base directory for saving (uses MODELS_DIR if None)
        
    Returns:
        Dictionary mapping artifact names to file paths
    """
    base_path = base_path or MODELS_DIR
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = {}
    
    try:
        # Save TF-IDF vectorizer
        tfidf_path = base_path / "tfidf_vectorizer.pkl"
        feature_extractor.save_vectorizer(tfidf_path)
        saved_paths['tfidf_vectorizer'] = tfidf_path
        
        # Save label encoder
        label_encoder_path = base_path / "label_encoder.pkl"
        joblib.dump(label_encoder, label_encoder_path)
        saved_paths['label_encoder'] = label_encoder_path
        
        # Save category mapping
        category_mapping_path = base_path / "category_mapping.pkl"
        with open(category_mapping_path, 'wb') as f:
            pickle.dump(CATEGORY_MAPPING, f)
        saved_paths['category_mapping'] = category_mapping_path
        
        logger.info(f"Feature engineering artifacts saved to: {base_path}")
        return saved_paths
        
    except Exception as e:
        logger.error(f"Failed to save feature engineering artifacts: {e}")
        raise FeatureEngineeringError(f"Artifact saving failed: {e}")


def load_feature_engineering_artifacts(
    base_path: Optional[Path] = None
) -> Tuple[TFIDFFeatureExtractor, CategoryLabelEncoder, Dict[int, str]]:
    """
    Load all feature engineering artifacts from disk.
    
    Args:
        base_path: Base directory for loading (uses MODELS_DIR if None)
        
    Returns:
        Tuple of (feature_extractor, label_encoder, category_mapping)
    """
    base_path = base_path or MODELS_DIR
    base_path = Path(base_path)
    
    try:
        # Load TF-IDF vectorizer
        tfidf_path = base_path / "tfidf_vectorizer.pkl"
        feature_extractor = TFIDFFeatureExtractor()
        feature_extractor.load_vectorizer(tfidf_path)
        
        # Load label encoder
        label_encoder_path = base_path / "label_encoder.pkl"
        label_encoder = joblib.load(label_encoder_path)
        
        # Load category mapping
        category_mapping_path = base_path / "category_mapping.pkl"
        with open(category_mapping_path, 'rb') as f:
            category_mapping = pickle.load(f)
        
        logger.info(f"Feature engineering artifacts loaded from: {base_path}")
        return feature_extractor, label_encoder, category_mapping
        
    except Exception as e:
        logger.error(f"Failed to load feature engineering artifacts: {e}")
        raise FeatureEngineeringError(f"Artifact loading failed: {e}")


if __name__ == "__main__":
    # Test the feature engineering functionality
    print("Testing TF-IDF feature engineering...")
    
    # Sample resume texts for testing
    sample_texts = [
        "python developer machine learning data science tensorflow pytorch",
        "java spring boot web development rest api microservices",
        "javascript react node.js frontend backend full stack developer",
        "data scientist analytics statistics machine learning algorithms",
        "devops engineer docker kubernetes aws cloud infrastructure",
        "business analyst requirements gathering stakeholder management",
        "network security engineer cybersecurity firewall penetration testing",
        "database administrator sql postgresql mysql data management"
    ]
    
    sample_categories = [
        "Python Developer", "Java Developer", "Web Developer", "Data Scientist",
        "Python Developer", "Java Developer", "Web Developer", "Data Scientist"  # Duplicate to allow stratification
    ]
    
    print(f"Testing with {len(sample_texts)} sample texts...")
    
    try:
        # Test TF-IDF feature extractor
        print("\\n--- Testing TF-IDF Feature Extractor ---")
        
        extractor = TFIDFFeatureExtractor()
        print(f"Created extractor with config: {extractor.config}")
        
        # Fit vectorizer
        vectorizer = extractor.fit_vectorizer(sample_texts)
        print(f"Fitted vectorizer - vocabulary size: {extractor.get_vocabulary_size()}")
        
        # Transform texts
        feature_matrix = extractor.transform_texts(sample_texts)
        print(f"Feature matrix shape: {feature_matrix.shape}")
        print(f"Matrix sparsity: {1 - feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1]):.3f}")
        
        # Test label encoder
        print("\\n--- Testing Label Encoder ---")
        
        label_encoder = CategoryLabelEncoder()
        label_encoder.fit_categories()
        print(f"Label encoder fitted with {len(label_encoder.get_categories())} categories")
        
        # Transform categories
        labels = label_encoder.transform_categories(sample_categories)
        print(f"Transformed labels: {labels}")
        
        # Inverse transform
        recovered_categories = label_encoder.inverse_transform_labels(labels)
        print(f"Recovered categories match: {sample_categories == recovered_categories}")
        
        # Test complete pipeline
        print("\\n--- Testing Complete Pipeline ---")
        
        X_train, X_test, y_train, y_test, feat_ext, lbl_enc = create_feature_pipeline(
            sample_texts, sample_categories, test_size=0.5  # Use 50% for small dataset
        )
        
        print(f"Pipeline results:")
        print(f"  - Training set: {X_train.shape}")
        print(f"  - Test set: {X_test.shape}")
        print(f"  - Training labels: {len(y_train)}")
        print(f"  - Test labels: {len(y_test)}")
        
        # Test feature importance
        print("\\n--- Testing Feature Importance ---")
        
        feature_names = feat_ext.get_feature_names()
        importance = analyze_feature_importance(X_train, y_train, feature_names, top_k=5)
        
        for category, features in importance.items():
            print(f"{category}: {[f[0] for f in features[:3]]}")
        
        # Test persistence
        print("\\n--- Testing Persistence ---")
        
        # Save artifacts
        saved_paths = save_feature_engineering_artifacts(feat_ext, lbl_enc)
        print(f"Saved artifacts: {list(saved_paths.keys())}")
        
        # Load artifacts
        loaded_extractor, loaded_encoder, loaded_mapping = load_feature_engineering_artifacts()
        print(f"Loaded artifacts successfully")
        print(f"Loaded vocabulary size: {loaded_extractor.get_vocabulary_size()}")
        print(f"Loaded categories: {len(loaded_encoder.get_categories())}")
        
        print("\\n✅ TF-IDF feature engineering test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()