#!/usr/bin/env python3
"""
Data loading and preprocessing module for the Resume Classification NLP System.

This module handles loading the resume dataset, preprocessing text data,
and preparing it for machine learning model training.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

from config import DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CATEGORY_MAPPING
from src.logger_setup import get_logger
from src.data_preprocessing import clean_text

# Initialize logger
logger = get_logger(__name__)


class DataLoadingError(Exception):
    """Custom exception for data loading errors."""
    pass


class ResumeDataLoader:
    """
    Resume data loader with preprocessing and train/test splitting capabilities.
    
    This class handles loading the resume dataset, cleaning text data,
    encoding labels, and preparing data for ML training.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Optional path to dataset file (uses default if None)
        """
        self.data_path = data_path or (RAW_DATA_DIR / "Resume1.csv")
        self.raw_data = None
        self.processed_data = None
        self.label_encoder = None
        self.category_mapping = {}
        
        logger.info(f"Initialized ResumeDataLoader with data path: {self.data_path}")
    
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw resume dataset from CSV file.
        
        Returns:
            DataFrame with raw resume data
            
        Raises:
            DataLoadingError: If data loading fails
        """
        try:
            if not self.data_path.exists():
                raise DataLoadingError(f"Dataset file not found: {self.data_path}")
            
            logger.info(f"Loading dataset from {self.data_path}")
            
            # Load CSV data
            df = pd.read_csv(self.data_path)
            
            # Handle different dataset formats
            if 'Resume_str' in df.columns:
                # Resume1.csv format
                df = df.rename(columns={'Resume_str': 'Resume'})
                required_columns = ['Category', 'Resume']
            else:
                # Original format
                required_columns = ['Category', 'Resume']
            
            # Validate required columns
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise DataLoadingError(f"Missing required columns: {missing_columns}")
            
            # Basic data validation
            if df.empty:
                raise DataLoadingError("Dataset is empty")
            
            # Remove any rows with missing data
            initial_rows = len(df)
            df = df.dropna(subset=['Category', 'Resume'])
            final_rows = len(df)
            
            if initial_rows != final_rows:
                logger.warning(f"Removed {initial_rows - final_rows} rows with missing data")
            
            # Store raw data
            self.raw_data = df
            
            logger.info(f"Successfully loaded {len(df)} resume records")
            logger.info(f"Categories found: {df['Category'].nunique()}")
            logger.info(f"Category distribution:\n{df['Category'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise DataLoadingError(f"Data loading failed: {e}")
    
    def preprocess_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess resume text data.
        
        Args:
            df: DataFrame with raw resume data
            
        Returns:
            DataFrame with preprocessed text
        """
        logger.info("Preprocessing resume text data...")
        
        try:
            processed_df = df.copy()
            
            # Clean resume text
            logger.info("Cleaning resume text...")
            processed_resumes = []
            
            for i, resume_text in enumerate(df['Resume']):
                try:
                    # Apply text cleaning pipeline
                    cleaned_text = clean_text(
                        resume_text,
                        remove_urls_flag=True,
                        remove_emails_phones_flag=True,
                        remove_special_chars_flag=True,
                        convert_to_lowercase=True,
                        remove_stop_words_flag=True,
                        expand_contractions_flag=True
                    )
                    
                    processed_resumes.append(cleaned_text)
                    
                    if (i + 1) % 50 == 0:
                        logger.debug(f"Processed {i + 1}/{len(df)} resumes")
                
                except Exception as e:
                    logger.warning(f"Failed to clean resume {i}: {e}")
                    # Use original text if cleaning fails
                    processed_resumes.append(str(resume_text))
            
            processed_df['Cleaned_Resume'] = processed_resumes
            
            # Add text statistics
            processed_df['Original_Length'] = df['Resume'].str.len()
            processed_df['Cleaned_Length'] = processed_df['Cleaned_Resume'].str.len()
            processed_df['Word_Count'] = processed_df['Cleaned_Resume'].str.split().str.len()
            
            # Filter out very short resumes (likely corrupted)
            min_length = 50
            initial_count = len(processed_df)
            processed_df = processed_df[processed_df['Cleaned_Length'] >= min_length]
            final_count = len(processed_df)
            
            if initial_count != final_count:
                logger.warning(f"Filtered out {initial_count - final_count} resumes shorter than {min_length} characters")
            
            self.processed_data = processed_df
            
            logger.info(f"Text preprocessing completed for {len(processed_df)} resumes")
            logger.info(f"Average text length: {processed_df['Cleaned_Length'].mean():.0f} characters")
            logger.info(f"Average word count: {processed_df['Word_Count'].mean():.0f} words")
            
            return processed_df
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            raise DataLoadingError(f"Text preprocessing failed: {e}")
    
    def encode_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, Dict[int, str]]:
        """
        Encode category labels for machine learning.
        
        Args:
            df: DataFrame with category labels
            
        Returns:
            Tuple of (df_with_encoded_labels, label_encoder, category_mapping)
        """
        logger.info("Encoding category labels...")
        
        try:
            df_encoded = df.copy()
            
            # Create label encoder
            self.label_encoder = LabelEncoder()
            
            # Fit and transform categories
            encoded_labels = self.label_encoder.fit_transform(df['Category'])
            df_encoded['Category_Encoded'] = encoded_labels
            
            # Create category mapping
            categories = self.label_encoder.classes_
            self.category_mapping = {i: category for i, category in enumerate(categories)}
            
            logger.info(f"Encoded {len(categories)} categories:")
            for i, category in self.category_mapping.items():
                count = sum(encoded_labels == i)
                logger.info(f"  {i}: {category} ({count} samples)")
            
            return df_encoded, self.label_encoder, self.category_mapping
            
        except Exception as e:
            logger.error(f"Label encoding failed: {e}")
            raise DataLoadingError(f"Label encoding failed: {e}")
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training and testing sets.
        
        Args:
            df: DataFrame with processed data
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, test_df)
        """
        logger.info(f"Splitting data into train/test sets (test_size={test_size})")
        
        try:
            # Stratified split to maintain category distribution
            train_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=random_state,
                stratify=df['Category_Encoded']
            )
            
            logger.info(f"Training set: {len(train_df)} samples")
            logger.info(f"Test set: {len(test_df)} samples")
            
            # Log category distribution in splits
            logger.info("Training set category distribution:")
            train_dist = train_df['Category'].value_counts()
            for category, count in train_dist.head(10).items():
                logger.info(f"  {category}: {count}")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Data splitting failed: {e}")
            raise DataLoadingError(f"Data splitting failed: {e}")
    
    def save_processed_data(
        self, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        save_path: Optional[Path] = None
    ) -> Dict[str, Path]:
        """
        Save processed data to files.
        
        Args:
            train_df: Training data
            test_df: Test data
            save_path: Optional base path (uses PROCESSED_DATA_DIR if None)
            
        Returns:
            Dictionary of saved file paths
        """
        save_path = save_path or PROCESSED_DATA_DIR
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        try:
            # Save train/test splits
            train_path = save_path / "train_data.csv"
            test_path = save_path / "test_data.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            
            saved_paths['train'] = train_path
            saved_paths['test'] = test_path
            
            # Save label encoder
            import joblib
            encoder_path = save_path / "label_encoder.pkl"
            joblib.dump(self.label_encoder, encoder_path)
            saved_paths['label_encoder'] = encoder_path
            
            # Save category mapping
            import json
            mapping_path = save_path / "category_mapping.json"
            with open(mapping_path, 'w') as f:
                json.dump(self.category_mapping, f, indent=2)
            saved_paths['category_mapping'] = mapping_path
            
            # Save data statistics
            stats = {
                'total_samples': len(train_df) + len(test_df),
                'train_samples': len(train_df),
                'test_samples': len(test_df),
                'num_categories': len(self.category_mapping),
                'categories': list(self.category_mapping.values()),
                'avg_text_length': float(train_df['Cleaned_Length'].mean()),
                'avg_word_count': float(train_df['Word_Count'].mean())
            }
            
            stats_path = save_path / "data_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            saved_paths['statistics'] = stats_path
            
            logger.info(f"Processed data saved to {save_path}")
            return saved_paths
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            raise DataLoadingError(f"Data saving failed: {e}")
    
    def load_processed_data(self, data_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously processed data.
        
        Args:
            data_path: Optional path to processed data directory
            
        Returns:
            Tuple of (train_df, test_df)
        """
        data_path = data_path or PROCESSED_DATA_DIR
        data_path = Path(data_path)
        
        try:
            train_path = data_path / "train_data.csv"
            test_path = data_path / "test_data.csv"
            
            if not train_path.exists() or not test_path.exists():
                raise DataLoadingError(f"Processed data files not found in {data_path}")
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Load label encoder and mapping
            encoder_path = data_path / "label_encoder.pkl"
            if encoder_path.exists():
                import joblib
                self.label_encoder = joblib.load(encoder_path)
            
            mapping_path = data_path / "category_mapping.json"
            if mapping_path.exists():
                import json
                with open(mapping_path, 'r') as f:
                    self.category_mapping = json.load(f)
                    # Convert string keys to int
                    self.category_mapping = {int(k): v for k, v in self.category_mapping.items()}
            
            logger.info(f"Loaded processed data: {len(train_df)} train, {len(test_df)} test samples")
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise DataLoadingError(f"Processed data loading failed: {e}")
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get training data in format ready for ML models.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) where X contains text and y contains encoded labels
        """
        if self.processed_data is None:
            raise DataLoadingError("No processed data available. Run full pipeline first.")
        
        # Split data
        train_df, test_df = self.split_data(self.processed_data)
        
        # Extract features and labels
        X_train = train_df['Cleaned_Resume'].values
        X_test = test_df['Cleaned_Resume'].values
        y_train = train_df['Category_Encoded'].values
        y_test = test_df['Category_Encoded'].values
        
        logger.info(f"Prepared training data: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, X_test, y_train, y_test


def load_and_prepare_data(
    data_path: Optional[Path] = None,
    save_processed: bool = True,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, Dict[int, str]]:
    """
    Complete data loading and preparation pipeline.
    
    Args:
        data_path: Optional path to dataset file
        save_processed: Whether to save processed data
        test_size: Fraction for test split
        random_state: Random seed
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, label_encoder, category_mapping)
    """
    logger.info("Starting complete data loading and preparation pipeline...")
    
    try:
        # Initialize data loader
        loader = ResumeDataLoader(data_path)
        
        # Load raw data
        raw_df = loader.load_raw_data()
        
        # Preprocess text
        processed_df = loader.preprocess_text_data(raw_df)
        
        # Encode labels
        encoded_df, label_encoder, category_mapping = loader.encode_labels(processed_df)
        
        # Split data
        train_df, test_df = loader.split_data(encoded_df, test_size, random_state)
        
        # Save processed data if requested
        if save_processed:
            loader.save_processed_data(train_df, test_df)
        
        # Extract training arrays
        X_train = train_df['Cleaned_Resume'].values
        X_test = test_df['Cleaned_Resume'].values
        y_train = train_df['Category_Encoded'].values
        y_test = test_df['Category_Encoded'].values
        
        logger.info("Data loading and preparation completed successfully")
        logger.info(f"Final data shapes: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return X_train, X_test, y_train, y_test, label_encoder, category_mapping
        
    except Exception as e:
        logger.error(f"Data preparation pipeline failed: {e}")
        raise DataLoadingError(f"Data preparation failed: {e}")


if __name__ == "__main__":
    # Test the data loading functionality
    print("Testing Resume Data Loading Pipeline...")
    
    try:
        # Test data loader
        print("\n--- Testing Data Loader ---")
        
        loader = ResumeDataLoader()
        print(f"Initialized loader with path: {loader.data_path}")
        
        # Load raw data
        raw_df = loader.load_raw_data()
        print(f"Loaded raw data: {raw_df.shape}")
        print(f"Categories: {raw_df['Category'].nunique()}")
        
        # Preprocess text
        processed_df = loader.preprocess_text_data(raw_df)
        print(f"Processed data: {processed_df.shape}")
        print(f"Average text length: {processed_df['Cleaned_Length'].mean():.0f}")
        
        # Encode labels
        encoded_df, label_encoder, category_mapping = loader.encode_labels(processed_df)
        print(f"Encoded data: {encoded_df.shape}")
        print(f"Categories encoded: {len(category_mapping)}")
        
        # Test complete pipeline
        print("\n--- Testing Complete Pipeline ---")
        
        X_train, X_test, y_train, y_test, encoder, mapping = load_and_prepare_data()
        
        print(f"Training data: {X_train.shape}")
        print(f"Test data: {X_test.shape}")
        print(f"Categories: {len(mapping)}")
        
        # Show sample data
        print(f"\nSample training text (first 200 chars):")
        print(f"'{X_train[0][:200]}...'")
        print(f"Label: {y_train[0]} -> {mapping[y_train[0]]}")
        
        print("\n✅ Data loading test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()