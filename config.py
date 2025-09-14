"""
Configuration settings for the Resume Classification NLP System.

This module contains all configuration parameters, category mappings,
and system settings used throughout the application.
"""

import os
from typing import Dict, List
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TEST_RESUMES_DIR = DATA_DIR / "test_resumes"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TESTS_DIR = PROJECT_ROOT / "tests"
CATEGORIZED_RESUMES_DIR = PROJECT_ROOT / "categorized_resumes"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model file paths
TFIDF_VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
CATEGORY_MAPPING_PATH = MODELS_DIR / "category_mapping.pkl"
MODEL_PERFORMANCE_PATH = MODELS_DIR / "model_performance.json"

# Job category mappings (25 categories as specified)
CATEGORY_MAPPING: Dict[int, str] = {
    0: "Data Scientist",
    1: "Java Developer", 
    2: "Python Developer",
    3: "Web Developer",
    4: "Business Analyst",
    5: "HR",
    6: "DevOps Engineer",
    7: "Software Engineer",
    8: "Testing",
    9: "Network Security Engineer",
    10: "SAP Developer",
    11: "Hardware",
    12: "Automation Testing",
    13: "Electrical Engineering",
    14: "Operations Manager",
    15: "PMO",
    16: "Database",
    17: "Hadoop",
    18: "ETL Developer",
    19: "DotNet Developer",
    20: "Blockchain",
    21: "Sales",
    22: "Mechanical Engineer",
    23: "Civil Engineer",
    24: "Arts"
}

# Reverse mapping for label encoding
CATEGORY_TO_ID: Dict[str, int] = {v: k for k, v in CATEGORY_MAPPING.items()}

# TF-IDF Configuration
TFIDF_CONFIG = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "stop_words": "english",
    "lowercase": True,
    "strip_accents": "unicode"
}

# Machine Learning Models Configuration
ML_MODELS_CONFIG = {
    "knn": {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto"
    },
    "logistic_regression": {
        "C": 1.0,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 1000,
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42
    },
    "svm": {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "random_state": 42
    },
    "naive_bayes": {
        "alpha": 1.0,
        "fit_prior": True
    }
}

# Cross-validation configuration
CV_CONFIG = {
    "cv_folds": 5,
    "scoring": "accuracy",
    "random_state": 42,
    "stratify": True
}

# Performance requirements
PERFORMANCE_REQUIREMENTS = {
    "min_accuracy": 0.90,
    "max_processing_time_per_resume": 5.0,  # seconds
    "max_memory_usage": 2.0,  # GB
    "max_error_rate": 0.05,
    "min_test_coverage": 0.80
}

# File processing configuration
FILE_CONFIG = {
    "max_file_size_mb": 10,
    "supported_formats": [".pdf"],
    "encoding_fallbacks": ["utf-8", "latin-1", "cp1252", "iso-8859-1"],
    "temp_dir": PROJECT_ROOT / "temp",
    "batch_size": 50
}

# Web application configuration
WEB_CONFIG = {
    "page_title": "Resume Classification System",
    "page_icon": "📄",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 200  # MB
}

# Logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "resume_classifier.log"),
            "mode": "a"
        },
        "error_file": {
            "class": "logging.FileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "errors.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file", "error_file"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}

# Text preprocessing configuration
TEXT_PREPROCESSING_CONFIG = {
    "remove_urls": True,
    "remove_emails": True,
    "remove_phone_numbers": True,
    "remove_special_chars": True,
    "convert_to_lowercase": True,
    "remove_stop_words": True,
    "remove_extra_whitespace": True,
    "min_text_length": 50,  # Minimum characters for valid resume
    "max_text_length": 50000  # Maximum characters to process
}

# Export configuration
EXPORT_CONFIG = {
    "csv_delimiter": ",",
    "include_confidence_scores": True,
    "include_processing_time": True,
    "include_timestamp": True,
    "date_format": "%Y-%m-%d %H:%M:%S"
}

def ensure_directories_exist() -> None:
    """Create all necessary directories if they don't exist."""
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TEST_RESUMES_DIR,
        MODELS_DIR, NOTEBOOKS_DIR, TESTS_DIR, CATEGORIZED_RESUMES_DIR,
        LOGS_DIR, FILE_CONFIG["temp_dir"]
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_category_folders() -> List[str]:
    """Get list of category folder names for file organization."""
    return [category.replace(" ", "_").replace("/", "_") for category in CATEGORY_MAPPING.values()]

def validate_config() -> bool:
    """Validate configuration settings."""
    try:
        # Check if all required directories can be created
        ensure_directories_exist()
        
        # Validate category mapping
        if len(CATEGORY_MAPPING) != 25:
            raise ValueError(f"Expected 25 categories, got {len(CATEGORY_MAPPING)}")
        
        # Validate TF-IDF configuration
        if TFIDF_CONFIG["max_features"] <= 0:
            raise ValueError("TF-IDF max_features must be positive")
        
        # Validate performance requirements
        if PERFORMANCE_REQUIREMENTS["min_accuracy"] <= 0 or PERFORMANCE_REQUIREMENTS["min_accuracy"] > 1:
            raise ValueError("Minimum accuracy must be between 0 and 1")
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Validate configuration when run directly
    if validate_config():
        print("Configuration validation successful!")
        print(f"Project root: {PROJECT_ROOT}")
        print(f"Number of categories: {len(CATEGORY_MAPPING)}")
        print("All directories created successfully.")
    else:
        print("Configuration validation failed!")