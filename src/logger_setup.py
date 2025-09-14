"""
Logging configuration and setup for the Resume Classification System.

This module provides centralized logging configuration and utilities
for consistent logging across all application components.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Optional

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import LOGGING_CONFIG, LOGS_DIR


def setup_logging(
    config_dict: Optional[dict] = None,
    log_level: str = "INFO"
) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        config_dict: Custom logging configuration dictionary
        log_level: Default logging level
        
    Returns:
        Configured logger instance
    """
    # Ensure logs directory exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use provided config or default
    config = config_dict or LOGGING_CONFIG
    
    try:
        # Apply logging configuration
        logging.config.dictConfig(config)
        logger = logging.getLogger(__name__)
        logger.info("Logging configuration applied successfully")
        return logger
    except Exception as e:
        # Fallback to basic configuration if dict config fails
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(LOGS_DIR / "resume_classifier.log")
            ]
        )
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to apply logging config: {e}")
        logger.info("Using fallback logging configuration")
        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information for debugging purposes.
    
    Args:
        logger: Logger instance to use
    """
    import platform
    import sys
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    logger.info("=== End System Information ===")


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration: float,
    memory_usage: Optional[float] = None,
    additional_metrics: Optional[dict] = None
) -> None:
    """
    Log performance metrics for operations.
    
    Args:
        logger: Logger instance to use
        operation: Name of the operation
        duration: Time taken in seconds
        memory_usage: Memory usage in MB (optional)
        additional_metrics: Additional metrics to log (optional)
    """
    metrics_msg = f"Performance - {operation}: {duration:.3f}s"
    
    if memory_usage is not None:
        metrics_msg += f", Memory: {memory_usage:.2f}MB"
    
    if additional_metrics:
        for key, value in additional_metrics.items():
            metrics_msg += f", {key}: {value}"
    
    logger.info(metrics_msg)


class PerformanceLogger:
    """Context manager for logging operation performance."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            log_performance_metrics(self.logger, self.operation, duration)
        else:
            self.logger.error(
                f"Operation failed: {self.operation} after {duration:.3f}s - {exc_val}"
            )


# Initialize default logger
default_logger = setup_logging()

if __name__ == "__main__":
    # Test logging setup
    test_logger = setup_logging()
    log_system_info(test_logger)
    
    test_logger.debug("Debug message test")
    test_logger.info("Info message test")
    test_logger.warning("Warning message test")
    test_logger.error("Error message test")
    
    # Test performance logger
    with PerformanceLogger(test_logger, "test_operation"):
        import time
        time.sleep(0.1)
    
    print("Logging setup test completed successfully!")