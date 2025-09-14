"""
Utility functions for the Resume Classification NLP System.

This module contains helper functions for PDF text extraction, file validation,
and other utility operations used throughout the application.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 not installed. Please run: pip install PyPDF2")
    sys.exit(1)

from config import FILE_CONFIG, PERFORMANCE_REQUIREMENTS
from src.logger_setup import get_logger, PerformanceLogger

# Initialize logger
logger = get_logger(__name__)


class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass


def validate_pdf_format(file_path: Union[str, Path]) -> bool:
    """
    Validate if a file is a valid PDF format.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        True if valid PDF, False otherwise
        
    Raises:
        PDFExtractionError: If file validation fails
    """
    file_path = Path(file_path)
    
    try:
        # Check if file exists
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False
        
        # Check file extension
        if file_path.suffix.lower() not in FILE_CONFIG["supported_formats"]:
            logger.error(f"Unsupported file format: {file_path.suffix}")
            return False
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > FILE_CONFIG["max_file_size_mb"]:
            logger.error(f"File too large: {file_size_mb:.2f}MB > {FILE_CONFIG['max_file_size_mb']}MB")
            return False
        
        # Try to open as PDF
        with open(file_path, 'rb') as file:
            try:
                pdf_reader = PdfReader(file)
                # Check if PDF has pages
                if len(pdf_reader.pages) == 0:
                    logger.error(f"PDF has no pages: {file_path}")
                    return False
                
                logger.debug(f"Valid PDF with {len(pdf_reader.pages)} pages: {file_path}")
                return True
                
            except Exception as e:
                logger.error(f"Invalid PDF format: {file_path} - {e}")
                return False
                
    except Exception as e:
        logger.error(f"File validation error: {file_path} - {e}")
        raise PDFExtractionError(f"File validation failed: {e}")


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Extract text content from a PDF file using PyPDF2.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content as string
        
    Raises:
        PDFExtractionError: If text extraction fails
    """
    file_path = Path(file_path)
    
    with PerformanceLogger(logger, f"PDF extraction: {file_path.name}"):
        try:
            # Validate PDF format first
            if not validate_pdf_format(file_path):
                raise PDFExtractionError(f"Invalid PDF file: {file_path}")
            
            extracted_text = ""
            
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                logger.debug(f"Extracting text from {num_pages} pages: {file_path.name}")
                
                # Extract text from each page
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\\n"
                        else:
                            logger.warning(f"No text found on page {page_num + 1}: {file_path.name}")
                    
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {file_path.name} - {e}")
                        continue
            
            # Clean up the extracted text
            extracted_text = extracted_text.strip()
            
            if not extracted_text:
                raise PDFExtractionError(f"No text could be extracted from: {file_path}")
            
            logger.info(f"Successfully extracted {len(extracted_text)} characters from: {file_path.name}")
            return extracted_text
            
        except PDFExtractionError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error extracting text from {file_path}: {e}")
            raise PDFExtractionError(f"Text extraction failed: {e}")


def batch_extract_texts(
    file_paths: List[Union[str, Path]], 
    progress_callback: Optional[callable] = None
) -> Dict[str, Union[str, Exception]]:
    """
    Extract text from multiple PDF files with progress tracking.
    
    Args:
        file_paths: List of paths to PDF files
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Dictionary mapping file paths to extracted text or exceptions
    """
    results = {}
    total_files = len(file_paths)
    
    logger.info(f"Starting batch extraction for {total_files} files")
    
    with PerformanceLogger(logger, f"Batch extraction: {total_files} files"):
        for i, file_path in enumerate(file_paths):
            file_path = Path(file_path)
            
            try:
                # Extract text from current file
                extracted_text = extract_text_from_pdf(file_path)
                results[str(file_path)] = extracted_text
                
                logger.debug(f"Successfully processed {i + 1}/{total_files}: {file_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                results[str(file_path)] = e
            
            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(i + 1, total_files, file_path.name)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
    
    # Log batch results
    successful = sum(1 for v in results.values() if isinstance(v, str))
    failed = total_files - successful
    error_rate = failed / total_files if total_files > 0 else 0
    
    logger.info(f"Batch extraction completed: {successful} successful, {failed} failed")
    
    if error_rate > PERFORMANCE_REQUIREMENTS["max_error_rate"]:
        logger.warning(f"Error rate {error_rate:.2%} exceeds threshold {PERFORMANCE_REQUIREMENTS['max_error_rate']:.2%}")
    
    return results


def get_pdf_metadata(file_path: Union[str, Path]) -> Dict[str, Union[str, int, None]]:
    """
    Extract metadata from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
    """
    file_path = Path(file_path)
    metadata = {
        "filename": file_path.name,
        "file_size_mb": 0,
        "num_pages": 0,
        "title": None,
        "author": None,
        "subject": None,
        "creator": None,
        "producer": None,
        "creation_date": None,
        "modification_date": None
    }
    
    try:
        # Get file size
        metadata["file_size_mb"] = round(file_path.stat().st_size / (1024 * 1024), 2)
        
        # Extract PDF metadata
        with open(file_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            metadata["num_pages"] = len(pdf_reader.pages)
            
            # Get document info if available
            if pdf_reader.metadata:
                doc_info = pdf_reader.metadata
                metadata.update({
                    "title": doc_info.get("/Title"),
                    "author": doc_info.get("/Author"),
                    "subject": doc_info.get("/Subject"),
                    "creator": doc_info.get("/Creator"),
                    "producer": doc_info.get("/Producer"),
                    "creation_date": str(doc_info.get("/CreationDate")) if doc_info.get("/CreationDate") else None,
                    "modification_date": str(doc_info.get("/ModDate")) if doc_info.get("/ModDate") else None
                })
        
        logger.debug(f"Extracted metadata for: {file_path.name}")
        
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {e}")
    
    return metadata


def detect_text_encoding(text: str) -> str:
    """
    Detect and handle text encoding issues.
    
    Args:
        text: Input text string
        
    Returns:
        Text with encoding issues resolved
    """
    try:
        # Try different encoding approaches
        for encoding in FILE_CONFIG["encoding_fallbacks"]:
            try:
                # Encode and decode to clean up encoding issues
                cleaned_text = text.encode(encoding, errors='ignore').decode(encoding)
                return cleaned_text
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        
        # If all encodings fail, remove non-ASCII characters
        logger.warning("Using ASCII fallback for text encoding")
        return ''.join(char for char in text if ord(char) < 128)
        
    except Exception as e:
        logger.error(f"Encoding detection failed: {e}")
        return text


def validate_extracted_text(text: str, min_length: int = 50) -> bool:
    """
    Validate if extracted text meets minimum quality requirements.
    
    Args:
        text: Extracted text to validate
        min_length: Minimum required text length
        
    Returns:
        True if text is valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    
    # Remove whitespace for length check
    clean_text = text.strip()
    
    if len(clean_text) < min_length:
        logger.warning(f"Text too short: {len(clean_text)} < {min_length} characters")
        return False
    
    # Check for reasonable text content (not just special characters)
    alphanumeric_chars = sum(1 for char in clean_text if char.isalnum())
    alphanumeric_ratio = alphanumeric_chars / len(clean_text)
    
    if alphanumeric_ratio < 0.3:  # At least 30% alphanumeric characters
        logger.warning(f"Text quality low: {alphanumeric_ratio:.2%} alphanumeric characters")
        return False
    
    return True


def create_temp_file(content: str, suffix: str = ".txt") -> Path:
    """
    Create a temporary file with the given content.
    
    Args:
        content: Content to write to the file
        suffix: File extension
        
    Returns:
        Path to the created temporary file
    """
    import tempfile
    
    temp_dir = FILE_CONFIG["temp_dir"]
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix=suffix, 
        dir=temp_dir, 
        delete=False,
        encoding='utf-8'
    ) as temp_file:
        temp_file.write(content)
        temp_path = Path(temp_file.name)
    
    logger.debug(f"Created temporary file: {temp_path}")
    return temp_path


def cleanup_temp_files(temp_dir: Optional[Path] = None) -> None:
    """
    Clean up temporary files.
    
    Args:
        temp_dir: Directory to clean up (defaults to configured temp directory)
    """
    temp_dir = temp_dir or FILE_CONFIG["temp_dir"]
    
    if not temp_dir.exists():
        return
    
    try:
        for temp_file in temp_dir.glob("*"):
            if temp_file.is_file():
                temp_file.unlink()
                logger.debug(f"Deleted temporary file: {temp_file}")
        
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
        
    except Exception as e:
        logger.error(f"Error cleaning up temporary files: {e}")


if __name__ == "__main__":
    # Test the PDF extraction utilities
    print("Testing PDF extraction utilities...")
    
    # Test with a sample file (if available)
    test_files = list(Path("data/test_resumes").glob("*.pdf"))
    
    if test_files:
        print(f"Found {len(test_files)} test files")
        
        for test_file in test_files[:2]:  # Test first 2 files
            print(f"\\nTesting: {test_file.name}")
            
            # Test validation
            is_valid = validate_pdf_format(test_file)
            print(f"Valid PDF: {is_valid}")
            
            if is_valid:
                # Test extraction
                try:
                    text = extract_text_from_pdf(test_file)
                    print(f"Extracted {len(text)} characters")
                    print(f"Text preview: {text[:200]}...")
                    
                    # Test metadata
                    metadata = get_pdf_metadata(test_file)
                    print(f"Metadata: {metadata}")
                    
                except Exception as e:
                    print(f"Extraction failed: {e}")
    else:
        print("No test PDF files found in data/test_resumes/")
        print("Add some PDF files to test the extraction functionality.")
    
    print("\\nPDF extraction utilities test completed!")