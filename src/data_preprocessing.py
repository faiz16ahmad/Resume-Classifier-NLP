"""
Text preprocessing pipeline for the Resume Classification NLP System.

This module provides comprehensive text cleaning and normalization functions
for preparing resume text data for machine learning processing.
"""

import re
import sys
import string
import unicodedata
from pathlib import Path
from typing import List, Optional, Set, Union
import logging

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import PorterStemmer, WordNetLemmatizer
except ImportError:
    print("NLTK not installed. Please run: pip install nltk")
    print("Then download required data: python -c \"import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')\"")
    sys.exit(1)

from config import TEXT_PREPROCESSING_CONFIG, FILE_CONFIG
from src.logger_setup import get_logger, PerformanceLogger

# Initialize logger
logger = get_logger(__name__)

# Initialize NLTK components
try:
    # Download required NLTK data if not present
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("Downloading required NLTK data...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

# Initialize NLTK tools
STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


class TextPreprocessingError(Exception):
    """Custom exception for text preprocessing errors."""
    pass


def remove_urls(text: str) -> str:
    """
    Remove URLs from text using regex patterns.
    
    Args:
        text: Input text containing URLs
        
    Returns:
        Text with URLs removed
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Comprehensive URL patterns
    url_patterns = [
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        r'[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]*\.(?:com|org|net|edu|gov|mil|int|co|io|ai|ly|me|us|uk|ca|de|fr|jp|cn|in|au|br|ru|it|es|nl|se|no|dk|fi|pl|cz|hu|ro|bg|hr|si|sk|lt|lv|ee|mt|cy|lu|be|at|ch|li|mc|sm|va|ad|gi|im|je|gg|fo|gl|is|ax|sj|bv|hm|tf|aq|gs|fk|pn|tk|to|nu|ws|as|gu|mp|pr|vi|um|fm|mh|pw|kp|kr|jp|cn|hk|mo|tw|sg|my|th|la|kh|vn|mm|bd|bt|np|lk|mv|in|pk|af|ir|iq|sy|lb|jo|il|ps|sa|ye|om|ae|qa|bh|kw|tr|cy|ge|am|az|kg|kz|uz|tm|tj|mn|ru|by|ua|md|ro|bg|mk|al|me|rs|ba|hr|si|sk|cz|hu|pl|lt|lv|ee|fi|se|no|dk|is|ie|gb|fr|es|pt|it|ch|at|de|nl|be|lu|mc|li|sm|va|ad|mt|gi|im|je|gg|fo|gl|ax|sj|bv|hm|tf|aq|gs|fk|pn|tk|to|nu|ws|as|gu|mp|pr|vi|um|fm|mh|pw)(?:/[^\\s]*)?',
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'  # Email addresses as URLs
    ]
    
    cleaned_text = text
    for pattern in url_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.IGNORECASE)
    
    return cleaned_text


def remove_emails_phones(text: str) -> str:
    """
    Remove email addresses and phone numbers from text.
    
    Args:
        text: Input text containing emails and phone numbers
        
    Returns:
        Text with emails and phone numbers removed
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Email patterns
    email_patterns = [
        r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        r'\b[A-Za-z0-9._%+-]+\s*@\s*[A-Za-z0-9.-]+\s*\.\s*[A-Za-z]{2,}\b'
    ]
    
    # Phone number patterns (various formats)
    phone_patterns = [
        r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b',  # US format
        r'\b[0-9]{3}[-.]?[0-9]{3}[-.]?[0-9]{4}\b',  # Simple format
        r'\([0-9]{3}\)\s*[0-9]{3}[-.]?[0-9]{4}',  # (xxx) xxx-xxxx
        r'\+[0-9]{1,3}[-.]?[0-9]{1,14}',  # International format
        r'\b[0-9]{10,15}\b'  # Long number sequences
    ]
    
    cleaned_text = text
    
    # Remove emails
    for pattern in email_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.IGNORECASE)
    
    # Remove phone numbers
    for pattern in phone_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text)
    
    return cleaned_text


def remove_special_chars(text: str) -> str:
    """
    Remove special characters and extra whitespace from text.
    
    Args:
        text: Input text with special characters
        
    Returns:
        Text with special characters cleaned
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove non-printable characters
    cleaned_text = ''.join(char for char in text if char.isprintable())
    
    # Replace multiple types of quotes and dashes with standard versions
    replacements = {
        '"': '"', '"': '"', ''': "'", ''': "'",
        '–': '-', '—': '-', '…': '...',
        '•': '*', '◦': '*', '▪': '*', '▫': '*'
    }
    
    for old, new in replacements.items():
        cleaned_text = cleaned_text.replace(old, new)
    
    # Remove excessive punctuation (more than 2 consecutive)
    cleaned_text = re.sub(r'[!.?]{3,}', '...', cleaned_text)
    cleaned_text = re.sub(r'[-]{3,}', '---', cleaned_text)
    
    # Remove special characters but keep basic punctuation
    # Keep: letters, numbers, spaces, basic punctuation (.,!?;:()[]{}'"-)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?;:()[\]{}\'"\-*+/=<>@#$%&_]', ' ', cleaned_text)
    
    # Clean up extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text.strip()


def remove_stop_words(text: str, custom_stop_words: Optional[Set[str]] = None) -> str:
    """
    Remove stop words from text using NLTK.
    
    Args:
        text: Input text containing stop words
        custom_stop_words: Additional stop words to remove
        
    Returns:
        Text with stop words removed
    """
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Tokenize text
        tokens = word_tokenize(text.lower())
        
        # Combine default and custom stop words
        stop_words_set = STOP_WORDS.copy()
        if custom_stop_words:
            stop_words_set.update(custom_stop_words)
        
        # Add common resume-specific stop words
        resume_stop_words = {
            'resume', 'cv', 'curriculum', 'vitae', 'page', 'pages',
            'email', 'phone', 'address', 'contact', 'information'
        }
        stop_words_set.update(resume_stop_words)
        
        # Filter out stop words and very short words
        filtered_tokens = [
            token for token in tokens 
            if token not in stop_words_set 
            and len(token) > 2 
            and token.isalpha()
        ]
        
        return ' '.join(filtered_tokens)
        
    except Exception as e:
        logger.warning(f"Error removing stop words: {e}")
        return text


def normalize_text(text: str) -> str:
    """
    Normalize text by converting to lowercase and handling encoding.
    
    Args:
        text: Input text to normalize
        
    Returns:
        Normalized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    try:
        # Convert to lowercase
        normalized = text.lower()
        
        # Normalize unicode characters
        normalized = unicodedata.normalize('NFKD', normalized)
        
        # Handle encoding issues
        for encoding in FILE_CONFIG["encoding_fallbacks"]:
            try:
                normalized = normalized.encode(encoding, errors='ignore').decode(encoding)
                break
            except (UnicodeEncodeError, UnicodeDecodeError):
                continue
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
        
    except Exception as e:
        logger.warning(f"Error normalizing text: {e}")
        return text.lower()


def expand_contractions(text: str) -> str:
    """
    Expand common English contractions.
    
    Args:
        text: Input text with contractions
        
    Returns:
        Text with expanded contractions
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Common contractions mapping
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
        "'m": " am", "let's": "let us", "that's": "that is",
        "who's": "who is", "what's": "what is", "here's": "here is",
        "there's": "there is", "where's": "where is", "how's": "how is",
        "it's": "it is", "he's": "he is", "she's": "she is",
        "we're": "we are", "they're": "they are", "you're": "you are",
        "i'm": "i am", "you've": "you have", "we've": "we have",
        "they've": "they have", "i've": "i have", "you'll": "you will",
        "we'll": "we will", "they'll": "they will", "i'll": "i will",
        "you'd": "you would", "we'd": "we would", "they'd": "they would",
        "i'd": "i would", "should've": "should have", "could've": "could have",
        "would've": "would have", "might've": "might have", "must've": "must have"
    }
    
    expanded_text = text
    for contraction, expansion in contractions.items():
        expanded_text = re.sub(
            r'\b' + re.escape(contraction) + r'\b',
            expansion,
            expanded_text,
            flags=re.IGNORECASE
        )
    
    return expanded_text


def remove_extra_whitespace(text: str) -> str:
    """
    Remove extra whitespace, tabs, and newlines.
    
    Args:
        text: Input text with extra whitespace
        
    Returns:
        Text with normalized whitespace
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Replace tabs and newlines with spaces
    cleaned = re.sub(r'[\t\n\r\f\v]', ' ', text)
    
    # Replace multiple spaces with single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # Remove leading and trailing whitespace
    return cleaned.strip()


def clean_text(
    text: str,
    remove_urls_flag: bool = True,
    remove_emails_phones_flag: bool = True,
    remove_special_chars_flag: bool = True,
    convert_to_lowercase: bool = True,
    remove_stop_words_flag: bool = True,
    expand_contractions_flag: bool = True,
    custom_stop_words: Optional[Set[str]] = None
) -> str:
    """
    Master text cleaning function that applies all preprocessing steps.
    
    Args:
        text: Input text to clean
        remove_urls_flag: Whether to remove URLs
        remove_emails_phones_flag: Whether to remove emails and phones
        remove_special_chars_flag: Whether to remove special characters
        convert_to_lowercase: Whether to convert to lowercase
        remove_stop_words_flag: Whether to remove stop words
        expand_contractions_flag: Whether to expand contractions
        custom_stop_words: Additional stop words to remove
        
    Returns:
        Cleaned and preprocessed text
        
    Raises:
        TextPreprocessingError: If preprocessing fails
    """
    if not text or not isinstance(text, str):
        return ""
    
    with PerformanceLogger(logger, "Text preprocessing"):
        try:
            original_length = len(text)
            cleaned_text = text
            
            # Apply preprocessing steps based on configuration
            config = TEXT_PREPROCESSING_CONFIG
            
            if remove_urls_flag and config.get("remove_urls", True):
                cleaned_text = remove_urls(cleaned_text)
                logger.debug("URLs removed")
            
            if remove_emails_phones_flag and config.get("remove_emails", True):
                cleaned_text = remove_emails_phones(cleaned_text)
                logger.debug("Emails and phones removed")
            
            if expand_contractions_flag:
                cleaned_text = expand_contractions(cleaned_text)
                logger.debug("Contractions expanded")
            
            if remove_special_chars_flag and config.get("remove_special_chars", True):
                cleaned_text = remove_special_chars(cleaned_text)
                logger.debug("Special characters removed")
            
            if convert_to_lowercase and config.get("convert_to_lowercase", True):
                cleaned_text = normalize_text(cleaned_text)
                logger.debug("Text normalized to lowercase")
            
            if config.get("remove_extra_whitespace", True):
                cleaned_text = remove_extra_whitespace(cleaned_text)
                logger.debug("Extra whitespace removed")
            
            if remove_stop_words_flag and config.get("remove_stop_words", True):
                cleaned_text = remove_stop_words(cleaned_text, custom_stop_words)
                logger.debug("Stop words removed")
            
            # Validate cleaned text
            if not cleaned_text or len(cleaned_text.strip()) < config.get("min_text_length", 50):
                raise TextPreprocessingError(f"Cleaned text too short: {len(cleaned_text)} characters")
            
            if len(cleaned_text) > config.get("max_text_length", 50000):
                logger.warning(f"Text truncated from {len(cleaned_text)} to {config['max_text_length']} characters")
                cleaned_text = cleaned_text[:config["max_text_length"]]
            
            final_length = len(cleaned_text)
            reduction_ratio = (original_length - final_length) / original_length if original_length > 0 else 0
            
            logger.info(f"Text preprocessing completed: {original_length} -> {final_length} chars ({reduction_ratio:.1%} reduction)")
            
            return cleaned_text.strip()
            
        except TextPreprocessingError:
            raise
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            raise TextPreprocessingError(f"Preprocessing failed: {e}")


def batch_clean_texts(
    texts: List[str],
    progress_callback: Optional[callable] = None,
    **kwargs
) -> List[Union[str, Exception]]:
    """
    Clean multiple texts with progress tracking.
    
    Args:
        texts: List of texts to clean
        progress_callback: Optional callback for progress updates
        **kwargs: Additional arguments for clean_text function
        
    Returns:
        List of cleaned texts or exceptions
    """
    results = []
    total_texts = len(texts)
    
    logger.info(f"Starting batch text cleaning for {total_texts} texts")
    
    with PerformanceLogger(logger, f"Batch text cleaning: {total_texts} texts"):
        for i, text in enumerate(texts):
            try:
                cleaned_text = clean_text(text, **kwargs)
                results.append(cleaned_text)
                logger.debug(f"Successfully cleaned text {i + 1}/{total_texts}")
                
            except Exception as e:
                logger.error(f"Failed to clean text {i + 1}: {e}")
                results.append(e)
            
            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(i + 1, total_texts)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")
    
    # Log batch results
    successful = sum(1 for r in results if isinstance(r, str))
    failed = total_texts - successful
    
    logger.info(f"Batch cleaning completed: {successful} successful, {failed} failed")
    
    return results


def get_text_statistics(text: str) -> dict:
    """
    Get statistics about text content.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary with text statistics
    """
    if not text or not isinstance(text, str):
        return {}
    
    try:
        tokens = word_tokenize(text.lower())
        
        stats = {
            "character_count": len(text),
            "word_count": len(tokens),
            "sentence_count": len(re.findall(r'[.!?]+', text)),
            "avg_word_length": sum(len(word) for word in tokens) / len(tokens) if tokens else 0,
            "unique_words": len(set(tokens)),
            "vocabulary_richness": len(set(tokens)) / len(tokens) if tokens else 0,
            "alphanumeric_ratio": sum(1 for char in text if char.isalnum()) / len(text) if text else 0,
            "uppercase_ratio": sum(1 for char in text if char.isupper()) / len(text) if text else 0,
            "digit_ratio": sum(1 for char in text if char.isdigit()) / len(text) if text else 0
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating text statistics: {e}")
        return {}


if __name__ == "__main__":
    # Test the text preprocessing functions
    print("Testing text preprocessing pipeline...")
    
    # Sample resume text for testing
    sample_text = """
    John Doe - Senior Software Engineer
    Email: john.doe@example.com | Phone: (555) 123-4567
    LinkedIn: https://linkedin.com/in/johndoe | Website: www.johndoe.com
    
    PROFESSIONAL EXPERIENCE:
    • I've worked as a Python developer for 5+ years
    • Can't imagine working without machine learning libraries
    • Developed REST APIs using Django & Flask frameworks
    • Won't compromise on code quality and testing
    
    TECHNICAL SKILLS:
    Programming: Python, Java, JavaScript, SQL, C++
    ML/AI: TensorFlow, PyTorch, scikit-learn, pandas, NumPy
    Web: Django, Flask, React, Node.js, HTML/CSS
    Databases: PostgreSQL, MySQL, MongoDB, Redis
    
    Special characters: @#$%^&*()_+{}|:<>?[]\\;'\",./ and more!!!
    """
    
    print(f"Original text length: {len(sample_text)} characters")
    print(f"Original text preview: {sample_text[:200]}...")
    
    # Test individual functions
    print("\\n--- Testing individual functions ---")
    
    # Test URL removal
    text_no_urls = remove_urls(sample_text)
    print(f"After URL removal: {len(text_no_urls)} characters")
    
    # Test email/phone removal
    text_no_contacts = remove_emails_phones(text_no_urls)
    print(f"After contact removal: {len(text_no_contacts)} characters")
    
    # Test special character removal
    text_no_special = remove_special_chars(text_no_contacts)
    print(f"After special char removal: {len(text_no_special)} characters")
    
    # Test complete cleaning
    print("\\n--- Testing complete cleaning pipeline ---")
    cleaned_text = clean_text(sample_text)
    print(f"Final cleaned text length: {len(cleaned_text)} characters")
    print(f"Cleaned text preview: {cleaned_text[:300]}...")
    
    # Test statistics
    print("\\n--- Text statistics ---")
    stats = get_text_statistics(cleaned_text)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Test batch processing
    print("\\n--- Testing batch processing ---")
    test_texts = [sample_text, "Short text", "Another resume with different content and structure."]
    
    def progress_callback(current, total):
        print(f"Progress: {current}/{total}")
    
    batch_results = batch_clean_texts(test_texts, progress_callback)
    print(f"Batch results: {len(batch_results)} processed")
    
    for i, result in enumerate(batch_results):
        if isinstance(result, str):
            print(f"Text {i+1}: {len(result)} characters")
        else:
            print(f"Text {i+1}: Error - {result}")
    
    print("\\nText preprocessing pipeline test completed!")