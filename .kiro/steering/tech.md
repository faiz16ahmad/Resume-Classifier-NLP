# Technology Stack

## Core Technologies
- **Python 3.8+**: Primary programming language
- **Streamlit**: Web application framework for UI
- **scikit-learn**: Machine learning algorithms and pipelines
- **NLTK**: Natural language processing and text preprocessing
- **pandas/numpy**: Data manipulation and numerical computing
- **PyPDF2**: PDF text extraction

## Machine Learning Stack
- **TF-IDF Vectorization**: Text feature extraction (max_features=5000, ngram_range=(1,2))
- **Multiple ML Models**: KNN, Logistic Regression, Random Forest, SVM, Naive Bayes
- **Model Persistence**: joblib for model serialization
- **Cross-validation**: 5-fold stratified CV for model evaluation

## Development Tools
- **pytest**: Testing framework with coverage reporting
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking
- **jupyter**: Notebook development

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Running the Application
```bash
# Start web interface
streamlit run src/web_app.py

# Initialize project structure
python config.py
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_classification_service.py
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Configuration
- All settings centralized in `config.py`
- Environment-specific configurations supported
- Logging configured via `LOGGING_CONFIG` dictionary
- Model hyperparameters in `ML_MODELS_CONFIG`