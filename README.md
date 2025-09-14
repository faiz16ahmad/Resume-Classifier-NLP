# Resume Classification NLP System

A production-ready application that automatically categorizes resumes into job categories using Natural Language Processing and Machine Learning techniques.

## 🎯 Project Overview

This system processes PDF resumes, extracts and cleans text content, applies machine learning models for classification, and provides a user-friendly web interface for batch processing with automated file organization and results export.

### Key Features

- **Automated Resume Classification**: Categorizes resumes into 25 predefined job categories
- **High Accuracy**: Achieves 90%+ classification accuracy using multiple ML algorithms
- **Web Interface**: Streamlit-based UI with drag-and-drop file upload
- **Batch Processing**: Handle multiple resumes simultaneously with real-time progress tracking
- **Automated Organization**: Automatically organizes classified resumes into category folders
- **Export Functionality**: Export results to CSV format with confidence scores
- **Comprehensive Testing**: Unit and integration tests with >80% code coverage

## 🏗️ Project Structure

```
resume-classification-nlp/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Cleaned data
│   └── test_resumes/           # Sample PDFs
├── models/
│   ├── tfidf_vectorizer.pkl    # Trained vectorizer
│   ├── best_model.pkl          # Best performing model
│   └── category_mapping.pkl    # Label mappings
├── src/
│   ├── data_preprocessing.py   # Text cleaning functions
│   ├── feature_engineering.py # TF-IDF implementation
│   ├── model_training.py       # ML pipeline
│   ├── web_app.py             # Streamlit interface
│   ├── utils.py               # Helper functions
│   └── logger_setup.py        # Logging configuration
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── model_development.ipynb
│   └── performance_evaluation.ipynb
├── tests/
│   ├── test_preprocessing.py
│   ├── test_models.py
│   └── test_web_app.py
├── categorized_resumes/        # Output directory
├── logs/                       # Application logs
├── requirements.txt
├── config.py                   # Configuration settings
├── README.md
└── .gitignore
```

## 🎯 Target Categories

The system classifies resumes into 25 job categories:

- Data Scientist
- Java Developer
- Python Developer
- Web Developer
- Business Analyst
- HR
- DevOps Engineer
- Software Engineer
- Testing
- Network Security Engineer
- SAP Developer
- Hardware
- Automation Testing
- Electrical Engineering
- Operations Manager
- PMO
- Database
- Hadoop
- ETL Developer
- DotNet Developer
- Blockchain
- Sales
- Mechanical Engineer
- Civil Engineer
- Arts

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd resume-classification-nlp
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

5. Initialize the project structure:

```bash
python config.py
```

### Running the Application

#### Option 1: Using the startup script (Recommended)

```bash
python run_web_app.py
```

#### Option 2: Direct Streamlit command

```bash
streamlit run src/web_app.py
```

The application will automatically:

- Validate configuration settings
- Create necessary directories
- Check for trained models
- Start the web interface at `http://localhost:8501`

### Using the Web Interface

1. **Upload Files**: Drag and drop PDF resumes or use the file browser
2. **Configure Options**: Choose file organization and export settings
3. **Start Processing**: Click "Start Classification" to begin
4. **View Results**: Check the "Results & Analytics" tab for detailed results
5. **Export Data**: Download CSV reports or summary documents
6. **Organize Files**: Automatically sort resumes by predicted category

## 🔧 Configuration

The system is highly configurable through `config.py`. Key settings include:

- **TF-IDF Parameters**: Max features (5000), n-gram range (1,2)
- **ML Models**: KNN, Logistic Regression, Random Forest, SVM, Naive Bayes
- **Performance Requirements**: 90% accuracy, <5s per resume processing
- **File Handling**: Supported formats, batch sizes, memory limits

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py
```

## 📊 Performance Metrics

- **Classification Accuracy**: >90%
- **Processing Speed**: <5 seconds per resume
- **Memory Usage**: <2GB for batch processing
- **Error Rate**: <5% on diverse resume formats
- **Test Coverage**: >80%

## 🛠️ Development

### Code Quality

The project follows strict code quality standards:

- **PEP 8 Compliance**: All Python code follows PEP 8 standards
- **Type Hints**: Comprehensive type annotations
- **Documentation**: Detailed docstrings for all functions
- **Error Handling**: Comprehensive error management
- **Logging**: Detailed logging for debugging and monitoring

### Adding New Categories

To add new job categories:

1. Update `CATEGORY_MAPPING` in `config.py`
2. Retrain models with new category data
3. Update tests and documentation

## 📈 Model Performance

The system compares multiple ML algorithms and automatically selects the best performer:

- **K-Nearest Neighbors (KNN)**
- **Logistic Regression** with L2 regularization
- **Random Forest** with feature importance
- **Support Vector Machine (SVM)** with RBF kernel
- **Multinomial Naive Bayes**

Performance is evaluated using:

- Accuracy, Precision, Recall, F1-score
- Confusion matrix analysis
- Cross-validation scores
- Feature importance analysis

## 🔒 Security Considerations

- File type validation for uploads
- Memory management for large files
- No persistent storage of resume content
- Secure CSV export functionality
- Input sanitization and validation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Support

For support and questions:

- Create an issue in the repository
- Check the documentation in the `docs/` directory
- Review the test files for usage examples

## 🎓 Learning Outcomes

This project demonstrates:

- Advanced NLP text preprocessing techniques
- Machine learning model comparison and selection
- Web application development with Streamlit
- Data pipeline creation and optimization
- Model persistence and deployment
- Production-ready code organization
- Comprehensive testing strategies

Perfect for showcasing in technical interviews and professional portfolios!
