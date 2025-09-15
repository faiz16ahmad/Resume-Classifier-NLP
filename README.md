# 🎯 Resume Classification Demo

A **single-file Streamlit application** that demonstrates AI-powered resume classification. Perfect for interviews, demos, and quick showcases!

## 🎯 Project Overview

This minimal application processes PDF and text resumes, extracts and cleans content, and classifies them into 24 job categories using a trained Random Forest model. Built as a standalone demo with 76.3% accuracy on 2,483 resumes.

### ✨ Key Features

- **Single File Application**: Everything in one `minimal_app.py` file
- **PDF & Text Support**: Upload PDF resumes or paste text directly
- **24 Job Categories**: Classifies into 24 professional categories
- **76.3% Accuracy**: Trained Random Forest model
- **Real-time Processing**: Instant classification results
- **Interactive UI**: Clean Streamlit interface with confidence scores
- **Easy Setup**: Just 6 dependencies, runs anywhere

## 📁 Project Structure

```
resume-classification-demo/
├── minimal_app.py              # Complete standalone application
├── requirements.txt            # Dependencies (only 6!)
├── README.md                   # This file
├── data/
│   └── raw/
│       └── Resume1.csv         # Training dataset (2,483 resumes)
└── models/                     # Pre-trained models
    ├── best_model.pkl         # Random Forest classifier
    ├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
    ├── label_encoder.pkl      # Label encoder
    └── category_mapping.pkl   # Category mappings
```

## 🎯 Target Categories

The system classifies resumes into 24 job categories:

- **ACCOUNTANT** - Accounting and financial professionals
- **ADVOCATE** - Legal professionals and lawyers
- **AGRICULTURE** - Agricultural and farming specialists
- **APPAREL** - Fashion and clothing industry professionals
- **ARTS** - Creative and artistic professionals
- **AUTOMOBILE** - Automotive industry specialists
- **AVIATION** - Aviation and aerospace professionals
- **BANKING** - Banking and financial services
- **BPO** - Business Process Outsourcing professionals
- **BUSINESS-DEVELOPMENT** - Business development and strategy
- **CHEF** - Culinary and food service professionals
- **CONSTRUCTION** - Construction and building industry
- **CONSULTANT** - Management and technical consultants
- **DESIGNER** - Design and creative professionals
- **DIGITAL-MEDIA** - Digital marketing and media specialists
- **ENGINEERING** - General engineering professionals
- **FINANCE** - Financial analysts and specialists
- **FITNESS** - Health and fitness professionals
- **HEALTHCARE** - Medical and healthcare professionals
- **HR** - Human resources professionals
- **INFORMATION-TECHNOLOGY** - IT and software professionals
- **PUBLIC-RELATIONS** - PR and communications specialists
- **SALES** - Sales and marketing professionals
- **TEACHER** - Education and teaching professionals

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/faiz16ahmad/Resume-Classifier-NLP.git
cd Resume-Classifier-NLP
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run minimal_app.py
```

4. **Open your browser:**
The app will automatically open at `http://localhost:8501`

**That's it! No complex setup, no configuration files, no additional scripts needed.** 🎉

### 📝 Using the Application

1. **Choose Input Method**: 
   - **Paste Text**: Copy and paste resume content directly
   - **Upload File**: Drag & drop PDF or text files

2. **Get Results**: 
   - Click "🚀 Classify Resume" 
   - View predicted category and confidence score
   - See processing statistics and text preview

3. **Understand Results**:
   - **High Confidence** (>70%): Very reliable prediction
   - **Medium Confidence** (50-70%): Good prediction
   - **Low Confidence** (<50%): Less certain, review manually

## ⚙️ Technical Details

The application uses:

- **Algorithm**: Random Forest Classifier
- **Features**: 5,000 TF-IDF features (unigrams + bigrams)
- **Text Processing**: NLTK for preprocessing and tokenization
- **PDF Support**: PyPDF2 for text extraction
- **Interface**: Streamlit for the web UI
- **Model Storage**: Joblib for model persistence

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

- **Classification Accuracy**: 76.3% (Random Forest model)
- **F1-Score**: 0.747 (excellent for multi-class classification)
- **Dataset Size**: 2,483 resumes across 24 categories
- **Processing Speed**: <5 seconds per resume
- **Memory Usage**: <2GB for batch processing
- **Feature Engineering**: 5,000 TF-IDF features

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
