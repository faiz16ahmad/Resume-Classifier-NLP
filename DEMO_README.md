# 🎯 Resume Classification Demo - Minimal App

A **single-file Streamlit application** that demonstrates AI-powered resume classification. Perfect for interviews, demos, and quick showcases!

## ✨ Features

- **Single File**: Everything in one `minimal_app.py` file
- **24 Categories**: Classifies resumes into 24 job categories
- **76.3% Accuracy**: Trained Random Forest model
- **Real-time Processing**: Instant classification results
- **Interactive UI**: Clean Streamlit interface
- **Text & File Input**: Paste text or upload .txt/.pdf files

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r minimal_requirements.txt
```

### 2. Run the Demo

```bash
streamlit run minimal_app.py
```

### 3. Open Browser

The app will automatically open at `http://localhost:8501`

## 📊 Model Details

- **Dataset**: 2,483 resumes across 24 categories
- **Algorithm**: Random Forest Classifier
- **Features**: 5,000 TF-IDF features (unigrams + bigrams)
- **Performance**: 76.3% accuracy, 0.747 F1-score
- **Processing**: <1 second per resume

## 🎯 Supported Categories

1. **ACCOUNTANT** - Accounting professionals
2. **ADVOCATE** - Legal professionals
3. **AGRICULTURE** - Agricultural specialists
4. **APPAREL** - Fashion industry
5. **ARTS** - Creative professionals
6. **AUTOMOBILE** - Automotive industry
7. **AVIATION** - Aviation professionals
8. **BANKING** - Banking services
9. **BPO** - Business Process Outsourcing
10. **BUSINESS-DEVELOPMENT** - Business strategy
11. **CHEF** - Culinary professionals
12. **CONSTRUCTION** - Construction industry
13. **CONSULTANT** - Management consultants
14. **DESIGNER** - Design professionals
15. **DIGITAL-MEDIA** - Digital marketing
16. **ENGINEERING** - General engineering
17. **FINANCE** - Financial specialists
18. **FITNESS** - Health & fitness
19. **HEALTHCARE** - Medical professionals
20. **HR** - Human resources
21. **INFORMATION-TECHNOLOGY** - IT professionals
22. **PUBLIC-RELATIONS** - PR specialists
23. **SALES** - Sales professionals
24. **TEACHER** - Education professionals

## 💡 Usage Tips

### For Best Results:

- Include **skills, experience, education**
- Use **complete sentences** and **professional language**
- Provide **detailed job descriptions** and **responsibilities**
- Include **industry-specific keywords**

### Sample Input:

```
Experienced software engineer with 5 years in Python development.
Skilled in Django, Flask, machine learning, and data analysis.
Built web applications and implemented ML algorithms.
Bachelor's degree in Computer Science.
```

## 🔧 Technical Implementation

The app demonstrates:

- **Text Preprocessing**: URL/email removal, contraction expansion, stop word filtering
- **Feature Engineering**: TF-IDF vectorization with 5,000 features
- **Classification**: Random Forest with probability scores
- **Real-time Processing**: Streamlit caching for model loading
- **Error Handling**: Graceful failure management

## 📁 File Structure

```
minimal_app.py              # Complete standalone application
minimal_requirements.txt    # Dependencies
models/                     # Pre-trained models (required)
├── best_model.pkl         # Random Forest classifier
├── tfidf_vectorizer.pkl   # TF-IDF vectorizer
├── label_encoder.pkl      # Label encoder
└── category_mapping.pkl   # Category mappings
```

## 🎪 Demo Features

- **Interactive UI**: Clean, professional interface
- **Real-time Results**: Instant classification with confidence scores
- **Multiple Input Methods**: Text area or PDF/text file upload
- **Detailed Analytics**: Processing statistics and text preview
- **Confidence Indicators**: Visual feedback on prediction quality
- **Category Browser**: Sidebar with all supported categories

## 🏆 Perfect For

- **Technical Interviews**: Showcase ML/NLP skills
- **Client Demos**: Quick proof-of-concept
- **Portfolio Projects**: Standalone demo piece
- **Educational Purposes**: Teaching NLP concepts
- **Rapid Prototyping**: Fast iteration and testing

## 🚀 Ready to Impress!

This minimal app packs the full power of the resume classification system into a single, easy-to-run file. Perfect for showcasing your NLP and ML capabilities in any setting!

---

_Built with ❤️ using Streamlit, scikit-learn, and NLTK_
