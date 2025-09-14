#!/usr/bin/env python3
"""
Minimal Resume Classification App - Single File Demo

A standalone Streamlit application for resume classification that demonstrates
the complete NLP pipeline in one file. Perfect for interviews and demos.

Usage: streamlit run minimal_app.py
"""

import streamlit as st
import joblib
import pickle
import re
import string
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

# NLTK imports with error handling
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data if not present
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Downloading required language data...")
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    
    STOP_WORDS = set(stopwords.words('english'))
    
except ImportError:
    st.error("NLTK not installed. Please run: pip install nltk")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Resume Classification Demo",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Category mapping (24 categories from Resume1 dataset)
CATEGORY_MAPPING = {
    0: "ACCOUNTANT",
    1: "ADVOCATE", 
    2: "AGRICULTURE",
    3: "APPAREL",
    4: "ARTS",
    5: "AUTOMOBILE",
    6: "AVIATION",
    7: "BANKING",
    8: "BPO",
    9: "BUSINESS-DEVELOPMENT",
    10: "CHEF",
    11: "CONSTRUCTION",
    12: "CONSULTANT",
    13: "DESIGNER",
    14: "DIGITAL-MEDIA",
    15: "ENGINEERING",
    16: "FINANCE",
    17: "FITNESS",
    18: "HEALTHCARE",
    19: "HR",
    20: "INFORMATION-TECHNOLOGY",
    21: "PUBLIC-RELATIONS",
    22: "SALES",
    23: "TEACHER"
}

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess resume text for classification.
    
    Args:
        text: Raw resume text
        
    Returns:
        Cleaned and preprocessed text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to string and strip
    text = str(text).strip()
    
    # Remove URLs
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(url_pattern, '', text)
    
    # Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '', text)
    
    # Remove phone numbers
    phone_pattern = r'\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b'
    text = re.sub(phone_pattern, '', text)
    
    # Expand common contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'ve": " have", "'ll": " will", "'d": " would",
        "'m": " am", "let's": "let us", "that's": "that is",
        "who's": "who is", "what's": "what is", "here's": "here is",
        "there's": "there is", "where's": "where is", "how's": "how is",
        "it's": "it is", "he's": "he is", "she's": "she is"
    }
    
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
        text = text.replace(contraction.title(), expansion.title())
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove stop words
    try:
        words = word_tokenize(text)
        words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
        text = ' '.join(words)
    except:
        # Fallback if tokenization fails
        words = text.split()
        words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
        text = ' '.join(words)
    
    return text.strip()


@st.cache_resource
def load_models():
    """
    Load the trained models with Streamlit caching for efficiency.
    
    Returns:
        Tuple of (model, vectorizer, label_encoder, category_mapping)
    """
    try:
        models_dir = Path("models")
        
        # Check if models directory exists
        if not models_dir.exists():
            st.error("Models directory not found. Please ensure the models are trained.")
            return None, None, None, None
        
        # Load the trained Random Forest model
        model_path = models_dir / "best_model.pkl"
        if not model_path.exists():
            st.error("Trained model not found. Please train the model first.")
            return None, None, None, None
        
        model = joblib.load(model_path)
        
        # Load TF-IDF vectorizer
        vectorizer_path = models_dir / "tfidf_vectorizer.pkl"
        if not vectorizer_path.exists():
            st.error("TF-IDF vectorizer not found.")
            return None, None, None, None
        
        vectorizer = joblib.load(vectorizer_path)
        
        # Load label encoder
        label_encoder_path = models_dir / "label_encoder.pkl"
        if label_encoder_path.exists():
            label_encoder = joblib.load(label_encoder_path)
        else:
            label_encoder = None
        
        # Load category mapping
        category_mapping_path = models_dir / "category_mapping.pkl"
        if category_mapping_path.exists():
            with open(category_mapping_path, 'rb') as f:
                category_mapping = pickle.load(f)
        else:
            category_mapping = CATEGORY_MAPPING
        
        return model, vectorizer, label_encoder, category_mapping
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


def classify_resume(text: str, model, vectorizer, label_encoder, category_mapping) -> tuple:
    """
    Classify a resume text and return the predicted category with confidence.
    
    Args:
        text: Preprocessed resume text
        model: Trained classification model
        vectorizer: TF-IDF vectorizer
        label_encoder: Label encoder (optional)
        category_mapping: Category mapping dictionary
        
    Returns:
        Tuple of (predicted_category, confidence_score)
    """
    try:
        # Transform text using TF-IDF vectorizer
        text_vector = vectorizer.transform([text])
        
        # Get prediction and probability
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        confidence = max(probabilities)
        
        # Convert prediction to category name
        if label_encoder:
            try:
                category = label_encoder.inverse_transform([prediction])[0]
            except:
                category = category_mapping.get(prediction, f"Category_{prediction}")
        else:
            category = category_mapping.get(prediction, f"Category_{prediction}")
        
        return category, confidence
        
    except Exception as e:
        st.error(f"Classification error: {e}")
        return "ERROR", 0.0


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🎯 Resume Classification Demo")
    st.markdown("""
    **AI-Powered Resume Categorization System**
    
    This demo showcases an NLP system that automatically classifies resumes into 24 job categories 
    using Random Forest and TF-IDF features. Trained on 2,483 resumes with 76.3% accuracy.
    """)
    
    # Load models
    with st.spinner("Loading AI models..."):
        model, vectorizer, label_encoder, category_mapping = load_models()
    
    if model is None:
        st.error("Failed to load models. Please check the models directory.")
        st.stop()
    
    st.success("✅ Models loaded successfully!")
    
    # Sidebar with information
    with st.sidebar:
        st.header("📊 Model Information")
        st.info(f"""
        **Dataset:** 2,483 resumes  
        **Categories:** 24 job types  
        **Algorithm:** Random Forest  
        **Accuracy:** 76.3%  
        **F1-Score:** 0.747  
        **Features:** 5,000 TF-IDF
        """)
        
        st.header("🎯 Supported Categories")
        categories_df = pd.DataFrame({
            'Category': list(category_mapping.values())
        })
        st.dataframe(categories_df, use_container_width=True)
    
    # Main interface
    st.header("📝 Enter Resume Text")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Paste Text", "Upload File"],
        horizontal=True
    )
    
    resume_text = ""
    
    if input_method == "Paste Text":
        resume_text = st.text_area(
            "Paste resume content here:",
            height=300,
            placeholder="Paste the resume text here... Include skills, experience, education, etc."
        )
    
    else:  # Upload File
        uploaded_file = st.file_uploader(
            "Upload resume file:",
            type=['txt'],
            help="Currently supports .txt files only"
        )
        
        if uploaded_file is not None:
            try:
                resume_text = str(uploaded_file.read(), "utf-8")
                st.success(f"File '{uploaded_file.name}' loaded successfully!")
                with st.expander("Preview uploaded content"):
                    st.text(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Classification button
    if st.button("🚀 Classify Resume", type="primary", use_container_width=True):
        if not resume_text.strip():
            st.warning("Please enter some resume text to classify.")
        else:
            with st.spinner("Analyzing resume..."):
                # Preprocess the text
                cleaned_text = preprocess_text(resume_text)
                
                if len(cleaned_text) < 20:
                    st.error("Resume text is too short after preprocessing. Please provide more detailed content.")
                else:
                    # Classify the resume
                    predicted_category, confidence = classify_resume(
                        cleaned_text, model, vectorizer, label_encoder, category_mapping
                    )
                    
                    # Display results
                    st.header("🎯 Classification Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            label="Predicted Category",
                            value=predicted_category.replace("-", " ").title()
                        )
                    
                    with col2:
                        st.metric(
                            label="Confidence Score",
                            value=f"{confidence:.1%}"
                        )
                    
                    # Confidence indicator
                    if confidence >= 0.7:
                        st.success(f"🎯 High confidence prediction!")
                    elif confidence >= 0.5:
                        st.warning(f"⚠️ Medium confidence prediction.")
                    else:
                        st.error(f"❌ Low confidence prediction.")
                    
                    # Additional details
                    with st.expander("📊 Processing Details"):
                        st.write("**Original text length:**", len(resume_text), "characters")
                        st.write("**Processed text length:**", len(cleaned_text), "characters")
                        st.write("**Text reduction:**", f"{(1 - len(cleaned_text)/len(resume_text)):.1%}")
                        
                        st.write("**Processed text preview:**")
                        st.text(cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Random Forest & TF-IDF | Built with Streamlit</p>
        <p>Demo of Resume Classification NLP System</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()