# Resume Classification NLP System

A production-ready application that automatically categorizes resumes into 24 predefined job categories using Natural Language Processing and Machine Learning techniques.

## Core Features
- **Automated Resume Classification**: Processes PDF resumes and categorizes them with 76.3% accuracy using Random Forest
- **Web Interface**: Streamlit-based UI with drag-and-drop file upload and batch processing
- **Automated Organization**: Organizes classified resumes into category folders
- **Export Functionality**: CSV export with confidence scores and processing metrics
- **Real-time Processing**: <5 seconds per resume with progress tracking

## Target Categories
24 job categories including ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, and TEACHER.

## Performance Requirements
- Classification accuracy: 76.3% (achieved with Random Forest)
- F1-Score: 0.747 (excellent for multi-class classification)
- Dataset size: 2,483 resumes across 24 categories
- Processing speed: <5 seconds per resume
- Memory usage: <2GB for batch processing
- Feature engineering: 5,000 TF-IDF features