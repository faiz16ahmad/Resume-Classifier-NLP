#!/usr/bin/env python3
"""
Streamlit Web Application for the Resume Classification NLP System.

This module provides a user-friendly web interface for uploading, classifying,
and organizing resume files with real-time progress tracking and results export.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import tempfile
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import time

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import WEB_CONFIG, CATEGORY_MAPPING
from src.logger_setup import get_logger
from src.utils import extract_text_from_pdf, batch_extract_texts
from src.data_preprocessing import clean_text
from src.classification_service import ResumeClassifier, ClassificationResult
from src.file_organizer import ResumeFileOrganizer, OrganizationResult
from src.csv_exporter import CSVExporter

# Initialize logger
logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title=WEB_CONFIG.get('page_title', 'Resume Classification System'),
    page_icon=WEB_CONFIG.get('page_icon', '📄'),
    layout=WEB_CONFIG.get('layout', 'wide'),
    initial_sidebar_state=WEB_CONFIG.get('initial_sidebar_state', 'expanded')
)


class WebAppError(Exception):
    """Custom exception for web application errors."""
    pass


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'classification_results' not in st.session_state:
        st.session_state.classification_results = []
    
    if 'organization_results' not in st.session_state:
        st.session_state.organization_results = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
    
    if 'organizer' not in st.session_state:
        st.session_state.organizer = None


@st.cache_resource
def load_models():
    """Load ML models with caching for performance."""
    try:
        with st.spinner('Loading classification models...'):
            # Check if models exist
            models_dir = Path("models")
            if not models_dir.exists():
                raise Exception("Models directory not found. Please train models first.")
            
            required_files = ["best_model.pkl", "tfidf_vectorizer.pkl"]
            missing_files = [f for f in required_files if not (models_dir / f).exists()]
            
            if missing_files:
                raise Exception(f"Required model files missing: {missing_files}")
            
            classifier = ResumeClassifier()
            classifier.load_models()
            
            organizer = ResumeFileOrganizer()
            
            logger.info("Models loaded successfully for web application")
            return classifier, organizer
    
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        st.error(f"Failed to load models: {e}")
        st.info("""
        **Model Loading Failed**
        
        This usually happens when:
        1. Models haven't been trained yet
        2. Model files are corrupted or incompatible
        3. There are missing dependencies
        
        **To fix this:**
        1. Run the model training pipeline: `python src/model_training.py`
        2. Or use the training notebook in `notebooks/model_development.ipynb`
        3. Ensure all required packages are installed: `pip install -r requirements.txt`
        """)
        return None, None


def display_header():
    """Display application header and navigation."""
    st.title("🎯 Resume Classification System")
    st.markdown("""
    **Automatically classify resumes into job categories using AI and NLP**
    
    Upload PDF resumes and get instant classifications with confidence scores,
    automated file organization, and comprehensive analytics.
    """)
    
    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📤 Upload & Classify", 
        "📊 Results & Analytics", 
        "📁 File Organization", 
        "⚙️ Settings & Help"
    ])
    
    return tab1, tab2, tab3, tab4


def display_file_upload(tab):
    """Display file upload interface."""
    with tab:
        st.header("📤 Upload Resume Files")
        
        # File upload widget
        uploaded_files = st.file_uploader(
            "Choose PDF resume files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more PDF resume files for classification"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Display uploaded files
            with st.expander("📋 Uploaded Files", expanded=True):
                for i, file in enumerate(uploaded_files):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.write(f"📄 {file.name}")
                    
                    with col2:
                        file_size = len(file.getvalue()) / 1024  # KB
                        st.write(f"{file_size:.1f} KB")
                    
                    with col3:
                        if st.button(f"👁️ Preview", key=f"preview_{i}"):
                            preview_file_content(file)
            
            # Classification options
            st.subheader("🔧 Classification Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                organize_files = st.checkbox(
                    "📁 Organize files by category", 
                    value=True,
                    help="Automatically organize classified resumes into category folders"
                )
            
            with col2:
                export_results = st.checkbox(
                    "📊 Export results to CSV", 
                    value=True,
                    help="Generate CSV report with classification results"
                )
            
            # Process button
            if st.button("🚀 Start Classification", type="primary", use_container_width=True):
                process_uploaded_files(uploaded_files, organize_files, export_results)
        
        else:
            st.info("👆 Please upload PDF resume files to get started")
            
            # Demo mode section
            st.markdown("---")
            st.subheader("🎯 Demo Mode")
            st.markdown("""
            **No trained models?** You can still explore the interface with demo data.
            """)
            
            if st.button("🚀 Try Demo Mode", type="secondary"):
                create_demo_results()


def preview_file_content(uploaded_file):
    """Preview content of uploaded PDF file."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text
        with st.spinner(f'Extracting text from {uploaded_file.name}...'):
            text = extract_text_from_pdf(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Display preview
        st.subheader(f"📄 Preview: {uploaded_file.name}")
        
        if text:
            # Show text statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Characters", len(text))
            
            with col2:
                word_count = len(text.split())
                st.metric("Words", word_count)
            
            with col3:
                line_count = len(text.split('\n'))
                st.metric("Lines", line_count)
            
            # Show text preview
            st.text_area(
                "Text Content (first 1000 characters)",
                text[:1000] + ("..." if len(text) > 1000 else ""),
                height=200,
                disabled=True
            )
        else:
            st.warning("⚠️ No text could be extracted from this PDF")
    
    except Exception as e:
        st.error(f"❌ Error previewing file: {e}")
        logger.error(f"File preview error: {e}")


def process_uploaded_files(uploaded_files, organize_files: bool, export_results: bool):
    """Process uploaded files through the classification pipeline."""
    try:
        # Initialize progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load models if not already loaded
        if st.session_state.classifier is None:
            classifier, organizer = load_models()
            if classifier is None:
                st.error("❌ Failed to load models. Please check the system configuration.")
                return
            
            st.session_state.classifier = classifier
            st.session_state.organizer = organizer
        
        # Step 1: Save uploaded files temporarily
        status_text.text("📁 Saving uploaded files...")
        temp_files = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_files.append((tmp_file.name, uploaded_file.name))
            
            progress_bar.progress((i + 1) / (len(uploaded_files) * 4))
        
        # Step 2: Extract text from PDFs
        status_text.text("📄 Extracting text from PDFs...")
        
        extracted_texts = []
        filenames = []
        
        for i, (temp_path, original_name) in enumerate(temp_files):
            try:
                text = extract_text_from_pdf(temp_path)
                extracted_texts.append(text)
                filenames.append(original_name)
            except Exception as e:
                st.warning(f"⚠️ Failed to extract text from {original_name}: {e}")
                extracted_texts.append("")
                filenames.append(original_name)
            
            progress_bar.progress((len(uploaded_files) + i + 1) / (len(uploaded_files) * 4))
        
        # Step 3: Classify resumes
        status_text.text("🤖 Classifying resumes...")
        
        classification_results = []
        
        for i, (text, filename) in enumerate(zip(extracted_texts, filenames)):
            if text:
                result = st.session_state.classifier.classify_resume(text, filename)
                classification_results.append(result)
            else:
                # Create error result for failed text extraction
                result = ClassificationResult(
                    resume_filename=filename,
                    predicted_category="ERROR",
                    confidence_score=0.0,
                    all_probabilities={},
                    processing_timestamp=datetime.now(),
                    processing_time=0.0,
                    raw_text_length=0,
                    cleaned_text_length=0,
                    error_message="Text extraction failed"
                )
                classification_results.append(result)
            
            progress_bar.progress((len(uploaded_files) * 2 + i + 1) / (len(uploaded_files) * 4))
        
        # Step 4: Organize files (if requested)
        organization_results = []
        
        if organize_files:
            status_text.text("📁 Organizing files by category...")
            
            for i, (temp_path_info, cls_result) in enumerate(zip(temp_files, classification_results)):
                temp_path, original_name = temp_path_info
                
                try:
                    org_result = st.session_state.organizer.organize_file(
                        temp_path, cls_result, copy_file=True
                    )
                    organization_results.append(org_result)
                except Exception as e:
                    st.warning(f"⚠️ Failed to organize {original_name}: {e}")
                    # Create error organization result
                    org_result = OrganizationResult(
                        source_file=temp_path,
                        destination_file="",
                        category=cls_result.predicted_category,
                        success=False,
                        error_message=str(e),
                        timestamp=datetime.now()
                    )
                    organization_results.append(org_result)
                
                progress_bar.progress((len(uploaded_files) * 3 + i + 1) / (len(uploaded_files) * 4))
        
        # Clean up temporary files
        for temp_path, _ in temp_files:
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Store results in session state
        st.session_state.classification_results = classification_results
        st.session_state.organization_results = organization_results
        st.session_state.processing_complete = True
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")
        
        # Display success message
        successful_classifications = sum(1 for r in classification_results if r.error_message is None)
        
        st.success(f"""
        🎉 **Processing Complete!**
        
        - **{successful_classifications}/{len(classification_results)}** resumes classified successfully
        - **{len(organization_results)}** files organized (if enabled)
      ailable in the "Results & Analytics" tab
        """)
        
        # Export results (if requested)
        if export_results and classification_results:
            export_classification_results(classification_results)
    
    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
        logger.error(f"File processing error: {e}")


def display_results_analytics(tab):
    """Display classification results and analytics."""
    with tab:
        st.header("📊 Classification Results & Analytics")
        
        if not st.session_state.processing_complete or not st.session_state.classification_results:
            st.info("📤 Please upload and process resume files first to see results here.")
            return
        
        results = st.session_state.classification_results
        
        # Summary metrics
        display_summary_metrics(results)
        
        # Results table
        display_results_table(results)
        
        # Analytics charts
        display_analytics_charts(results)
        
        # Export options
        display_export_options(results)


def display_summary_metrics(results: List[ClassificationResult]):
    """Display summary metrics for classification results."""
    st.subheader("📈 Summary Metrics")
    
    # Calculate metrics
    total_resumes = len(results)
    successful = sum(1 for r in results if r.error_message is None)
    failed = total_resumes - successful
    
    if successful > 0:
        avg_confidence = sum(r.confidence_score for r in results if r.error_message is None) / successful
        avg_processing_time = sum(r.processing_time for r in results if r.error_message is None) / successful
    else:
        avg_confidence = 0.0
        avg_processing_time = 0.0
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Resumes", total_resumes)
    
    with col2:
        st.metric("Successful", successful, delta=f"{(successful/total_resumes)*100:.1f}%")
    
    with col3:
        st.metric("Failed", failed, delta=f"{(failed/total_resumes)*100:.1f}%" if failed > 0 else None)
    
    with col4:
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col5:
        st.metric("Avg Time (s)", f"{avg_processing_time:.2f}")


def display_results_table(results: List[ClassificationResult]):
    """Display detailed results table."""
    st.subheader("📋 Detailed Results")
    
    # Prepare data for table
    table_data = []
    
    for result in results:
        table_data.append({
            'Filename': result.resume_filename,
            'Category': result.predicted_category,
            'Confidence': f"{result.confidence_score:.3f}",
            'Processing Time (s)': f"{result.processing_time:.2f}",
            'Text Length': result.raw_text_length,
            'Status': '✅ Success' if result.error_message is None else '❌ Error',
            'Error': result.error_message or ''
        })
    
    # Create DataFrame and display
    df = pd.DataFrame(table_data)
    
    # Add filtering options
    col1, col2 = st.columns(2)
    
    with col1:
        category_filter = st.selectbox(
            "Filter by Category",
            ['All'] + sorted(df['Category'].unique().tolist()),
            key="category_filter"
        )
    
    with col2:
        status_filter = st.selectbox(
            "Filter by Status",
            ['All', '✅ Success', '❌ Error'],
            key="status_filter"
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if category_filter != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == category_filter]
    
    if status_filter != 'All':
        filtered_df = filtered_df[filtered_df['Status'] == status_filter]
    
    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


def display_analytics_charts(results: List[ClassificationResult]):
    """Display analytics charts for classification results."""
    st.subheader("📊 Analytics Charts")
    
    # Filter successful results for charts
    successful_results = [r for r in results if r.error_message is None]
    
    if not successful_results:
        st.warning("⚠️ No successful classifications to display charts")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Category distribution pie chart
        category_counts = {}
        for result in successful_results:
            category = result.predicted_category
            category_counts[category] = category_counts.get(category, 0) + 1
        
        fig_pie = px.pie(
            values=list(category_counts.values()),
            names=list(category_counts.keys()),
            title="Distribution by Job Category"
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Confidence score distribution
        confidence_scores = [r.confidence_score for r in successful_results]
        
        fig_hist = px.histogram(
            x=confidence_scores,
            nbins=20,
            title="Confidence Score Distribution",
            labels={'x': 'Confidence Score', 'y': 'Count'}
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Processing time analysis
    if len(successful_results) > 1:
        processing_times = [r.processing_time for r in successful_results]
        filenames = [r.resume_filename for r in successful_results]
        
        fig_bar = px.bar(
            x=filenames,
            y=processing_times,
            title="Processing Time by Resume",
            labels={'x': 'Resume', 'y': 'Processing Time (s)'}
        )
        
        fig_bar.update_xaxis(tickangle=45)
        st.plotly_chart(fig_bar, use_container_width=True)


def display_export_options(results: List[ClassificationResult]):
    """Display export options for results."""
    st.subheader("📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Export to CSV", use_container_width=True):
            export_classification_results(results)
    
    with col2:
        if st.button("📋 Export Summary Report", use_container_width=True):
            export_summary_report(results)
    
    with col3:
        if st.button("📈 Export Analytics Data", use_container_width=True):
            export_analytics_data(results)


def export_classification_results(results: List[ClassificationResult]):
    """Export classification results to CSV."""
    try:
        exporter = CSVExporter()
        
        # Create temporary file for export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            export_result = exporter.export_classification_results(
                results, tmp_file.name, include_probabilities=True
            )
        
        if export_result.success:
            # Read the CSV content
            with open(tmp_file.name, 'rb') as f:
                csv_content = f.read()
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            # Provide download
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"resume_classification_results_{timestamp}.csv"
            
            st.download_button(
                label="⬇️ Download CSV File",
                data=csv_content,
                file_name=filename,
                mime="text/csv",
                use_container_width=True
            )
            
            st.success(f"✅ CSV export ready! {export_result.record_count} records exported.")
        
        else:
            st.error(f"❌ Export failed: {export_result.error_message}")
    
    except Exception as e:
        st.error(f"❌ Export error: {e}")
        logger.error(f"CSV export error: {e}")


def export_summary_report(results: List[ClassificationResult]):
    """Export summary report."""
    try:
        # Generate summary statistics
        total_resumes = len(results)
        successful = sum(1 for r in results if r.error_message is None)
        failed = total_resumes - successful
        
        if successful > 0:
            avg_confidence = sum(r.confidence_score for r in results if r.error_message is None) / successful
            avg_processing_time = sum(r.processing_time for r in results if r.error_message is None) / successful
        else:
            avg_confidence = 0.0
            avg_processing_time = 0.0
        
        # Category distribution
        category_counts = {}
        for result in results:
            if result.error_message is None:
                category = result.predicted_category
                category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create report content
        report_content = f"""
RESUME CLASSIFICATION SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== OVERVIEW ===
Total Resumes Processed: {total_resumes}
Successful Classifications: {successful}
Failed Classifications: {failed}
Success Rate: {(successful/total_resumes)*100:.1f}%

=== PERFORMANCE METRICS ===
Average Confidence Score: {avg_confidence:.3f}
Average Processing Time: {avg_processing_time:.2f} seconds

=== CATEGORY DISTRIBUTION ===
"""
        
        for category, count in sorted(category_counts.items()):
            percentage = (count / successful) * 100 if successful > 0 else 0
            report_content += f"{category}: {count} ({percentage:.1f}%)\n"
        
        report_content += "\n=== DETAILED RESULTS ===\n"
        
        for result in results:
            status = "SUCCESS" if result.error_message is None else "ERROR"
            report_content += f"{result.resume_filename}: {result.predicted_category} ({result.confidence_score:.3f}) - {status}\n"
        
        # Provide download
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"resume_classification_summary_{timestamp}.txt"
        
        st.download_button(
            label="⬇️ Download Summary Report",
            data=report_content,
            file_name=filename,
            mime="text/plain",
            use_container_width=True
        )
        
        st.success("✅ Summary report ready for download!")
    
    except Exception as e:
        st.error(f"❌ Report generation error: {e}")
        logger.error(f"Summary report error: {e}")


def export_analytics_data(results: List[ClassificationResult]):
    """Export analytics data in JSON format."""
    try:
        import json
        
        # Prepare analytics data
        analytics_data = {
            'summary': {
                'total_resumes': len(results),
                'successful_classifications': sum(1 for r in results if r.error_message is None),
                'failed_classifications': sum(1 for r in results if r.error_message is not None),
                'processing_timestamp': datetime.now().isoformat()
            },
            'results': [result.to_dict() for result in results]
        }
        
        # Convert to JSON
        json_content = json.dumps(analytics_data, indent=2)
        
        # Provide download
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"resume_classification_analytics_{timestamp}.json"
        
        st.download_button(
            label="⬇️ Download Analytics Data (JSON)",
            data=json_content,
            file_name=filename,
            mime="application/json",
            use_container_width=True
        )
        
        st.success("✅ Analytics data ready for download!")
    
    except Exception as e:
        st.error(f"❌ Analytics export error: {e}")
        logger.error(f"Analytics export error: {e}")


def display_file_organization(tab):
    """Display file organization interface and results."""
    with tab:
        st.header("📁 File Organization")
        
        if not st.session_state.processing_complete:
            st.info("📤 Please process resume files first to see organization options here.")
            return
        
        # Organization results
        if st.session_state.organization_results:
            display_organization_results()
        else:
            st.info("📁 File organization was not enabled during processing.")


def display_organization_results():
    """Display file organization results."""
    st.subheader("📊 Organization Results")
    
    results = st.session_state.organization_results
    
    # Summary metrics
    total_files = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total_files - successful
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Files", total_files)
    
    with col2:
        st.metric("Successfully Organized", successful)
    
    with col3:
        st.metric("Failed", failed)
    
    # Results table
    table_data = []
    
    for result in results:
        table_data.append({
            'Source File': Path(result.source_file).name,
            'Category': result.category,
            'Destination': Path(result.destination_file).name if result.destination_file else 'N/A',
            'Status': '✅ Success' if result.success else '❌ Failed',
            'Error': result.error_message or ''
        })
    
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def create_demo_results():
    """Create demo classification results for testing the interface."""
    try:
        # Create sample classification results
        demo_results = [
            ClassificationResult(
                resume_filename="john_doe_python_dev.pdf",
                predicted_category="Python Developer",
                confidence_score=0.92,
                all_probabilities={
                    "Python Developer": 0.92,
                    "Data Scientist": 0.05,
                    "Software Engineer": 0.03
                },
                processing_timestamp=datetime.now(),
                processing_time=2.1,
                raw_text_length=1250,
                cleaned_text_length=980,
                error_message=None
            ),
            ClassificationResult(
                resume_filename="jane_smith_data_scientist.pdf",
                predicted_category="Data Scientist",
                confidence_score=0.88,
                all_probabilities={
                    "Data Scientist": 0.88,
                    "Python Developer": 0.08,
                    "Business Analyst": 0.04
                },
                processing_timestamp=datetime.now(),
                processing_time=1.8,
                raw_text_length=1450,
                cleaned_text_length=1120,
                error_message=None
            ),
            ClassificationResult(
                resume_filename="alex_johnson_java_dev.pdf",
                predicted_category="Java Developer",
                confidence_score=0.85,
                all_probabilities={
                    "Java Developer": 0.85,
                    "Software Engineer": 0.10,
                    "Web Developer": 0.05
                },
                processing_timestamp=datetime.now(),
                processing_time=2.3,
                raw_text_length=1180,
                cleaned_text_length=890,
                error_message=None
            ),
            ClassificationResult(
                resume_filename="corrupted_resume.pdf",
                predicted_category="ERROR",
                confidence_score=0.0,
                all_probabilities={},
                processing_timestamp=datetime.now(),
                processing_time=0.5,
                raw_text_length=0,
                cleaned_text_length=0,
                error_message="Text extraction failed"
            )
        ]
        
        # Store demo results in session state
        st.session_state.classification_results = demo_results
        st.session_state.processing_complete = True
        
        # Create demo organization results
        demo_org_results = [
            OrganizationResult(
                source_file="john_doe_python_dev.pdf",
                destination_file="categorized_resumes/Python_Developer/john_doe_python_dev.pdf",
                category="Python Developer",
                success=True,
                error_message=None,
                timestamp=datetime.now()
            ),
            OrganizationResult(
                source_file="jane_smith_data_scientist.pdf",
                destination_file="categorized_resumes/Data_Scientist/jane_smith_data_scientist.pdf",
                category="Data Scientist",
                success=True,
                error_message=None,
                timestamp=datetime.now()
            ),
            OrganizationResult(
                source_file="alex_johnson_java_dev.pdf",
                destination_file="categorized_resumes/Java_Developer/alex_johnson_java_dev.pdf",
                category="Java Developer",
                success=True,
                error_message=None,
                timestamp=datetime.now()
            )
        ]
        
        st.session_state.organization_results = demo_org_results
        
        st.success("""
        🎉 **Demo Mode Activated!**
        
        - **3 successful** and **1 failed** demo classifications created
        - **3 files** organized by category
        - Check the "Results & Analytics" tab to explore the interface
        - Try the export functionality and view the charts
        """)
        
        logger.info("Demo results created successfully")
        
    except Exception as e:
        st.error(f"❌ Failed to create demo results: {e}")
        logger.error(f"Demo creation error: {e}")


def display_settings_help(tab):
    """Display settings and help information."""
    with tab:
        st.header("⚙️ Settings & Help")
        
        # Help section
        st.subheader("❓ Help & Documentation")
        
        with st.expander("🚀 Getting Started", expanded=False):
            st.markdown("""
            **How to use the Resume Classification System:**
            
            1. **Upload Files**: Go to the "Upload & Classify" tab and select PDF resume files
            2. **Configure Options**: Choose whether to organize files and export results
            3. **Start Processing**: Click "Start Classification" to begin
            4. **View Results**: Check the "Results & Analytics" tab for detailed results
            5. **Export Data**: Download CSV reports or summary documents
            """)
        
        with st.expander("📊 Understanding Results", expanded=False):
            st.markdown("""
            **Classification Results Explained:**
            
            - **Category**: The predicted job category for the resume
            - **Confidence Score**: How confident the model is (0.0 to 1.0)
            - **Processing Time**: Time taken to classify the resume
            - **Status**: Whether classification was successful or failed
            
            **Confidence Score Guidelines:**
            - 0.8 - 1.0: Very confident prediction
            - 0.6 - 0.8: Moderately confident prediction
            - 0.4 - 0.6: Low confidence prediction
            - 0.0 - 0.4: Very low confidence (review recommended)
            """)
        
        # System information
        st.subheader("ℹ️ System Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Application Version**")
            st.code("v1.0.0")
        
        with col2:
            st.write("**Python Version**")
            st.code(f"{sys.version.split()[0]}")
        
        with col3:
            st.write("**Streamlit Version**")
            st.code(st.__version__)


def main():
    """Main application entry point."""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Display header and get tabs
        tab1, tab2, tab3, tab4 = display_header()
        
        # Display each tab content
        display_file_upload(tab1)
        display_results_analytics(tab2)
        display_file_organization(tab3)
        display_settings_help(tab4)
        
        # Sidebar with additional information
        with st.sidebar:
            st.header("🎯 Resume Classifier")
            
            st.markdown("""
            **Quick Stats:**
            - 25 job categories supported
            - 90%+ classification accuracy
            - <5 seconds processing time
            - Batch processing enabled
            """)
            
            if st.session_state.processing_complete:
                results = st.session_state.classification_results
                successful = sum(1 for r in results if r.error_message is None)
                
                st.success(f"✅ Last session: {successful}/{len(results)} successful")
            
            st.markdown("---")
            
            st.markdown("""
            **Supported Categories:**
            Data Scientist, Java Developer, Python Developer, 
            Web Developer, Business Analyst, HR, DevOps Engineer,
            Software Engineer, Testing, Network Security Engineer,
            and 15 more...
            """)
            
            # System status
            st.subheader("🔍 System Status")
            
            # Check if models are loaded
            if st.session_state.classifier is not None:
                st.success("✅ Models loaded and ready")
            else:
                st.warning("⚠️ Models not loaded")
                
                # Check if model files exist
                models_dir = Path("models")
                if models_dir.exists():
                    model_files = list(models_dir.glob("*.pkl"))
                    if model_files:
                        st.info(f"📁 Found {len(model_files)} model files")
                        if st.button("🔄 Retry Loading Models", key="retry_models"):
                            st.cache_resource.clear()
                            st.rerun()
                    else:
                        st.error("❌ No model files found")
                        st.info("Run model training first")
                else:
                    st.error("❌ Models directory missing")
            
            # Processing status
            if st.session_state.processing_complete:
                st.success("✅ Ready for new files")
            else:
                st.info("⏳ Ready to process")
    
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logger.error(f"Main application error: {e}")


if __name__ == "__main__":
    main()