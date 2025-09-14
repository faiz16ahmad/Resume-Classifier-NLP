"""
CSV export functionality for the Resume Classification NLP System.

This module provides comprehensive CSV export capabilities for classification results,
organization results, and system analytics with proper formatting and validation.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
import logging

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

from config import EXPORT_CONFIG
from src.logger_setup import get_logger, PerformanceLogger
from src.classification_service import ClassificationResult
from src.file_organizer import OrganizationResult

# Initialize logger
logger = get_logger(__name__)


class CSVExportError(Exception):
    """Custom exception for CSV export errors."""
    pass


@dataclass
class ExportResult:
    """Data class for storing export operation results."""
    
    export_file: str
    record_count: int
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    file_size: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class CSVExporter:
    """
    CSV export system for resume classification results and analytics.
    
    This class handles the export of classification results, organization results,
    and system statistics to CSV format with proper formatting and validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the CSV exporter.
        
        Args:
            config: Optional export configuration (uses EXPORT_CONFIG if None)
        """
        self.config = config or EXPORT_CONFIG.copy()
        logger.info("Initialized CSV exporter with configuration")
    
    def export_classification_results(
        self,
        results: List[ClassificationResult],
        output_file: Union[str, Path],
        include_probabilities: bool = True
    ) -> ExportResult:
        """Export classification results to CSV format."""
        output_path = Path(output_file)
        start_time = datetime.now()
        
        logger.info(f"Exporting {len(results)} classification results to {output_path}")
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            headers = [
                'resume_filename', 'predicted_category', 'confidence_score',
                'processing_time', 'raw_text_length', 'cleaned_text_length',
                'processing_timestamp', 'error_message'
            ]
            
            if include_probabilities and results:
                all_categories = set()
                for result in results:
                    if result.all_probabilities:
                        all_categories.update(result.all_probabilities.keys())
                
                prob_headers = [f'prob_{cat.lower().replace(" ", "_")}' for cat in sorted(all_categories)]
                headers.extend(prob_headers)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(
                    csvfile, 
                    fieldnames=headers,
                    delimiter=self.config.get('csv_delimiter', ',')
                )
                
                writer.writeheader()
                
                for result in results:
                    row = {
                        'resume_filename': result.resume_filename,
                        'predicted_category': result.predicted_category,
                        'confidence_score': f"{result.confidence_score:.4f}",
                        'processing_time': f"{result.processing_time:.3f}",
                        'raw_text_length': result.raw_text_length,
                        'cleaned_text_length': result.cleaned_text_length,
                        'processing_timestamp': result.processing_timestamp.strftime(
                            self.config.get('date_format', '%Y-%m-%d %H:%M:%S')
                        ),
                        'error_message': result.error_message or ''
                    }
                    
                    if include_probabilities and result.all_probabilities:
                        for cat in sorted(all_categories):
                            prob_key = f'prob_{cat.lower().replace(" ", "_")}'
                            prob_value = result.all_probabilities.get(cat, 0.0)
                            row[prob_key] = f"{prob_value:.4f}"
                    
                    writer.writerow(row)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            file_size = output_path.stat().st_size
            
            logger.info(f"Successfully exported {len(results)} records to {output_path}")
            
            return ExportResult(
                export_file=str(output_path),
                record_count=len(results),
                success=True,
                processing_time=processing_time,
                file_size=file_size,
                timestamp=start_time
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to export classification results: {e}")
            
            return ExportResult(
                export_file=str(output_path),
                record_count=0,
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                timestamp=start_time
            )    

    def generate_export_filename(
        self,
        base_name: str,
        export_type: str = "results",
        include_timestamp: bool = True
    ) -> str:
        """Generate standardized export filename."""
        timestamp_str = ""
        if include_timestamp:
            timestamp_str = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return f"{base_name}_{export_type}{timestamp_str}.csv"
    
    def validate_export_data(
        self,
        data: List[Any],
        data_type: str = "classification"
    ) -> bool:
        """Validate export data before processing."""
        if not data:
            logger.warning(f"No {data_type} data provided for export")
            return False
        
        if not isinstance(data, list):
            logger.error(f"Export data must be a list, got {type(data)}")
            return False
        
        if data_type == "classification":
            if not all(isinstance(item, ClassificationResult) for item in data):
                logger.error("All items must be ClassificationResult objects")
                return False
        elif data_type == "organization":
            if not all(isinstance(item, OrganizationResult) for item in data):
                logger.error("All items must be OrganizationResult objects")
                return False
        
        logger.debug(f"Validated {len(data)} {data_type} records for export")
        return True


def export_classification_results_csv(
    results: List[ClassificationResult],
    output_file: Union[str, Path],
    include_probabilities: bool = True
) -> ExportResult:
    """Convenience function to export classification results to CSV."""
    try:
        exporter = CSVExporter()
        return exporter.export_classification_results(results, output_file, include_probabilities)
    except Exception as e:
        logger.error(f"Classification results export failed: {e}")
        raise CSVExportError(f"Export failed: {e}")


if __name__ == "__main__":
    # Test the CSV export functionality
    print("Testing CSV Export System...")
    
    try:
        print("\\n--- Testing CSV Exporter Initialization ---")
        
        exporter = CSVExporter()
        print(f"✅ Created CSV exporter")
        
        # Test filename generation
        print("\\n--- Testing Filename Generation ---")
        
        test_filenames = [
            exporter.generate_export_filename("resume_results", "classification"),
            exporter.generate_export_filename("organization_data", "organization", False),
            exporter.generate_export_filename("system_stats", "statistics")
        ]
        
        for filename in test_filenames:
            print(f"   📄 Generated: {filename}")
        
        # Test data validation
        print("\\n--- Testing Data Validation ---")
        
        mock_classification_results = []
        for i in range(3):
            result = ClassificationResult(
                resume_filename=f"resume_{i}.pdf",
                predicted_category=["Python Developer", "Java Developer", "Data Scientist"][i],
                confidence_score=0.8 + i * 0.05,
                all_probabilities={
                    "Python Developer": 0.8 + i * 0.05,
                    "Java Developer": 0.1,
                    "Data Scientist": 0.1 - i * 0.05
                },
                processing_timestamp=datetime.now(),
                processing_time=1.0 + i * 0.2,
                raw_text_length=1000 + i * 100,
                cleaned_text_length=800 + i * 80
            )
            mock_classification_results.append(result)
        
        is_valid = exporter.validate_export_data(mock_classification_results, "classification")
        print(f"   ✅ Classification data validation: {is_valid}")
        
        # Test CSV export
        print("\\n--- Testing CSV Export Operations ---")
        
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            cls_export_file = temp_path / "classification_results.csv"
            cls_export_result = exporter.export_classification_results(
                mock_classification_results, cls_export_file, include_probabilities=True
            )
            
            print(f"   📊 Classification export: {cls_export_result.success}")
            print(f"      Records: {cls_export_result.record_count}")
            print(f"      File size: {cls_export_result.file_size} bytes")
            print(f"      Processing time: {cls_export_result.processing_time:.3f}s")
            
            if cls_export_file.exists():
                with open(cls_export_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:3]
                    print(f"   📄 CSV preview (first 3 lines):")
                    for line in lines:
                        print(f"      {line.strip()}")
        
        print("\\n✅ CSV export system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()