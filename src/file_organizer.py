"""
Automated file organization system for the Resume Classification NLP System.

This module provides functionality to automatically organize classified resumes
into category-specific folders with proper naming conventions and conflict resolution.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import re

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

from config import CATEGORIZED_RESUMES_DIR, CATEGORY_MAPPING, get_category_folders
from src.logger_setup import get_logger, PerformanceLogger
from src.classification_service import ClassificationResult

# Initialize logger
logger = get_logger(__name__)


class FileOrganizationError(Exception):
    """Custom exception for file organization errors."""
    pass


@dataclass
class OrganizationResult:
    """Data class for storing file organization results."""
    
    source_file: str
    destination_file: str
    category: str
    success: bool
    error_message: Optional[str] = None
    processing_time: float = 0.0
    file_size: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, any]:
        """Convert result to dictionary for serialization."""
        return {
            'source_file': self.source_file,
            'destination_file': self.destination_file,
            'category': self.category,
            'success': self.success,
            'error_message': self.error_message,
            'processing_time': float(self.processing_time),
            'file_size': int(self.file_size),
            'timestamp': self.timestamp.isoformat()
        }


class ResumeFileOrganizer:
    """
    Automated file organization system for classified resumes.
    
    This class handles the organization of resume files into category-specific
    directories based on classification results, with proper naming conventions
    and conflict resolution.
    """
    
    def __init__(self, base_output_dir: Optional[Path] = None):
        """
        Initialize the file organizer.
        
        Args:
            base_output_dir: Optional base directory for organized files (uses CATEGORIZED_RESUMES_DIR if None)
        """
        self.base_output_dir = Path(base_output_dir) if base_output_dir else CATEGORIZED_RESUMES_DIR
        self.category_mapping = CATEGORY_MAPPING.copy()
        self.category_folders = get_category_folders()
        
        logger.info(f"Initialized ResumeFileOrganizer with base directory: {self.base_output_dir}")
    
    def create_category_directories(self) -> Dict[str, Path]:
        """
        Create directory structure for all job categories.
        
        Returns:
            Dictionary mapping category names to directory paths
            
        Raises:
            FileOrganizationError: If directory creation fails
        """
        logger.info("Creating category directory structure...")
        
        try:
            # Ensure base directory exists
            self.base_output_dir.mkdir(parents=True, exist_ok=True)
            
            category_dirs = {}
            
            # Create directory for each category
            for category_id, category_name in self.category_mapping.items():
                # Sanitize category name for folder
                folder_name = self._sanitize_folder_name(category_name)
                category_dir = self.base_output_dir / folder_name
                
                # Create directory
                category_dir.mkdir(parents=True, exist_ok=True)
                category_dirs[category_name] = category_dir
                
                logger.debug(f"Created directory: {category_dir}")
            
            # Create additional utility directories
            utility_dirs = ['_unclassified', '_errors', '_duplicates']
            for util_dir in utility_dirs:
                util_path = self.base_output_dir / util_dir
                util_path.mkdir(parents=True, exist_ok=True)
                category_dirs[util_dir] = util_path
            
            logger.info(f"Created {len(category_dirs)} category directories")
            return category_dirs
            
        except Exception as e:
            logger.error(f"Failed to create category directories: {e}")
            raise FileOrganizationError(f"Directory creation failed: {e}")
    
    def _sanitize_folder_name(self, category_name: str) -> str:
        """
        Sanitize category name for use as folder name.
        
        Args:
            category_name: Original category name
            
        Returns:
            Sanitized folder name
        """
        # Replace spaces and special characters with underscores
        sanitized = re.sub(r'[^\w\s-]', '', category_name)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        sanitized = sanitized.strip('_').lower()
        
        return sanitized
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe file system operations.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'[^\w\s.-]', '', sanitized)
        sanitized = re.sub(r'\s+', '_', sanitized)
        
        # Ensure filename isn't too long (max 255 characters)
        if len(sanitized) > 200:  # Leave room for extensions and suffixes
            name, ext = os.path.splitext(sanitized)
            sanitized = name[:200-len(ext)] + ext
        
        return sanitized
    
    def _generate_unique_filename(self, target_dir: Path, filename: str) -> str:
        """
        Generate a unique filename to avoid conflicts.
        
        Args:
            target_dir: Target directory
            filename: Desired filename
            
        Returns:
            Unique filename
        """
        base_name, extension = os.path.splitext(filename)
        counter = 1
        unique_filename = filename
        
        while (target_dir / unique_filename).exists():
            unique_filename = f"{base_name}_{counter:03d}{extension}"
            counter += 1
            
            # Prevent infinite loop
            if counter > 9999:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                unique_filename = f"{base_name}_{timestamp}{extension}"
                break
        
        return unique_filename
    
    def organize_file(
        self, 
        source_file_path: Union[str, Path], 
        classification_result: ClassificationResult,
        copy_file: bool = True
    ) -> OrganizationResult:
        """
        Organize a single file based on classification result.
        
        Args:
            source_file_path: Path to the source file
            classification_result: Classification result containing predicted category
            copy_file: Whether to copy (True) or move (False) the file
            
        Returns:
            OrganizationResult with operation details
        """
        source_path = Path(source_file_path)
        start_time = datetime.now()
        
        logger.debug(f"Organizing file: {source_path} -> {classification_result.predicted_category}")
        
        try:
            # Validate source file
            if not source_path.exists():
                raise FileOrganizationError(f"Source file not found: {source_path}")
            
            # Get file size
            file_size = source_path.stat().st_size
            
            # Determine target category and directory
            category = classification_result.predicted_category
            
            # Handle error cases
            if classification_result.error_message or category == "ERROR":
                category_folder = "_errors"
            elif category not in self.category_mapping.values():
                category_folder = "_unclassified"
                logger.warning(f"Unknown category '{category}', moving to unclassified")
            else:
                category_folder = self._sanitize_folder_name(category)
            
            # Create target directory
            target_dir = self.base_output_dir / category_folder
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare filename
            original_filename = source_path.name
            sanitized_filename = self._sanitize_filename(original_filename)
            
            # Add confidence score to filename if available
            if hasattr(classification_result, 'confidence_score') and classification_result.confidence_score > 0:
                name, ext = os.path.splitext(sanitized_filename)
                confidence_str = f"_conf{classification_result.confidence_score:.3f}".replace('.', 'p')
                sanitized_filename = f"{name}{confidence_str}{ext}"
            
            # Generate unique filename
            unique_filename = self._generate_unique_filename(target_dir, sanitized_filename)
            target_path = target_dir / unique_filename
            
            # Perform file operation
            if copy_file:
                shutil.copy2(source_path, target_path)
                operation = "copied"
            else:
                shutil.move(str(source_path), str(target_path))
                operation = "moved"
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = OrganizationResult(
                source_file=str(source_path),
                destination_file=str(target_path),
                category=category,
                success=True,
                processing_time=processing_time,
                file_size=file_size,
                timestamp=start_time
            )
            
            logger.info(f"Successfully {operation} {source_path.name} to {category_folder}/{unique_filename}")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Failed to organize {source_path}: {e}")
            
            return OrganizationResult(
                source_file=str(source_path),
                destination_file="",
                category=classification_result.predicted_category if classification_result else "ERROR",
                success=False,
                error_message=str(e),
                processing_time=processing_time,
                file_size=0,
                timestamp=start_time
            )
    
    def organize_batch(
        self,
        file_paths: List[Union[str, Path]],
        classification_results: List[ClassificationResult],
        copy_files: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[OrganizationResult]:
        """
        Organize multiple files based on classification results.
        
        Args:
            file_paths: List of source file paths
            classification_results: List of classification results
            copy_files: Whether to copy (True) or move (False) files
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of OrganizationResult objects
            
        Raises:
            FileOrganizationError: If input validation fails
        """
        if len(file_paths) != len(classification_results):
            raise FileOrganizationError("Number of files and classification results must match")
        
        logger.info(f"Starting batch organization of {len(file_paths)} files")
        
        # Create category directories
        self.create_category_directories()
        
        results = []
        successful = 0
        failed = 0
        
        with PerformanceLogger(logger, f"Batch file organization: {len(file_paths)} files"):
            for i, (file_path, classification_result) in enumerate(zip(file_paths, classification_results)):
                try:
                    result = self.organize_file(file_path, classification_result, copy_files)
                    results.append(result)
                    
                    if result.success:
                        successful += 1
                    else:
                        failed += 1
                    
                    # Call progress callback if provided
                    if progress_callback:
                        try:
                            progress_callback(i + 1, len(file_paths), file_path, result)
                        except Exception as e:
                            logger.warning(f"Progress callback error: {e}")
                
                except Exception as e:
                    logger.error(f"Batch organization error for {file_path}: {e}")
                    failed += 1
                    
                    # Create error result
                    error_result = OrganizationResult(
                        source_file=str(file_path),
                        destination_file="",
                        category="ERROR",
                        success=False,
                        error_message=str(e),
                        timestamp=datetime.now()
                    )
                    results.append(error_result)
        
        logger.info(f"Batch organization completed: {successful} successful, {failed} failed")
        return results
    
    def get_organization_statistics(self, results: List[OrganizationResult]) -> Dict[str, any]:
        """
        Generate statistics from organization results.
        
        Args:
            results: List of organization results
            
        Returns:
            Dictionary with organization statistics
        """
        if not results:
            return {}
        
        try:
            # Filter successful results
            successful_results = [r for r in results if r.success]
            
            # Category distribution
            category_counts = {}
            total_file_size = 0
            processing_times = []
            
            for result in successful_results:
                category = result.category
                category_counts[category] = category_counts.get(category, 0) + 1
                total_file_size += result.file_size
                processing_times.append(result.processing_time)
            
            # Calculate statistics
            stats = {
                'total_files': len(results),
                'successful_organizations': len(successful_results),
                'failed_organizations': len(results) - len(successful_results),
                'success_rate': len(successful_results) / len(results) if results else 0.0,
                'category_distribution': category_counts,
                'total_file_size_mb': total_file_size / (1024 * 1024),
                'average_file_size_mb': (total_file_size / len(successful_results) / (1024 * 1024)) if successful_results else 0.0,
                'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0.0,
                'total_processing_time': sum(processing_times),
                'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
                'unique_categories': len(category_counts)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics calculation failed: {e}")
            return {'error': str(e)}
    
    def cleanup_empty_directories(self) -> int:
        """
        Remove empty category directories.
        
        Returns:
            Number of directories removed
        """
        removed_count = 0
        
        try:
            for category_dir in self.base_output_dir.iterdir():
                if category_dir.is_dir() and not any(category_dir.iterdir()):
                    category_dir.rmdir()
                    removed_count += 1
                    logger.debug(f"Removed empty directory: {category_dir}")
            
            logger.info(f"Cleaned up {removed_count} empty directories")
            return removed_count
            
        except Exception as e:
            logger.error(f"Directory cleanup failed: {e}")
            return 0
    
    def get_directory_structure(self) -> Dict[str, Dict[str, any]]:
        """
        Get current directory structure and file counts.
        
        Returns:
            Dictionary with directory structure information
        """
        structure = {}
        
        try:
            if not self.base_output_dir.exists():
                return structure
            
            for category_dir in self.base_output_dir.iterdir():
                if category_dir.is_dir():
                    files = list(category_dir.glob("*"))
                    file_count = len([f for f in files if f.is_file()])
                    total_size = sum(f.stat().st_size for f in files if f.is_file())
                    
                    structure[category_dir.name] = {
                        'file_count': file_count,
                        'total_size_mb': total_size / (1024 * 1024),
                        'path': str(category_dir)
                    }
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to get directory structure: {e}")
            return {'error': str(e)}


def organize_classified_resumes(
    file_paths: List[Union[str, Path]],
    classification_results: List[ClassificationResult],
    output_dir: Optional[Path] = None,
    copy_files: bool = True
) -> Tuple[List[OrganizationResult], Dict[str, any]]:
    """
    Convenience function to organize classified resumes.
    
    Args:
        file_paths: List of source file paths
        classification_results: List of classification results
        output_dir: Optional output directory (uses default if None)
        copy_files: Whether to copy (True) or move (False) files
        
    Returns:
        Tuple of (organization_results, statistics)
    """
    try:
        organizer = ResumeFileOrganizer(output_dir)
        results = organizer.organize_batch(file_paths, classification_results, copy_files)
        stats = organizer.get_organization_statistics(results)
        
        logger.info("Resume organization completed successfully")
        return results, stats
        
    except Exception as e:
        logger.error(f"Resume organization failed: {e}")
        raise FileOrganizationError(f"Organization failed: {e}")


if __name__ == "__main__":
    # Test the file organization functionality
    print("Testing Resume File Organization System...")
    
    try:
        # Test organizer initialization
        print("\\n--- Testing File Organizer Initialization ---")
        
        organizer = ResumeFileOrganizer()
        print(f"✅ Created organizer with base directory: {organizer.base_output_dir}")
        print(f"📁 Category folders: {len(organizer.category_folders)}")
        
        # Test directory creation
        print("\\n--- Testing Directory Creation ---")
        
        category_dirs = organizer.create_category_directories()
        print(f"✅ Created {len(category_dirs)} directories")
        
        for category, path in list(category_dirs.items())[:5]:  # Show first 5
            print(f"   📂 {category}: {path}")
        
        # Test filename sanitization
        print("\\n--- Testing Filename Sanitization ---")
        
        test_filenames = [
            "Resume with spaces.pdf",
            "Resume<>with|special*chars?.pdf",
            "Very_Long_Filename_That_Might_Exceed_System_Limits_And_Cause_Issues.pdf",
            "normal_resume.pdf"
        ]
        
        for filename in test_filenames:
            sanitized = organizer._sanitize_filename(filename)
            print(f"   📝 '{filename}' -> '{sanitized}'")
        
        # Test folder name sanitization
        print("\\n--- Testing Folder Name Sanitization ---")
        
        test_categories = [
            "Python Developer",
            "Network Security Engineer", 
            "C++ Developer",
            "Data Scientist & Analyst"
        ]
        
        for category in test_categories:
            sanitized = organizer._sanitize_folder_name(category)
            print(f"   📂 '{category}' -> '{sanitized}'")
        
        # Test mock organization (without actual files)
        print("\\n--- Testing Mock File Organization ---")
        
        # Create mock classification results
        from src.classification_service import ClassificationResult
        from datetime import datetime
        
        mock_results = [
            ClassificationResult(
                resume_filename="python_dev.pdf",
                predicted_category="Python Developer",
                confidence_score=0.85,
                all_probabilities={"Python Developer": 0.85},
                processing_timestamp=datetime.now(),
                processing_time=1.0,
                raw_text_length=1000,
                cleaned_text_length=800
            ),
            ClassificationResult(
                resume_filename="java_dev.pdf",
                predicted_category="Java Developer",
                confidence_score=0.92,
                all_probabilities={"Java Developer": 0.92},
                processing_timestamp=datetime.now(),
                processing_time=1.2,
                raw_text_length=1200,
                cleaned_text_length=900
            ),
            ClassificationResult(
                resume_filename="error_resume.pdf",
                predicted_category="ERROR",
                confidence_score=0.0,
                all_probabilities={},
                processing_timestamp=datetime.now(),
                processing_time=0.5,
                raw_text_length=500,
                cleaned_text_length=0,
                error_message="Classification failed"
            )
        ]
        
        # Test statistics calculation
        print("\\n--- Testing Statistics Calculation ---")
        
        # Create mock organization results
        mock_org_results = []
        for result in mock_results:
            org_result = OrganizationResult(
                source_file=f"test_files/{result.resume_filename}",
                destination_file=f"organized/{result.predicted_category.lower()}/{result.resume_filename}",
                category=result.predicted_category,
                success=result.error_message is None,
                error_message=result.error_message,
                processing_time=0.1,
                file_size=1024 * 100,  # 100KB
                timestamp=datetime.now()
            )
            mock_org_results.append(org_result)
        
        stats = organizer.get_organization_statistics(mock_org_results)
        print(f"   📊 Total files: {stats.get('total_files', 0)}")
        print(f"   ✅ Successful: {stats.get('successful_organizations', 0)}")
        print(f"   ❌ Failed: {stats.get('failed_organizations', 0)}")
        print(f"   📈 Success rate: {stats.get('success_rate', 0):.2%}")
        print(f"   📋 Category distribution: {stats.get('category_distribution', {})}")
        print(f"   💾 Total size: {stats.get('total_file_size_mb', 0):.2f} MB")
        
        # Test directory structure
        print("\\n--- Testing Directory Structure ---")
        
        structure = organizer.get_directory_structure()
        print(f"   📂 Found {len(structure)} directories")
        
        for dir_name, info in list(structure.items())[:5]:  # Show first 5
            print(f"   📁 {dir_name}: {info.get('file_count', 0)} files, {info.get('total_size_mb', 0):.2f} MB")
        
        print("\\n✅ File organization system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()