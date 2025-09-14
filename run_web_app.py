#!/usr/bin/env python3
"""
Startup script for the Resume Classification Web Application.

This script initializes the project structure and starts the Streamlit web application.
"""

import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    """Initialize project and start web application."""
    try:
        # Import and run configuration setup
        from config import ensure_directories_exist, validate_config
        
        print("🚀 Starting Resume Classification Web Application...")
        print("=" * 50)
        
        # Validate configuration
        print("📋 Validating configuration...")
        if not validate_config():
            print("❌ Configuration validation failed!")
            return 1
        print("✅ Configuration validated successfully!")
        
        # Ensure directories exist
        print("📁 Creating project directories...")
        ensure_directories_exist()
        print("✅ Project directories created!")
        
        # Check if models exist
        from config import MODELS_DIR
        model_files = list(MODELS_DIR.glob("*.pkl"))
        
        if not model_files:
            print("⚠️  Warning: No trained models found!")
            print("   Please run the training pipeline first:")
            print("   python src/training_pipeline.py")
            print()
        else:
            print(f"✅ Found {len(model_files)} model files")
        
        # Start Streamlit application
        print("🌐 Starting Streamlit web application...")
        print("   The application will open in your default browser")
        print("   Press Ctrl+C to stop the application")
        print("=" * 50)
        
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(project_root / "src" / "web_app.py"),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1

if __name__ == "__main__":
    exit(main())