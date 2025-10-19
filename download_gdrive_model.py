#!/usr/bin/env python3
"""
Google Drive Model Downloader for BARYONYX System
Downloads the trained model from Google Drive and integrates it into the system
"""

import os
import json
import logging
import requests
from pathlib import Path
from datetime import datetime

class GoogleDriveModelDownloader:
    """Handles downloading and integrating models from Google Drive"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models_dir = Path('models')
        self.models_dir.mkdir(exist_ok=True)
        
        # Google Drive file ID from the URL
        self.gdrive_file_id = "1UPz_r23OqYrPWh76Tx0Vh4BqnyaVWXJR"
        self.model_path = Path("best_model9.pth")
        
    def download_model(self):
        """Download the model from Google Drive"""
        try:
            # Check if model already exists
            if self.model_path.exists():
                self.logger.info(f"Model already exists at {self.model_path}")
                return str(self.model_path)
            
            self.logger.info("Downloading model from Google Drive...")
            
            # Try gdown first (more reliable for Google Drive)
            try:
                import gdown
                self.logger.info("Using gdown for download...")
                gdown.download(
                    f"https://drive.google.com/uc?id={self.gdrive_file_id}",
                    str(self.model_path),
                    quiet=False
                )
            except ImportError:
                self.logger.warning("gdown not available, falling back to requests...")
                # Use the direct download URL with confirmation
                download_url = "https://drive.usercontent.google.com/download"
                params = {
                    'id': self.gdrive_file_id,
                    'export': 'download',
                    'confirm': 't',
                    'uuid': '35d1368e-e658-4d5a-828c-ed08202a23ee'
                }
                
                # Download the file with proper session handling
                session = requests.Session()
                response = session.get(download_url, params=params, stream=True)
                response.raise_for_status()
                
                # Save the file
                with open(self.model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
            
            # Verify the file was downloaded properly
            if self.model_path.exists() and self.model_path.stat().st_size > 0:
                file_size_mb = self.model_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"Model downloaded successfully ({file_size_mb:.2f} MB)")
                return str(self.model_path)
            else:
                self.logger.error("Downloaded file is empty or doesn't exist")
                if self.model_path.exists():
                    self.model_path.unlink()  # Remove empty file
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to download model from Google Drive: {e}")
            return None
    
    def integrate_with_system(self):
        """Integrate the downloaded model with the existing system"""
        try:
            if not self.model_path.exists():
                self.logger.error("Model file not found")
                return False
            
            # Copy to models directory
            models_model_path = self.models_dir / "best_model9.pth"
            if not models_model_path.exists():
                import shutil
                shutil.copy2(self.model_path, models_model_path)
                self.logger.info(f"Model copied to {models_model_path}")
            
            # Create integration report
            integration_report = {
                'download_time': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'models_dir_path': str(models_model_path),
                'file_size_mb': self.model_path.stat().st_size / (1024 * 1024),
                'integration_status': 'ready_for_use'
            }
            
            # Save integration report
            report_path = self.models_dir / 'model_integration_report.json'
            with open(report_path, 'w') as f:
                json.dump(integration_report, f, indent=2)
            
            self.logger.info(f"Integration report saved to {report_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate model: {e}")
            return False
    
    def get_model_path(self):
        """Get the model path, downloading if necessary"""
        # First check if model exists locally
        if self.model_path.exists():
            return str(self.model_path)
        
        # Try to download from Google Drive
        downloaded_path = self.download_model()
        if downloaded_path:
            return downloaded_path
        
        self.logger.warning("No model found locally or on Google Drive")
        return None
    
    def is_model_available(self):
        """Check if model is available (local or downloadable)"""
        return self.model_path.exists()

def download_and_integrate_model():
    """Main function to download and integrate the model"""
    logging.basicConfig(level=logging.INFO)
    downloader = GoogleDriveModelDownloader()
    
    print("🔍 Checking model availability...")
    if downloader.is_model_available():
        print("✅ Model is available")
        model_path = downloader.get_model_path()
        print(f"📁 Model path: {model_path}")
    else:
        print("❌ No model available")
        print("🔄 Attempting to download from Google Drive...")
        model_path = downloader.download_model()
        if model_path:
            print(f"✅ Model downloaded to: {model_path}")
        else:
            print("❌ Failed to download model")
            return False
    
    # Integrate with system
    print("\n🔧 Integrating model with system...")
    if downloader.integrate_with_system():
        print("✅ Model integration completed")
        return True
    else:
        print("❌ Model integration failed")
        return False

if __name__ == "__main__":
    success = download_and_integrate_model()
    if success:
        print("\n🎉 Model download and integration completed successfully!")
        print("📋 The model is now ready to use in your BARYONYX system")
    else:
        print("\n❌ Model download and integration failed!")
