#!/usr/bin/env python3
"""
Google Drive Model Downloader for PythonAnywhere
Downloads the trained model from Google Drive when local model is not available
"""

import os
import logging
import requests
from pathlib import Path

class GoogleDriveModelDownloader:
    """Handles downloading models from Google Drive"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_dir = Path('models')
        self.model_dir.mkdir(exist_ok=True)
        
        # Google Drive file ID from the URL
        self.gdrive_file_id = "1UPz_r23OqYrPWh76Tx0Vh4BqnyaVWXJR"
        self.local_model_path = self.model_dir / "best_model9.pth"
        
    def download_model(self):
        """Download the model from Google Drive"""
        try:
            # Check if model already exists
            if self.local_model_path.exists():
                self.logger.info(f"Model already exists at {self.local_model_path}")
                return str(self.local_model_path)
            
            self.logger.info("Downloading model from Google Drive...")
            
            # Try gdown first (more reliable for Google Drive)
            try:
                import gdown
                self.logger.info("Using gdown for download...")
                gdown.download(
                    f"https://drive.google.com/uc?id={self.gdrive_file_id}",
                    str(self.local_model_path),
                    quiet=False
                )
            except ImportError:
                self.logger.warning("gdown not available, falling back to requests...")
                # Fallback to requests method
                download_url = f"https://drive.google.com/uc?export=download&id={self.gdrive_file_id}"
                
                # Download the file with proper session handling
                session = requests.Session()
                
                # First request to get the confirmation page
                response = session.get(download_url, stream=True)
                response.raise_for_status()
                
                # Check if we got a confirmation page (Google Drive shows this for large files)
                if 'download' in response.text.lower() and 'confirm' in response.text.lower():
                    # Extract confirmation token
                    import re
                    confirm_token = re.search(r'confirm=([^&]+)', response.text)
                    if confirm_token:
                        confirm_url = f"https://drive.google.com/uc?export=download&confirm={confirm_token.group(1)}&id={self.gdrive_file_id}"
                        response = session.get(confirm_url, stream=True)
                        response.raise_for_status()
                
                # Save the file
                with open(self.local_model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
            
            # Verify the file was downloaded properly
            if self.local_model_path.exists() and self.local_model_path.stat().st_size > 0:
                file_size_mb = self.local_model_path.stat().st_size / (1024 * 1024)
                self.logger.info(f"Model downloaded successfully to {self.local_model_path} ({file_size_mb:.2f} MB)")
                return str(self.local_model_path)
            else:
                self.logger.error("Downloaded file is empty or doesn't exist")
                if self.local_model_path.exists():
                    self.local_model_path.unlink()  # Remove empty file
                return None
            
        except Exception as e:
            self.logger.error(f"Failed to download model from Google Drive: {e}")
            return None
    
    def get_model_path(self):
        """Get the model path, downloading if necessary"""
        # First check if model exists locally
        if self.local_model_path.exists():
            return str(self.local_model_path)
        
        # Try to download from Google Drive
        downloaded_path = self.download_model()
        if downloaded_path:
            return downloaded_path
        
        # Fallback to checking in root directory
        root_model_path = Path("best_model1.pth")
        if root_model_path.exists():
            return str(root_model_path)
        
        # Check for alternative model files
        alt_model_path = Path("best_model1.2.pth")
        if alt_model_path.exists():
            return str(alt_model_path)
        
        # Check for the new model9
        model9_path = Path("best_model9.pth")
        if model9_path.exists():
            return str(model9_path)
        
        self.logger.warning("No model file found locally or on Google Drive")
        return None
    
    def is_model_available(self):
        """Check if any model is available (local or downloadable)"""
        return (
            self.local_model_path.exists() or
            Path("best_model1.pth").exists() or
            Path("best_model1.2.pth").exists() or
            Path("best_model9.pth").exists()
        )

def get_model_path():
    """Convenience function to get the model path"""
    downloader = GoogleDriveModelDownloader()
    return downloader.get_model_path()

def ensure_model_available():
    """Ensure a model is available, downloading if necessary"""
    downloader = GoogleDriveModelDownloader()
    return downloader.get_model_path() is not None

if __name__ == "__main__":
    # Test the downloader
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
