"""
PythonAnywhere specific configuration
This file contains settings optimized for PythonAnywhere deployment
"""

import os

# PythonAnywhere specific settings
class PythonAnywhereConfig:
    # Database configuration for PythonAnywhere
    SQLALCHEMY_DATABASE_URI = 'sqlite:///lostfound.db'
    
    # Static files configuration
    STATIC_URL_PATH = '/static'
    STATIC_FOLDER = 'static'
    
    # Upload configuration
    UPLOAD_FOLDER = 'static/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Security settings
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-here')
    
    # Email configuration (set these in PythonAnywhere environment variables)
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', '')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', '')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'smartlostandfound@gmail.com')
    
    # Disable training and heavy processing for PythonAnywhere
    PROCESSING_MODE = 'ultra_fast'
    ENABLE_TRAINING = 'false'
    
    # Disable COCO evaluation on startup for faster loading
    RENDER = 'true'
