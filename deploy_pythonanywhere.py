#!/usr/bin/env python3
"""
PythonAnywhere Deployment Script
This script helps set up the application for PythonAnywhere deployment
"""

import os
import sys
import subprocess

def create_directories():
    """Create necessary directories for the application"""
    directories = [
        'static/uploads',
        'static/img',
        'training_data/images',
        'training_data/labels',
        'models',
        'instance'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def setup_environment():
    """Set up environment variables for PythonAnywhere"""
    env_vars = {
        'FLASK_APP': 'app.py',
        'FLASK_ENV': 'production',
        'PROCESSING_MODE': 'ultra_fast',
        'ENABLE_TRAINING': 'false',
        'RENDER': 'true'
    }
    
    print("Environment variables to set in PythonAnywhere:")
    for key, value in env_vars.items():
        print(f"  {key}={value}")

def install_requirements():
    """Install requirements with PythonAnywhere optimizations"""
    print("Installing requirements...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--user', '-r', 'requirements_pythonanywhere.txt'], check=True)
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("You may need to install some packages manually in PythonAnywhere")

def main():
    print("🚀 Setting up BARYONYX for PythonAnywhere deployment...")
    
    create_directories()
    setup_environment()
    install_requirements()
    
    print("\n📋 Next steps:")
    print("1. Upload all files to PythonAnywhere")
    print("2. Set up environment variables in PythonAnywhere")
    print("3. Configure WSGI file path")
    print("4. Set up static files mapping")
    print("5. Reload your web app")
    
    print("\n🔧 PythonAnywhere Configuration:")
    print("- WSGI file: /home/fraarroyo/mysite/wsgi.py")
    print("- Static files: /home/fraarroyo/mysite/static/")
    print("- Media files: /home/fraarroyo/mysite/static/uploads/")

if __name__ == "__main__":
    main()
