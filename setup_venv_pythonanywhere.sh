#!/bin/bash
# Virtual Environment Setup for PythonAnywhere
# Run this script in your PythonAnywhere console after cloning the repository

echo "🐍 Setting up virtual environment for PythonAnywhere..."

# Create virtual environment
echo "Creating virtual environment..."
python3.10 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements in virtual environment..."
pip install -r requirements_pythonanywhere.txt

echo "✓ Virtual environment created and configured"
echo ""
echo "📋 To activate the virtual environment in PythonAnywhere:"
echo "   source /home/smartlostandfound/mysite/venv/bin/activate"
echo ""
echo "📋 Update your WSGI file to use the virtual environment:"
echo "   Change the shebang to: #!/home/smartlostandfound/mysite/venv/bin/python"
