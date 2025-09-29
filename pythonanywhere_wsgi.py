#!/home/smartlostandfound/mysite/venv/bin/python

import sys
import os

# Add your project directory to the Python path
project_home = '/home/smartlostandfound/mysite'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Add virtual environment site-packages to path
venv_site_packages = '/home/smartlostandfound/mysite/venv/lib/python3.10/site-packages'
if venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# Set the environment variable for the Flask app
os.environ['FLASK_APP'] = 'app.py'

# Import the Flask application
from app import app as application

if __name__ == "__main__":
    application.run()
