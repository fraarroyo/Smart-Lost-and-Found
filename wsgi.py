#!/usr/bin/python3.10

import sys
import os

# Add your project directory to the Python path
project_home = '/home/fraarroyo/mysite'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Set the environment variable for the Flask app
os.environ['FLASK_APP'] = 'app.py'

# Import the Flask application
from app import app as application

if __name__ == "__main__":
    application.run()
