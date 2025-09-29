# PythonAnywhere Deployment Guide

This guide will help you deploy your BARYONYX Lost and Found Management System to PythonAnywhere.

## Prerequisites

1. **PythonAnywhere Account**: Sign up at [pythonanywhere.com](https://pythonanywhere.com)
2. **Git Access**: Your code is already on GitHub at `https://github.com/fraarroyo/Smart-Lost-and-Found.git`

## Step 1: Set Up PythonAnywhere Environment

### 1.1 Create a New Web App
1. Log into PythonAnywhere
2. Go to the **Web** tab
3. Click **Add a new web app**
4. Choose **Manual configuration**
5. Select **Python 3.10** (or latest available)
6. Click **Next**

### 1.2 Configure the Web App
- **Source code**: `/home/smartlostandfound/mysite`
- **Working directory**: `/home/smartlostandfound/mysite`
- **WSGI file**: `/home/smartlostandfound/mysite/wsgi.py`

## Step 2: Upload Your Code

### 2.1 Clone from GitHub
In the PythonAnywhere **Consoles** tab, open a Bash console and run:

```bash
cd /home/smartlostandfound
git clone https://github.com/fraarroyo/Smart-Lost-and-Found.git mysite
cd mysite
```

### 2.2 Create Virtual Environment
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 2.3 Install Dependencies
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Install requirements
pip install -r requirements_pythonanywhere.txt
```

**Note**: Some packages like `pycocotools` might need manual installation or may not be available. The app will work without them.

## Step 3: Configure Environment Variables

In the PythonAnywhere **Web** tab, go to your web app configuration and add these environment variables:

```
FLASK_APP=app.py
FLASK_ENV=production
PROCESSING_MODE=ultra_fast
ENABLE_TRAINING=false
RENDER=true
FLASK_SECRET_KEY=your-secret-key-here
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=true
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_DEFAULT_SENDER=your-email@gmail.com
```

## Step 4: Configure Static Files

In the **Web** tab, set up static files mapping:

| URL | Directory |
|-----|-----------|
| `/static/` | `/home/smartlostandfound/mysite/static/` |
| `/media/` | `/home/smartlostandfound/mysite/static/uploads/` |

## Step 5: Database Setup

### 5.1 Initialize Database
In a Bash console:
```bash
cd /home/smartlostandfound/mysite
source venv/bin/activate
python -c "from app import app, db; app.app_context().push(); db.create_all(); print('Database created')"
```

### 5.2 Create Admin User
```bash
cd /home/smartlostandfound/mysite
source venv/bin/activate
python -c "
from app import app, db, User
from werkzeug.security import generate_password_hash
with app.app_context():
    admin = User(username='admin', email='admin@example.com', password_hash=generate_password_hash('admin123'), is_admin=True)
    db.session.add(admin)
    db.session.commit()
    print('Admin user created: admin / admin123')
"
```

## Step 6: Configure WSGI File

The `wsgi.py` file is already created. Make sure it points to the correct path:

```python
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
```

## Step 7: Reload and Test

1. In the **Web** tab, click **Reload** for your web app
2. Visit your PythonAnywhere URL: `https://smartlostandfound.pythonanywhere.com`
3. Test the application:
   - Register a new user
   - Login with admin credentials (admin / admin123)
   - Upload an item
   - Test the search functionality

## Troubleshooting

### Common Issues

1. **Import Errors**: Some ML packages might not be available. The app will work with reduced functionality.

2. **Database Issues**: Make sure the database file has proper permissions:
   ```bash
   chmod 664 /home/fraarroyo/mysite/lostfound.db
   ```

3. **Static Files Not Loading**: Check the static files mapping in the Web tab.

4. **Memory Issues**: PythonAnywhere has memory limits. The app is configured for minimal memory usage.

### Performance Optimization

1. **Disable Heavy Features**: The app is configured to skip heavy ML processing on startup
2. **Use SQLite**: Database is set to SQLite for simplicity
3. **Optimize Images**: Images are automatically resized to reduce memory usage

## Security Notes

1. **Change Default Passwords**: Update admin credentials after deployment
2. **Environment Variables**: Keep sensitive data in environment variables
3. **HTTPS**: PythonAnywhere provides HTTPS by default

## Monitoring

- Check the **Web** tab for error logs
- Monitor the **Tasks** tab for any background processes
- Use the **Files** tab to check file permissions and structure

## Support

If you encounter issues:
1. Check the PythonAnywhere error logs
2. Verify all environment variables are set correctly
3. Ensure all dependencies are installed
4. Check file permissions

Your Lost and Found Management System should now be live on PythonAnywhere! 🎉
