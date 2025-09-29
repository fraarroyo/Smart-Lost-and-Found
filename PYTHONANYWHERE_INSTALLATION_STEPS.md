# PythonAnywhere Installation Steps

## Step-by-Step Installation Guide

### 1. Clone and Setup
```bash
cd /home/smartlostandfound
git clone https://github.com/fraarroyo/Smart-Lost-and-Found.git mysite
cd mysite
```

### 2. Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 3. Install Dependencies

#### Option 1: Simple installation (recommended):
```bash
pip install -r requirements_simple.txt
```

#### Option 2: Full installation (if you want ML features):
```bash
# First install basic packages
pip install -r requirements_simple.txt

# Then try to add ML packages one by one
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
pip install transformers==4.30.0
pip install scikit-learn==1.3.0
pip install scipy==1.11.0
pip install opencv-python-headless==4.10.0.84
pip install joblib==1.3.2
```

#### Option 3: Minimal installation (if ML packages cause issues):
```bash
pip install -r requirements_minimal.txt
```

### 4. Test Installation
```bash
python -c "import flask; print('Flask OK')"
python -c "import numpy; print('NumPy OK')"
python -c "import torch; print('PyTorch OK')"
python -c "from app import app; print('App import OK')"
```

### 5. Initialize Database
```bash
python -c "from app import app, db; app.app_context().push(); db.create_all(); print('Database created')"
```

### 6. Create Admin User
```bash
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

### 7. Configure Web App
- Source code: `/home/smartlostandfound/mysite`
- Working directory: `/home/smartlostandfound/mysite`
- WSGI file: `/home/smartlostandfound/mysite/wsgi.py`

### 8. Set Static Files
- URL: `/static/` → Directory: `/home/smartlostandfound/mysite/static/`
- URL: `/media/` → Directory: `/home/smartlostandfound/mysite/static/uploads/`

### 9. Reload Web App
Click "Reload" in the Web tab.

## Troubleshooting

### If you get dependency conflicts:
1. Try installing packages one by one
2. Use the minimal requirements first
3. Skip problematic packages (the app will work with reduced functionality)

### If you get import errors:
1. Make sure virtual environment is activated
2. Check that all paths are correct
3. Verify the WSGI file path

### Alternative: Skip ML packages
If ML packages cause too many conflicts, you can run the app without them:
```bash
pip install -r requirements_minimal.txt
```
The app will work with basic functionality (user management, item posting, search) but without advanced ML features.
