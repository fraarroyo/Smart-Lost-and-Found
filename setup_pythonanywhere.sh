#!/bin/bash
# PythonAnywhere Setup Script
# Run this script in your PythonAnywhere console after cloning the repository

echo "🚀 Setting up BARYONYX for PythonAnywhere..."

# Create necessary directories
mkdir -p static/uploads
mkdir -p static/img
mkdir -p training_data/images
mkdir -p training_data/labels
mkdir -p models
mkdir -p instance

echo "✓ Created necessary directories"

# Set up environment variables
export FLASK_APP=app.py
export FLASK_ENV=production
export PROCESSING_MODE=ultra_fast
export ENABLE_TRAINING=false
export RENDER=true

echo "✓ Set environment variables"

# Install requirements
echo "📦 Installing requirements..."
pip3.10 install --user -r requirements_pythonanywhere.txt

echo "✓ Requirements installed"

# Initialize database
echo "🗄️ Initializing database..."
python3.10 -c "from app import app, db; app.app_context().push(); db.create_all(); print('Database created')"

# Create admin user
echo "👤 Creating admin user..."
python3.10 -c "
from app import app, db, User
from werkzeug.security import generate_password_hash
with app.app_context():
    # Check if admin already exists
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(username='admin', email='admin@example.com', password_hash=generate_password_hash('admin123'), is_admin=True)
        db.session.add(admin)
        db.session.commit()
        print('Admin user created: admin / admin123')
    else:
        print('Admin user already exists')
"

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Go to the Web tab in PythonAnywhere"
echo "2. Configure your web app with these settings:"
echo "   - Source code: /home/smartlostandfound/mysite"
echo "   - Working directory: /home/smartlostandfound/mysite"
echo "   - WSGI file: /home/smartlostandfound/mysite/wsgi.py"
echo "3. Set up static files mapping:"
echo "   - URL: /static/ -> Directory: /home/smartlostandfound/mysite/static/"
echo "   - URL: /media/ -> Directory: /home/smartlostandfound/mysite/static/uploads/"
echo "4. Add environment variables in the Web tab"
echo "5. Reload your web app"
echo ""
echo "🔗 Your app will be available at: https://smartlostandfound.pythonanywhere.com"
