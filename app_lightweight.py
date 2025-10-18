#!/usr/bin/env python3
"""
Lightweight version of the Lost and Found app for PythonAnywhere
This version removes heavy ML dependencies and focuses on core functionality
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import uuid
import time
import signal
from functools import wraps, lru_cache
import json
import secrets
from itsdangerous import URLSafeTimedSerializer
from PIL import Image, ImageDraw
from collections import Counter
import webcolors
from sklearn.cluster import KMeans
import numpy as np
from flask_migrate import Migrate
from sqlalchemy import text
import hashlib
import io
import zipfile
from collections import defaultdict
import threading
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.urandom(32))

# Database configuration with fallback options
if os.environ.get('PYTHONANYWHERE_DOMAIN'):
    # PythonAnywhere - try multiple database locations
    possible_paths = [
        '/home/smartlostandfound/lostfound.db',
        '/home/smartlostandfound/Smart-Lost-and-Found/lostfound.db',
        '/tmp/lostfound.db'
    ]
    
    db_path = None
    for path in possible_paths:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            test_file = path + '.test'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            db_path = path
            break
        except (OSError, PermissionError):
            continue
    
    if db_path:
        app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
        print(f"Using database: {db_path}")
    else:
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        print("Warning: Using in-memory database")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lostfound.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
app.config['SECURITY_PASSWORD_SALT'] = secrets.token_hex(16)

# Email configuration
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
mail = Mail(app)
migrate = Migrate(app, db)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_guest = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    category = db.Column(db.String(100), nullable=True)
    location = db.Column(db.String(200), nullable=True)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(500), nullable=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    is_found = db.Column(db.Boolean, default=False)
    contact_info = db.Column(db.String(500), nullable=True)
    status = db.Column(db.String(50), default='active')  # active, resolved, archived

# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Utility functions
def guest_restricted(f):
    """Decorator to restrict guest users from certain actions"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.is_authenticated and current_user.is_guest:
            flash('This feature is not available for guest users. Please register for a full account.', 'warning')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function

def get_dominant_colors(image_path, num_colors=5):
    """Get dominant colors from an image using basic color analysis"""
    try:
        image = Image.open(image_path)
        image = image.convert('RGB')
        
        # Resize image for faster processing
        image.thumbnail((150, 150))
        
        # Convert to numpy array
        data = np.array(image)
        data = data.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=num_colors, random_state=42)
        kmeans.fit(data)
        
        colors = kmeans.cluster_centers_.astype(int)
        return colors.tolist()
    except Exception as e:
        print(f"Color analysis failed: {e}")
        return []

def analyze_text_similarity(text1, text2):
    """Basic text similarity analysis"""
    try:
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    except Exception:
        return 0.0

# Routes
@app.route('/')
def index():
    """Home page showing all items"""
    try:
        items = Item.query.order_by(Item.date.desc()).all()
        return render_template('index.html', items=items)
    except Exception as e:
        print(f"Database error: {e}")
        return render_template('index.html', items=[])

@app.route('/add_item', methods=['GET', 'POST'])
@login_required
@guest_restricted
def add_item():
    """Add a new lost/found item"""
    if request.method == 'POST':
        try:
            title = request.form['title']
            description = request.form['description']
            category = request.form['category']
            location = request.form['location']
            is_found = 'is_found' in request.form
            contact_info = request.form.get('contact_info', '')
            
            # Handle image upload
            image_path = None
            if 'image' in request.files:
                file = request.files['image']
                if file and file.filename:
                    filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4()}_{filename}"
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    file.save(file_path)
                    image_path = file_path
                    
                    # Analyze colors
                    colors = get_dominant_colors(file_path)
            
            # Create new item
            item = Item(
                title=title,
                description=description,
                category=category,
                location=location,
                image_path=image_path,
                user_id=current_user.id,
                is_found=is_found,
                contact_info=contact_info
            )
            
            db.session.add(item)
            db.session.commit()
            
            flash('Item added successfully!', 'success')
            return redirect(url_for('index'))
            
        except Exception as e:
            print(f"Error adding item: {e}")
            flash('Error adding item. Please try again.', 'error')
    
    return render_template('add_item.html')

@app.route('/search')
def search():
    """Search for items"""
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    is_found = request.args.get('is_found', '')
    
    items = Item.query
    
    if query:
        items = items.filter(
            (Item.title.contains(query)) | 
            (Item.description.contains(query)) |
            (Item.location.contains(query))
        )
    
    if category:
        items = items.filter(Item.category == category)
    
    if is_found:
        items = items.filter(Item.is_found == (is_found == 'true'))
    
    items = items.order_by(Item.date.desc()).all()
    
    return render_template('search.html', items=items, query=query, category=category, is_found=is_found)

@app.route('/item/<int:item_id>')
def view_item(item_id):
    """View a specific item"""
    item = Item.query.get_or_404(item_id)
    return render_template('view_item.html', item=item)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password_hash, password):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    return redirect(url_for('index'))

# Initialize database
with app.app_context():
    try:
        db.create_all()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Database initialization error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
