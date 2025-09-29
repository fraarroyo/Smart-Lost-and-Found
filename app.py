from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_mail import Mail, Message
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import uuid
import time
import signal
from functools import wraps, lru_cache
import json
import secrets

def guest_restricted(f):
    """Decorator to restrict guest users from certain actions"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if current_user.is_authenticated and current_user.is_guest:
            flash('This feature is not available for guest users. Please register for a full account.', 'warning')
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
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

app = Flask(__name__)
# Use a stable secret key from env if provided; otherwise generate a random one
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', os.urandom(32))

# Update database URI to use SQLite
# For PythonAnywhere, use absolute path to avoid permission issues
if os.environ.get('PYTHONANYWHERE_DOMAIN'):
    # Running on PythonAnywhere
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////home/smartlostandfound/mysite/lostfound.db'
else:
    # Running locally
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lostfound.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Fix uploads directory path
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)  # Remember me duration
app.config['SECURITY_PASSWORD_SALT'] = secrets.token_hex(16)  # For password reset tokens

# Email configuration
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'smartlostandfound@gmail.com')

# Training data directories
TRAINING_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
TRAINING_IMAGES_DIR = os.path.join(TRAINING_DATA_DIR, 'images')
TRAINING_LABELS_DIR = os.path.join(TRAINING_DATA_DIR, 'labels')
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

# Create training directories
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
os.makedirs(TRAINING_IMAGES_DIR, exist_ok=True)
os.makedirs(TRAINING_LABELS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Import ML models - these are required for the application to function
# Import will be handled in the try block below

# Use the custom best_model1.pth checkpoint for object detection
# Clear any environment variables that might interfere with custom model loading
os.environ.pop('USE_WEB_DETECTOR', None)
# Keep BEST_MODEL1 to use best_model1.pth

# Set ultra_fast processing mode for maximum speed
os.environ['PROCESSING_MODE'] = 'ultra_fast'
os.environ['ENABLE_TRAINING'] = 'false'  # Disable training for speed

# PythonAnywhere optimizations
if os.environ.get('PYTHONANYWHERE_DOMAIN'):
    # Additional optimizations for PythonAnywhere
    os.environ['RENDER'] = 'true'  # Skip heavy startup evaluation
    os.environ['PYTHONANYWHERE_OPTIMIZED'] = 'true'

# Initialize unified ML model
try:
    from ml_models import UnifiedModel, ObjectDetector, TextAnalyzer, ModelTrainer, SequenceProcessor
    from rnn_models import rnn_manager
    from image_rnn_analyzer import image_rnn_analyzer
    from enhanced_image_processor import enhanced_image_processor
    unified_model = UnifiedModel()
    object_detector = unified_model  # For backward compatibility
    text_analyzer = TextAnalyzer()  # Uses unified model internally
    model_trainer = ModelTrainer()  # Uses unified model internally
    sequence_processor = SequenceProcessor()  # For sequence processing
except ImportError as e:
    # ML dependencies are required - fail fast if not available
    print(f"ERROR: Required ML dependencies not available: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    raise e

sequence_processor = SequenceProcessor()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize Flask-Mail
mail = Mail(app)

# Initialize email service
from email_service import init_email_service, get_email_service
init_email_service(mail)

# Initialize password reset serializer
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Analysis result cache to improve performance
analysis_cache = {}
CACHE_MAX_SIZE = 100
CACHE_TTL = 3600  # 1 hour in seconds

def get_cache_key(image_path):
    """Generate cache key for image analysis"""
    try:
        with open(image_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"analysis_{file_hash}"
    except:
        return f"analysis_{os.path.basename(image_path)}_{os.path.getmtime(image_path)}"

def get_cached_analysis(image_path):
    """Get cached analysis result if available and not expired"""
    cache_key = get_cache_key(image_path)
    if cache_key in analysis_cache:
        result, timestamp = analysis_cache[cache_key]
        if time.time() - timestamp < CACHE_TTL:
            return result
        else:
            # Remove expired entry
            del analysis_cache[cache_key]
    return None

def cache_analysis_result(image_path, result):
    """Cache analysis result"""
    cache_key = get_cache_key(image_path)
    
    # Remove oldest entries if cache is full
    if len(analysis_cache) >= CACHE_MAX_SIZE:
        oldest_key = min(analysis_cache.keys(), key=lambda k: analysis_cache[k][1])
        del analysis_cache[oldest_key]
    
    analysis_cache[cache_key] = (result, time.time())

# Add these mappings at the top of the file after imports
ITEM_CATEGORY_MAPPINGS = {
    'laptop': 'electronics',
    'computer': 'electronics',
    'phone': 'electronics',
    'mobile': 'electronics',
    'camera': 'electronics',
    'headphones': 'electronics',
    'earphones': 'electronics',
    'watch': 'electronics',
    'shirt': 'clothing',
    'pants': 'clothing',
    'jacket': 'clothing',
    'dress': 'clothing',
    'shoes': 'clothing',
    'bag': 'accessories',
    'backpack': 'accessories',
    'wallet': 'accessories',
    'purse': 'accessories',
    'glasses': 'accessories',
    'sunglasses': 'accessories',
    'document': 'documents',
    'book': 'documents',
    'notebook': 'documents',
    'id card': 'documents',
    'passport': 'documents'
}

# Expand allowed classes with common synonyms
ALLOWED_CLASSES = {
    "phone", "mobile", "cell phone", "cellphone", "smartphone", "computer mouse", "mouse", "wallet", "eyeglasses", "eye glasses", "glasses", "spectacles", "id card", "id", "identity card", "tumbler", "tablet", "ipad", "bottle", "umbrella", "wrist watch", "watch", "usb", "flash drive", "thumb drive", "pen drive"
}

def normalize_class_name(name: str) -> str:
    """Normalize detector class names to canonical forms used by the UI filters.
    Examples: 'mobile_phone' -> 'cell phone', 'cellphone' -> 'cell phone'.
    """
    if not name:
        return ''
    n = str(name).strip().lower().replace('_', ' ').replace('-', ' ')
    synonyms = {
        'mobile phone': 'cell phone',
        'cellphone': 'cell phone',
        'smart phone': 'smartphone',
        'eye glasses': 'eyeglasses',
        'identity card': 'id card',
        'thumbdrive': 'thumb drive',
        'pendrive': 'pen drive',
    }
    return synonyms.get(n, n)

def classify_item(detected_objects):
    """Classify multiple items based on detected objects and suggest categories."""
    if not detected_objects:
        return [("Unknown Item", "other")]
        
    # Sort objects by confidence
    sorted_objects = sorted(detected_objects, key=lambda x: x['confidence'], reverse=True)
    
    # Process all objects with confidence > 0.5
    items = []
    for obj in sorted_objects:
        if obj['confidence'] < 0.5:
            continue
            
        item_type = obj['class'].lower()
        
        # Try to find a matching category
        suggested_category = "other"
        for key, category in ITEM_CATEGORY_MAPPINGS.items():
            if key in item_type:
                suggested_category = category
                break
        
        # Format the item type nicely
        item_type = item_type.replace('_', ' ').title()
        items.append((item_type, suggested_category))
    
    # If no items were found, return unknown
    if not items:
        return [("Unknown Item", "other")]
        
    return items

def is_screen_cracked(image_path, box=None):
    """
    Placeholder for phone screen crack detection.
    Returns True if cracked, False if not, or None if unsure.
    """
    # TODO: Replace with actual crack detection logic
    # For now, always return None (unknown)
    return None

def is_item_damaged(image_path, box=None):
    """
    Placeholder for general item damage detection.
    Returns True if damaged, False if not, or None if unsure.
    """
    # TODO: Replace with actual damage detection logic
    # For now, always return None (unknown)
    return None

def generate_description(detected_objects, color=None, size=None, category=None, location=None, crack_status=None, damage_status=None, rnn_analysis=None):
    """Generate a detailed natural language description from the most relevant detected object and attributes."""
    if not detected_objects:
        return "No objects detected in the image."
    
    best_obj = max(detected_objects, key=lambda x: x['confidence'])
    cls = best_obj['class'].replace('_', ' ').title()
    confidence = best_obj.get('confidence', 0)
    
    # Use RNN analysis if available for enhanced description
    if rnn_analysis and rnn_analysis.get('details'):
        rnn_details = rnn_analysis['details']
        
        # Build enhanced description using RNN analysis
        desc_parts = []
        
        # Size from RNN or fallback
        if rnn_details.get('size') and rnn_details['size'] != 'unknown':
            size_str = rnn_details['size'].replace('_', ' ').title()
            desc_parts.append(size_str)
        elif size:
            desc_parts.append(size.title())
        
        # Colors from RNN or fallback
        if rnn_details.get('colors'):
            color_str = ', '.join(rnn_details['colors'][:2]).title()
            desc_parts.append(color_str)
        elif color:
            desc_parts.append(color.title())
        
        # Material from RNN
        if rnn_details.get('materials') and rnn_details['materials'][0] != 'unknown':
            material_str = rnn_details['materials'][0].replace('_', ' ').title()
            desc_parts.append(material_str)
        
        # Item type
        desc_parts.append(cls)
        
        # Brand from RNN
        if rnn_details.get('brands') and rnn_details['brands'][0] != 'unknown':
            brand_str = rnn_details['brands'][0].replace('_', ' ').title()
            desc_parts.append(f"({brand_str} brand)")
        
        # Category
        if category:
            desc_parts.append(f"({category.title()} category)")
        
        # Style from RNN
        if rnn_details.get('styles') and rnn_details['styles'][0] != 'unknown':
            style_str = rnn_details['styles'][0].replace('_', ' ').title()
            desc_parts.append(f"in {style_str} style")
        
        # Condition from RNN
        if rnn_details.get('condition') and rnn_details['condition'] != 'unknown':
            condition_str = rnn_details['condition'].replace('_', ' ').title()
            desc_parts.append(f"in {condition_str} condition")
        
        desc = " ".join(desc_parts)
        desc += f", detected with {confidence*100:.0f}% confidence."
        
        # Add RNN confidence if available
        if rnn_analysis.get('confidence', 0) > 0:
            rnn_conf = rnn_analysis['confidence'] * 100
            desc += f" RNN analysis confidence: {rnn_conf:.0f}%."
        
    else:
        # Fallback to original description generation
        desc_parts = []
        if size:
            desc_parts.append(size.title())
        if color:
            desc_parts.append(color.title())
        desc_parts.append(cls)
        if category:
            desc_parts.append(f"({category.title()} category)")
        desc = " ".join(desc_parts)
        desc += f", detected with {confidence*100:.0f}% confidence."
    
    # Add crack and damage status
    if crack_status is not None:
        desc += f" Screen is {'cracked' if crack_status else 'not cracked'}."
    if damage_status is not None:
        desc += f" Item is {'damaged' if damage_status else 'not damaged'}."
    if location:
        desc += f" Location: {location}."
    
    return desc

def rgb_to_name(rgb):
    """Convert RGB color to name with better fallback handling."""
    try:
        if isinstance(rgb, str):
            # Handle hex color codes
            if rgb.startswith('#'):
                rgb = tuple(int(rgb[i:i+2], 16) for i in (1, 3, 5))
            else:
                rgb_tuple = tuple(int(x) for x in rgb.strip('rgb()').replace('(', '').replace(')', '').split(','))
        else:
            rgb_tuple = rgb
            
        try:
            return webcolors.rgb_to_name(rgb_tuple)
        except ValueError:
            # If exact match fails, find closest color
            def distance(c1, c2):
                return sum((a - b) ** 2 for a, b in zip(c1, c2))
                
            # Get available color names
            if hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
                color_names = webcolors.CSS3_NAMES_TO_HEX.keys()
            elif hasattr(webcolors, "CSS3_NAMES"):
                color_names = webcolors.CSS3_NAMES
            else:
                # Fallback to basic colors if webcolors is not available
                color_names = [
                    'black', 'white', 'gray', 'silver', 'red', 'green', 'blue',
                    'yellow', 'purple', 'brown', 'orange', 'pink', 'gold', 'navy',
                    'teal', 'maroon', 'olive', 'lime', 'aqua', 'fuchsia'
                ]
                
            # Convert color names to RGB and find closest match
            colors = {name: webcolors.name_to_rgb(name) for name in color_names}
            closest = min(colors.items(), key=lambda kv: distance(rgb_tuple, kv[1]))
            return closest[0]
            
    except Exception as e:
        print(f"rgb_to_name error: {e}")
        # Return a default color based on RGB values
        r, g, b = rgb_tuple if isinstance(rgb_tuple, tuple) else (0, 0, 0)
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif abs(r - g) < 30 and abs(g - b) < 30:
            return "gray"
        else:
            return "unknown"

def extract_dominant_color(image_path, exclude_colors=None):
    """Color analysis disabled for maximum speed."""
    return "unknown"

def extract_case_color(image_path, box, border_width=8):
    """Color analysis disabled for maximum speed."""
    return "unknown"

def _extract_case_color_fallback(image_path, box, border_width=8):
    """Fallback case color extraction method."""
    try:
        image = Image.open(image_path).convert('RGB')
        x1, y1, x2, y2 = map(int, box)
        # Clamp to image size
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        # Create a mask for the border
        mask = Image.new('L', image.size, 0)
        draw = ImageDraw.Draw(mask)
        # Outer rectangle
        draw.rectangle([x1, y1, x2, y2], outline=255, width=border_width)
        # Get border pixels
        border_pixels = [image.getpixel((x, y)) for x in range(x1, x2) for y in range(y1, y2) if mask.getpixel((x, y)) > 0]
        if not border_pixels:
            return "Unknown"
        color_counts = Counter(border_pixels)
        most_common = color_counts.most_common(10)
        color_names = []
        for color_tuple, _ in most_common:
            name = rgb_to_name(color_tuple)
            if name != "Unknown" and name not in color_names:
                color_names.append(name)
            if len(color_names) >= 2:
                break
        if len(color_names) < 2:
            default_colors = ['black', 'white', 'gray', 'silver']
            for color in default_colors:
                if color not in color_names:
                    color_names.append(color)
                if len(color_names) >= 2:
                    break
        return ", ".join(color_names[:2])
    except Exception as e:
        print(f"extract_case_color error: {e}")
        return "black, white"

def extract_tumbler_color(image_path, box):
    """Color analysis disabled for maximum speed."""
    return "unknown"

def _extract_tumbler_color_fallback(image_path, box):
    """Fallback tumbler color extraction method."""
    try:
        image = Image.open(image_path).convert('RGB')
        x1, y1, x2, y2 = map(int, box)
        height = y2 - y1
        y1_central = int(y1 + 0.2 * height)
        y2_central = int(y2 - 0.2 * height)
        x1, y1_central = max(0, x1), max(0, y1_central)
        x2, y2_central = min(image.width, x2), min(image.height, y2_central)
        pixels = [image.getpixel((x, y)) for x in range(x1, x2) for y in range(y1_central, y2_central)]
        if not pixels:
            return "Unknown"
        # Use k-means clustering to find the two most dominant colors
        arr = np.array(pixels)
        kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
        labels = kmeans.fit_predict(arr)
        centers = kmeans.cluster_centers_.astype(int)
        # Sort clusters by size
        counts = np.bincount(labels)
        sorted_indices = np.argsort(-counts)
        color_names = []
        for idx in sorted_indices:
            rgb_tuple = tuple(centers[idx])
            name = rgb_to_name(rgb_tuple)
            if name != "Unknown" and name not in color_names:
                color_names.append(name)
            if len(color_names) >= 2:
                break
        if len(color_names) < 2:
            default_colors = ['black', 'white', 'gray', 'silver']
            for color in default_colors:
                if color not in color_names:
                    color_names.append(color)
                if len(color_names) >= 2:
                    break
        return ", ".join(color_names[:2])
    except Exception as e:
        print(f"extract_tumbler_color error: {e}")
        return "black, white"

def estimate_size(detected_objects, image_path):
    if not detected_objects:
        return None
    image = Image.open(image_path)
    width, height = image.size
    # Use the largest object's bounding box
    largest_area = 0
    size_label = None
    for obj in detected_objects:
        box = obj.get('box')
        if box:
            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
    if largest_area == 0:
        return None
    image_area = width * height
    ratio = largest_area / image_area
    if ratio > 0.5:
        size_label = 'large'
    elif ratio > 0.2:
        size_label = 'medium'
    else:
        size_label = 'small'
    return size_label

# Models
class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    item_expiry_days = db.Column(db.Integer, default=30)
    max_image_size = db.Column(db.Integer, default=16)  # MB
    matching_threshold = db.Column(db.Integer, default=60)  # %
    enable_email_notifications = db.Column(db.Boolean, default=True)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)  # Allow NULL for guest users
    password_hash = db.Column(db.String(128), nullable=True)  # Allow NULL for guest users
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expiry = db.Column(db.DateTime)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    is_guest = db.Column(db.Boolean, default=False)  # New field for guest users
    date_joined = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    items = db.relationship('Item', backref='owner', lazy=True)

    def set_password(self, password):
        print(f"Setting password for user: {self.username}")  # Debug log
        self.password_hash = generate_password_hash(password)
        print(f"Generated password hash: {self.password_hash}")  # Debug log

    def check_password(self, password):
        print(f"Checking password for user: {self.username}")  # Debug log
        print(f"Stored hash: {self.password_hash}")  # Debug log
        if self.is_guest:
            return False  # Guest users don't have passwords
        result = check_password_hash(self.password_hash, password)
        print(f"Password check result: {result}")  # Debug log
        return result

    @staticmethod
    def create_guest_user():
        """Create a temporary guest user"""
        try:
            guest_username = f"guest_{uuid.uuid4().hex[:8]}"
            print(f"Creating guest user with username: {guest_username}")  # Debug log
            
            # Check if username already exists (very unlikely but just in case)
            existing_user = User.query.filter_by(username=guest_username).first()
            if existing_user:
                print(f"Username {guest_username} already exists, generating new one")  # Debug log
                guest_username = f"guest_{uuid.uuid4().hex[:8]}"
            
            guest_user = User(
                username=guest_username,
                email=None,
                password_hash=None,
                is_guest=True,
                is_active=True,
                is_admin=False
            )
            print(f"Guest user object created: {guest_user}")  # Debug log
            
            db.session.add(guest_user)
            print(f"Guest user added to session")  # Debug log
            
            db.session.commit()
            print(f"Guest user committed to database")  # Debug log
            
            return guest_user
        except Exception as e:
            print(f"Error in create_guest_user: {e}")  # Debug log
            db.session.rollback()  # Rollback on error
            raise e

    def generate_reset_token(self):
        if self.is_guest:
            return None  # Guest users can't reset passwords
        token = serializer.dumps(self.email, salt=app.config['SECURITY_PASSWORD_SALT'])
        self.reset_token = token
        self.reset_token_expiry = datetime.utcnow() + timedelta(hours=1)
        db.session.commit()
        return token

    @staticmethod
    def verify_reset_token(token):
        try:
            email = serializer.loads(token, salt=app.config['SECURITY_PASSWORD_SALT'], max_age=3600)
            user = User.query.filter_by(email=email).first()
            if user and user.reset_token == token and user.reset_token_expiry > datetime.utcnow():
                return user
        except:
            return None
        return None

class Item(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    brand = db.Column(db.String(100))  # Brand field for items
    status = db.Column(db.String(20), nullable=False)  # 'lost' or 'found'
    location = db.Column(db.String(100))
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    image_path = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    detected_objects = db.Column(db.Text)  # JSON string of detected objects
    text_embedding = db.Column(db.Text)  # JSON string of text embedding
    color = db.Column(db.String(50))
    size = db.Column(db.String(50))
    image_hash = db.Column(db.String(64))  # New field for image hash
    
    # Additional image information fields
    brand = db.Column(db.String(100))  # Brand/manufacturer
    model = db.Column(db.String(100))  # Model name/number
    material = db.Column(db.String(100))  # Material (leather, plastic, metal, etc.)
    condition = db.Column(db.String(50))  # Condition (new, good, fair, poor)
    estimated_value = db.Column(db.Float)  # Estimated monetary value
    serial_number = db.Column(db.String(100))  # Serial number if available
    distinctive_features = db.Column(db.Text)  # Special markings, scratches, etc.
    color_secondary = db.Column(db.String(50))  # Secondary color
    size_dimensions = db.Column(db.String(100))  # Specific dimensions (e.g., "10x15x5 cm")
    weight = db.Column(db.String(50))  # Weight if known
    purchase_date = db.Column(db.Date)  # When it was purchased
    last_seen_location = db.Column(db.String(200))  # More specific location details
    additional_notes = db.Column(db.Text)  # Additional notes from user
    
    # Contact information fields
    owner_contact = db.Column(db.String(20))  # Owner's contact number
    finder_contact = db.Column(db.String(20))  # Finder's contact number
    owner_email = db.Column(db.String(120))  # Owner's email address
    finder_email = db.Column(db.String(120))  # Finder's email address
    contact_preference = db.Column(db.String(50))  # Preferred contact method
    best_time_to_contact = db.Column(db.String(50))  # Best time to contact
    
    # AI-generated analysis fields
    checkpoint_embedding = db.Column(db.Text)  # JSON string of checkpoint model embedding
    image_quality_score = db.Column(db.Float)  # AI-assessed image quality (0-1)
    confidence_score = db.Column(db.Float)  # Overall confidence in detection/analysis
    suggested_category = db.Column(db.String(50))  # AI-suggested category
    suggested_tags = db.Column(db.Text)  # JSON string of suggested tags

class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lost_item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    found_item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    match_date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    match_score = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='pending')  # pending, confirmed, rejected
    notes = db.Column(db.Text)
    
    # Relationships
    lost_item = db.relationship('Item', foreign_keys=[lost_item_id], backref='found_matches')
    found_item = db.relationship('Item', foreign_keys=[found_item_id], backref='lost_matches')

class TrainingData(db.Model):
    """Model for storing training data and user feedback."""
    id = db.Column(db.Integer, primary_key=True)
    image_path = db.Column(db.String(500))
    detected_objects = db.Column(db.Text)  # JSON string of detected objects
    user_feedback = db.Column(db.Text)  # JSON string of user feedback
    true_labels = db.Column(db.Text)  # JSON string of true labels
    feedback_type = db.Column(db.String(50))  # 'correction', 'confirmation', 'rejection'
    confidence_adjustment = db.Column(db.Float)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    is_processed = db.Column(db.Boolean, default=False)
    
    # Relationships
    user = db.relationship('User', backref='training_data')

class ModelMetrics(db.Model):
    """Model for storing model performance metrics."""
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)  # 'object_detector', 'text_analyzer'
    metric_name = db.Column(db.String(100), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    class_name = db.Column(db.String(100))  # For class-specific metrics
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    @classmethod
    def add_metric(cls, model_type, metric_name, metric_value, class_name=None):
        """Add a new metric record."""
        metric = cls(
            model_type=model_type,
            metric_name=metric_name,
            metric_value=metric_value,
            class_name=class_name
        )
        db.session.add(metric)
        db.session.commit()
        return metric

class MatchFeedback(db.Model):
    """Model for tracking user feedback on potential matches."""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    lost_item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    found_item_id = db.Column(db.Integer, db.ForeignKey('item.id'), nullable=False)
    feedback_type = db.Column(db.String(20), nullable=False)  # 'not_match', 'confirmed_match'
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='match_feedbacks')
    lost_item = db.relationship('Item', foreign_keys=[lost_item_id], backref='lost_feedbacks')
    found_item = db.relationship('Item', foreign_keys=[found_item_id], backref='found_feedbacks')
    
    @classmethod
    def add_feedback(cls, user_id, lost_item_id, found_item_id, feedback_type):
        """Add a new feedback record."""
        # Check if feedback already exists
        existing = cls.query.filter_by(
            user_id=user_id,
            lost_item_id=lost_item_id,
            found_item_id=found_item_id
        ).first()
        
        if existing:
            # Update existing feedback
            existing.feedback_type = feedback_type
            existing.created_at = datetime.utcnow()
            return existing
        else:
            # Create new feedback
            feedback = cls(
                user_id=user_id,
                lost_item_id=lost_item_id,
                found_item_id=found_item_id,
                feedback_type=feedback_type
            )
            db.session.add(feedback)
            db.session.commit()
            return feedback

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Routes
@app.route('/')
def index():
    items = Item.query.order_by(Item.date.desc()).all()
    return render_template('index.html', items=items)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already registered')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please login.')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = request.form.get('remember') == 'on'
        
        print(f"\nLogin attempt for username: {username}")  # Debug log
        
        user = User.query.filter_by(username=username).first()
        print(f"User found: {user is not None}")  # Debug log
        
        if user:
            print(f"Stored password hash: {user.password_hash}")  # Debug log
            print(f"Is admin: {user.is_admin}")  # Debug log
            password_check = user.check_password(password)
            print(f"Password check result: {password_check}")  # Debug log
            
            if password_check:
                login_user(user, remember=remember)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
        
        flash('Invalid username or password')
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/guest-login')
def guest_login():
    """Create and login a guest user"""
    try:
        print(f"Creating guest user...")  # Debug log
        guest_user = User.create_guest_user()
        print(f"Guest user created: {guest_user.username}, ID: {guest_user.id}")  # Debug log
        login_user(guest_user, remember=True)
        print(f"Guest user logged in successfully")  # Debug log
        flash('Logged in as guest user. You can browse and search items, but some features may be limited.', 'info')
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error creating guest user: {e}")  # Debug log
        import traceback
        traceback.print_exc()  # Print full traceback
        flash(f'Failed to create guest account: {str(e)}', 'error')
        return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    user_was_guest = current_user.is_guest if current_user.is_authenticated else False
    logout_user()
    if user_was_guest:
        flash('Guest session ended. Thank you for using our service!', 'info')
    return redirect(url_for('index'))

def enhanced_image_analysis(image_path):
    """Enhanced image analysis using both object detection and checkpoint model."""
    try:
        # Basic object detection
        detected_objects = object_detector.detect_objects(image_path)
        
        # Relabel objects for better categorization
        for obj in detected_objects:
            cls = obj.get('class', '').lower()
            if cls == 'suitcase':
                obj['class'] = 'wallet'
            if cls == 'bottle':
                obj['class'] = 'tumbler'
            if cls == 'clock':
                obj['class'] = 'watch'
            if cls in ['flash drive', 'thumb drive', 'pen drive']:
                obj['class'] = 'usb'
        
        # Get checkpoint model analysis
        checkpoint_embedding = None
        image_quality_score = 0.0
        confidence_score = 0.0
        suggested_category = "other"
        suggested_tags = []
        
        try:
            # Get checkpoint embedding
            checkpoint_embedding = unified_model.get_checkpoint_embedding(image_path)
            
            # Calculate image quality score based on various factors
            image = Image.open(image_path)
            width, height = image.size
            
            # Quality factors
            resolution_score = min(1.0, (width * height) / (224 * 224))  # Normalize to 224x224
            aspect_ratio_score = 1.0 - abs(1.0 - (width / height))  # Prefer square-ish images
            
            # Check for blur (simplified)
            gray = image.convert('L')
            gray_array = np.array(gray)
            blur_score = 1.0 - (np.std(gray_array) / 255.0)  # Higher std = less blur
            
            image_quality_score = (resolution_score + aspect_ratio_score + blur_score) / 3.0
            
            # Calculate overall confidence
            if detected_objects:
                confidence_score = max([obj.get('confidence', 0.0) for obj in detected_objects])
            else:
                confidence_score = 0.0
            
            # Suggest category based on detected objects
            if detected_objects:
                best_obj = max(detected_objects, key=lambda x: x.get('confidence', 0.0))
                obj_class = best_obj.get('class', '').lower()
                
                # Map object classes to categories
                category_mapping = {
                    'cell phone': 'electronics',
                    'laptop': 'electronics',
                    'keyboard': 'electronics',
                    'mouse': 'electronics',
                    'book': 'books',
                    'backpack': 'accessories',
                    'handbag': 'accessories',
                    'wallet': 'accessories',
                    'watch': 'accessories',
                    'car': 'vehicles',
                    'bicycle': 'vehicles',
                    'person': 'clothing',
                    'shirt': 'clothing',
                    'pants': 'clothing',
                    'shoes': 'clothing'
                }
                
                suggested_category = category_mapping.get(obj_class, 'other')
                
                # Generate suggested tags
                suggested_tags = [obj_class]
                if 'color' in best_obj:
                    suggested_tags.append(best_obj['color'])
                if 'features' in best_obj:
                    suggested_tags.extend(best_obj['features'])
        
        except Exception as e:
            print(f"Error in checkpoint analysis: {e}")
        
        # Extract additional information from image (color analysis disabled)
        color_info = {'primary_color': 'unknown', 'secondary_color': ''}
        size_info = estimate_detailed_size(detected_objects, image_path)
        material_info = estimate_material(detected_objects, image_path)
        
        # RNN-based detailed analysis
        rnn_analysis = None
        try:
            rnn_analysis = image_rnn_analyzer.analyze_image_details(image_path)
            print(f"[RNN ANALYSIS] Detailed analysis completed for {image_path}")
        except Exception as e:
            print(f"[RNN ANALYSIS] Error in RNN analysis: {e}")
            rnn_analysis = {
                'details': {},
                'caption': "RNN analysis unavailable",
                'confidence': 0.0
            }
        
        return {
            'detected_objects': detected_objects,
            'checkpoint_embedding': checkpoint_embedding,
            'image_quality_score': image_quality_score,
            'confidence_score': confidence_score,
            'suggested_category': suggested_category,
            'suggested_tags': suggested_tags,
            'color_info': color_info,
            'size_info': size_info,
            'material_info': material_info,
            'rnn_analysis': rnn_analysis
        }
        
    except Exception as e:
        print(f"Error in enhanced image analysis: {e}")
        return {
            'detected_objects': [],
            'checkpoint_embedding': None,
            'image_quality_score': 0.0,
            'confidence_score': 0.0,
            'suggested_category': 'other',
            'suggested_tags': [],
            'color_info': {},
            'size_info': {},
            'material_info': {}
        }

def extract_detailed_color_info(image_path):
    """Extract detailed color information from image using enhanced color detection."""
    try:
        from enhanced_color_detection import enhance_existing_color_detection
        from advanced_color_detection import enhance_existing_color_detection_advanced
        
        # Use advanced color detection
        color_analysis = enhance_existing_color_detection_advanced(image_path)
        
        if color_analysis.get('error'):
            # Fallback to original method
            return _extract_detailed_color_info_fallback(image_path)
        
        primary = color_analysis.get('primary_color')
        secondary = color_analysis.get('secondary_color')
        
        return {
            'primary_color': primary.get('name', 'unknown') if primary else 'unknown',
            'primary_rgb': primary.get('rgb', [0, 0, 0]) if primary else [0, 0, 0],
            'secondary_color': secondary.get('name', '') if secondary else '',
            'secondary_rgb': secondary.get('rgb', None) if secondary else None,
            'color_palette': [color.get('rgb', [0, 0, 0]) for color in color_analysis.get('dominant_colors', [])],
            'enhanced_analysis': color_analysis  # Store full analysis for advanced features
        }
        
    except Exception as e:
        print(f"Enhanced color detection error, using fallback: {e}")
        return _extract_detailed_color_info_fallback(image_path)

def _extract_detailed_color_info_fallback(image_path):
    """Fallback color extraction method."""
    try:
        image = Image.open(image_path)
        image_array = np.array(image)
        
        # Convert to RGB if needed
        if image_array.shape[2] == 4:  # RGBA
            image_array = image_array[:, :, :3]
        
        # Reshape to list of pixels
        pixels = image_array.reshape(-1, 3)
        
        # Use KMeans to find dominant colors
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors
        dominant_colors = kmeans.cluster_centers_.astype(int)
        color_counts = np.bincount(kmeans.labels_)
        
        # Find the most dominant color
        primary_color_idx = np.argmax(color_counts)
        primary_color = dominant_colors[primary_color_idx]
        
        # Convert RGB to color name
        try:
            primary_color_name = webcolors.rgb_to_name(tuple(primary_color))
        except ValueError:
            primary_color_name = f"RGB({primary_color[0]}, {primary_color[1]}, {primary_color[2]})"
        
        # Find secondary color
        secondary_color = None
        secondary_color_name = None
        if len(dominant_colors) > 1:
            secondary_color_idx = np.argsort(color_counts)[-2]
            secondary_color = dominant_colors[secondary_color_idx]
            try:
                secondary_color_name = webcolors.rgb_to_name(tuple(secondary_color))
            except ValueError:
                secondary_color_name = f"RGB({secondary_color[0]}, {secondary_color[1]}, {secondary_color[2]})"
        
        return {
            'primary_color': primary_color_name,
            'primary_rgb': primary_color.tolist(),
            'secondary_color': secondary_color_name,
            'secondary_rgb': secondary_color.tolist() if secondary_color is not None else None,
            'color_palette': dominant_colors.tolist()
        }
        
    except Exception as e:
        print(f"Error extracting color info: {e}")
        return {}

def estimate_detailed_size(detected_objects, image_path):
    """Estimate detailed size information."""
    try:
        image = Image.open(image_path)
        width, height = image.size
        
        if not detected_objects:
            return {'size_category': 'unknown', 'relative_size': 0.0}
        
        # Find the largest detected object
        largest_area = 0
        largest_obj = None
        
        for obj in detected_objects:
            box = obj.get('box')
            if box:
                x1, y1, x2, y2 = box
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_obj = obj
        
        if largest_obj:
            image_area = width * height
            relative_size = largest_area / image_area
            
            # Categorize size
            if relative_size > 0.7:
                size_category = 'large'
            elif relative_size > 0.3:
                size_category = 'medium'
            else:
                size_category = 'small'
            
            return {
                'size_category': size_category,
                'relative_size': relative_size,
                'bounding_box': largest_obj.get('box'),
                'estimated_dimensions': f"{int((x2-x1) * 0.1)}x{int((y2-y1) * 0.1)} cm"  # Rough estimate
            }
        
        return {'size_category': 'unknown', 'relative_size': 0.0}
        
    except Exception as e:
        print(f"Error estimating size: {e}")
        return {}

def estimate_material(detected_objects, image_path):
    """Estimate material based on detected objects and image analysis."""
    try:
        if not detected_objects:
            return {'material': 'unknown', 'confidence': 0.0}
        
        # Material mapping based on object classes
        material_mapping = {
            'cell phone': ['plastic', 'glass', 'metal'],
            'laptop': ['plastic', 'metal', 'glass'],
            'book': ['paper', 'cardboard'],
            'backpack': ['fabric', 'plastic', 'leather'],
            'handbag': ['leather', 'fabric', 'plastic'],
            'wallet': ['leather', 'fabric', 'plastic'],
            'watch': ['metal', 'plastic', 'leather'],
            'car': ['metal', 'plastic', 'glass'],
            'bicycle': ['metal', 'plastic'],
            'shirt': ['cotton', 'polyester', 'fabric'],
            'pants': ['cotton', 'denim', 'fabric'],
            'shoes': ['leather', 'fabric', 'rubber']
        }
        
        best_obj = max(detected_objects, key=lambda x: x.get('confidence', 0.0))
        obj_class = best_obj.get('class', '').lower()
        
        possible_materials = material_mapping.get(obj_class, ['unknown'])
        
        return {
            'material': possible_materials[0],  # Most likely material
            'possible_materials': possible_materials,
            'confidence': best_obj.get('confidence', 0.0)
        }
        
    except Exception as e:
        print(f"Error estimating material: {e}")
        return {'material': 'unknown', 'confidence': 0.0}

@app.route('/process_image', methods=['POST'])
@login_required
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected'}), 400
        
    temp_path = None
    try:
        filename = secure_filename(f"{uuid.uuid4()}_{image.filename}")
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image.save(temp_path)
        if not os.path.exists(temp_path):
            raise Exception("Image file was not saved successfully")
        
        # Check cache first for performance optimization
        cached_result = get_cached_analysis(temp_path)
        if cached_result:
            print(f"[CACHE HIT] Using cached analysis for {temp_path}")
            comprehensive_analysis = cached_result.get('comprehensive_analysis', {})
            analysis_result = cached_result.get('analysis_result', {})
        else:
            # Choose processing mode based on configuration (default to ultra-fast mode for speed)
            processing_mode = os.getenv('PROCESSING_MODE', 'ultra_fast').lower()
            
            if processing_mode == 'ultra_fast':
                print(f"[ULTRA-FAST ANALYSIS] Starting ultra-fast analysis for {temp_path}")
                comprehensive_analysis = enhanced_image_processor.process_image_ultra_fast(
                    temp_path, object_detector
                )
            elif processing_mode == 'fast':
                print(f"[FAST ANALYSIS] Starting fast analysis for {temp_path}")
                comprehensive_analysis = enhanced_image_processor.process_image_fast(
                    temp_path, object_detector
                )
            else:
                # Use comprehensive R-CNN + RNN + BERT analysis with timeout
                print(f"[COMPREHENSIVE ANALYSIS] Starting R-CNN + RNN + BERT analysis for {temp_path}")
                try:
                    # Set a timeout for the analysis (60 seconds)
                    import threading
                    result = [None]
                    exception = [None]
                    
                    def run_analysis():
                        try:
                            result[0] = enhanced_image_processor.process_image_comprehensive(
                                temp_path, object_detector, image_rnn_analyzer, text_analyzer
                            )
                        except Exception as e:
                            exception[0] = e
                    
                    thread = threading.Thread(target=run_analysis)
                    thread.daemon = True
                    thread.start()
                    thread.join(timeout=15)  # 15 second timeout for faster response
                    
                    if thread.is_alive():
                        print("[TIMEOUT] Analysis taking too long, falling back to fast mode")
                        comprehensive_analysis = enhanced_image_processor.process_image_fast(
                            temp_path, object_detector
                        )
                    elif exception[0]:
                        print(f"[ERROR] Analysis failed: {exception[0]}, falling back to fast mode")
                        comprehensive_analysis = enhanced_image_processor.process_image_fast(
                            temp_path, object_detector
                        )
                    else:
                        comprehensive_analysis = result[0]
                        
                except Exception as e:
                    print(f"[ERROR] Analysis error: {e}, falling back to fast mode")
                    comprehensive_analysis = enhanced_image_processor.process_image_fast(
                        temp_path, object_detector
                    )
        
        # Extract results from comprehensive analysis
        rcnn_analysis = comprehensive_analysis.get('rcnn_analysis', {})
        rnn_analysis = comprehensive_analysis.get('rnn_analysis', {})
        bert_analysis = comprehensive_analysis.get('bert_analysis', {})
        fused_analysis = comprehensive_analysis.get('fused_analysis', {})
        enhanced_description = comprehensive_analysis.get('enhanced_description', '')
        
        detected_objects = rcnn_analysis.get('objects', [])
        print(f"[COMPREHENSIVE ANALYSIS] R-CNN: {len(detected_objects)} objects, RNN: {rnn_analysis.get('confidence', 0):.2f}, BERT: {bert_analysis.get('text_confidence', 0):.2f}")
        
        # Create analysis_result from comprehensive analysis (avoid duplicate processing)
        if not cached_result:
            # Extract data from comprehensive analysis to create compatible analysis_result
            analysis_result = {
                'detected_objects': rcnn_analysis.get('objects', []),
                'checkpoint_embedding': None,  # Will be filled by checkpoint model if needed
                'image_quality_score': 0.8,  # Default quality score
                'confidence_score': rcnn_analysis.get('object_confidence', 0.0),
                'suggested_category': 'other',
                'suggested_tags': [],
                'color_info': rnn_analysis.get('details', {}).get('colors', []),
                'size_info': rnn_analysis.get('details', {}).get('size', 'unknown'),
                'material_info': rnn_analysis.get('details', {}).get('materials', []),
                'rnn_analysis': rnn_analysis,
                'comprehensive_analysis': comprehensive_analysis
            }
            
            # Cache the results for future use
            cache_analysis_result(temp_path, {
                'comprehensive_analysis': comprehensive_analysis,
                'analysis_result': analysis_result
            })
            print(f"[CACHE] Cached analysis results for {temp_path}")
        else:
            # Use cached analysis_result
            analysis_result = cached_result.get('analysis_result', {})
        
        # Log detected classes for debugging (only in debug mode)
        if os.getenv('DEBUG_MODE', 'false').lower() == 'true':
            print("Detected classes:", [obj.get('class', '').lower() for obj in detected_objects])
        
        # Filter objects (simplified for speed)
        filtered_objects = []
        image_area = None
        # Skip image area calculation for speed unless needed
        if len(detected_objects) > 0:
            try:
                img = Image.open(temp_path)
                image_area = img.width * img.height
                img.close()
            except:
                pass
        # Simplified object filtering for speed
        for obj in detected_objects:
            obj_class = normalize_class_name(obj.get('class', ''))
            # Basic filtering only
            if obj_class in ALLOWED_CLASSES and obj_class != 'tv':
                obj['class'] = obj_class
                filtered_objects.append(obj)
        # Process only the detected item with the highest confidence
        if filtered_objects:
            best_obj = max(filtered_objects, key=lambda x: x['confidence'])
            obj = best_obj
            obj_class = normalize_class_name(obj.get('class', ''))
            box = obj.get('box')
            
            # Skip complex heuristics in ultra-fast mode
            if processing_mode != 'ultra_fast':
                # Heuristic: If class is 'tv' and bounding box is small, relabel as 'phone'
                if obj_class == 'tv' and box:
                    try:
                        img = Image.open(temp_path)
                        img_w, img_h = img.size
                        x1, y1, x2, y2 = map(int, box)
                        box_w, box_h = x2 - x1, y2 - y1
                        box_area = box_w * box_h
                        img_area = img_w * img_h
                        # If box is less than 30% of image area, likely a phone
                        if box_area / img_area < 0.3:
                            obj_class = 'phone'
                            obj['class'] = 'phone'
                    except Exception as e:
                        print(f"TV/phone heuristic error: {e}")
            # Crop the image to the bounding box for this object
            cropped_path = None
            if box:
                try:
                    img = Image.open(temp_path)
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(img.width, x2), min(img.height, y2)
                    cropped = img.crop((x1, y1, x2, y2))
                    cropped_path = temp_path + f"_crop_{x1}_{y1}_{x2}_{y2}.png"
                    cropped.save(cropped_path)
                    img.close()
                except Exception as e:
                    print(f"Cropping error: {e}")
                    cropped_path = temp_path  # fallback
            else:
                cropped_path = temp_path
            # Simplified processing for ultra-fast mode
            if processing_mode == 'ultra_fast':
                # Skip expensive operations in ultra-fast mode
                color = 'unknown'
                size = 'unknown'
                item_type = obj_class
                category = 'other'
                crack_status = None
                damage_status = None
            else:
                # Color analysis disabled for maximum speed
                color = 'unknown'
                # Get size using only the cropped region
                size = estimate_size([obj], cropped_path)
                # Get item type and category
                item_type, category = classify_item([obj])[0]
                # Crack detection for phones
                crack_status = None
                if obj_class in ["phone", "mobile", "cell phone", "cellphone", "smartphone"]:
                    crack_status = is_screen_cracked(cropped_path, box)
                # Damage detection for all items
                damage_status = is_item_damaged(cropped_path, box)
            # Generate enhanced description using comprehensive analysis
            comprehensive_analysis = analysis_result.get('comprehensive_analysis', {})
            enhanced_description = comprehensive_analysis.get('enhanced_description', '')
            
            if enhanced_description and enhanced_description != "Enhanced description generation failed":
                # Use comprehensive analysis description
                description = enhanced_description
                print(f"[COMPREHENSIVE DESCRIPTION] Using enhanced description: {description[:100]}...")
            else:
                # Fallback to traditional RNN-enhanced description
                rnn_analysis = analysis_result.get('rnn_analysis', {})
                description = generate_description([obj], color, size, category, None, crack_status, damage_status, rnn_analysis)
                print(f"[FALLBACK DESCRIPTION] Using traditional description: {description[:100]}...")
            items_info = [{
                'item_type': item_type,
                'description': description,
                'category': category,
                'color': color,
                'size': size,
                'confidence': obj['confidence'],
                'box': box
            }]
            # Clean up cropped file
            if cropped_path and cropped_path != temp_path:
                try:
                    os.remove(cropped_path)
                except Exception as cleanup_err:
                    print(f"Cropped cleanup error: {cleanup_err}")
        else:
            items_info = []
        # Skip expensive training operations in ultra-fast mode or when disabled
        enable_training = os.getenv('ENABLE_TRAINING', 'true').lower() in ['true', '1', 'yes']
        if processing_mode != 'ultra_fast' and enable_training and filtered_objects:
            try:
                # Create a temporary item-like object for training
                temp_item = type('TempItem', (), {
                    'id': f"processed_{uuid.uuid4()}",
                    'image_path': temp_path,
                    'category': 'other',  # Default category
                    'status': 'found',   # Default status
                    'detected_objects': '[]'
                })()
                
                # Log metrics for processed images (async)
                import threading
                def log_metrics_async():
                    try:
                        log_training_metrics(f"processed_{uuid.uuid4()}", filtered_objects, 'processed')
                        add_item_to_training_dataset(temp_item, filtered_objects)
                        print(f"[TRAINING] Added processed image to training dataset")
                    except Exception as e:
                        print(f"[TRAINING] Error adding processed image to training: {e}")
                
                # Run training operations in background
                training_thread = threading.Thread(target=log_metrics_async)
                training_thread.daemon = True
                training_thread.start()
                
            except Exception as e:
                print(f"[TRAINING] Error setting up training: {e}")
        
        # Skip RNN tracking in ultra-fast mode
        if processing_mode != 'ultra_fast' and current_user.is_authenticated and items_info:
            try:
                item_type = items_info[0].get('item_type', '')
                rnn_manager.add_user_action(
                    user_id=str(current_user.id),
                    action='upload',
                    item_type=item_type,
                    timestamp=datetime.utcnow(),
                    confidence=items_info[0].get('confidence', 1.0)
                )
                
                # Add temporal data for pattern analysis
                rnn_manager.temporal_data.append({
                    'item_type': item_type,
                    'timestamp': datetime.utcnow(),
                    'location': 'upload',
                    'user_id': str(current_user.id)
                })
            except Exception as e:
                print(f"[RNN] Error tracking user behavior: {e}")
        
        # Prepare analysis summary (simplified for ultra-fast mode)
        if processing_mode == 'ultra_fast':
            analysis_summary = {
                'rcnn_objects_detected': len(detected_objects),
                'overall_confidence': comprehensive_analysis.get('processing_confidence', 0.0),
                'processing_method': 'Ultra-Fast R-CNN',
                'enhanced_features': {}
            }
        else:
            analysis_summary = {
                'rcnn_objects_detected': len(detected_objects),
                'rnn_confidence': rnn_analysis.get('confidence', 0.0),
                'bert_confidence': bert_analysis.get('text_confidence', 0.0),
                'overall_confidence': comprehensive_analysis.get('processing_confidence', 0.0),
                'processing_method': 'R-CNN + RNN + BERT',
                'enhanced_features': {
                    'colors': rnn_analysis.get('details', {}).get('colors', []),
                    'materials': rnn_analysis.get('details', {}).get('materials', []),
                    'condition': rnn_analysis.get('details', {}).get('condition', 'unknown'),
                    'size': rnn_analysis.get('details', {}).get('size', 'unknown'),
                    'brands': rnn_analysis.get('details', {}).get('brands', []),
                    'styles': rnn_analysis.get('details', {}).get('styles', [])
                }
            }
        
        return jsonify({
            'items': items_info,
            'detected_objects': filtered_objects,
            'description': items_info[0]['description'] if items_info else "No objects detected in the image.",
            'comprehensive_analysis': analysis_summary
        })
    except Exception as e:
        print(f"process_image error: {e}")
        # Instead of returning an error, return a valid response with no items and status 200
        return jsonify({
            'items': [],
            'detected_objects': [],
            'description': 'No objects detected in the image.'
        }), 200
    finally:
        # Clean up the temporary file (immediate cleanup for speed)
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"Cleanup error: {cleanup_err}")
                # Schedule cleanup for later if immediate removal fails
                try:
                    import atexit
                    atexit.register(lambda: os.remove(temp_path) if os.path.exists(temp_path) else None)
                except:
                    pass

def find_potential_matches(item):
    """Find potential matches for a lost/found item using text similarity with boosts for category, color, size, type, and location proximity."""
    # Expand the time window slightly to improve recall
    opposite_status = 'found' if item.status == 'lost' else 'lost'
    cutoff_date = datetime.utcnow() - timedelta(days=30)
    potential_matches = Item.query.filter(
        Item.status == opposite_status,
        Item.date >= cutoff_date
    ).all()

    def normalize_tokens(value: str) -> set:
        if not value:
            return set()
        # Simple tokenization; keep alphanumerics lowercased
        import re
        tokens = re.findall(r"[a-z0-9]+", value.lower())
        return set(tokens)

    matches = []
    for other_item in potential_matches:
        # Skip if same user
        if other_item.user_id == item.user_id:
            continue

        # Calculate semantic similarity from title+description
        similarity = text_analyzer.compute_similarity(
            f"{item.title} {item.description}",
            f"{other_item.title} {other_item.description}"
        )

        # Category
        category_match = item.category == other_item.category

        # Colors
        color_match = False
        if item.color and other_item.color:
            item_colors = set(c.strip().lower() for c in item.color.split(','))
            other_colors = set(c.strip().lower() for c in other_item.color.split(','))
            color_match = bool(item_colors.intersection(other_colors))

        # Size
        size_match = item.size == other_item.size

        # Type (last token of description as a rough proxy)
        item_type = item.description.split()[-1].lower() if item.description else ''
        other_type = other_item.description.split()[-1].lower() if other_item.description else ''
        type_match = item_type == other_type and item_type != ''

        # Location token overlap
        location_match = False
        try:
            item_loc_tokens = normalize_tokens(item.location)
            other_loc_tokens = normalize_tokens(other_item.location)
            common_loc = item_loc_tokens.intersection(other_loc_tokens)
            location_match = len(common_loc) > 0
        except Exception:
            location_match = False

        # Force perfect score on identical image hash with same category
        if (
            hasattr(item, 'image_hash') and hasattr(other_item, 'image_hash') and item.image_hash and other_item.image_hash and item.image_hash == other_item.image_hash
            and item.category == other_item.category
        ):
            match_score = 1.0
        else:
            # Aggregate score with interpretable boosts
            match_score = similarity
            if category_match:
                match_score += 0.20
            if color_match:
                match_score += 0.15
            if size_match:
                match_score += 0.10
            if type_match:
                match_score += 0.15
            if location_match:
                match_score += 0.10
            # cap at 1.0
            if match_score > 1.0:
                match_score = 1.0

        # Thresholding: require category match to avoid cross-category false positives
        threshold = 0.60
        print(f"[DEBUG] Match score for item {other_item.id}: {match_score:.3f} (threshold: {threshold})")
        if match_score >= threshold and category_match:
            print(f"[DEBUG] --> Potential match found: {item.id} <-> {other_item.id}")
            matches.append({
                'item': other_item,
                'score': match_score,
                'similarity': similarity,
                'category_match': category_match,
                'color_match': color_match,
                'size_match': size_match,
                'type_match': type_match,
                'location_match': location_match
            })

    # Sort matches by score in descending order
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches

@app.route('/add_item', methods=['GET', 'POST'])
@login_required
@guest_restricted
def add_item():
    if request.method == 'POST':
        title = request.form.get('title')
        category = request.form.get('category')
        status = request.form.get('status')
        location = request.form.get('location')
        # Removed true label usage
        
        # Additional information fields
        brand = request.form.get('brand', '')
        model = request.form.get('model', '')
        material = request.form.get('material', '')
        condition = request.form.get('condition', '')
        estimated_value = request.form.get('estimated_value', '')
        serial_number = request.form.get('serial_number', '')
        distinctive_features = request.form.get('distinctive_features', '')
        color_secondary = request.form.get('color_secondary', '')
        size_dimensions = request.form.get('size_dimensions', '')
        weight = request.form.get('weight', '')
        purchase_date = request.form.get('purchase_date', '')
        last_seen_location = request.form.get('last_seen_location', '')
        additional_notes = request.form.get('additional_notes', '')
        
        # Contact information - context-aware based on status
        if status == 'lost':
            # For lost items, user is the owner
            owner_contact = request.form.get('owner_contact', '')
            owner_email = request.form.get('owner_email', '')
            finder_contact = ''  # Not applicable for lost items
            finder_email = ''    # Not applicable for lost items
        else:
            # For found items, user is the finder
            finder_contact = request.form.get('finder_contact', '')
            finder_email = request.form.get('finder_email', '')
            owner_contact = ''   # Not applicable for found items
            owner_email = ''     # Not applicable for found items
        
        
        image = request.files.get('image')
        if not image:
            flash('Please upload an image of the item')
            return redirect(url_for('add_item'))
            
        try:
            # Ensure filename is secure and unique
            original_filename = secure_filename(image.filename)
            file_ext = os.path.splitext(original_filename)[1]
            unique_filename = f"{uuid.uuid4()}{file_ext}"
            
            # Create uploads directory if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            
            # Save the image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            image.save(image_path)
            
            # Store relative path for database (relative to static folder)
            relative_path = os.path.join('uploads', unique_filename).replace("\\", "/")
            
            # Use enhanced image analysis
            analysis_result = enhanced_image_analysis(image_path)
            detected_objects = analysis_result['detected_objects']
            
            # Only keep relevant item classes (disable person, chair, tv, bench, etc.)
            RELEVANT_CLASSES = {
                "cell phone", "mouse", "wallet", "tumbler", "umbrella", "laptop", "keyboard", "book", "backpack", "handbag", "watch"
            }
            filtered_objects = [obj for obj in detected_objects if obj['class'].lower() in RELEVANT_CLASSES]
            if filtered_objects:
                main_obj = max(filtered_objects, key=lambda x: x['confidence'])
                used_objects = [main_obj]
            elif detected_objects:
                # Fallback: use the highest confidence object
                main_obj = max(detected_objects, key=lambda x: x['confidence'])
                used_objects = [main_obj]
            else:
                main_obj = None
                used_objects = []

            # True label field removed; no augmentation of detections

            # Use AI-suggested category if no category provided or if confidence is high
            if not category or category == 'other':
                if analysis_result['confidence_score'] > 0.7:
                    category = analysis_result['suggested_category']

            # Color analysis disabled for maximum speed
            color = 'unknown'
            color_secondary = ''

            # Extract size information from analysis
            size_info = analysis_result.get('size_info', {})
            size = size_info.get('size_category', estimate_size(used_objects, image_path))
            if not size_dimensions and 'estimated_dimensions' in size_info:
                size_dimensions = size_info['estimated_dimensions']

            # Extract material information from analysis
            material_info = analysis_result.get('material_info', {})
            if not material and 'material' in material_info:
                material = material_info['material']

            # Use improved description
            description = generate_description(used_objects, color, size, category, location)
            detected_objects_json = json.dumps(used_objects)

            # Generate text embedding for search
            text_embedding = text_analyzer.analyze_text(f"{title} {description}")
            text_embedding = json.dumps(text_embedding.tolist())

            # Store checkpoint embedding
            checkpoint_embedding = analysis_result.get('checkpoint_embedding')
            checkpoint_embedding_json = json.dumps(checkpoint_embedding) if checkpoint_embedding else None

            # Store suggested tags
            suggested_tags = analysis_result.get('suggested_tags', [])
            suggested_tags_json = json.dumps(suggested_tags) if suggested_tags else None

            # Compute image hash
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
                image_hash = hashlib.md5(image_bytes).hexdigest()

            # Convert numeric fields
            try:
                estimated_value = float(estimated_value) if estimated_value else None
            except (ValueError, TypeError):
                estimated_value = None

            # Convert purchase date
            purchase_date_obj = None
            if purchase_date:
                try:
                    purchase_date_obj = datetime.strptime(purchase_date, '%Y-%m-%d').date()
                except (ValueError, TypeError):
                    purchase_date_obj = None

            item = Item(
                title=title,
                description=description,
                category=category,
                status=status,
                location=location,
                image_path=relative_path,
                user_id=current_user.id,
                detected_objects=detected_objects_json,
                text_embedding=text_embedding,
                color=color,
                size=size,
                image_hash=image_hash,
                
                # Additional information fields
                brand=brand,
                model=model,
                material=material,
                condition=condition,
                estimated_value=estimated_value,
                serial_number=serial_number,
                distinctive_features=distinctive_features,
                color_secondary=color_secondary,
                size_dimensions=size_dimensions,
                weight=weight,
                purchase_date=purchase_date_obj,
                last_seen_location=last_seen_location,
                additional_notes=additional_notes,
                
                # Contact information
                owner_contact=owner_contact,
                finder_contact=finder_contact,
                owner_email=owner_email,
                finder_email=finder_email,
                
                # AI-generated analysis fields
                checkpoint_embedding=checkpoint_embedding_json,
                image_quality_score=analysis_result.get('image_quality_score', 0.0),
                confidence_score=analysis_result.get('confidence_score', 0.0),
                suggested_category=analysis_result.get('suggested_category', 'other'),
                suggested_tags=suggested_tags_json
            )
            
            db.session.add(item)
            db.session.commit()

            # Automatically add item to training dataset
            detected_objects_list = json.loads(detected_objects_json) if detected_objects_json else []
            add_item_to_training_dataset(item, detected_objects_list)

            # Automatically create Match records for high-confidence matches only
            matches = find_potential_matches(item)
            HIGH_CONFIDENCE_THRESHOLD = 0.80
            created_matches = 0
            for match in matches:
                # Only add if not already matched
                lost_id = match['item'].id if item.status == 'found' else item.id
                found_id = item.id if item.status == 'found' else match['item'].id
                existing = Match.query.filter_by(lost_item_id=lost_id, found_item_id=found_id).first()
                # Consider identical image hashes as high-confidence
                hashes_identical = False
                try:
                    hashes_identical = (
                        getattr(item, 'image_hash', None)
                        and getattr(match['item'], 'image_hash', None)
                        and item.image_hash == match['item'].image_hash
                    )
                except Exception:
                    hashes_identical = False
                if not existing and (match['score'] >= HIGH_CONFIDENCE_THRESHOLD or hashes_identical):
                    print(f"[DEBUG] Creating new Match: lost_id={lost_id}, found_id={found_id}, score={match['score']}")
                    new_match = Match(
                        lost_item_id=lost_id,
                        found_item_id=found_id,
                        match_score=match['score'],
                        status='confirmed'
                    )
                    db.session.add(new_match)
                    created_matches += 1
                    
                    # Send email notification to the owner of the lost item
                    try:
                        lost_item = db.session.get(Item, lost_id)
                        found_item = db.session.get(Item, found_id)
                        if lost_item and found_item:
                            email_service = get_email_service()
                            if email_service:
                                email_service.send_match_notification(
                                    lost_item, found_item, match['score']
                                )
                                print(f"[EMAIL] Match notification sent for items {lost_id} <-> {found_id}")
                    except Exception as e:
                        print(f"[EMAIL] Failed to send match notification: {e}")
            db.session.commit()

            if created_matches > 0:
                flash(f'Automatically found {created_matches} potential match(es).', 'success')
            else:
                # No auto-matches found; guide user to manual/assisted matching
                flash('No automatic matches found. Showing similar items you can review.', 'info')
                return redirect(url_for('similar_items', item_id=item.id))

            flash('Item added successfully!')
            return redirect(url_for('index'))

        except Exception as e:
            # Clean up on error
            try:
                if 'image_path' in locals() and os.path.exists(image_path):
                    os.remove(image_path)
            except:
                pass
            print(f"Error adding item: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            flash(f'Error adding item: {str(e)}')
            return redirect(url_for('add_item'))

    return render_template('add_item.html', 
                         status=request.args.get('status', 'lost'),
                         allowed_classes=sorted(ALLOWED_CLASSES))

@app.route('/edit_item/<int:item_id>', methods=['GET', 'POST'])
@login_required
@guest_restricted
def edit_item(item_id):
    """Edit an existing item with additional information fields."""
    item = Item.query.get_or_404(item_id)
    
    # Check if user owns the item or is admin
    if item.user_id != current_user.id and not current_user.is_admin:
        flash('You can only edit your own items.', 'error')
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        try:
            # Update basic fields
            item.title = request.form.get('title', item.title)
            item.category = request.form.get('category', item.category)
            item.status = request.form.get('status', item.status)
            item.location = request.form.get('location', item.location)
            item.description = request.form.get('description', item.description)
            
            # Update additional information fields
            item.brand = request.form.get('brand', item.brand)
            item.model = request.form.get('model', item.model)
            item.material = request.form.get('material', item.material)
            item.condition = request.form.get('condition', item.condition)
            item.serial_number = request.form.get('serial_number', item.serial_number)
            item.distinctive_features = request.form.get('distinctive_features', item.distinctive_features)
            item.color_secondary = request.form.get('color_secondary', item.color_secondary)
            item.size_dimensions = request.form.get('size_dimensions', item.size_dimensions)
            item.weight = request.form.get('weight', item.weight)
            item.last_seen_location = request.form.get('last_seen_location', item.last_seen_location)
            item.additional_notes = request.form.get('additional_notes', item.additional_notes)
            
            # Update contact information - context-aware based on status
            if item.status == 'lost':
                # For lost items, user is the owner
                item.owner_contact = request.form.get('owner_contact', item.owner_contact)
                item.owner_email = request.form.get('owner_email', item.owner_email)
                # Clear finder fields for lost items
                item.finder_contact = ''
                item.finder_email = ''
            else:
                # For found items, user is the finder
                item.finder_contact = request.form.get('finder_contact', item.finder_contact)
                item.finder_email = request.form.get('finder_email', item.finder_email)
                # Clear owner fields for found items
                item.owner_contact = ''
                item.owner_email = ''
            
            
            # Convert numeric fields
            try:
                estimated_value = request.form.get('estimated_value', '')
                item.estimated_value = float(estimated_value) if estimated_value else None
            except (ValueError, TypeError):
                item.estimated_value = None
                

            # Convert purchase date
            purchase_date = request.form.get('purchase_date', '')
            if purchase_date:
                try:
                    item.purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d').date()
                except (ValueError, TypeError):
                    pass  # Keep existing value if conversion fails
            
            # Handle image update if new image provided
            if 'image' in request.files and request.files['image'].filename:
                new_image = request.files['image']
                if new_image.filename:
                    # Save new image
                    original_filename = secure_filename(new_image.filename)
                    file_ext = os.path.splitext(original_filename)[1]
                    unique_filename = f"{uuid.uuid4()}{file_ext}"
                    
                    new_image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                    new_image.save(new_image_path)
                    
                    # Update image path
                    relative_path = os.path.join('uploads', unique_filename).replace("\\", "/")
                    item.image_path = relative_path
                    
                    # Re-analyze image with enhanced analysis
                    analysis_result = enhanced_image_analysis(new_image_path)
                    
                    # Update AI-generated fields
                    item.checkpoint_embedding = json.dumps(analysis_result.get('checkpoint_embedding')) if analysis_result.get('checkpoint_embedding') else None
                    item.image_quality_score = analysis_result.get('image_quality_score', 0.0)
                    item.confidence_score = analysis_result.get('confidence_score', 0.0)
                    item.suggested_category = analysis_result.get('suggested_category', 'other')
                    item.suggested_tags = json.dumps(analysis_result.get('suggested_tags', [])) if analysis_result.get('suggested_tags') else None
                    
                    # Update detected objects
                    detected_objects = analysis_result.get('detected_objects', [])
                    item.detected_objects = json.dumps(detected_objects)
                    
                    # Update color and size if not manually set
                    if not request.form.get('color_override'):
                        color_info = analysis_result.get('color_info', {})
                    item.color = color_info.get('primary_color', item.color)
                    if not item.color_secondary:
                        item.color_secondary = color_info.get('secondary_color', '')
                
                if not request.form.get('size_override'):
                    size_info = analysis_result.get('size_info', {})
                    item.size = size_info.get('size_category', item.size)
                    if not item.size_dimensions and 'estimated_dimensions' in size_info:
                        item.size_dimensions = size_info['estimated_dimensions']
                
                if not item.material:
                    material_info = analysis_result.get('material_info', {})
                    item.material = material_info.get('material', item.material)
                
                # Update text embedding
                text_embedding = text_analyzer.analyze_text(f"{item.title} {item.description}")
                item.text_embedding = json.dumps(text_embedding.tolist())
                
                # Update image hash
                with open(new_image_path, 'rb') as f:
                    image_bytes = f.read()
                item.image_hash = hashlib.md5(image_bytes).hexdigest()
            
            db.session.commit()
            flash('Item updated successfully!', 'success')
            return redirect(url_for('item_detail', item_id=item.id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating item: {str(e)}', 'error')
    
    # Prepare form data for display
    form_data = {
        'title': item.title,
        'category': item.category,
        'status': item.status,
        'location': item.location,
        'description': item.description,
        'brand': item.brand or '',
        'model': item.model or '',
        'material': item.material or '',
        'condition': item.condition or '',
        'estimated_value': item.estimated_value or '',
        'serial_number': item.serial_number or '',
        'distinctive_features': item.distinctive_features or '',
        'color_secondary': item.color_secondary or '',
        'size_dimensions': item.size_dimensions or '',
        'weight': item.weight or '',
        'purchase_date': item.purchase_date.strftime('%Y-%m-%d') if item.purchase_date else '',
        'last_seen_location': item.last_seen_location or '',
        'contact_preference': item.contact_preference or '',
        'additional_notes': item.additional_notes or ''
    }
    
    return render_template('edit_item.html', item=item, form_data=form_data)

@app.route('/item/<int:item_id>')
def view_item(item_id):
    item = Item.query.get_or_404(item_id)
    detected_objects = json.loads(item.detected_objects) if item.detected_objects else None
    return render_template('view_item.html', item=item, detected_objects=detected_objects)

@app.route('/item/<int:item_id>/modify-description', methods=['GET', 'POST'])
@login_required
def modify_description(item_id):
    """Modify item description."""
    item = Item.query.get_or_404(item_id)
    
    # Check if user owns the item or is admin
    if item.user_id != current_user.id and not current_user.is_admin:
        flash('You can only modify descriptions of your own items.', 'error')
        return redirect(url_for('view_item', item_id=item_id))
    
    if request.method == 'POST':
        try:
            # Get the new description
            new_description = request.form.get('description', '').strip()
            
            if not new_description:
                flash('Description cannot be empty.', 'error')
                return redirect(url_for('modify_description', item_id=item_id))
            
            # Update the description
            old_description = item.description
            item.description = new_description
            
            # Update text embedding with new description
            try:
                text_embedding = text_analyzer.analyze_text(f"{item.title} {item.description}")
                item.text_embedding = json.dumps(text_embedding.tolist())
            except Exception as e:
                print(f"Error updating text embedding: {e}")
            
            # Save changes
            db.session.commit()
            
            flash('Description updated successfully!', 'success')
            return redirect(url_for('view_item', item_id=item_id))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating description: {str(e)}', 'error')
            return redirect(url_for('modify_description', item_id=item_id))
    
    # Get original AI-generated description if available
    original_description = None
    if hasattr(item, 'ai_generated_description') and item.ai_generated_description:
        original_description = item.ai_generated_description
    elif item.description and len(item.description) > 50:  # Assume longer descriptions are AI-generated
        original_description = item.description
    
    return render_template('modify_description.html', item=item, original_description=original_description)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    status = request.args.get('status', '')
    
    # Track user behavior for RNN analysis
    if current_user.is_authenticated:
        rnn_manager.add_user_action(
            user_id=str(current_user.id),
            action='search',
            item_type=category,
            timestamp=datetime.utcnow()
        )
        
        # Get RNN predictions for user intent
        user_intent = rnn_manager.predict_user_intent(str(current_user.id))
    else:
        user_intent = {"prediction": "unknown", "confidence": 0.0, "suggestions": []}
    
    items = Item.query
    
    if query:
        # Get query embedding
        query_embedding = text_analyzer.analyze_text(query)
        
        # Analyze description with RNN
        description_analysis = rnn_manager.analyze_description_sequence(query)
        
        # Get all items and compute similarity
        all_items = Item.query.all()
        similarities = []
        
        for item in all_items:
            item_embedding = json.loads(item.text_embedding)
            similarity = text_analyzer.compute_similarity(query, f"{item.title} {item.description}")
            
            # Enhance similarity with RNN analysis
            rnn_similarity = description_analysis.get('similarity_score', 0.0)
            enhanced_similarity = (similarity + rnn_similarity) / 2.0
            
            similarities.append((item, enhanced_similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        items = [item for item, _ in similarities]
    else:
        if category:
            items = items.filter_by(category=category)
        if status:
            items = items.filter_by(status=status)
        items = items.order_by(Item.date.desc()).all()
    
    # Get temporal pattern predictions
    temporal_prediction = rnn_manager.predict_temporal_patterns(category)
    
    return render_template('search.html', items=items, query=query, category=category, status=status,
                         user_intent=user_intent, temporal_prediction=temporal_prediction)

@app.route('/similar_items/<int:item_id>')
def similar_items(item_id):
    item = Item.query.get_or_404(item_id)

    # Strategy: prioritize identical image hash, else rank by appearance (color/size/category)
    all_items = Item.query.filter(Item.id != item_id).all()
    
    # Get user feedback on matches if user is authenticated
    user_feedback = {}
    if current_user.is_authenticated:
        feedback_records = MatchFeedback.query.filter_by(user_id=current_user.id).all()
        for feedback in feedback_records:
            key = f"{feedback.lost_item_id}_{feedback.found_item_id}"
            user_feedback[key] = feedback.feedback_type

    def color_tokens(value: str):
        if not value:
            return set()
        return set(c.strip().lower() for c in value.split(','))

    candidate_scores = []
    for other_item in all_items:
        # Check if user has marked this as "not_match"
        if current_user.is_authenticated:
            feedback_key = f"{item.id}_{other_item.id}"
            reverse_key = f"{other_item.id}_{item.id}"
            if user_feedback.get(feedback_key) == 'not_match' or user_feedback.get(reverse_key) == 'not_match':
                continue  # Skip items marked as not a match
        
        # Identical hash with same category is best possible
        if getattr(item, 'image_hash', None) and getattr(other_item, 'image_hash', None):
            if item.image_hash == other_item.image_hash and item.category == other_item.category:
                candidate_scores.append((other_item, 1.0))
                continue

        score = 0.0
        # Category alignment
        if item.category == other_item.category:
            score += 0.30

        # Color overlap
        item_colors = color_tokens(item.color)
        other_colors = color_tokens(other_item.color)
        if item_colors and other_colors:
            intersection = len(item_colors.intersection(other_colors))
            union = len(item_colors.union(other_colors)) or 1
            color_similarity = intersection / union
            score += 0.50 * color_similarity

        # Size match gives small boost
        if item.size and other_item.size and item.size == other_item.size:
            score += 0.10

        # Very small text assist to break ties only
        try:
            text_sim = text_analyzer.compute_similarity(
                f"{item.title} {item.description}",
                f"{other_item.title} {other_item.description}"
            )
        except Exception:
            text_sim = 0.0
        score += 0.10 * text_sim

        if score > 1.0:
            score = 1.0

        candidate_scores.append((other_item, score))

    # Pick only the single best candidate
    best_item = None
    if candidate_scores:
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_item, _ = candidate_scores[0]

    # Prepare similar items with match status
    similar_items = []
    if best_item:
        # Check if this is a confirmed match
        is_confirmed_match = False
        if current_user.is_authenticated:
            feedback_key = f"{item.id}_{best_item.id}"
            reverse_key = f"{best_item.id}_{item.id}"
            is_confirmed_match = (user_feedback.get(feedback_key) == 'confirmed_match' or 
                                user_feedback.get(reverse_key) == 'confirmed_match')
        
        similar_items.append({
            'item': best_item,
            'is_confirmed_match': is_confirmed_match
        })

    return render_template('similar_items.html', item=item, similar_items=similar_items)

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        
        if user:
            token = user.generate_reset_token()
            reset_url = url_for('reset_password', token=token, _external=True)
            # In a real application, you would send this URL via email
            # For now, we'll just flash it to the user
            flash(f'Password reset link: {reset_url}')
            return redirect(url_for('login'))
        else:
            flash('Email not found')
            
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = User.verify_reset_token(token)
    if not user:
        flash('Invalid or expired reset token')
        return redirect(url_for('forgot_password'))
        
    if request.method == 'POST':
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        if password != confirm_password:
            flash('Passwords do not match')
            return redirect(url_for('reset_password', token=token))
            
        user.set_password(password)
        user.reset_token = None
        user.reset_token_expiry = None
        db.session.commit()
        
        flash('Your password has been reset successfully')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

@app.route('/matches/<int:item_id>')
@login_required
def view_matches(item_id):
    item = Item.query.get_or_404(item_id)
    matches = session.get('potential_matches', [])
    
    # Get full item objects for matches
    match_items = []
    for match in matches:
        match_item = db.session.get(Item, match['id'])
        if match_item:
            match['item'] = match_item
            match_items.append(match)
    
    return render_template('matches.html', item=item, matches=match_items)

@app.route('/admin/users')
@login_required
def admin_users():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    users = User.query.order_by(User.date_joined.desc()).all()
    return render_template('admin_users.html', users=users)

@app.route('/my-items')
@login_required
def my_items():
    """User's own listed items"""
    status = request.args.get('status', 'all')
    items = Item.query.filter_by(user_id=current_user.id)

    if status == 'lost' or status == 'found':
        items = items.filter_by(status=status)
    elif status == 'matched':
        # Get IDs of items that are part of a confirmed match
        matched_lost_ids = db.session.query(Match.lost_item_id).filter_by(status='confirmed')
        matched_found_ids = db.session.query(Match.found_item_id).filter_by(status='confirmed')
        items = items.filter(
            (Item.id.in_(matched_lost_ids)) | (Item.id.in_(matched_found_ids))
        )

    items = items.order_by(Item.date.desc()).all()
    return render_template('my_items.html', items=items, status=status)

@app.route('/admin/items')
@login_required
def admin_items():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    status = request.args.get('status', 'all')
    items = Item.query

    if status == 'lost' or status == 'found':
        items = items.filter_by(status=status)
    elif status == 'matched':
        # Get IDs of items that are part of a confirmed match
        matched_lost_ids = db.session.query(Match.lost_item_id).filter_by(status='confirmed')
        matched_found_ids = db.session.query(Match.found_item_id).filter_by(status='confirmed')
        items = items.filter(
            (Item.id.in_(matched_lost_ids)) | (Item.id.in_(matched_found_ids))
        )

    items = items.order_by(Item.date.desc()).all()
    return render_template('admin_items.html', items=items, status=status)

@app.route('/admin/matches')
@login_required
def admin_matches():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    status = request.args.get('status', 'all')
    matches = Match.query
    
    if status != 'all':
        matches = matches.filter_by(status=status)
    
    matches = matches.order_by(Match.match_date.desc()).all()
    print(f"[DEBUG] Filtering matches by status: {status}, found: {len(matches)}")
    return render_template('admin_matches.html', matches=matches, status=status)

@app.route('/admin')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    # Get statistics
    total_users = User.query.count()
    total_items = Item.query.count()
    lost_items = Item.query.filter_by(status='lost').count()
    found_items = Item.query.filter_by(status='found').count()
    matches_made = Match.query.count()
    confirmed_matches = Match.query.filter_by(status='confirmed').count()
    pending_matches = Match.query.filter_by(status='pending').count()
    
    # Calculate success rate
    success_rate = (confirmed_matches / matches_made * 100) if matches_made > 0 else 0
    
    # Get recent data
    recent_users = User.query.order_by(User.date_joined.desc()).limit(5).all()
    recent_items = Item.query.order_by(Item.date.desc()).limit(5).all()
    recent_matches = Match.query.order_by(Match.match_date.desc()).limit(5).all()
    
    return render_template('admin_dashboard.html',
                         total_users=total_users,
                         total_items=total_items,
                         lost_items=lost_items,
                         found_items=found_items,
                         matches_made=matches_made,
                         confirmed_matches=confirmed_matches,
                         pending_matches=pending_matches,
                         success_rate=success_rate,
                         recent_users=recent_users,
                         recent_items=recent_items,
                         recent_matches=recent_matches)

@app.route('/admin/users/add', methods=['POST'])
@login_required
def admin_add_user():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    is_admin = data.get('is_admin', False)
    
    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password are required'}), 400
    
    if User.query.filter_by(username=username).first():
        return jsonify({'success': False, 'message': 'Username already exists'}), 400
        
    if email and User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'message': 'Email already registered'}), 400
    
    try:
        user = User(username=username, email=email, is_admin=is_admin)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        return jsonify({'success': True, 'message': 'User added successfully'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/admin/users/<int:user_id>')
@login_required
def admin_get_user(user_id):
    """Get user details for viewing/editing"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = User.query.get_or_404(user_id)
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'is_admin': user.is_admin,
        'is_guest': user.is_guest,
        'is_active': user.is_active,
        'date_joined': user.date_joined.strftime('%Y-%m-%d') if user.date_joined else None,
        'last_login': user.last_login.strftime('%Y-%m-%d %H:%M') if user.last_login else None
    })

@app.route('/admin/users/<int:user_id>', methods=['PUT'])
@login_required
def admin_update_user(user_id):
    """Update user information"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    try:
        # Update username
        if 'username' in data and data['username']:
            user.username = data['username']
        
        # Update email
        if 'email' in data:
            user.email = data['email'] if data['email'] else None
        
        # Update password if provided
        if 'password' in data and data['password']:
            user.password_hash = generate_password_hash(data['password'])
        
        # Update admin status
        if 'is_admin' in data:
            user.is_admin = data['is_admin']
        
        db.session.commit()
        return jsonify({'success': True, 'message': 'User updated successfully'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/admin/users/<int:user_id>/toggle-status', methods=['POST'])
@login_required
def admin_toggle_user_status(user_id):
    """Toggle user active/inactive status"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = User.query.get_or_404(user_id)
    data = request.get_json()
    
    try:
        user.is_active = data.get('activate', True)
        db.session.commit()
        return jsonify({'success': True, 'message': f'User {"activated" if user.is_active else "deactivated"} successfully'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/admin/users/<int:user_id>/delete', methods=['DELETE'])
@login_required
def admin_delete_user(user_id):
    """Delete user"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({'success': False, 'message': 'Cannot delete yourself'}), 400
    
    if user.is_admin:
        return jsonify({'success': False, 'message': 'Cannot delete admin users'}), 400
        
    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'success': True, 'message': 'User deleted successfully'})
    
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)}), 400

@app.route('/admin/items/<int:item_id>/edit-data', methods=['GET'])
@login_required
def admin_get_item_data(item_id):
    """Get item data for admin editing"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        item = Item.query.get_or_404(item_id)
        
        # Prepare item data for editing
        item_data = {
            'id': item.id,
            'title': item.title,
            'category': item.category,
            'status': item.status,
            'location': item.location or '',
            'description': item.description,
            'brand': item.brand or '',
            'model': item.model or '',
            'color': item.color or '',
            'size': item.size or '',
            'additional_notes': item.additional_notes or '',
            'owner_username': item.owner.username,
            'date': item.date.strftime('%Y-%m-%d %H:%M')
        }
        
        return jsonify({
            'success': True,
            'item': item_data
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to get item data {item_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to load item data.'
        }), 500

@app.route('/admin/items/<int:item_id>/update', methods=['POST'])
@login_required
def admin_update_item(item_id):
    """Update item data from admin panel"""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        item = Item.query.get_or_404(item_id)
        
        # Update basic fields
        item.title = request.form.get('title', item.title)
        item.category = request.form.get('category', item.category)
        item.status = request.form.get('status', item.status)
        item.location = request.form.get('location', item.location)
        item.description = request.form.get('description', item.description)
        
        # Update additional fields
        item.brand = request.form.get('brand', item.brand)
        item.model = request.form.get('model', item.model)
        item.color = request.form.get('color', item.color)
        item.size = request.form.get('size', item.size)
        item.additional_notes = request.form.get('additional_notes', item.additional_notes)
        
        # Log the update for audit purposes
        print(f"[ADMIN] User {current_user.username} updated item {item_id}: {item.title}")
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Item "{item.title}" updated successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Failed to update item {item_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to update item. Please try again.'
        }), 500

@app.route('/admin/items/<int:item_id>/delete', methods=['POST'])
@login_required
def admin_delete_item(item_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        item = Item.query.get_or_404(item_id)
        
        # Log the deletion for audit purposes
        print(f"[ADMIN] User {current_user.username} deleted item {item_id}: {item.title}")
        
        # Delete associated matches first
        Match.query.filter(
            (Match.lost_item_id == item_id) | (Match.found_item_id == item_id)
        ).delete()
        
        # Delete the item
        db.session.delete(item)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Item "{item.title}" deleted successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Failed to delete item {item_id}: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to delete item. Please try again.'
        }), 500

@app.route('/api/mark-match', methods=['POST'])
@login_required
def mark_match():
    """Mark two items as a confirmed match"""
    try:
        data = request.get_json()
        lost_item_id = data.get('lost_item_id')
        found_item_id = data.get('found_item_id')
        status = data.get('status', 'confirmed')
        
        if not lost_item_id or not found_item_id:
            return jsonify({'success': False, 'error': 'Missing item IDs'}), 400
        
        # Get the items
        lost_item = db.session.get(Item, lost_item_id)
        found_item = db.session.get(Item, found_item_id)
        
        if not lost_item or not found_item:
            return jsonify({'success': False, 'error': 'Items not found'}), 404
        
        # Check if user owns one of the items
        if lost_item.user_id != current_user.id and found_item.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'You can only mark matches for your own items'}), 403
        
        # Check if match already exists
        existing_match = Match.query.filter(
            ((Match.lost_item_id == lost_item_id) & (Match.found_item_id == found_item_id)) |
            ((Match.lost_item_id == found_item_id) & (Match.found_item_id == lost_item_id))
        ).first()
        
        if existing_match:
            return jsonify({'success': False, 'error': 'Match already exists'}), 400
        
        # Create new match
        match = Match(
            lost_item_id=lost_item_id,
            found_item_id=found_item_id,
            status=status,
            created_by=current_user.id
        )
        db.session.add(match)
        
        # Record positive feedback
        MatchFeedback.add_feedback(
            user_id=current_user.id,
            lost_item_id=lost_item_id,
            found_item_id=found_item_id,
            feedback_type='confirmed_match'
        )
        
        # Send email notifications to both owners
        try:
            email_service = EmailService()
            
            # Email to lost item owner
            if lost_item.user_id != current_user.id:
                email_service.send_match_notification(
                    lost_item.owner.email,
                    lost_item.owner.username,
                    lost_item.title,
                    found_item.title,
                    found_item.owner.username,
                    found_item.image_path
                )
            
            # Email to found item owner
            if found_item.user_id != current_user.id:
                email_service.send_match_notification(
                    found_item.owner.email,
                    found_item.owner.username,
                    found_item.title,
                    lost_item.title,
                    lost_item.owner.username,
                    lost_item.image_path
                )
        except Exception as e:
            print(f"[ERROR] Failed to send match notifications: {e}")
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Match confirmed and notifications sent'
        })
        
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Failed to mark match: {e}")
        return jsonify({'success': False, 'error': 'Failed to mark match'}), 500

@app.route('/api/not-match', methods=['POST'])
@login_required
def not_match():
    """Record that two items are not a match"""
    try:
        data = request.get_json()
        lost_item_id = data.get('lost_item_id')
        found_item_id = data.get('found_item_id')
        
        if not lost_item_id or not found_item_id:
            return jsonify({'success': False, 'error': 'Missing item IDs'}), 400
        
        # Get the items
        lost_item = db.session.get(Item, lost_item_id)
        found_item = db.session.get(Item, found_item_id)
        
        if not lost_item or not found_item:
            return jsonify({'success': False, 'error': 'Items not found'}), 404
        
        # Check if user owns one of the items
        if lost_item.user_id != current_user.id and found_item.user_id != current_user.id:
            return jsonify({'success': False, 'error': 'You can only provide feedback for your own items'}), 403
        
        # Record negative feedback
        MatchFeedback.add_feedback(
            user_id=current_user.id,
            lost_item_id=lost_item_id,
            found_item_id=found_item_id,
            feedback_type='not_match'
        )
        print(f"[FEEDBACK] User {current_user.username} marked items {lost_item_id} and {found_item_id} as not a match")
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded'
        })
        
    except Exception as e:
        print(f"[ERROR] Failed to record not-match feedback: {e}")
        return jsonify({'success': False, 'error': 'Failed to record feedback'}), 500

@app.route('/admin/export', methods=['POST'])
@login_required
def admin_export_data():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    export_format = request.form.get('export_format', 'csv')
    export_users = request.form.get('export_users') == 'on'
    export_items = request.form.get('export_items') == 'on'
    export_matches = request.form.get('export_matches') == 'on'
    
    # Implementation for data export
    # This is a placeholder - you'll need to implement the actual export logic
    
    flash('Data exported successfully')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/settings', methods=['POST'])
@login_required
def admin_update_settings():
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    settings = Settings.query.first()
    if not settings:
        settings = Settings()
        
    settings.item_expiry_days = request.form.get('item_expiry_days', type=int)
    settings.max_image_size = request.form.get('max_image_size', type=int)
    settings.matching_threshold = request.form.get('matching_threshold', type=int)
    settings.enable_email_notifications = request.form.get('enable_email_notifications') == 'on'
    
    db.session.add(settings)
    db.session.commit()
    
    flash('Settings updated successfully')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/matches/<int:match_id>/confirm', methods=['POST'])
@login_required
def admin_confirm_match(match_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    match = Match.query.get_or_404(match_id)
    match.status = 'confirmed'
    db.session.commit()
    
    return jsonify({'message': 'Match confirmed successfully'})

@app.route('/admin/matches/<int:match_id>/reject', methods=['POST'])
@login_required
def admin_reject_match(match_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    match = Match.query.get_or_404(match_id)
    match.status = 'rejected'
    db.session.commit()
    
    return jsonify({'message': 'Match rejected successfully'})

def log_training_metrics(item_id, detected_objects, source='user_upload'):
    """Log training metrics for analysis."""
    try:
        if detected_objects:
            # Log object detection metrics
            for obj in detected_objects:
                obj_class = obj.get('class', 'unknown')
                confidence = obj.get('confidence', 0.0)
                
                # Log detection confidence
                ModelMetrics.add_metric(
                    'unified_model',
                    'detection_confidence',
                    confidence,
                    class_name=obj_class
                )
                
                # Log object count by class
                ModelMetrics.add_metric(
                    'unified_model',
                    'object_count',
                    1.0,
                    class_name=obj_class
                )
            
            # Log sample metrics
            ModelMetrics.add_metric(
                'unified_model',
                'samples_added',
                1.0
            )
            
            ModelMetrics.add_metric(
                'unified_model',
                'objects_per_sample',
                len(detected_objects)
            )
            
            # Log source metrics
            ModelMetrics.add_metric(
                'unified_model',
                f'samples_from_{source}',
                1.0
            )
            
    except Exception as e:
        print(f"[METRICS] Error logging training metrics: {e}")

def add_item_to_training_dataset(item, detected_objects):
    """Automatically add uploaded item to unified training dataset and integrate into current system."""
    try:
        # Create unified training sample
        training_sample = {
            'item_id': item.id,
            'image_path': item.image_path,
            'detected_objects': detected_objects,
            'user_labels': [],  # Will be populated when users provide feedback
            'auto_labels': detected_objects,  # Use detected objects as initial labels
            'category': getattr(item, 'category', 'other'),
            'status': getattr(item, 'status', 'found'),
            'timestamp': datetime.now().isoformat(),
            'source': 'user_upload'
        }
        
        # Save training sample to file
        training_file = os.path.join(TRAINING_DATA_DIR, f'training_sample_{item.id}.json')
        with open(training_file, 'w') as f:
            json.dump(training_sample, f, indent=2)
        
        # Copy image to training images directory
        if item.image_path:
            # Try multiple possible source paths
            possible_paths = [
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', item.image_path),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), item.image_path),
                item.image_path
            ]
            
            source_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    source_path = path
                    break
            
            if source_path:
                # Create category subdirectory
                category = getattr(item, 'category', 'other').lower().replace(' ', '_')
                category_dir = os.path.join(TRAINING_IMAGES_DIR, category)
                os.makedirs(category_dir, exist_ok=True)
                
                # Copy image with item ID
                dest_path = os.path.join(category_dir, f'item_{item.id}_{os.path.basename(item.image_path)}')
                import shutil
                shutil.copy2(source_path, dest_path)
                
                # Update training sample with new image path
                training_sample['training_image_path'] = os.path.relpath(dest_path, TRAINING_DATA_DIR)
                print(f"[TRAINING] Copied image to training directory: {dest_path}")
            else:
                print(f"[TRAINING] Warning: Could not find source image for {item.image_path}")
        
        # Log training metrics
        log_training_metrics(item.id, detected_objects, 'user_upload')
        
        # Add to unified model trainer
        if model_trainer.add_feedback(
            item.image_path,
            detected_objects,
            [],  # No user feedback initially
            None  # Do not use true labels
        ):
            print(f"[UNIFIED TRAINING] Added item {item.id} to unified training dataset")
            
            # Integrate into current system by updating model parameters
            integrate_training_data_into_system(item, detected_objects)
            
            return True
        else:
            print(f"[UNIFIED TRAINING] Failed to add item {item.id} to unified training dataset")
            return False
            
    except Exception as e:
        print(f"[UNIFIED TRAINING] Error adding item {item.id} to unified training dataset: {e}")
        return False

def integrate_training_data_into_system(item, detected_objects):
    """Integrate new training data into the current system for immediate use."""
    try:
        # Update object detection confidence adjustments
        for obj in detected_objects:
            obj_class = obj.get('class', 'unknown').lower()
            confidence = obj.get('confidence', 0.0)
            
            # If confidence is high, boost future detections of this class
            if confidence > 0.8:
                if obj_class not in unified_model.confidence_adjustments:
                    unified_model.confidence_adjustments[obj_class] = []
                
                unified_model.confidence_adjustments[obj_class].append({
                    'original_confidence': confidence,
                    'feedback_confidence': min(1.0, confidence + 0.05),
                    'adjustment': 0.05,
                    'feedback_type': 'high_confidence_boost',
                    'timestamp': datetime.now().isoformat()
                })
        
        # Update text similarity adjustments based on item descriptions
        if hasattr(item, 'description') and item.description:
            # Create text similarity training data
            text_similarity_data = {
                'text1': item.title or '',
                'text2': item.description,
                'user_score': 0.9,  # High similarity for same item
                'original_score': 0.0
            }
            
            unified_model.update_similarity_adjustments(text_similarity_data)
        
        # Save updated adjustments
        unified_model.save_adjustments()
        
        print(f"[SYSTEM INTEGRATION] Integrated training data for item {item.id} into current system")
        
    except Exception as e:
        print(f"[SYSTEM INTEGRATION] Error integrating training data: {e}")

def get_training_dataset_stats():
    """Get comprehensive statistics about the training dataset."""
    try:
        # Count training samples
        training_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.json')]
        
        # Count images by category
        category_counts = {}
        if os.path.exists(TRAINING_IMAGES_DIR):
            for category_dir in os.listdir(TRAINING_IMAGES_DIR):
                category_path = os.path.join(TRAINING_IMAGES_DIR, category_dir)
                if os.path.isdir(category_path):
                    image_count = len([f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))])
                    category_counts[category_dir] = image_count
        
        # Get total images
        total_images = sum(category_counts.values())
        
        # Analyze training samples for detailed metrics
        sample_metrics = {
            'total_samples': len(training_files),
            'total_images': total_images,
            'category_counts': category_counts,
            'categories': list(category_counts.keys()),
            'samples_with_objects': 0,
            'samples_with_feedback': 0,
            'total_detected_objects': 0,
            'object_class_distribution': {},
            'confidence_stats': {
                'average': 0.0,
                'min': 1.0,
                'max': 0.0
            },
            'recent_activity': {
                'last_7_days': 0,
                'last_30_days': 0
            },
            'source_breakdown': {
                'user_upload': 0,
                'retroactive': 0,
                'processed': 0
            }
        }
        
        # Analyze each training sample
        confidences = []
        now = datetime.now()
        week_ago = now - timedelta(days=7)
        month_ago = now - timedelta(days=30)
        
        for file in training_files:
            try:
                file_path = os.path.join(TRAINING_DATA_DIR, file)
                with open(file_path, 'r') as f:
                    sample = json.load(f)
                
                # Count samples with detected objects
                if sample.get('detected_objects'):
                    sample_metrics['samples_with_objects'] += 1
                    sample_metrics['total_detected_objects'] += len(sample['detected_objects'])
                    
                    # Analyze object classes and confidence
                    for obj in sample['detected_objects']:
                        obj_class = obj.get('class', 'unknown')
                        confidence = obj.get('confidence', 0.0)
                        
                        # Object class distribution
                        if obj_class not in sample_metrics['object_class_distribution']:
                            sample_metrics['object_class_distribution'][obj_class] = 0
                        sample_metrics['object_class_distribution'][obj_class] += 1
                        
                        # Confidence statistics
                        confidences.append(confidence)
                
                # Count samples with user feedback
                if sample.get('user_labels') or sample.get('user_feedback'):
                    sample_metrics['samples_with_feedback'] += 1
                
                # Source breakdown
                source = sample.get('source', 'unknown')
                if source in sample_metrics['source_breakdown']:
                    sample_metrics['source_breakdown'][source] += 1
                else:
                    sample_metrics['source_breakdown']['retroactive'] += 1
                
                # Recent activity
                timestamp_str = sample.get('timestamp', '')
                if timestamp_str:
                    try:
                        sample_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        if sample_time >= week_ago:
                            sample_metrics['recent_activity']['last_7_days'] += 1
                        if sample_time >= month_ago:
                            sample_metrics['recent_activity']['last_30_days'] += 1
                    except:
                        pass
                        
            except Exception as e:
                print(f"[TRAINING] Error analyzing sample {file}: {e}")
        
        # Calculate confidence statistics
        if confidences:
            sample_metrics['confidence_stats'] = {
                'average': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'count': len(confidences)
            }
        
        return sample_metrics
        
    except Exception as e:
        print(f"[TRAINING] Error getting dataset stats: {e}")
        return {
            'total_samples': 0,
            'total_images': 0,
            'category_counts': {},
            'categories': [],
            'samples_with_objects': 0,
            'samples_with_feedback': 0,
            'total_detected_objects': 0,
            'object_class_distribution': {},
            'confidence_stats': {'average': 0.0, 'min': 1.0, 'max': 0.0, 'count': 0},
            'recent_activity': {'last_7_days': 0, 'last_30_days': 0},
            'source_breakdown': {'user_upload': 0, 'retroactive': 0, 'processed': 0}
        }

def retroactive_match_all():
    lost_items = Item.query.filter_by(status='lost').all()
    found_items = Item.query.filter_by(status='found').all()
    count = 0
    HIGH_CONFIDENCE_THRESHOLD = 0.80
    for lost_item in lost_items:
        matches = find_potential_matches(lost_item)
        for match in matches:
            found_item = match['item']
            found_id = found_item.id
            existing = Match.query.filter_by(lost_item_id=lost_item.id, found_item_id=found_id).first()
            # Identical hashes qualify regardless of score
            hashes_identical = False
            try:
                hashes_identical = (
                    getattr(lost_item, 'image_hash', None)
                    and getattr(found_item, 'image_hash', None)
                    and lost_item.image_hash == found_item.image_hash
                )
            except Exception:
                hashes_identical = False
            if not existing and (match['score'] >= HIGH_CONFIDENCE_THRESHOLD or hashes_identical):
                new_match = Match(
                    lost_item_id=lost_item.id,
                    found_item_id=found_id,
                    match_score=match['score'],
                    status='confirmed'
                )
                db.session.add(new_match)
                count += 1
                
                # Send email notification to the owner of the lost item
                try:
                    found_item = db.session.get(Item, found_id)
                    if found_item:
                        email_service = get_email_service()
                        if email_service:
                            email_service.send_match_notification(
                                lost_item, found_item, match['score']
                            )
                            print(f"[EMAIL] Retroactive match notification sent for items {lost_item.id} <-> {found_id}")
                except Exception as e:
                    print(f"[EMAIL] Failed to send retroactive match notification: {e}")
    db.session.commit()
    print(f"[DEBUG] Retroactive matching complete. {count} new matches created.")

def add_all_existing_items_to_training():
    """Add all existing items to the training dataset."""
    items = Item.query.all()
    count = 0
    for item in items:
        detected_objects = json.loads(item.detected_objects) if item.detected_objects else []
        if add_item_to_training_dataset(item, detected_objects):
            count += 1
    
    print(f"[TRAINING] Added {count} existing items to training dataset")
    return count

def add_all_uploaded_images_to_training():
    """Add all uploaded images to training dataset and train the model with them."""
    uploads_dir = app.config['UPLOAD_FOLDER']
    static_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'img')
    
    count = 0
    errors = 0
    training_samples = []
    
    # Process images from uploads directory
    if os.path.exists(uploads_dir):
        for filename in os.listdir(uploads_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_path = os.path.join(uploads_dir, filename)
                try:
                    # Create a temporary item-like object for training
                    temp_item = type('TempItem', (), {
                        'id': f"upload_{filename}",
                        'image_path': os.path.join('uploads', filename).replace("\\", "/"),
                        'category': 'other',  # Default category
                        'status': 'found',   # Default status
                        'detected_objects': '[]'
                    })()
                    
                    # Detect objects in the image
                    detected_objects = object_detector.detect_objects(image_path)
                    
                    # Log metrics for retroactive training
                    log_training_metrics(f"upload_{filename}", detected_objects, 'retroactive')
                    
                    # Add to training dataset
                    if add_item_to_training_dataset(temp_item, detected_objects):
                        count += 1
                        training_samples.append({
                            'image_path': image_path,
                            'detected_objects': detected_objects,
                            'source': 'upload'
                        })
                        print(f"[TRAINING] Added uploaded image: {filename}")
                    
                except Exception as e:
                    errors += 1
                    print(f"[TRAINING] Error processing {filename}: {e}")
    
    # Process images from static/img directory
    if os.path.exists(static_img_dir):
        for filename in os.listdir(static_img_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                image_path = os.path.join(static_img_dir, filename)
                try:
                    # Create a temporary item-like object for training
                    temp_item = type('TempItem', (), {
                        'id': f"static_{filename}",
                        'image_path': os.path.join('img', filename).replace("\\", "/"),
                        'category': 'other',  # Default category
                        'status': 'found',   # Default status
                        'detected_objects': '[]'
                    })()
                    
                    # Detect objects in the image
                    detected_objects = object_detector.detect_objects(image_path)
                    
                    # Log metrics for retroactive training
                    log_training_metrics(f"static_{filename}", detected_objects, 'retroactive')
                    
                    # Add to training dataset
                    if add_item_to_training_dataset(temp_item, detected_objects):
                        count += 1
                        training_samples.append({
                            'image_path': image_path,
                            'detected_objects': detected_objects,
                            'source': 'static'
                        })
                        print(f"[TRAINING] Added static image: {filename}")
                    
                except Exception as e:
                    errors += 1
                    print(f"[TRAINING] Error processing {filename}: {e}")
    
    # Train the model with all collected samples
    if training_samples:
        print(f"[TRAINING] Training model with {len(training_samples)} new samples...")
        train_model_with_samples(training_samples)
    
    print(f"[TRAINING] Added {count} uploaded images to training dataset, {errors} errors")
    return count, errors

def train_model_with_samples(training_samples):
    """Train the unified model with a collection of training samples."""
    try:
        print(f"[MODEL TRAINING] Starting training with {len(training_samples)} samples")
        
        # Prepare training data for the unified model
        for sample in training_samples:
            # Create training sample data
            sample_data = {
                'image_path': sample['image_path'],
                'detected_objects': sample['detected_objects'],
                'user_feedback': [],  # No user feedback initially
                'true_labels': sample['detected_objects'],  # Use detected objects as true labels
                'timestamp': datetime.now().isoformat(),
                'source': sample['source']
            }
            
            # Add to unified model training
            model_trainer.add_feedback(
                sample['image_path'],
                sample['detected_objects'],
                [],  # No user feedback
                sample['detected_objects']  # Use detected objects as true labels
            )
        
        # Retrain the unified model
        success = model_trainer.retrain_models()
        
        if success:
            print(f"[MODEL TRAINING] Successfully trained model with {len(training_samples)} samples")
            
            # Log training completion metrics
            ModelMetrics.add_metric('unified_model', 'training_samples_processed', len(training_samples))
            ModelMetrics.add_metric('unified_model', 'model_retrain_success', 1.0)
            
            return True
        else:
            print(f"[MODEL TRAINING] Failed to retrain model")
            ModelMetrics.add_metric('unified_model', 'model_retrain_failure', 1.0)
            return False
            
    except Exception as e:
        print(f"[MODEL TRAINING] Error during training: {e}")
        ModelMetrics.add_metric('unified_model', 'model_training_error', 1.0)
        return False

@app.route('/admin/retroactive_match')
@login_required
def admin_retroactive_match():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    retroactive_match_all()
    flash('Retroactive matching complete!')
    return redirect(url_for('admin_matches'))

# Background periodic retroactive matching
def _retroactive_matching_worker(interval_minutes=30):
    with app.app_context():
        while True:
            try:
                retroactive_match_all()
            except Exception as e:
                print(f"[BACKGROUND] Retroactive matching error: {e}")
            time.sleep(max(5, int(interval_minutes) * 60))

def start_background_matching():
    try:
        t = threading.Thread(target=_retroactive_matching_worker, kwargs={"interval_minutes": 60}, daemon=True)
        t.start()
        print("[BACKGROUND] Retroactive matching thread started (every 60 minutes)")
    except Exception as e:
        print(f"[BACKGROUND] Failed to start matching thread: {e}")

# Start background matcher after app initialization
start_background_matching()

@app.route('/admin/download_training_data.zip')
@login_required
def download_training_data():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))

    # Create an in-memory ZIP of the training_data directory
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        base_dir = TRAINING_DATA_DIR
        for root, dirs, files in os.walk(base_dir):
            for file_name in files:
                abs_path = os.path.join(root, file_name)
                # Store paths inside the zip relative to training_data/
                arcname = os.path.relpath(abs_path, base_dir)
                zf.write(abs_path, arcname)
    memory_file.seek(0)

    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f'training_data_{timestamp}.zip'
    )

# Training and Feedback Routes
@app.route('/feedback/<int:item_id>', methods=['GET', 'POST'])
@login_required
def provide_feedback(item_id):
    """Allow users to provide feedback on detected objects."""
    item = Item.query.get_or_404(item_id)
    
    if request.method == 'POST':
        feedback_data = request.get_json()
        detected_objects = json.loads(item.detected_objects) if item.detected_objects else []
        user_feedback = feedback_data.get('feedback', [])
        
        # Store feedback in database
        training_record = TrainingData(
            image_path=item.image_path,
            detected_objects=item.detected_objects,
            user_feedback=json.dumps(user_feedback),
            user_id=current_user.id,
            feedback_type='correction'
        )
        db.session.add(training_record)
        db.session.commit()
        
        # Add to model trainer
        if model_trainer.add_feedback(item.image_path, detected_objects, user_feedback):
            flash('Thank you for your feedback! It will help improve the model.')
        else:
            flash('Error processing feedback. Please try again.')
        
        return jsonify({'success': True})
    
    detected_objects = json.loads(item.detected_objects) if item.detected_objects else []
    return render_template('feedback.html', item=item, detected_objects=detected_objects)

@app.route('/admin/training')
@login_required
def admin_training():
    """Admin interface for managing model training."""
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    # Get unified training statistics and adapt to template structure
    unified_stats = model_trainer.get_training_statistics()
    training_stats = {
        'total_training_samples': unified_stats.get('total_samples', 0),
        'object_detector': {
            'total_samples': unified_stats.get('object_detection_samples', 0),
            'average_confidence_adjustment': unified_stats.get('average_confidence_adjustment', 0.0),
            'classes_feedback': unified_stats.get('classes_feedback', {})
        },
        'text_analyzer': {
            'total_samples': unified_stats.get('text_analysis_samples', 0),
            'average_similarity_adjustment': unified_stats.get('average_similarity_adjustment', 0.0)
        }
    }
    
    # Get recent training data
    recent_training = TrainingData.query.order_by(TrainingData.created_at.desc()).limit(10).all()
    
    # Get unified model metrics
    object_metrics = ModelMetrics.query.filter_by(model_type='unified_model').order_by(ModelMetrics.timestamp.desc()).limit(20).all()
    text_metrics = ModelMetrics.query.filter_by(model_type='unified_model').order_by(ModelMetrics.timestamp.desc()).limit(20).all()
    
    return render_template('admin_training.html', 
                         training_stats=training_stats,
                         recent_training=recent_training,
                         object_metrics=object_metrics,
                         text_metrics=text_metrics)

@app.route('/admin/rnn_analytics')
@login_required
def admin_rnn_analytics():
    """RNN Analytics Dashboard"""
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    # Get user behavior statistics
    user_stats = {}
    for user_id, actions in rnn_manager.user_sequences.items():
        if actions:
            user_stats[user_id] = {
                'total_actions': len(actions),
                'last_action': actions[-1][2].strftime('%Y-%m-%d %H:%M:%S'),
                'action_types': list(set([a[0] for a in actions])),
                'item_types': list(set([a[1] for a in actions if a[1]]))
            }
    
    # Get temporal pattern statistics
    temporal_stats = {
        'total_items': len(rnn_manager.temporal_data),
        'by_type': {},
        'by_hour': {},
        'by_day': {}
    }
    
    for data in rnn_manager.temporal_data:
        item_type = data.get('item_type', 'unknown')
        timestamp = data.get('timestamp', datetime.utcnow())
        
        # Count by type
        temporal_stats['by_type'][item_type] = temporal_stats['by_type'].get(item_type, 0) + 1
        
        # Count by hour
        hour = timestamp.hour
        temporal_stats['by_hour'][hour] = temporal_stats['by_hour'].get(hour, 0) + 1
        
        # Count by day of week
        day = timestamp.weekday()
        temporal_stats['by_day'][day] = temporal_stats['by_day'].get(day, 0) + 1
    
    return render_template('admin_rnn_analytics.html', 
                         user_stats=user_stats, 
                         temporal_stats=temporal_stats)

@app.route('/admin/training/retrain', methods=['POST'])
@login_required
def admin_retrain_models():
    """Retrain models with accumulated data."""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        success = model_trainer.retrain_models()
        if success:
            flash('Unified model retrained successfully!')
        else:
            flash('Not enough training data for retraining (need at least 5 samples)')
        
        return redirect(url_for('admin_training'))
    except Exception as e:
        flash(f'Error retraining models: {str(e)}')
        return redirect(url_for('admin_training'))

@app.route('/admin/training/feedback/<int:feedback_id>', methods=['GET', 'POST'])
@login_required
def admin_view_feedback(feedback_id):
    """View and process individual feedback."""
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    training_record = TrainingData.query.get_or_404(feedback_id)
    
    if request.method == 'POST':
        action = request.form.get('action')
        if action == 'approve':
            training_record.is_processed = True
            db.session.commit()
            flash('Feedback approved and processed.')
        elif action == 'reject':
            db.session.delete(training_record)
            db.session.commit()
            flash('Feedback rejected and deleted.')
        
        return redirect(url_for('admin_training'))
    
    detected_objects = json.loads(training_record.detected_objects) if training_record.detected_objects else []
    user_feedback = json.loads(training_record.user_feedback) if training_record.user_feedback else []
    
    return render_template('admin_feedback_detail.html', 
                         training_record=training_record,
                         detected_objects=detected_objects,
                         user_feedback=user_feedback)

@app.route('/admin/training/metrics')
@login_required
def admin_model_metrics():
    """View detailed model performance metrics."""
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    # Get unified model metrics
    object_metrics = ModelMetrics.query.filter_by(model_type='unified_model').all()
    text_metrics = ModelMetrics.query.filter_by(model_type='unified_model').all()
    
    # Get checkpoint model metrics
    checkpoint_metrics = ModelMetrics.query.filter_by(model_type='checkpoint_model').all()
    
    # Calculate summary statistics
    def calculate_metrics(metrics):
        if not metrics:
            return {}
        
        metric_names = set(m.metric_name for m in metrics)
        summary = {}
        for name in metric_names:
            values = [m.metric_value for m in metrics if m.metric_name == name]
            summary[name] = {
                'count': len(values),
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }
        return summary
    
    object_summary = calculate_metrics(object_metrics)
    text_summary = calculate_metrics(text_metrics)
    checkpoint_summary = calculate_metrics(checkpoint_metrics)
    
    # Get latest checkpoint evaluation results
    latest_checkpoint_results = {}
    if checkpoint_metrics:
        # Get the most recent evaluation date
        latest_evaluation = ModelMetrics.query.filter_by(model_type='checkpoint_model').order_by(ModelMetrics.timestamp.desc()).first()
        
        latest_checkpoint_results = {
            'accuracy': next((m.metric_value for m in checkpoint_metrics if m.metric_name == 'coco_accuracy'), 0),
            'f1_weighted': next((m.metric_value for m in checkpoint_metrics if m.metric_name == 'coco_f1_weighted'), 0),
            'f1_macro': next((m.metric_value for m in checkpoint_metrics if m.metric_name == 'coco_f1_macro'), 0),
            'avg_confidence': next((m.metric_value for m in checkpoint_metrics if m.metric_name == 'coco_avg_confidence'), 0),
            'avg_processing_time': next((m.metric_value for m in checkpoint_metrics if m.metric_name == 'coco_avg_processing_time'), 0),
            'num_samples': next((m.metric_value for m in checkpoint_metrics if m.metric_name == 'coco_num_samples'), 0),
            'evaluation_date': latest_evaluation.timestamp if latest_evaluation else None
        }
    
    return render_template('admin_model_metrics.html',
                         object_metrics=object_metrics,
                         text_metrics=text_metrics,
                         checkpoint_metrics=checkpoint_metrics,
                         object_summary=object_summary,
                         text_summary=text_summary,
                         checkpoint_summary=checkpoint_summary,
                         latest_checkpoint_results=latest_checkpoint_results)

@app.route('/admin/training/dataset/metrics')
@login_required
def admin_dataset_metrics():
    """View comprehensive training dataset metrics."""
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    # Get comprehensive dataset statistics
    dataset_stats = get_training_dataset_stats()
    
    # Get recent training data for activity analysis
    recent_training = TrainingData.query.order_by(TrainingData.created_at.desc()).limit(50).all()
    
    # Calculate additional metrics
    feedback_rate = 0.0
    if dataset_stats['total_samples'] > 0:
        feedback_rate = (dataset_stats['samples_with_feedback'] / dataset_stats['total_samples']) * 100
    
    avg_objects_per_sample = 0.0
    if dataset_stats['samples_with_objects'] > 0:
        avg_objects_per_sample = dataset_stats['total_detected_objects'] / dataset_stats['samples_with_objects']
    
    return render_template('admin_dataset_metrics.html',
                         dataset_stats=dataset_stats,
                         recent_training=recent_training,
                         feedback_rate=feedback_rate,
                         avg_objects_per_sample=avg_objects_per_sample)

def _compute_classification_metrics(y_true, y_pred, labels):
    """Compute comprehensive classification metrics including F1, MAP, and accuracy."""
    if not y_true or not y_pred:
        return {
            'accuracy': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'weighted_f1': 0.0,
            'per_class': {},
            'confusion_matrix': {}
        }
    
    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0.0

    # Per-class precision/recall/F1
    per_class = {}
    eps = 1e-9
    total_support = 0
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t != label and p != label)
        
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        support = sum(1 for t in y_true if t == label)
        
        per_class[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
        total_support += support

    # Macro F1 (average of per-class F1 scores)
    macro_f1 = sum(v['f1'] for v in per_class.values()) / (len(per_class) or 1)

    # Micro F1 (global precision and recall)
    total_tp = sum(v['tp'] for v in per_class.values())
    total_fp = sum(v['fp'] for v in per_class.values())
    total_fn = sum(v['fn'] for v in per_class.values())
    
    micro_precision = total_tp / (total_tp + total_fp + eps)
    micro_recall = total_tp / (total_tp + total_fn + eps)
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall + eps)

    # Weighted F1 (weighted by support)
    weighted_f1 = sum(v['f1'] * v['support'] for v in per_class.values()) / (total_support or 1)

    # Confusion matrix
    confusion_matrix = {}
    for true_label in labels:
        confusion_matrix[true_label] = {}
        for pred_label in labels:
            confusion_matrix[true_label][pred_label] = sum(
                1 for t, p in zip(y_true, y_pred) 
                if t == true_label and p == pred_label
            )

    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'micro_f1': micro_f1,
        'weighted_f1': weighted_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'per_class': per_class,
        'confusion_matrix': confusion_matrix,
        'total_samples': len(y_true),
        'num_classes': len(labels)
    }

def _compute_map(ground_truth, predictions):
    """Compute mean Average Precision for retrieval-like ranking.
    ground_truth: dict[item_id] -> set of true relevant item_ids
    predictions: dict[item_id] -> list of ranked predicted item_ids
    """
    def average_precision(relevant_set, ranked_list):
        if not ranked_list or not relevant_set:
            return 0.0
        hits = 0
        sum_precisions = 0.0
        for idx, pred in enumerate(ranked_list, 1):
            if pred in relevant_set:
                hits += 1
                sum_precisions += hits / idx
        return sum_precisions / len(relevant_set)

    def precision_at_k(relevant_set, ranked_list, k):
        """Compute precision at k."""
        if not ranked_list or not relevant_set:
            return 0.0
        top_k = ranked_list[:k]
        hits = sum(1 for pred in top_k if pred in relevant_set)
        return hits / min(k, len(ranked_list))

    def recall_at_k(relevant_set, ranked_list, k):
        """Compute recall at k."""
        if not ranked_list or not relevant_set:
            return 0.0
        top_k = ranked_list[:k]
        hits = sum(1 for pred in top_k if pred in relevant_set)
        return hits / len(relevant_set)

    ap_values = []
    precision_at_5 = []
    precision_at_10 = []
    recall_at_5 = []
    recall_at_10 = []
    
    for qid, rel in ground_truth.items():
        pred_list = predictions.get(qid, [])
        ap_values.append(average_precision(rel, pred_list))
        precision_at_5.append(precision_at_k(rel, pred_list, 5))
        precision_at_10.append(precision_at_k(rel, pred_list, 10))
        recall_at_5.append(recall_at_k(rel, pred_list, 5))
        recall_at_10.append(recall_at_k(rel, pred_list, 10))
    
    return {
        'map': sum(ap_values) / (len(ap_values) or 1),
        'precision_at_5': sum(precision_at_5) / (len(precision_at_5) or 1),
        'precision_at_10': sum(precision_at_10) / (len(precision_at_10) or 1),
        'recall_at_5': sum(recall_at_5) / (len(recall_at_5) or 1),
        'recall_at_10': sum(recall_at_10) / (len(recall_at_10) or 1),
        'num_queries': len(ap_values)
    }

@app.route('/admin/training/evaluate_coco', methods=['POST'])
@login_required
def admin_evaluate_coco():
    """Run COCO evaluation using best_model1.pth."""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        import subprocess
        import os
        
        # Run COCO evaluation script
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate_coco_detection.py')
        result = subprocess.run([
            sys.executable, script_path, 
            '--dataset', 'image recog.v1i.coco-mmdetection',
            '--split', 'test',
            '--limit', '50'  # Limit to 50 images for quick test
        ], capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
        
        if result.returncode == 0:
            # Find the generated results file
            import glob
            result_files = glob.glob('coco_eval_test_*.json')
            if result_files:
                latest_file = max(result_files, key=os.path.getctime)
                with open(latest_file, 'r') as f:
                    results = json.load(f)
                
                # Store metrics in database
                metrics = results.get('metrics', {})
                ModelMetrics.add_metric('coco_eval', 'map', metrics.get('map', 0.0))
                ModelMetrics.add_metric('coco_eval', 'ap50', metrics.get('ap50', 0.0))
                ModelMetrics.add_metric('coco_eval', 'ap75', metrics.get('ap75', 0.0))
                
                return jsonify({
                    'success': True,
                    'message': 'COCO evaluation completed successfully',
                    'results_file': latest_file,
                    'metrics': metrics,
                    'output': result.stdout
                })
            else:
                return jsonify({'error': 'COCO evaluation completed but no results file found'}), 500
        else:
            return jsonify({
                'error': 'COCO evaluation failed',
                'stderr': result.stderr,
                'stdout': result.stdout
            }), 500
            
    except Exception as e:
        return jsonify({'error': f'Error running COCO evaluation: {str(e)}'}), 500

@app.route('/admin/training/evaluate_checkpoint', methods=['POST'])
@login_required
def admin_evaluate_checkpoint():
    """Evaluate checkpoint model on COCO dataset."""
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    try:
        from coco_evaluator import COCOEvaluator
        
        # Initialize evaluator
        evaluator = COCOEvaluator()
        
        if evaluator.model is None:
            flash('Failed to load checkpoint model.', 'error')
            return redirect(url_for('admin_model_metrics'))
        
        # Evaluate on test split
        test_results = evaluator.evaluate_split('test')
        
        if test_results is None:
            flash('Failed to evaluate test split.', 'error')
            return redirect(url_for('admin_model_metrics'))
        
        # Calculate metrics
        test_metrics = evaluator.calculate_metrics(test_results)
        
        if test_metrics is None:
            flash('Failed to calculate metrics.', 'error')
            return redirect(url_for('admin_model_metrics'))
        
        # Store metrics in database
        timestamp = datetime.now()
        
        # Store checkpoint evaluation metrics
        ModelMetrics.add_metric('checkpoint_model', 'coco_accuracy', float(test_metrics['accuracy']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_precision_weighted', float(test_metrics['precision_weighted']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_recall_weighted', float(test_metrics['recall_weighted']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_f1_weighted', float(test_metrics['f1_weighted']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_precision_macro', float(test_metrics['precision_macro']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_recall_macro', float(test_metrics['recall_macro']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_f1_macro', float(test_metrics['f1_macro']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_avg_confidence', float(test_metrics['avg_confidence']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_avg_processing_time', float(test_metrics['avg_processing_time']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_avg_quality', float(test_metrics['avg_quality']))
        ModelMetrics.add_metric('checkpoint_model', 'coco_num_samples', float(test_metrics['num_samples']))
        
        # Save detailed results
        results_file = evaluator.save_results(test_results, test_metrics, 'test')
        
        flash(f'Checkpoint evaluation completed successfully! Results saved to {results_file}', 'success')
        return redirect(url_for('admin_model_metrics'))
        
    except Exception as e:
        flash(f'Error during checkpoint evaluation: {str(e)}', 'error')
        return redirect(url_for('admin_model_metrics'))

@app.route('/admin/training/evaluate', methods=['POST'])
@login_required
def admin_evaluate_models():
    """Evaluate current model with comprehensive metrics including F1, MAP, and accuracy."""
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))

    try:
        # Build a comprehensive classification dataset from items with detected objects
        items = Item.query.all()
        y_true, y_pred = [], []
        labels_set = set()
        
        for item in items:
            try:
                det = json.loads(item.detected_objects) if item.detected_objects else []
            except Exception:
                det = []
            
            if not det:
                continue
                
            # Use the highest confidence detected object as prediction
            best_obj = max(det, key=lambda x: x.get('confidence', 0))
            predicted_class = best_obj.get('class', 'unknown')
            
            # Use category as true label (or detected class if no category)
            true_label = item.category or predicted_class
            
            y_true.append(true_label)
            y_pred.append(predicted_class)
            labels_set.update([true_label, predicted_class])
            
        if not y_true:
            flash('No items with detected objects found for evaluation.', 'warning')
            return redirect(url_for('admin_model_metrics'))

        # Compute comprehensive classification metrics
        classification_metrics = _compute_classification_metrics(y_true, y_pred, sorted(labels_set))

        # Compute MAP for retrieval evaluation
        ground_truth = defaultdict(set)
        predictions = {}
        for item in items:
            # Relevant items: same category, opposite status
            relevant = set(
                i.id for i in items
                if i.id != item.id and i.category == item.category and i.status != item.status
            )
            ground_truth[item.id] = relevant

            # Use our matching algorithm to generate predicted ranking
            matches = find_potential_matches(item)
            ranked = [m['item'].id for m in matches]
            predictions[item.id] = ranked

        map_metrics = _compute_map(ground_truth, predictions)

        # Store all metrics in database
        timestamp = datetime.now()
        
        # Overall metrics
        ModelMetrics.add_metric('unified_model', 'accuracy', float(classification_metrics['accuracy']))
        ModelMetrics.add_metric('unified_model', 'macro_f1', float(classification_metrics['macro_f1']))
        ModelMetrics.add_metric('unified_model', 'micro_f1', float(classification_metrics['micro_f1']))
        ModelMetrics.add_metric('unified_model', 'weighted_f1', float(classification_metrics['weighted_f1']))
        ModelMetrics.add_metric('unified_model', 'micro_precision', float(classification_metrics['micro_precision']))
        ModelMetrics.add_metric('unified_model', 'micro_recall', float(classification_metrics['micro_recall']))
        
        # MAP metrics
        ModelMetrics.add_metric('unified_model', 'mean_average_precision', float(map_metrics['map']))
        ModelMetrics.add_metric('unified_model', 'precision_at_5', float(map_metrics['precision_at_5']))
        ModelMetrics.add_metric('unified_model', 'precision_at_10', float(map_metrics['precision_at_10']))
        ModelMetrics.add_metric('unified_model', 'recall_at_5', float(map_metrics['recall_at_5']))
        ModelMetrics.add_metric('unified_model', 'recall_at_10', float(map_metrics['recall_at_10']))
        
        # Per-class metrics
        for label, vals in classification_metrics['per_class'].items():
            ModelMetrics.add_metric('unified_model', 'class_f1', float(vals['f1']), class_name=label)
            ModelMetrics.add_metric('unified_model', 'class_precision', float(vals['precision']), class_name=label)
            ModelMetrics.add_metric('unified_model', 'class_recall', float(vals['recall']), class_name=label)
            ModelMetrics.add_metric('unified_model', 'class_support', float(vals['support']), class_name=label)

        # Store evaluation summary
        evaluation_summary = {
            'timestamp': timestamp.isoformat(),
            'total_samples': classification_metrics['total_samples'],
            'num_classes': classification_metrics['num_classes'],
            'classification_metrics': classification_metrics,
            'map_metrics': map_metrics
        }
        
        # Store as a special metric
        ModelMetrics.add_metric('unified_model', 'evaluation_summary', 1.0)
        
        flash(f'Evaluation complete! Processed {classification_metrics["total_samples"]} samples across {classification_metrics["num_classes"]} classes. Metrics recorded.', 'success')
        
    except Exception as e:
        flash(f'Error during evaluation: {str(e)}', 'error')
        print(f"[EVALUATION] Error: {e}")
    
    return redirect(url_for('admin_model_metrics'))

@app.route('/admin/training/dataset')
@login_required
def admin_training_dataset():
    """View and manage the training dataset."""
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    # Get training dataset statistics
    dataset_stats = get_training_dataset_stats()
    
    # Get recent training samples
    training_samples = []
    try:
        training_files = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.json')]
        training_files.sort(key=lambda x: os.path.getmtime(os.path.join(TRAINING_DATA_DIR, x)), reverse=True)
        
        for file in training_files[:20]:  # Show last 20 samples
            file_path = os.path.join(TRAINING_DATA_DIR, file)
            with open(file_path, 'r') as f:
                sample = json.load(f)
                training_samples.append(sample)
    except Exception as e:
        print(f"Error loading training samples: {e}")
    
    return render_template('admin_training_dataset.html',
                         dataset_stats=dataset_stats,
                         training_samples=training_samples)

@app.route('/admin/training/dataset/add_existing', methods=['POST'])
@login_required
def admin_add_existing_to_training():
    """Add all existing items to training dataset."""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        count = add_all_existing_items_to_training()
        flash(f'Successfully added {count} existing items to training dataset!')
        return redirect(url_for('admin_training_dataset'))
    except Exception as e:
        flash(f'Error adding existing items to training dataset: {str(e)}')
        return redirect(url_for('admin_training_dataset'))

@app.route('/admin/training/dataset/add_uploaded_images', methods=['POST'])
@login_required
def admin_add_uploaded_images_to_training():
    """Add all uploaded images to training dataset and train the model."""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        count, errors = add_all_uploaded_images_to_training()
        if errors > 0:
            flash(f'Successfully added {count} uploaded images to training dataset and trained the model! ({errors} errors occurred)')
        else:
            flash(f'Successfully added {count} uploaded images to training dataset and trained the model!')
        return redirect(url_for('admin_training_dataset'))
    except Exception as e:
        flash(f'Error adding uploaded images to training dataset: {str(e)}')
        return redirect(url_for('admin_training_dataset'))

@app.route('/admin/training/dataset/train_with_uploaded_images', methods=['POST'])
@login_required
def admin_train_with_uploaded_images():
    """Comprehensive training using all uploaded images."""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        # Get all uploaded images
        uploads_dir = app.config['UPLOAD_FOLDER']
        static_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'img')
        
        all_images = []
        
        # Collect all image paths
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    all_images.append(os.path.join(uploads_dir, filename))
        
        if os.path.exists(static_img_dir):
            for filename in os.listdir(static_img_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                    all_images.append(os.path.join(static_img_dir, filename))
        
        if not all_images:
            flash('No uploaded images found for training.', 'warning')
            return redirect(url_for('admin_training_dataset'))
        
        # Process all images for training
        training_samples = []
        processed_count = 0
        error_count = 0
        
        for image_path in all_images:
            try:
                # Detect objects in the image
                detected_objects = object_detector.detect_objects(image_path)
                
                if detected_objects:
                    # Create training sample
                    sample = {
                        'image_path': image_path,
                        'detected_objects': detected_objects,
                        'source': 'uploaded_image'
                    }
                    training_samples.append(sample)
                    processed_count += 1
                    
                    # Log metrics
                    log_training_metrics(f"training_{os.path.basename(image_path)}", detected_objects, 'uploaded_training')
                    
            except Exception as e:
                error_count += 1
                print(f"[TRAINING] Error processing {image_path}: {e}")
        
        # Train the model with all samples
        if training_samples:
            success = train_model_with_samples(training_samples)
            if success:
                flash(f'Successfully trained model with {processed_count} uploaded images! Model has been updated with new data.', 'success')
            else:
                flash(f'Added {processed_count} images to dataset but model training failed. Please retry.', 'warning')
        else:
            flash('No valid training samples found in uploaded images.', 'warning')
        
        if error_count > 0:
            flash(f'Note: {error_count} images had processing errors.', 'info')
        
        return redirect(url_for('admin_training_dataset'))
        
    except Exception as e:
        flash(f'Error during comprehensive training: {str(e)}', 'error')
        return redirect(url_for('admin_training_dataset'))

@app.route('/admin/training/dataset/export', methods=['POST'])
@login_required
def admin_export_training_dataset():
    """Export training dataset."""
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    try:
        import zipfile
        from datetime import datetime
        
        # Create export filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_filename = f'training_dataset_export_{timestamp}.zip'
        export_path = os.path.join(TRAINING_DATA_DIR, export_filename)
        
        # Create zip file
        with zipfile.ZipFile(export_path, 'w') as zipf:
            # Add training data files
            for file in os.listdir(TRAINING_DATA_DIR):
                if file.endswith('.json'):
                    file_path = os.path.join(TRAINING_DATA_DIR, file)
                    zipf.write(file_path, f'training_data/{file}')
            
            # Add images
            if os.path.exists(TRAINING_IMAGES_DIR):
                for root, dirs, files in os.walk(TRAINING_IMAGES_DIR):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, TRAINING_DATA_DIR)
                        zipf.write(file_path, f'images/{arcname}')
        
        # Return the file for download
        return send_file(export_path, as_attachment=True, download_name=export_filename)
        
    except Exception as e:
        flash(f'Error exporting training dataset: {str(e)}')
        return redirect(url_for('admin_training_dataset'))

# Initialize Flask-Migrate
migrate = Migrate(app, db)

def init_db():
    with app.app_context():
        # Create all tables
        db.create_all()
        
        # Create default settings if they don't exist
        if not Settings.query.first():
            default_settings = Settings()
            db.session.add(default_settings)
            db.session.commit()
        
        # Create default admin if it doesn't exist
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@example.com',
                is_admin=True,
                is_active=True
            )
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Default admin account created!")
        else:
            print("Default admin account already exists!")

def run_startup_coco_evaluation():
    """Run COCO dataset evaluation on application startup."""
    try:
        print("🔄 Running COCO dataset evaluation on startup...")
        
        # Import the COCO evaluator
        from coco_evaluator import COCOEvaluator
        
        # Initialize evaluator
        evaluator = COCOEvaluator()
        
        # Run evaluation on test split
        print("📊 Evaluating checkpoint model against COCO dataset...")
        results = evaluator.evaluate_split('test')
        
        if results:
            # Calculate metrics
            metrics = evaluator.calculate_metrics(results)
            
            # Save results to database
            with app.app_context():
                # Clear existing checkpoint metrics
                ModelMetrics.query.filter_by(model_type='checkpoint_model').delete()
                
                # Add new metrics
                ModelMetrics.add_metric('checkpoint_model', 'coco_accuracy', metrics.get('accuracy', 0.0))
                ModelMetrics.add_metric('checkpoint_model', 'coco_f1_weighted', metrics.get('f1_weighted', 0.0))
                ModelMetrics.add_metric('checkpoint_model', 'coco_f1_macro', metrics.get('f1_macro', 0.0))
                ModelMetrics.add_metric('checkpoint_model', 'coco_precision', metrics.get('precision', 0.0))
                ModelMetrics.add_metric('checkpoint_model', 'coco_recall', metrics.get('recall', 0.0))
                ModelMetrics.add_metric('checkpoint_model', 'coco_avg_confidence', metrics.get('avg_confidence', 0.0))
                ModelMetrics.add_metric('checkpoint_model', 'coco_avg_processing_time', metrics.get('avg_processing_time', 0.0))
                ModelMetrics.add_metric('checkpoint_model', 'coco_num_samples', metrics.get('samples_evaluated', 0))
                
                db.session.commit()
                print(f"✅ COCO evaluation completed successfully!")
                print(f"   - Accuracy: {metrics.get('accuracy', 0.0):.3f}")
                print(f"   - F1 Score: {metrics.get('f1_score', 0.0):.3f}")
                print(f"   - Samples: {metrics.get('samples_evaluated', 0)}")
                print(f"   - Avg Confidence: {metrics.get('avg_confidence', 0.0):.3f}")
        else:
            print("⚠️  COCO evaluation completed but no results generated")
            
    except Exception as e:
        print(f"❌ Error during startup COCO evaluation: {e}")
        print("   Application will continue without evaluation results")

# Run the app
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✓ Database initialized")
    
    # Skip heavy startup evaluation on Render to open port quickly
    if not os.environ.get('RENDER', '').lower() in ['true', '1']:
        run_startup_coco_evaluation()
    
    print("✓ Starting Flask application...")
    print("✓ Admin login: admin / admin123")
    port = int(os.environ.get('PORT', 4000))
    print(f"✓ Access at: http://localhost:{port}")
    app.run(host='0.0.0.0', port=port)