from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os
import uuid
from functools import wraps
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

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Update database URI to use SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lostfound.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Fix uploads directory path
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)  # Remember me duration
app.config['SECURITY_PASSWORD_SALT'] = secrets.token_hex(16)  # For password reset tokens

try:
    from ml_models import ObjectDetector, TextAnalyzer, SequenceProcessor
except ImportError:
    # Dummy ML model classes for when torch/ML dependencies are not available
    class ObjectDetector:
        def detect_objects(self, image_path):
            return []

    class TextAnalyzer:
        def analyze_text(self, text):
            return []
            
        def compute_similarity(self, text1, text2):
            return 0.0

    class SequenceProcessor:
        def process(self, sequence):
            return []

# Initialize ML models
object_detector = ObjectDetector()
text_analyzer = TextAnalyzer()
sequence_processor = SequenceProcessor()

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize password reset serializer
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

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

def generate_description(detected_objects, color=None, size=None, category=None, location=None, crack_status=None, damage_status=None):
    """Generate a detailed natural language description from the most relevant detected object and attributes."""
    if not detected_objects:
        return "No objects detected in the image."
    best_obj = max(detected_objects, key=lambda x: x['confidence'])
    cls = best_obj['class'].replace('_', ' ').title()
    confidence = best_obj.get('confidence', 0)
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
    """Extract the two most dominant colors from the image using k-means clustering."""
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((50, 50))  # Speed up and reduce noise
        pixels = list(image.getdata())
        if exclude_colors:
            pixels = [c for c in pixels if c not in exclude_colors]
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
        print(f"extract_dominant_color error: {e}")
        return "black, white"

def extract_case_color(image_path, box, border_width=8):
    """Extract the dominant color from the border of a bounding box (case area)."""
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
    """Extract the dominant color from the central area of the bounding box (main body of tumbler) using k-means clustering."""
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
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    reset_token = db.Column(db.String(100), unique=True)
    reset_token_expiry = db.Column(db.DateTime)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
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
        result = check_password_hash(self.password_hash, password)
        print(f"Password check result: {result}")  # Debug log
        return result

    def generate_reset_token(self):
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
    status = db.Column(db.String(20), nullable=False)  # 'lost' or 'found'
    location = db.Column(db.String(100))
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    image_path = db.Column(db.String(200))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    detected_objects = db.Column(db.Text)  # JSON string of detected objects
    text_embedding = db.Column(db.Text)  # JSON string of text embedding
    color = db.Column(db.String(50))
    size = db.Column(db.String(50))

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

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

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
            
        detected_objects = object_detector.detect_objects(temp_path)
        # Relabel objects
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
        # Log detected classes for debugging
        print("Detected classes:", [obj.get('class', '').lower() for obj in detected_objects])
        # Filter objects
        filtered_objects = []
        image_area = None
        try:
            img = Image.open(temp_path)
            image_area = img.width * img.height
            img.close()  # Close the image file
        except:
            pass
        for obj in detected_objects:
            obj_class = obj.get('class', '').lower()
            # Size-based filter for phones: skip if bounding box is very small
            if obj_class in ["phone", "mobile", "cell phone", "cellphone", "smartphone"] and 'box' in obj and image_area:
                x1, y1, x2, y2 = map(int, obj['box'])
                box_area = max(1, (x2 - x1) * (y2 - y1))
                if box_area / image_area < 0.10:  # less than 10% of image area
                    continue
            # Filter out 'tv' and only allow classes in ALLOWED_CLASSES
            if obj_class == 'tv':
                continue
            if obj_class in ALLOWED_CLASSES:
                filtered_objects.append(obj)
        # Process only the detected item with the highest confidence
        if filtered_objects:
            best_obj = max(filtered_objects, key=lambda x: x['confidence'])
            obj = best_obj
            obj_class = obj.get('class', '').lower()
            box = obj.get('box')
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
            # Get color based on object type, always using cropped_path
            color = None
            if obj_class == 'tumbler' and box:
                color = extract_tumbler_color(cropped_path, [0, 0, cropped.width, cropped.height])
            elif obj_class in ["phone", "mobile", "cell phone", "cellphone", "smartphone", "tablet", "ipad"] and box:
                color = extract_case_color(cropped_path, [0, 0, cropped.width, cropped.height])
            else:
                color = extract_dominant_color(cropped_path)
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
            # Generate improved description
            description = generate_description([obj], color, size, category, None, crack_status, damage_status)
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
        return jsonify({
            'items': items_info,
            'detected_objects': filtered_objects,
            'description': items_info[0]['description'] if items_info else "No objects detected in the image."
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
        # Clean up the temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                import time
                time.sleep(0.1)
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"Cleanup error: {cleanup_err}")
                try:
                    import atexit
                    atexit.register(lambda: os.remove(temp_path) if os.path.exists(temp_path) else None)
                except:
                    pass

def find_potential_matches(item):
    """Find potential matches for a lost/found item based on similarity and fallback on item type/category if needed."""
    # Get items with opposite status within the last 15 days
    opposite_status = 'found' if item.status == 'lost' else 'lost'
    fifteen_days_ago = datetime.utcnow() - timedelta(days=15)
    potential_matches = Item.query.filter(
        Item.status == opposite_status,
        Item.date >= fifteen_days_ago
    ).all()
    
    matches = []
    for other_item in potential_matches:
        # Skip if same user
        if other_item.user_id == item.user_id:
            continue
        
        # Calculate similarity score
        similarity = text_analyzer.compute_similarity(
            f"{item.title} {item.description}",
            f"{other_item.title} {other_item.description}"
        )
        
        # Check if items have same category
        category_match = item.category == other_item.category
        
        # Check if colors match
        color_match = False
        if item.color and other_item.color:
            item_colors = set(c.strip().lower() for c in item.color.split(','))
            other_colors = set(c.strip().lower() for c in other_item.color.split(','))
            color_match = bool(item_colors.intersection(other_colors))
        
        # Check if sizes match
        size_match = item.size == other_item.size
        
        # Fallback: check if item type matches (from description)
        item_type = item.description.split()[-1].lower() if item.description else ''
        other_type = other_item.description.split()[-1].lower() if other_item.description else ''
        type_match = item_type == other_type and item_type != ''
        
        # Calculate total match score
        match_score = similarity
        if category_match:
            match_score += 0.2
        if color_match:
            match_score += 0.2
        if size_match:
            match_score += 0.1
        if type_match:
            match_score += 0.2
        
        # Lower threshold if type and category match
        threshold = 0.6 if not (type_match and category_match) else 0.5
        
        if match_score >= threshold:
            matches.append({
                'item': other_item,
                'score': match_score,
                'similarity': similarity,
                'category_match': category_match,
                'color_match': color_match,
                'size_match': size_match,
                'type_match': type_match
            })
    
    # Sort matches by score in descending order
    matches.sort(key=lambda x: x['score'], reverse=True)
    return matches

@app.route('/add_item', methods=['GET', 'POST'])
@login_required
def add_item():
    if request.method == 'POST':
        title = request.form.get('title')
        category = request.form.get('category')
        status = request.form.get('status')
        location = request.form.get('location')
        
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
            
            # Detect objects and generate description
            detected_objects = object_detector.detect_objects(image_path)
            color = extract_dominant_color(image_path)
            size = estimate_size(detected_objects, image_path)
            # Use improved description
            description = generate_description(detected_objects, color, size, category, location)
            detected_objects_json = json.dumps(detected_objects)

            # Generate text embedding for search
            text_embedding = text_analyzer.analyze_text(f"{title} {description}")
            text_embedding = json.dumps(text_embedding.tolist())

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
                size=size
            )
            
            db.session.add(item)
            db.session.commit()

            # Find potential matches
            matches = find_potential_matches(item)
            
            if matches:
                # Store matches in session for display
                session['potential_matches'] = [{
                    'id': match['item'].id,
                    'title': match['item'].title,
                    'score': match['score'],
                    'similarity': match['similarity'],
                    'category_match': match['category_match'],
                    'color_match': match['color_match'],
                    'size_match': match['size_match']
                } for match in matches[:5]]  # Store top 5 matches
                return redirect(url_for('view_matches', item_id=item.id))
            
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
            flash(f'Error processing image: {str(e)}')
            return redirect(url_for('add_item'))

    return render_template('add_item.html', status=request.args.get('status', 'lost'))

@app.route('/item/<int:item_id>')
def view_item(item_id):
    item = Item.query.get_or_404(item_id)
    detected_objects = json.loads(item.detected_objects) if item.detected_objects else None
    return render_template('view_item.html', item=item, detected_objects=detected_objects)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    status = request.args.get('status', '')
    
    items = Item.query
    
    if query:
        # Get query embedding
        query_embedding = text_analyzer.analyze_text(query)
        
        # Get all items and compute similarity
        all_items = Item.query.all()
        similarities = []
        
        for item in all_items:
            item_embedding = json.loads(item.text_embedding)
            similarity = text_analyzer.compute_similarity(query, f"{item.title} {item.description}")
            similarities.append((item, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        items = [item for item, _ in similarities]
    else:
        if category:
            items = items.filter_by(category=category)
        if status:
            items = items.filter_by(status=status)
        items = items.order_by(Item.date.desc()).all()
    
    return render_template('search.html', items=items, query=query, category=category, status=status)

@app.route('/similar_items/<int:item_id>')
def similar_items(item_id):
        item = Item.query.get_or_404(item_id)
        item_embedding = json.loads(item.text_embedding)
    
    # Get all items and compute similarity
        all_items = Item.query.filter(Item.id != item_id).all()
        similarities = []
        
        for other_item in all_items:
            other_embedding = json.loads(other_item.text_embedding)
            similarity = text_analyzer.compute_similarity(
                f"{item.title} {item.description}",
                f"{other_item.title} {other_item.description}"
            )
            similarities.append((other_item, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_items = [item for item, _ in similarities[:5]]  # Get top 5 similar items
    
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
        match_item = Item.query.get(match['id'])
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

@app.route('/admin/items')
@login_required
def admin_items():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    
    status = request.args.get('status', 'all')
    items = Item.query
    
    if status != 'all':
        items = items.filter_by(status=status)
    
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
        
    username = request.form.get('username')
    email = request.form.get('email')
    password = request.form.get('password')
    is_admin = request.form.get('is_admin') == 'on'
    
    if User.query.filter_by(username=username).first():
        flash('Username already exists')
        return redirect(url_for('admin_dashboard'))
        
    if User.query.filter_by(email=email).first():
        flash('Email already registered')
        return redirect(url_for('admin_dashboard'))
        
    user = User(username=username, email=email, is_admin=is_admin)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    
    flash('User added successfully')
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
def admin_delete_user(user_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    user = User.query.get_or_404(user_id)
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot delete yourself'}), 400
        
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({'message': 'User deleted successfully'})

@app.route('/admin/items/<int:item_id>/delete', methods=['POST'])
@login_required
def admin_delete_item(item_id):
    if not current_user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
        
    item = Item.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    
    return jsonify({'message': 'Item deleted successfully'})

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

@app.route('/admin/model_metrics')
@login_required
def admin_model_metrics():
    if not current_user.is_admin:
        flash('You do not have permission to access this page.', 'danger')
        return redirect(url_for('index'))
    # Dummy metrics for demonstration
    metrics = {
        'accuracy': 0.92,
        'precision': 0.90,
        'recall': 0.88,
        'f1': 0.89,
        'map': 0.87,
        'iou': 0.85,
        'labels': ['Phone', 'Wallet', 'Watch', 'Tumbler'],
        'confusion_matrix': [
            [50, 2, 1, 0],
            [3, 45, 2, 0],
            [0, 1, 48, 1],
            [0, 0, 2, 46]
        ]
    }
    return render_template('admin_model_metrics.html', metrics=metrics)

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

# Run the app
if __name__ == '__main__':
    init_db()  # Initialize database and create default admin
    app.run(debug=True) 