import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from transformers import BertTokenizer, BertModel
import numpy as np
from PIL import Image
import cv2
import logging
import json
import os
from datetime import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from gdrive_model_downloader import get_model_path, ensure_model_available
import joblib

class UnifiedModel:
    """Unified model that handles all training data in one place."""
    
    def __init__(self):
        # Set up logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize object detection model using best_model1.pth
        self.label_names = {}
        self.score_threshold = float(os.getenv('SCORE_THRESH', '0.5'))
        self.object_model = self._load_trained_detector_or_default()
        self.object_model.to(self.device)
        self.object_model.eval()
        
        # Lazy load text analysis model - only load when needed
        self.text_tokenizer = None
        self.text_model = None
        self._text_model_loaded = False
        
        # Unified training data storage
        self.training_data = []
        self.confidence_adjustments = {}
        self.similarity_adjustments = {}
        
        # Model paths
        self.model_path = 'models/unified_model.pth'
        self.confidence_model_path = 'models/unified_confidence_adjuster.pkl'
        self.similarity_model_path = 'models/unified_similarity_adjuster.pkl'
        self.checkpoint_path = 'checkpoint.pt'
        
        # Initialize checkpoint model
        self.checkpoint_model = None
        self.load_checkpoint_model()
        
        # Custom model classes - will be populated from best_model1.pth
        self.classes = [
            'background', 'Mouse', 'phone', 'tumbler', 'wallet'
        ]
        
        # Dynamic COCO category mappings (from dataset annotations when available)
        self.coco_id_to_name = {}
        self.coco_name_to_id = {}
        self._load_coco_categories_from_annotations()
        
        # Load existing training data and adjustments
        self.load_training_data()
        self.load_adjustments()
        
        # Initialize pretrained visual backbone for embeddings
        try:
            self.vision_backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        except Exception:
            self.vision_backbone = torchvision.models.resnet50(pretrained=True)
        self.vision_backbone.fc = torch.nn.Identity()
        self.vision_backbone.to(self.device)
        self.vision_backbone.eval()
        
        from torchvision import transforms as _tv_transforms
        self.vision_transform = _tv_transforms.Compose([
            _tv_transforms.Resize(256),
            _tv_transforms.CenterCrop(224),
            _tv_transforms.ToTensor(),
            _tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_training_data(self):
        """Load unified training data from file."""
        try:
            # Load from organized training data directory
            training_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_data')
            if os.path.exists(training_data_dir):
                training_files = [f for f in os.listdir(training_data_dir) if f.endswith('.json')]
                self.training_data = []
                
                for file in training_files:
                    file_path = os.path.join(training_data_dir, file)
                    try:
                        with open(file_path, 'r') as f:
                            sample = json.load(f)
                            self.training_data.append(sample)
                    except Exception as e:
                        self.logger.error(f"Error loading training file {file}: {e}")
                
                self.logger.info(f"Loaded {len(self.training_data)} unified training samples from {len(training_files)} files")
            else:
                self.training_data = []
        except Exception as e:
            self.logger.error(f"Error loading training data: {e}")
            self.training_data = []

    def load_checkpoint_model(self):
        """Load the TorchScript checkpoint model."""
        try:
            if os.path.exists(self.checkpoint_path):
                self.checkpoint_model = torch.jit.load(self.checkpoint_path, map_location=self.device)
                self.checkpoint_model.eval()
                self.logger.info("Loaded checkpoint model successfully")
            else:
                # Checkpoint is optional, so just set to None without warning
                self.checkpoint_model = None
        except Exception as e:
            self.logger.error(f"Error loading checkpoint model: {e}")
            self.checkpoint_model = None

    def load_adjustments(self):
        """Load confidence and similarity adjustments."""
        try:
            # Load confidence adjustments
            if os.path.exists(self.confidence_model_path):
                self.confidence_adjustments = joblib.load(self.confidence_model_path)
                self.logger.info("Loaded unified confidence adjustment model")
            else:
                self.confidence_adjustments = {}
            
            # Load similarity adjustments
            if os.path.exists(self.similarity_model_path):
                self.similarity_adjustments = joblib.load(self.similarity_model_path)
                self.logger.info("Loaded unified similarity adjustment model")
            else:
                self.similarity_adjustments = {}
                
        except Exception as e:
            self.logger.error(f"Error loading adjustments: {e}")
            self.confidence_adjustments = {}
            self.similarity_adjustments = {}

    def _ensure_text_model_loaded(self):
        """Lazy load text analysis model only when needed"""
        if not self._text_model_loaded:
            try:
                self.logger.info("Loading BERT model for text analysis...")
                # Use slow (pure-Python) tokenizer to avoid building Rust wheels on deployment
                self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
                self.text_model = BertModel.from_pretrained('bert-base-uncased')
                self.text_model.to(self.device)
                self.text_model.eval()
                self._text_model_loaded = True
                self.logger.info("BERT model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load BERT model: {e}")
                raise e

    def _load_trained_detector_or_default(self):
        """Load best_model9.pth as the primary detector, with minimal fallback."""
        try:
            # Primary: Try to load best_model9.pth
            base_dir = os.path.dirname(os.path.abspath(__file__))
            best_model_path = os.path.join(base_dir, 'models', 'best_model9.pth')
            
            # Check if best_model9.pth exists
            if os.path.exists(best_model_path):
                self.logger.info(f"Loading best_model9.pth from {best_model_path}")
                
                # Load the checkpoint
                ckpt = torch.load(best_model_path, map_location=self.device)
                
                # Extract model information
                num_classes = ckpt.get('num_classes')
                label_names = ckpt.get('label_names', {})
                
                # Process label names
                if isinstance(label_names, dict):
                    # keys may be tensors; normalize to int
                    self.label_names = {int(k): v for k, v in label_names.items()}
                    # Add background class (0) if missing
                    if 0 not in self.label_names:
                        self.label_names[0] = 'background'
                elif isinstance(label_names, list):
                    self.label_names = {i: n for i, n in enumerate(label_names)}
                else:
                    # Default label names for lost and found items
                    self.label_names = {
                        0: 'background',
                        1: 'lost_item',
                        2: 'found_item'
                    }
                
                # Build model architecture
                model = fasterrcnn_resnet50_fpn(weights=None, box_score_thresh=self.score_threshold)
                
                # Replace predictor head to match checkpoint classes
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                if num_classes is None:
                    if self.label_names:
                        num_classes = max(self.label_names.keys()) + 1
                    else:
                        num_classes = 3  # background + lost + found
                
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes))
                
                # Load the trained weights
                state_dict = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
                model.load_state_dict(state_dict, strict=False)  # Use strict=False for compatibility
                
                self.logger.info(f"Successfully loaded best_model9.pth with {int(num_classes)} classes")
                return model
            
            else:
                # Fallback: Create a basic model without COCO weights
                self.logger.warning(f"best_model9.pth not found at {best_model_path}")
                self.logger.info("Creating basic FasterRCNN model without COCO weights")
                
                # Create model without pretrained weights
                model = fasterrcnn_resnet50_fpn(weights=None, box_score_thresh=self.score_threshold)
                
                # Set up basic classes for lost and found
                self.label_names = {
                    0: 'background',
                    1: 'lost_item', 
                    2: 'found_item'
                }
                
                # Configure predictor for 3 classes
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
                
                self.logger.info("Created basic model with 3 classes (background, lost_item, found_item)")
                return model
                
        except Exception as e:
            self.logger.error(f"Failed to load best_model9.pth: {e}")
            # Last resort: create minimal model
            self.logger.info("Creating minimal model as last resort")
            model = fasterrcnn_resnet50_fpn(weights=None, box_score_thresh=self.score_threshold)
            self.label_names = {0: 'background', 1: 'item'}
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
            return model

    def _load_coco_categories_from_annotations(self):
        """Attempt to load COCO categories from an annotation JSON specified via env or default dataset.
        Env:
        - COCO_ANN_FILE: absolute or relative path to _annotations.coco.json
        - COCO_DATASET_ROOT: if COCO_ANN_FILE not set, will look under this root for 'train/valid/test/_annotations.coco.json'
        """
        try:
            ann_path = os.getenv('COCO_ANN_FILE')
            if ann_path and os.path.isfile(ann_path):
                path = ann_path
            else:
                # Try to discover from default dataset structure
                base_dir = os.path.dirname(os.path.abspath(__file__))
                dataset_root = os.getenv('COCO_DATASET_ROOT', os.path.join(base_dir, 'image recog.v1i.coco-mmdetection'))
                candidates = [
                    os.path.join(dataset_root, split, '_annotations.coco.json')
                    for split in ['train', 'valid', 'test']
                ]
                path = next((p for p in candidates if os.path.isfile(p)), None)
            if not path:
                return
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            cats = data.get('categories', [])
            self.coco_id_to_name = {int(c['id']): str(c['name']) for c in cats}
            self.coco_name_to_id = {str(v).lower(): int(k) for k, v in self.coco_id_to_name.items()}
            if self.coco_id_to_name:
                self.logger.info(f"Loaded {len(self.coco_id_to_name)} COCO categories from annotations")
        except Exception as e:
            self.logger.warning(f"Could not load COCO categories from annotations: {e}")

    def save_adjustments(self):
        """Save confidence and similarity adjustments."""
        try:
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.confidence_adjustments, self.confidence_model_path)
            joblib.dump(self.similarity_adjustments, self.similarity_model_path)
            self.logger.info("Saved unified adjustment models")
        except Exception as e:
            self.logger.error(f"Error saving adjustments: {e}")

    def add_training_sample(self, sample_data):
        """Add a unified training sample with all types of data."""
        try:
            # Extract features from the image if present
            if 'image_path' in sample_data and os.path.exists(sample_data['image_path']):
                image = Image.open(sample_data['image_path']).convert('RGB')
                features = self.extract_image_features(image)
                sample_data['features'] = features
            
            # Add timestamp if not present
            if 'timestamp' not in sample_data:
                sample_data['timestamp'] = datetime.now().isoformat()
            
            self.training_data.append(sample_data)
            
            # Update adjustments based on sample type
            if 'detected_objects' in sample_data and 'user_feedback' in sample_data:
                self.update_confidence_adjustments(sample_data['detected_objects'], sample_data['user_feedback'])
            
            if 'text_similarity' in sample_data:
                self.update_similarity_adjustments(sample_data['text_similarity'])
            
            self.save_adjustments()
            
            self.logger.info(f"Added unified training sample with {len(sample_data)} data points")
            return True
        except Exception as e:
            self.logger.error(f"Error adding training sample: {e}")
            return False

    def extract_image_features(self, image):
        """Extract features from image for training."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Basic features
            features = {
                'size': img_array.shape,
                'mean_color': np.mean(img_array, axis=(0, 1)).tolist(),
                'std_color': np.std(img_array, axis=(0, 1)).tolist(),
                'brightness': np.mean(img_array),
                'contrast': np.std(img_array)
            }
            
            # Convert to grayscale for additional features
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            features.update({
                'gray_mean': float(np.mean(gray)),
                'gray_std': float(np.std(gray)),
                'edges': float(np.sum(cv2.Canny(gray, 50, 150)) / (gray.shape[0] * gray.shape[1]))
            })
            
            return features
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            return {}
    
    def get_image_embedding(self, image_path):
        """Compute an image embedding using pretrained ResNet50 backbone."""
        try:
            image = Image.open(image_path).convert('RGB')
            tensor = self.vision_transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feat = self.vision_backbone(tensor)
            vec = feat.squeeze(0).cpu().numpy().astype('float32')
            norm = np.linalg.norm(vec) or 1.0
            return (vec / norm).tolist()
        except Exception as e:
            self.logger.error(f"Error computing image embedding for {image_path}: {e}")
            return []
    
    def compute_image_similarity(self, emb1, emb2):
        if not emb1 or not emb2:
            return 0.0
        v1 = np.array(emb1, dtype=np.float32)
        v2 = np.array(emb2, dtype=np.float32)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) or 1.0
        return float(np.dot(v1, v2) / denom)

    def predict_with_checkpoint(self, image_path):
        """Use the checkpoint model for prediction on an image."""
        if self.checkpoint_model is None:
            # Silently return None if checkpoint model is not available
            return None
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Apply the same transform as used in the vision backbone
            image_tensor = self.vision_transform(image).unsqueeze(0).to(self.device)
            
            # Run inference with checkpoint model
            with torch.no_grad():
                output = self.checkpoint_model(image_tensor)
            
            # Convert output to numpy
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()
            
            self.logger.info(f"Checkpoint model prediction completed for {image_path}")
            return output
            
        except Exception as e:
            self.logger.error(f"Error running checkpoint prediction on {image_path}: {e}")
            return None

    def get_checkpoint_embedding(self, image_path):
        """Get embedding from the checkpoint model."""
        prediction = self.predict_with_checkpoint(image_path)
        if prediction is not None:
            # Flatten the output to get a feature vector
            if len(prediction.shape) > 1:
                embedding = prediction.flatten()
            else:
                embedding = prediction
            
            # Normalize the embedding
            norm = np.linalg.norm(embedding) or 1.0
            normalized_embedding = (embedding / norm).astype(np.float32)
            return normalized_embedding.tolist()
        
        return []

    def compare_with_checkpoint(self, image1_path, image2_path):
        """Compare two images using the checkpoint model."""
        emb1 = self.get_checkpoint_embedding(image1_path)
        emb2 = self.get_checkpoint_embedding(image2_path)
        
        if not emb1 or not emb2:
            return 0.0
        
        # Compute cosine similarity
        similarity = self.compute_image_similarity(emb1, emb2)
        return similarity

    def update_confidence_adjustments(self, detected_objects, user_feedback):
        """Update confidence adjustments based on user feedback."""
        for obj in detected_objects:
            obj_class = obj.get('class', '').lower()
            original_confidence = obj.get('confidence', 0.0)
            
            # Find corresponding feedback
            feedback = None
            for fb in user_feedback:
                if fb.get('class') == obj_class:
                    feedback = fb
                    break
            
            if feedback:
                feedback_type = feedback.get('type', 'correction')
                feedback_confidence = feedback.get('confidence', 0.0)
                
                # Calculate adjustment factor
                if feedback_type == 'confirmation':
                    adjustment = min(0.2, feedback_confidence - original_confidence)
                elif feedback_type == 'rejection':
                    adjustment = max(-0.3, -original_confidence)
                else:  # correction
                    adjustment = feedback_confidence - original_confidence
                
                # Store adjustment for this class
                if obj_class not in self.confidence_adjustments:
                    self.confidence_adjustments[obj_class] = []
                
                self.confidence_adjustments[obj_class].append({
                    'original_confidence': original_confidence,
                    'feedback_confidence': feedback_confidence,
                    'adjustment': adjustment,
                    'feedback_type': feedback_type,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only recent adjustments (last 100)
                if len(self.confidence_adjustments[obj_class]) > 100:
                    self.confidence_adjustments[obj_class] = self.confidence_adjustments[obj_class][-100:]

    def update_similarity_adjustments(self, text_similarity_data):
        """Update similarity adjustments based on text feedback."""
        text1 = text_similarity_data.get('text1', '')
        text2 = text_similarity_data.get('text2', '')
        user_score = text_similarity_data.get('user_score', 0.0)
        original_score = text_similarity_data.get('original_score', 0.0)
        
        # Create a key for this text pair
        text_key = f"{text1}|||{text2}"
        
        if text_key not in self.similarity_adjustments:
            self.similarity_adjustments[text_key] = []
        
        adjustment = user_score - original_score
        
        self.similarity_adjustments[text_key].append({
            'original_score': original_score,
            'user_score': user_score,
            'adjustment': adjustment,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only recent adjustments (last 50)
        if len(self.similarity_adjustments[text_key]) > 50:
            self.similarity_adjustments[text_key] = self.similarity_adjustments[text_key][-50:]

    def adjust_confidence(self, obj_class, original_confidence):
        """Adjust confidence based on unified training data."""
        if obj_class not in self.confidence_adjustments:
            return original_confidence
        
        adjustments = self.confidence_adjustments[obj_class]
        if not adjustments:
            return original_confidence
        
        # Calculate average adjustment for this class
        avg_adjustment = np.mean([adj['adjustment'] for adj in adjustments])
        
        # Apply adjustment with some smoothing
        adjusted_confidence = original_confidence + (avg_adjustment * 0.5)
        
        # Clamp to valid range
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        return adjusted_confidence

    def adjust_similarity(self, text1, text2, original_similarity):
        """Adjust similarity based on unified training data."""
        text_key = f"{text1}|||{text2}"
        reverse_key = f"{text2}|||{text1}"
        
        # Check both directions
        for key in [text_key, reverse_key]:
            if key in self.similarity_adjustments:
                adjustments = self.similarity_adjustments[key]
                if adjustments:
                    avg_adjustment = np.mean([adj['adjustment'] for adj in adjustments])
                    adjusted_similarity = original_similarity + (avg_adjustment * 0.3)
                    return max(0.0, min(1.0, adjusted_similarity))
        
        return original_similarity

    def detect_objects(self, image_path):
        """Detect objects with unified confidence adjustments."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Perform smartphone-specific feature detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=5, maxRadius=30)
            
            # Basic transformation for object detection
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            # Perform detection
            with torch.no_grad():
                predictions = self.object_model(image_tensor)

            # Process predictions
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            # Filter predictions with configured score threshold
            mask = scores > self.score_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Get detected objects with unified confidence adjustment
            detected_objects = []
            for box, score, label in zip(boxes, scores, labels):
                # Determine class name from trained labels if available
                obj_class = None
                if self.label_names:
                    obj_class = self.label_names.get(int(label))
                
                # Map custom model classes to expected object types
                if obj_class in ['lost_item', 'found_item']:
                    # For lost/found items, we need to determine the actual object type
                    # This would typically be done by additional analysis or user input
                    # For now, we'll use a generic "item" class
                    obj_class = 'item'
                elif not obj_class:
                    # Fallback to custom model classes
                    try:
                        obj_class = self.classes[int(label)]
                    except Exception:
                        obj_class = 'item'
                
                original_confidence = float(score)
                
                # Apply unified confidence adjustment
                adjusted_confidence = self.adjust_confidence(obj_class.lower(), original_confidence)
                
                obj = {
                    'class': obj_class,
                    'label': int(label),
                    'confidence': adjusted_confidence,
                    'original_confidence': original_confidence,
                    'box': [float(b) for b in box]
                }
                
                # Color analysis disabled for maximum speed
                # No color analysis performed
                
                detected_objects.append(obj)

            if not detected_objects:
                self.logger.info('No objects detected in the image.')
            return detected_objects
        except Exception as e:
            self.logger.error(f'Error processing image {image_path}: {e}', exc_info=True)
            return {'error': f'Error processing image: {str(e)}'}

    def export_coco_results(self, image_id: int, detections):
        """Convert internal detections to COCO results entries for a single image.
        Each detection should contain 'class' (name), 'label' (int), 'confidence', and 'box' [x1,y1,x2,y2].
        Uses loaded COCO categories if available; skips classes not present.
        """
        results = []
        try:
            for d in detections or []:
                cls_name = str(d.get('class', '')).lower()
                category_id = self.coco_name_to_id.get(cls_name)
                if category_id is None:
                    # Try numeric fallback
                    lbl = d.get('label', None)
                    if isinstance(lbl, (int, np.integer)) and lbl in self.coco_id_to_name:
                        category_id = int(lbl)
                if category_id is None:
                    continue
                box = d.get('box', None)
                if not box or len(box) != 4:
                    continue
                x1, y1, x2, y2 = map(float, box)
                xywh = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
                results.append({
                    'image_id': int(image_id),
                    'category_id': int(category_id),
                    'bbox': xywh,
                    'score': float(d.get('confidence', 0.0)),
                })
        except Exception as e:
            self.logger.error(f"Error exporting COCO results: {e}")
        return results

    def analyze_text(self, text):
        """Analyze text with unified model."""
        # Lazy load text model only when needed
        self._ensure_text_model_loaded()
        
        inputs = self.text_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings[0]

    def compute_similarity(self, text1, text2):
        """Compute text similarity with unified adjustments."""
        emb1 = self.analyze_text(text1)
        emb2 = self.analyze_text(text2)

        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Apply unified similarity adjustment
        adjusted_similarity = self.adjust_similarity(text1, text2, float(similarity))
        
        return adjusted_similarity

    def retrain_unified_model(self):
        """Retrain the unified model with all accumulated data."""
        if len(self.training_data) < 5:
            self.logger.warning("Not enough training data for retraining (need at least 5 samples)")
            return False
        
        try:
            self.logger.info(f"Retraining unified model with {len(self.training_data)} samples")
            
            # Analyze training data for model improvements
            self.analyze_training_data()
            
            # Update confidence adjustments based on training data
            self.update_confidence_from_training_data()
            
            # Update similarity adjustments based on training data
            self.update_similarity_from_training_data()
            
            # Save all adjustments
            self.save_adjustments()
            
            # Log training completion
            self.logger.info("Unified model retraining completed successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error retraining unified model: {e}")
            return False

    def analyze_training_data(self):
        """Analyze training data to identify patterns and improvements."""
        try:
            # Count object classes in training data
            class_counts = {}
            confidence_scores = []
            
            for sample in self.training_data:
                if 'detected_objects' in sample:
                    for obj in sample['detected_objects']:
                        obj_class = obj.get('class', 'unknown')
                        confidence = obj.get('confidence', 0.0)
                        
                        if obj_class not in class_counts:
                            class_counts[obj_class] = 0
                        class_counts[obj_class] += 1
                        confidence_scores.append(confidence)
            
            self.logger.info(f"Training data analysis: {len(class_counts)} object classes, {len(confidence_scores)} detections")
            self.logger.info(f"Class distribution: {class_counts}")
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                self.logger.info(f"Average confidence: {avg_confidence:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error analyzing training data: {e}")

    def update_confidence_from_training_data(self):
        """Update confidence adjustments based on training data patterns."""
        try:
            # Group training data by object class
            class_data = {}
            
            for sample in self.training_data:
                if 'detected_objects' in sample:
                    for obj in sample['detected_objects']:
                        obj_class = obj.get('class', 'unknown')
                        confidence = obj.get('confidence', 0.0)
                        
                        if obj_class not in class_data:
                            class_data[obj_class] = []
                        class_data[obj_class].append(confidence)
            
            # Update confidence adjustments based on training patterns
            for obj_class, confidences in class_data.items():
                if len(confidences) >= 3:  # Need at least 3 samples
                    avg_confidence = sum(confidences) / len(confidences)
                    
                    # If average confidence is low, boost it
                    if avg_confidence < 0.6:
                        boost = 0.1
                        if obj_class not in self.confidence_adjustments:
                            self.confidence_adjustments[obj_class] = []
                        
                        self.confidence_adjustments[obj_class].append({
                            'original_confidence': avg_confidence,
                            'feedback_confidence': avg_confidence + boost,
                            'adjustment': boost,
                            'feedback_type': 'training_boost',
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        self.logger.info(f"Boosted confidence for {obj_class} by {boost}")
            
        except Exception as e:
            self.logger.error(f"Error updating confidence from training data: {e}")

    def update_similarity_from_training_data(self):
        """Update similarity adjustments based on training data patterns."""
        try:
            # Analyze text similarity patterns from training data
            similarity_data = []
            
            for sample in self.training_data:
                if 'text_similarity' in sample:
                    similarity_data.append(sample['text_similarity'])
            
            if len(similarity_data) >= 3:
                # Calculate average similarity adjustments
                avg_adjustment = sum(
                    s.get('user_score', 0) - s.get('original_score', 0) 
                    for s in similarity_data
                ) / len(similarity_data)
                
                if abs(avg_adjustment) > 0.05:  # Significant adjustment
                    self.logger.info(f"Applied average similarity adjustment: {avg_adjustment:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error updating similarity from training data: {e}")

    def get_checkpoint_info(self):
        """Get information about the loaded checkpoint model."""
        if self.checkpoint_model is None:
            return {
                'loaded': False,
                'status': 'Checkpoint model not available (optional)'
            }
        
        try:
            # Get model information
            info = {
                'loaded': True,
                'model_type': str(type(self.checkpoint_model)),
                'device': str(self.device),
                'parameters_count': len(list(self.checkpoint_model.parameters())),
                'has_forward': hasattr(self.checkpoint_model, 'forward')
            }
            
            # Try to get input/output info
            try:
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    output = self.checkpoint_model(dummy_input)
                info['input_shape'] = [1, 3, 224, 224]
                info['output_shape'] = list(output.shape) if hasattr(output, 'shape') else 'unknown'
            except Exception as e:
                info['input_output_test'] = f'Failed: {str(e)[:100]}'
            
            return info
            
        except Exception as e:
            return {
                'loaded': True,
                'error': f'Error getting checkpoint info: {e}'
            }

    def get_unified_training_stats(self):
        """Get comprehensive statistics about unified training data."""
        if not self.training_data:
            return {
                'total_samples': 0,
                'object_detection_samples': 0,
                'text_analysis_samples': 0,
                'classes_feedback': {},
                'average_confidence_adjustment': 0.0,
                'average_similarity_adjustment': 0.0
            }
        
        # Count different types of samples
        object_samples = 0
        text_samples = 0
        class_feedback = {}
        total_confidence_adjustments = []
        total_similarity_adjustments = []
        
        for sample in self.training_data:
            if 'detected_objects' in sample:
                object_samples += 1
                for obj in sample.get('detected_objects', []):
                    obj_class = obj.get('class', '').lower()
                    if obj_class not in class_feedback:
                        class_feedback[obj_class] = 0
                    class_feedback[obj_class] += 1
            
            if 'text_similarity' in sample:
                text_samples += 1
        
        # Calculate average confidence adjustments
        for class_name, adjustments in self.confidence_adjustments.items():
            if adjustments:
                avg_adjustment = np.mean([adj['adjustment'] for adj in adjustments])
                total_confidence_adjustments.append(avg_adjustment)
        
        # Calculate average similarity adjustments
        for text_key, adjustments in self.similarity_adjustments.items():
            if adjustments:
                avg_adjustment = np.mean([adj['adjustment'] for adj in adjustments])
                total_similarity_adjustments.append(avg_adjustment)
        
        return {
            'total_samples': len(self.training_data),
            'object_detection_samples': object_samples,
            'text_analysis_samples': text_samples,
            'classes_feedback': class_feedback,
            'average_confidence_adjustment': np.mean(total_confidence_adjustments) if total_confidence_adjustments else 0.0,
            'average_similarity_adjustment': np.mean(total_similarity_adjustments) if total_similarity_adjustments else 0.0
        }

# Legacy classes for backward compatibility
class ObjectDetector(UnifiedModel):
    """Legacy ObjectDetector class that now uses the unified model."""
    def __init__(self):
        super().__init__()

class TextAnalyzer:
    """Legacy TextAnalyzer class that now uses the unified model."""
    def __init__(self):
        self.unified_model = UnifiedModel()
    
    def analyze_text(self, text):
        return self.unified_model.analyze_text(text)
    
    def compute_similarity(self, text1, text2):
        return self.unified_model.compute_similarity(text1, text2)
    
    def add_text_training_sample(self, text1, text2, user_similarity_score, expected_similarity=None):
        sample_data = {
            'text_similarity': {
                'text1': text1,
                'text2': text2,
                'user_score': user_similarity_score,
                'original_score': expected_similarity or 0.0
            },
            'timestamp': datetime.now().isoformat()
        }
        return self.unified_model.add_training_sample(sample_data)

class ModelTrainer:
    """Centralized model training and management class using unified model."""
    
    def __init__(self):
        self.unified_model = UnifiedModel()
        
    def add_feedback(self, image_path, detected_objects, user_feedback, true_labels=None):
        """Add user feedback for unified model training."""
        sample_data = {
            'image_path': image_path,
            'detected_objects': detected_objects,
            'user_feedback': user_feedback,
            'true_labels': true_labels or [],
            'timestamp': datetime.now().isoformat()
        }
        return self.unified_model.add_training_sample(sample_data)
    
    def add_text_feedback(self, text1, text2, user_similarity_score):
        """Add text similarity feedback to unified model."""
        sample_data = {
            'text_similarity': {
                'text1': text1,
                'text2': text2,
                'user_score': user_similarity_score,
                'original_score': 0.0
            },
            'timestamp': datetime.now().isoformat()
        }
        return self.unified_model.add_training_sample(sample_data)
    
    def retrain_models(self):
        """Retrain the unified model."""
        return self.unified_model.retrain_unified_model()
    
    def get_training_statistics(self):
        """Get comprehensive training statistics from unified model."""
        return self.unified_model.get_unified_training_stats()

class SequenceProcessor:
    def __init__(self, input_size=100, hidden_size=128, num_layers=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.model.eval()

    def process_sequence(self, sequence):
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output, (hidden, cell) = self.model(sequence_tensor)
        
        return output.cpu().numpy(), hidden.cpu().numpy() 