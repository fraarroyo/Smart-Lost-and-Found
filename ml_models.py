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
import joblib

class UnifiedModel:
    """Unified model that handles all training data in one place."""
    
    def __init__(self):
        # Set up logging first
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize object detection model (prefer trained checkpoint if available)
        self.label_names = {}
        self.score_threshold = float(os.getenv('SCORE_THRESH', '0.5'))
        self.object_model = self._load_trained_detector_or_default()
        self.object_model.to(self.device)
        self.object_model.eval()
        
        # Initialize text analysis model
        # Use slow (pure-Python) tokenizer to avoid building Rust wheels on deployment
        self.text_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast=False)
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_model.to(self.device)
        self.text_model.eval()
        
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
        
        # Enhanced class names with specific smartphone features (COCO fallback)
        self.classes = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
            'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
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
                self.logger.warning(f"Checkpoint file not found at {self.checkpoint_path}")
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

    def _load_trained_detector_or_default(self):
        """Load a FasterRCNN detector from a checkpoint with label names if available."""
        try:
            # Resolve checkpoint path from env or default under this project
            base_dir = os.path.dirname(os.path.abspath(__file__))
            default_ckpt = os.path.join(base_dir, 'outputs', 'best_model.pth')
            ckpt_path = os.getenv('BEST_MODEL', default_ckpt)

            # Build model with torchvision weights and configured score threshold
            weights = None
            try:
                weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            except Exception:
                weights = None
            model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=self.score_threshold)

            if ckpt_path and os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                # Expecting a dict with keys 'model', 'num_classes', 'label_names'
                num_classes = ckpt.get('num_classes')
                label_names = ckpt.get('label_names', {})
                if isinstance(label_names, dict):
                    # keys may be tensors; normalize to int
                    self.label_names = {int(k): v for k, v in label_names.items()}
                elif isinstance(label_names, list):
                    self.label_names = {i: n for i, n in enumerate(label_names)}
                else:
                    self.label_names = {}

                # Replace predictor head to match checkpoint classes
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                if num_classes is None:
                    if self.label_names:
                        num_classes = max(self.label_names.keys()) + 1
                    else:
                        raise ValueError('num_classes missing in checkpoint and cannot be inferred')
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, int(num_classes))

                state_dict = ckpt['model'] if 'model' in ckpt else ckpt
                model.load_state_dict(state_dict, strict=True)
                self.logger.info(f"Loaded trained detector from {ckpt_path} with {int(num_classes)} classes")
                return model

            # Fallback: COCO-pretrained model
            self.logger.warning(f"BEST_MODEL checkpoint not found at {ckpt_path}; using COCO-pretrained model")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load trained detector, using default COCO model: {e}")
            try:
                return fasterrcnn_resnet50_fpn(pretrained=True)
            except Exception:
                return fasterrcnn_resnet50_fpn()

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
            self.logger.warning("Checkpoint model not loaded")
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
                # Determine class name from trained labels if available, else COCO fallback
                obj_class = None
                if self.label_names:
                    obj_class = self.label_names.get(int(label))
                if not obj_class:
                    try:
                        obj_class = self.classes[int(label)]
                    except Exception:
                        obj_class = str(int(label))
                original_confidence = float(score)
                
                # Apply unified confidence adjustment
                adjusted_confidence = self.adjust_confidence(obj_class.lower(), original_confidence)
                
                obj = {
                    'class': obj_class,
                    'confidence': adjusted_confidence,
                    'original_confidence': original_confidence,
                    'box': [float(b) for b in box]
                }
                
                # If it's a cell phone, analyze specific features
                if str(obj_class).lower() in ['cell phone', 'cellphone', 'smartphone', 'phone']:
                    x1, y1, x2, y2 = map(int, box)
                    h, w = cv_image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        roi = cv_image[y1:y2, x1:x2]
                        if roi.size > 0:
                            avg_color = np.mean(roi, axis=(0,1))
                            obj['color'] = {
                                'B': int(avg_color[0]),
                                'G': int(avg_color[1]),
                                'R': int(avg_color[2])
                            }
                            
                            if circles is not None:
                                obj['features'] = ['dual_camera_detected']
                            
                            blue_mask = cv2.inRange(roi, 
                                                  np.array([100, 0, 0]), 
                                                  np.array([255, 100, 100]))
                            if np.sum(blue_mask) > 100:
                                if 'features' not in obj:
                                    obj['features'] = []
                                obj['features'].append('fingerprint_sensor_detected')
                
                detected_objects.append(obj)

            if not detected_objects:
                self.logger.info('No objects detected in the image.')
            return detected_objects
        except Exception as e:
            self.logger.error(f'Error processing image {image_path}: {e}', exc_info=True)
            return {'error': f'Error processing image: {str(e)}'}

    def analyze_text(self, text):
        """Analyze text with unified model."""
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
                'error': 'Checkpoint model not loaded'
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