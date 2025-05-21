import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from transformers import BertTokenizer, BertModel
import numpy as np
from PIL import Image
import cv2
import logging

class ObjectDetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Enhanced class names with specific smartphone features
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
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def detect_objects(self, image_path):
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            # Convert PIL image to OpenCV format for additional processing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Perform smartphone-specific feature detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect camera module using circle detection
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                     param1=50, param2=30, minRadius=5, maxRadius=30)
            
            # Basic transformation for object detection
            transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            # Perform detection
            with torch.no_grad():
                predictions = self.model(image_tensor)

            # Process predictions
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()

            # Filter predictions with confidence > 0.5
            mask = scores > 0.5
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]

            # Get detected objects with enhanced features
            detected_objects = []
            for box, score, label in zip(boxes, scores, labels):
                obj = {
                    'class': self.classes[label],
                    'confidence': float(score),
                    'box': [float(b) for b in box]
                }
                
                # If it's a cell phone, analyze specific features
                if self.classes[label] == 'cell phone':
                    # Get the region of interest
                    x1, y1, x2, y2 = map(int, box)
                    # Ensure coordinates are within image bounds
                    h, w = cv_image.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if x2 > x1 and y2 > y1:
                        roi = cv_image[y1:y2, x1:x2]
                        if roi.size > 0:  # Check if ROI is valid
                            # Detect color
                            avg_color = np.mean(roi, axis=(0,1))
                            obj['color'] = {
                                'B': int(avg_color[0]),
                                'G': int(avg_color[1]),
                                'R': int(avg_color[2])
                            }
                            
                            # Check for camera modules
                            if circles is not None:
                                obj['features'] = ['dual_camera_detected']
                            
                            # Check for fingerprint sensor (blue-ish region)
                            blue_mask = cv2.inRange(roi, 
                                                  np.array([100, 0, 0]), 
                                                  np.array([255, 100, 100]))
                            if np.sum(blue_mask) > 100:
                                if 'features' not in obj:
                                    obj['features'] = []
                                obj['features'].append('fingerprint_sensor_detected')
                        else:
                            self.logger.warning(f'ROI size is zero for box: {box}')
                    else:
                        self.logger.warning(f'Invalid ROI coordinates: {x1},{y1},{x2},{y2} for image size {w}x{h}')
                
                detected_objects.append(obj)

            if not detected_objects:
                self.logger.info('No objects detected in the image.')
            return detected_objects
        except Exception as e:
            self.logger.error(f'Error processing image {image_path}: {e}', exc_info=True)
            return {'error': f'Error processing image: {str(e)}'}

class TextAnalyzer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def analyze_text(self, text):
        # Tokenize and prepare input
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        return embeddings[0]

    def compute_similarity(self, text1, text2):
        # Get embeddings for both texts
        emb1 = self.analyze_text(text1)
        emb2 = self.analyze_text(text2)

        # Compute cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

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
        # Convert sequence to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Process through LSTM
        with torch.no_grad():
            output, (hidden, cell) = self.model(sequence_tensor)
        
        return output.cpu().numpy(), hidden.cpu().numpy() 