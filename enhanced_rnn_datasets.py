#!/usr/bin/env python3
"""
Enhanced RNN Training Datasets for BARYONYX Lost & Found System
Creates comprehensive, realistic training datasets with real-world scenarios
"""

import json
import csv
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
from collections import Counter
import re

class EnhancedDatasetGenerator:
    """Generator for enhanced, realistic training datasets"""
    
    def __init__(self):
        self.item_types = {
            'phone': {
                'brands': ['iPhone', 'Samsung', 'Google', 'OnePlus', 'Huawei', 'Xiaomi', 'Motorola'],
                'models': ['12', '13', '14', '15', 'Galaxy S23', 'Pixel 7', '11', 'Nord', 'P60', '13 Pro'],
                'colors': ['black', 'white', 'silver', 'gold', 'blue', 'red', 'green', 'purple'],
                'conditions': ['new', 'good', 'fair', 'cracked', 'damaged', 'scratched'],
                'accessories': ['case', 'screen protector', 'charger', 'headphones', 'cable']
            },
            'wallet': {
                'materials': ['leather', 'synthetic', 'canvas', 'fabric', 'metal'],
                'colors': ['brown', 'black', 'blue', 'red', 'gray', 'green', 'tan'],
                'types': ['bifold', 'trifold', 'card holder', 'money clip', 'zipper', 'slim'],
                'contents': ['cards', 'money', 'ID', 'photos', 'receipts', 'keys', 'coins']
            },
            'mouse': {
                'brands': ['Logitech', 'Razer', 'Corsair', 'SteelSeries', 'Microsoft', 'Apple', 'HP', 'Dell'],
                'types': ['wireless', 'gaming', 'optical', 'Bluetooth', 'ergonomic', 'trackball'],
                'colors': ['black', 'white', 'red', 'blue', 'green', 'silver', 'RGB'],
                'features': ['RGB', 'wireless', 'gaming', 'ergonomic', 'precision', 'scroll']
            },
            'tumbler': {
                'materials': ['stainless steel', 'plastic', 'ceramic', 'glass', 'aluminum'],
                'colors': ['silver', 'black', 'white', 'blue', 'red', 'green', 'pink', 'clear'],
                'types': ['water bottle', 'coffee cup', 'mug', 'tumbler', 'thermos', 'sports bottle'],
                'features': ['insulated', 'leak-proof', 'handle', 'straw', 'lid', 'measurement', 'logo']
            }
        }
        
        self.locations = {
            'office': ['desk', 'conference room', 'break room', 'lobby', 'parking lot'],
            'home': ['living room', 'bedroom', 'kitchen', 'bathroom', 'garage', 'garden'],
            'public': ['restaurant', 'cafe', 'library', 'mall', 'park', 'gym', 'theater'],
            'transport': ['bus', 'train', 'taxi', 'car', 'airport', 'station', 'platform']
        }
        
        self.time_patterns = {
            'morning': {'hours': range(6, 12), 'activities': ['commute', 'work', 'breakfast']},
            'afternoon': {'hours': range(12, 18), 'activities': ['lunch', 'work', 'shopping']},
            'evening': {'hours': range(18, 22), 'activities': ['dinner', 'social', 'entertainment']},
            'night': {'hours': list(range(22, 24)) + list(range(0, 6)), 'activities': ['home', 'sleep', 'late work']}
        }
    
    def generate_item_description(self, item_type: str, is_lost: bool = True) -> str:
        """Generate realistic item description"""
        if item_type not in self.item_types:
            return f"{'Lost' if is_lost else 'Found'} {item_type}"
        
        item_info = self.item_types[item_type]
        prefix = "Lost" if is_lost else "Found"
        
        # Select random attributes
        if item_type == 'phone':
            brand = random.choice(item_info['brands'])
            model = random.choice(item_info['models'])
            color = random.choice(item_info['colors'])
            condition = random.choice(item_info['conditions'])
            accessory = random.choice(item_info['accessories'])
            
            description = f"{prefix} {brand} {model} {color}"
            if condition != 'new':
                description += f" with {condition} screen"
            if random.random() < 0.3:
                description += f" and {accessory}"
        
        elif item_type == 'wallet':
            material = random.choice(item_info['materials'])
            color = random.choice(item_info['colors'])
            wallet_type = random.choice(item_info['types'])
            contents = random.choice(item_info['contents'])
            
            description = f"{prefix} {color} {material} {wallet_type}"
            if random.random() < 0.4:
                description += f" with {contents}"
        
        elif item_type == 'mouse':
            brand = random.choice(item_info['brands'])
            mouse_type = random.choice(item_info['types'])
            color = random.choice(item_info['colors'])
            feature = random.choice(item_info['features'])
            
            description = f"{prefix} {brand} {mouse_type} {color}"
            if random.random() < 0.3:
                description += f" with {feature}"
        
        elif item_type == 'tumbler':
            material = random.choice(item_info['materials'])
            color = random.choice(item_info['colors'])
            tumbler_type = random.choice(item_info['types'])
            feature = random.choice(item_info['features'])
            
            description = f"{prefix} {material} {tumbler_type} {color}"
            if random.random() < 0.3:
                description += f" with {feature}"
        
        return description
    
    def generate_user_behavior_sequence(self, user_id: str, num_actions: int = 20) -> List[Dict]:
        """Generate realistic user behavior sequence"""
        actions = []
        current_time = datetime.now() - timedelta(days=random.randint(1, 30))
        
        # User profile characteristics
        user_profile = {
            'activity_level': random.uniform(0.3, 1.0),
            'preferred_times': random.choice(['morning', 'afternoon', 'evening']),
            'preferred_items': random.sample(list(self.item_types.keys()), k=random.randint(1, 3)),
            'search_frequency': random.uniform(0.1, 0.8),
            'upload_frequency': random.uniform(0.05, 0.4)
        }
        
        for i in range(num_actions):
            # Determine action based on user profile and time
            hour = current_time.hour
            time_period = self._get_time_period(hour)
            
            # Action probabilities based on user profile and time
            if time_period == user_profile['preferred_times']:
                action_probs = [0.4, 0.25, 0.25, 0.08, 0.02]  # search, view, upload, browse, logout
            else:
                action_probs = [0.3, 0.2, 0.15, 0.3, 0.05]
            
            # Adjust based on user activity level
            action_probs = [p * user_profile['activity_level'] for p in action_probs]
            action_probs = [p / sum(action_probs) for p in action_probs]  # Normalize
            
            action = np.random.choice(['search', 'view', 'upload', 'browse', 'logout'], p=action_probs)
            
            # Select item type if relevant
            item_type = None
            if action in ['search', 'view', 'upload']:
                if random.random() < 0.7:  # 70% chance to use preferred items
                    item_type = random.choice(user_profile['preferred_items'])
                else:
                    item_type = random.choice(list(self.item_types.keys()))
            
            # Create feature vector
            features = self._create_behavior_features(
                current_time, action, item_type, user_profile, i, actions
            )
            
            # Determine next action (for training label)
            next_action = self._predict_next_action(action, user_profile, time_period)
            
            actions.append({
                'user_id': user_id,
                'timestamp': current_time,
                'action': action,
                'item_type': item_type,
                'features': features,
                'next_action': next_action,
                'time_period': time_period,
                'session_id': f"session_{i // 5}"  # New session every 5 actions
            })
            
            # Advance time
            time_increment = random.randint(5, 180)  # 5 minutes to 3 hours
            current_time += timedelta(minutes=time_increment)
        
        return actions
    
    def _get_time_period(self, hour: int) -> str:
        """Get time period from hour"""
        for period, info in self.time_patterns.items():
            if hour in info['hours']:
                return period
        return 'night'
    
    def _create_behavior_features(self, timestamp: datetime, action: str, item_type: str, 
                                user_profile: Dict, action_index: int, previous_actions: List[Dict]) -> List[float]:
        """Create comprehensive behavior feature vector"""
        features = [0.0] * 15
        
        # Time features
        features[0] = timestamp.hour / 24.0
        features[1] = timestamp.weekday() / 7.0
        features[2] = timestamp.day / 31.0
        features[3] = timestamp.month / 12.0
        
        # Action encoding
        action_encoding = {'search': 1.0, 'view': 2.0, 'upload': 3.0, 'browse': 4.0, 'logout': 5.0}
        features[4] = action_encoding[action] / 5.0
        
        # Item type encoding
        if item_type:
            item_encoding = {item: (i+1) for i, item in enumerate(self.item_types.keys())}
            features[5] = item_encoding[item_type] / len(self.item_types)
        
        # User profile features
        features[6] = user_profile['activity_level']
        features[7] = user_profile['search_frequency']
        features[8] = user_profile['upload_frequency']
        
        # Session features
        features[9] = min(action_index / 20.0, 1.0)  # Session progress
        features[10] = len([a for a in previous_actions if a['action'] == 'search']) / max(action_index, 1)
        features[11] = len([a for a in previous_actions if a['action'] == 'upload']) / max(action_index, 1)
        
        # Time since last action
        if previous_actions:
            last_time = previous_actions[-1]['timestamp']
            time_diff = (timestamp - last_time).total_seconds() / 3600.0  # Hours
            features[12] = min(time_diff / 24.0, 1.0)  # Normalized to 1 day
        else:
            features[12] = 0.0
        
        # Device and location context (simulated)
        features[13] = random.uniform(0.0, 1.0)  # Device type (mobile/desktop)
        features[14] = random.uniform(0.0, 1.0)  # Location type
        
        return features
    
    def _predict_next_action(self, current_action: str, user_profile: Dict, time_period: str) -> str:
        """Predict next likely action based on current action and context"""
        if current_action == 'search':
            return random.choices(['search', 'view', 'upload'], weights=[0.5, 0.4, 0.1])[0]
        elif current_action == 'view':
            return random.choices(['search', 'view', 'upload', 'browse'], weights=[0.3, 0.3, 0.2, 0.2])[0]
        elif current_action == 'upload':
            return random.choices(['view', 'search', 'browse'], weights=[0.5, 0.3, 0.2])[0]
        elif current_action == 'browse':
            return random.choices(['search', 'view', 'upload'], weights=[0.4, 0.4, 0.2])[0]
        else:  # logout
            return 'logout'
    
    def generate_item_matching_dataset(self, num_samples: int = 2000) -> List[Dict]:
        """Generate dataset for item matching with realistic scenarios"""
        dataset = []
        
        for i in range(num_samples):
            # Generate two items (lost and found)
            item_type = random.choice(list(self.item_types.keys()))
            lost_item = self.generate_item_description(item_type, is_lost=True)
            found_item = self.generate_item_description(item_type, is_lost=False)
            
            # Create feature vectors for both items
            lost_features = self._create_item_features(lost_item, item_type, is_lost=True)
            found_features = self._create_item_features(found_item, item_type, is_lost=False)
            
            # Calculate similarity score
            similarity_score = self._calculate_similarity(lost_item, found_item, item_type)
            
            # Determine if it's a match
            is_match = similarity_score > 0.7
            
            dataset.append({
                'lost_item': lost_item,
                'found_item': found_item,
                'lost_features': lost_features,
                'found_features': found_features,
                'similarity_score': similarity_score,
                'is_match': is_match,
                'item_type': item_type,
                'match_confidence': similarity_score if is_match else 1.0 - similarity_score
            })
        
        return dataset
    
    def _create_item_features(self, description: str, item_type: str, is_lost: bool) -> List[float]:
        """Create comprehensive item feature vector"""
        features = [0.0] * 20
        
        # Basic item features
        item_encoding = {item: (i+1) for i, item in enumerate(self.item_types.keys())}
        features[0] = item_encoding[item_type] / len(self.item_types)
        
        # Color detection from description
        colors = ['black', 'white', 'silver', 'gold', 'blue', 'red', 'green', 'purple', 'brown', 'gray']
        color_found = any(color in description.lower() for color in colors)
        features[1] = 1.0 if color_found else 0.0
        
        # Size indicators
        size_indicators = ['small', 'large', 'big', 'tiny', 'compact', 'slim']
        size_found = any(size in description.lower() for size in size_indicators)
        features[2] = 1.0 if size_found else 0.0
        
        # Condition indicators
        condition_indicators = ['new', 'good', 'fair', 'cracked', 'damaged', 'scratched', 'broken']
        condition_found = any(condition in description.lower() for condition in condition_indicators)
        features[3] = 1.0 if condition_found else 0.0
        
        # Location type (simulated)
        location_types = ['office', 'home', 'public', 'transport']
        location_type = random.choice(location_types)
        location_encoding = {loc: (i+1) for i, loc in enumerate(location_types)}
        features[4] = location_encoding[location_type] / len(location_types)
        
        # Location confidence
        features[5] = random.uniform(0.6, 1.0)
        
        # Temporal features
        now = datetime.now()
        features[6] = now.hour / 24.0
        features[7] = now.weekday() / 7.0
        features[8] = now.month / 12.0
        features[9] = ((now.month - 1) // 3) / 4.0  # Season
        
        # User behavior features (simulated)
        features[10] = random.uniform(0.3, 1.0)  # User activity level
        features[11] = random.uniform(0.1, 0.8)  # Search frequency
        features[12] = random.uniform(0.05, 0.4)  # Upload frequency
        
        # Item characteristics
        features[13] = 1.0 if 'image' in description.lower() else 0.0  # Has image
        features[14] = min(len(description.split()) / 20.0, 1.0)  # Description length
        features[15] = random.uniform(0.5, 1.0)  # Description quality
        
        # Matching features
        features[16] = random.uniform(0.0, 1.0)  # Similarity score
        features[17] = random.uniform(0.5, 1.0)  # Match confidence
        features[18] = random.uniform(0.7, 1.0)  # Category consistency
        features[19] = random.uniform(0.6, 1.0)  # Temporal relevance
        
        return features
    
    def _calculate_similarity(self, lost_item: str, found_item: str, item_type: str) -> float:
        """Calculate similarity score between lost and found items"""
        # Basic text similarity
        lost_words = set(lost_item.lower().split())
        found_words = set(found_item.lower().split())
        
        if not lost_words or not found_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(lost_words.intersection(found_words))
        union = len(lost_words.union(found_words))
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # Boost similarity if same item type
        type_boost = 0.3 if item_type in lost_item.lower() and item_type in found_item.lower() else 0.0
        
        # Boost similarity for color matches
        color_boost = 0.2
        for color in ['black', 'white', 'silver', 'gold', 'blue', 'red', 'green', 'purple', 'brown', 'gray']:
            if color in lost_item.lower() and color in found_item.lower():
                color_boost = 0.2
                break
        else:
            color_boost = 0.0
        
        # Final similarity score
        similarity = min(jaccard_sim + type_boost + color_boost, 1.0)
        return similarity
    
    def generate_temporal_dataset(self, num_samples: int = 1500) -> List[Dict]:
        """Generate dataset for temporal pattern recognition"""
        dataset = []
        
        for i in range(num_samples):
            # Generate random time
            hour = random.randint(0, 23)
            day = random.randint(1, 31)
            month = random.randint(1, 12)
            
            # Determine time period
            time_period = self._get_time_period(hour)
            time_period_encoding = {'morning': 0, 'afternoon': 1, 'evening': 2, 'night': 3}
            
            # Select item type and location
            item_type = random.choice(list(self.item_types.keys()))
            location_type = random.choice(list(self.locations.keys()))
            
            # Create temporal features
            features = [0.0] * 10
            features[0] = hour / 24.0
            features[1] = day / 31.0
            features[2] = month / 12.0
            features[3] = ((month - 1) // 3) / 4.0  # Season
            
            # Location encoding
            location_encoding = {loc: (i+1) for i, loc in enumerate(self.locations.keys())}
            features[4] = location_encoding[location_type] / len(self.locations)
            
            # Item frequency (simulated based on item type)
            item_frequency = {'phone': 0.8, 'wallet': 0.7, 'mouse': 0.4, 'tumbler': 0.3}
            features[5] = item_frequency.get(item_type, 0.5)
            
            # Weather factor (simplified)
            features[6] = random.uniform(0.2, 0.8)
            
            # Holiday factor
            holiday_months = [12, 1, 7, 11]  # December, January, July, November
            features[7] = 0.8 if month in holiday_months else 0.2
            
            # Day of week factor
            features[8] = random.randint(0, 6) / 7.0
            
            # Time since last similar event
            features[9] = random.uniform(0.1, 1.0)
            
            dataset.append({
                'features': features,
                'label': time_period_encoding[time_period],
                'time_period': time_period,
                'item_type': item_type,
                'location_type': location_type,
                'hour': hour,
                'month': month
            })
        
        return dataset
    
    def save_datasets(self, output_dir: str = 'enhanced_datasets/'):
        """Save all enhanced datasets"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 Generating Enhanced RNN Training Datasets")
        print("=" * 60)
        
        # Generate user behavior dataset
        print("👤 Generating user behavior dataset...")
        behavior_data = []
        for user_id in range(50):  # 50 users
            user_actions = self.generate_user_behavior_sequence(f"user_{user_id}", 20)
            behavior_data.extend(user_actions)
        
        with open(os.path.join(output_dir, 'enhanced_user_behavior.json'), 'w') as f:
            json.dump(behavior_data, f, indent=2, default=str)
        
        print(f"✅ User behavior dataset: {len(behavior_data)} samples")
        
        # Generate item matching dataset
        print("🔍 Generating item matching dataset...")
        matching_data = self.generate_item_matching_dataset(2000)
        
        with open(os.path.join(output_dir, 'enhanced_item_matching.json'), 'w') as f:
            json.dump(matching_data, f, indent=2, default=str)
        
        print(f"✅ Item matching dataset: {len(matching_data)} samples")
        
        # Generate temporal dataset
        print("⏰ Generating temporal dataset...")
        temporal_data = self.generate_temporal_dataset(1500)
        
        with open(os.path.join(output_dir, 'enhanced_temporal.json'), 'w') as f:
            json.dump(temporal_data, f, indent=2, default=str)
        
        print(f"✅ Temporal dataset: {len(temporal_data)} samples")
        
        # Generate text descriptions dataset
        print("📝 Generating text descriptions dataset...")
        text_data = []
        for item_type in self.item_types.keys():
            for _ in range(500):  # 500 samples per item type
                lost_desc = self.generate_item_description(item_type, is_lost=True)
                found_desc = self.generate_item_description(item_type, is_lost=False)
                
                # Create labels
                item_encoding = {item: i for i, item in enumerate(self.item_types.keys())}
                
                text_data.extend([
                    {
                        'text': lost_desc,
                        'label': item_encoding[item_type],
                        'category': item_type,
                        'is_lost': True
                    },
                    {
                        'text': found_desc,
                        'label': item_encoding[item_type],
                        'category': item_type,
                        'is_lost': False
                    }
                ])
        
        with open(os.path.join(output_dir, 'enhanced_text_descriptions.json'), 'w') as f:
            json.dump(text_data, f, indent=2, default=str)
        
        print(f"✅ Text descriptions dataset: {len(text_data)} samples")
        
        # Create dataset summary
        summary = {
            'total_samples': len(behavior_data) + len(matching_data) + len(temporal_data) + len(text_data),
            'user_behavior_samples': len(behavior_data),
            'item_matching_samples': len(matching_data),
            'temporal_samples': len(temporal_data),
            'text_description_samples': len(text_data),
            'generated_at': datetime.now().isoformat(),
            'item_types': list(self.item_types.keys()),
            'locations': list(self.locations.keys()),
            'time_periods': list(self.time_patterns.keys())
        }
        
        with open(os.path.join(output_dir, 'dataset_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Total samples: {summary['total_samples']:,}")
        print(f"  User behavior: {summary['user_behavior_samples']:,}")
        print(f"  Item matching: {summary['item_matching_samples']:,}")
        print(f"  Temporal: {summary['temporal_samples']:,}")
        print(f"  Text descriptions: {summary['text_description_samples']:,}")
        print(f"\n✅ All datasets saved to {output_dir}")

def main():
    """Main function to generate enhanced datasets"""
    generator = EnhancedDatasetGenerator()
    generator.save_datasets()

if __name__ == "__main__":
    main()
