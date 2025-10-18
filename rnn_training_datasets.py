#!/usr/bin/env python3
"""
RNN Training Datasets for BARYONYX Lost & Found System
Creates comprehensive training datasets for all three RNN models
"""

import json
import csv
import random
from datetime import datetime, timedelta
import numpy as np

def create_user_behavior_dataset(num_samples=1000):
    """
    Create training dataset for UserBehaviorLSTM
    Features: [hour, day_of_week, action_type, item_type, confidence, 
              search_count, upload_count, view_count, time_since_last, session_length]
    Labels: [search, upload, view, browse, logout]
    """
    
    # Action types and their probabilities
    actions = ['search', 'upload', 'view', 'browse', 'logout']
    action_weights = [0.4, 0.2, 0.25, 0.1, 0.05]  # More searches than uploads
    # Normalize to sum to 1.0
    action_weights = np.array(action_weights)
    action_weights = action_weights / action_weights.sum()
    
    # Item types
    item_types = ['phone', 'wallet', 'mouse', 'tumbler']
    
    # Time patterns (people are more active during certain hours)
    hour_weights = [0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01]
    # Normalize to sum to 1.0
    hour_weights = np.array(hour_weights)
    hour_weights = hour_weights / hour_weights.sum()
    
    dataset = []
    
    for i in range(num_samples):
        # Generate realistic time features
        hour = np.random.choice(24, p=hour_weights)
        day_of_week = random.randint(0, 6)
        
        # Generate action based on time patterns
        if 6 <= hour <= 9:  # Morning commute
            time_action_weights = [0.5, 0.1, 0.3, 0.05, 0.05]  # search, upload, view, browse, logout
        elif 12 <= hour <= 14:  # Lunch time
            time_action_weights = [0.3, 0.2, 0.4, 0.05, 0.05]
        elif 17 <= hour <= 19:  # Evening commute
            time_action_weights = [0.4, 0.15, 0.35, 0.05, 0.05]
        elif 20 <= hour <= 23:  # Evening browsing
            time_action_weights = [0.35, 0.25, 0.3, 0.05, 0.05]
        else:  # Other times
            time_action_weights = [0.4, 0.2, 0.25, 0.1, 0.05]
        
        # Normalize time-specific weights
        time_action_weights = np.array(time_action_weights)
        time_action_weights = time_action_weights / time_action_weights.sum()
        
        action = np.random.choice(actions, p=time_action_weights)
        item_type = random.choice(item_types) if action in ['search', 'upload', 'view'] else None
        
        # Generate user behavior features
        features = [0.0] * 10
        
        # Time features
        features[0] = hour / 24.0
        features[1] = day_of_week / 7.0
        
        # Action type encoding
        action_encoding = {'search': 1.0, 'upload': 2.0, 'view': 3.0, 'browse': 4.0, 'logout': 5.0}
        features[2] = action_encoding[action] / 5.0
        
        # Item type encoding
        if item_type:
            item_encoding = {item: (i+1) for i, item in enumerate(item_types)}
            features[3] = item_encoding[item_type] / len(item_types)
        
        # Simulate recent activity counts
        features[5] = random.uniform(0.1, 0.8)  # search_count
        features[6] = random.uniform(0.05, 0.4)  # upload_count
        features[7] = random.uniform(0.1, 0.6)  # view_count
        
        # Time since last action (in hours, normalized)
        features[8] = random.uniform(0.1, 1.0)
        
        # Session length (normalized)
        features[9] = random.uniform(0.1, 0.9)
        
        # Confidence based on action type
        confidence = random.uniform(0.6, 1.0) if action in ['search', 'view'] else random.uniform(0.4, 0.9)
        features[4] = confidence
        
        # Create label (next likely action)
        if action == 'search':
            next_action = random.choices(['search', 'view', 'upload'], weights=[0.6, 0.3, 0.1])[0]
        elif action == 'upload':
            next_action = random.choices(['view', 'search', 'browse'], weights=[0.5, 0.3, 0.2])[0]
        elif action == 'view':
            next_action = random.choices(['search', 'view', 'upload'], weights=[0.4, 0.4, 0.2])[0]
        elif action == 'browse':
            next_action = random.choices(['search', 'view', 'upload'], weights=[0.5, 0.3, 0.2])[0]
        else:  # logout
            next_action = 'logout'
        
        label = actions.index(next_action)
        
        dataset.append({
            'features': features,
            'label': label,
            'action': action,
            'next_action': next_action,
            'item_type': item_type,
            'timestamp': datetime.now() - timedelta(hours=random.randint(0, 24))
        })
    
    return dataset

def create_description_dataset(num_samples=800):
    """
    Create training dataset for BidirectionalDescriptionRNN
    Input: Text descriptions of lost/found items
    Labels: Item categories (0-3) - phone, mouse, wallet, tumbler
    """
    
    # Item categories and their descriptions - focused on 4 specific items
    categories = {
        0: 'phone',      # phones, smartphones
        1: 'mouse',      # computer mice, wireless mice
        2: 'wallet',     # wallets, purses
        3: 'tumbler'     # tumblers, water bottles, drink containers
    }
    
    # Sample descriptions for each category
    descriptions = {
        0: [  # Phone descriptions
            "Lost black iPhone 12 with cracked screen",
            "Found Samsung Galaxy phone with blue case",
            "Lost iPhone 13 Pro Max gold",
            "Found Google Pixel phone black",
            "Lost OnePlus phone with red case",
            "Found Xiaomi phone white",
            "Lost Huawei phone with cracked screen",
            "Found Motorola phone blue",
            "Lost iPhone 14 Pro silver",
            "Found Samsung Galaxy S23 black",
            "Lost iPhone 15 with clear case",
            "Found Google Pixel 7 white",
            "Lost OnePlus 11 with black case",
            "Found Xiaomi 13 Pro blue",
            "Lost Huawei P60 with gold case",
            "Found Motorola Edge 40 black",
            "Lost iPhone SE with pink case",
            "Found Samsung Galaxy A54 white",
            "Lost Google Pixel 6a black",
            "Found OnePlus Nord with blue case"
        ],
        1: [  # Mouse descriptions
            "Lost computer mouse wireless black",
            "Found gaming mouse red with RGB",
            "Lost wireless mouse Logitech black",
            "Found optical mouse white",
            "Lost Bluetooth mouse silver",
            "Found gaming mouse Razer green",
            "Lost wireless mouse Microsoft blue",
            "Found ergonomic mouse black",
            "Lost gaming mouse Corsair RGB",
            "Found wireless mouse Apple white",
            "Lost optical mouse Logitech black",
            "Found gaming mouse SteelSeries blue",
            "Lost wireless mouse HP black",
            "Found trackball mouse red",
            "Lost gaming mouse ASUS ROG",
            "Found wireless mouse Dell black",
            "Lost Bluetooth mouse Lenovo silver",
            "Found gaming mouse HyperX red",
            "Lost wireless mouse Acer blue",
            "Found optical mouse Trust black"
        ],
        2: [  # Wallet descriptions
            "Lost wallet brown leather with cards",
            "Found wallet black leather bifold",
            "Lost wallet red leather with money",
            "Found wallet blue canvas",
            "Lost wallet black synthetic leather",
            "Found wallet brown leather trifold",
            "Lost wallet green leather with ID",
            "Found wallet black leather RFID",
            "Lost wallet gray leather with coins",
            "Found wallet brown leather card holder",
            "Lost wallet black leather with photos",
            "Found wallet red leather money clip",
            "Lost wallet blue leather with receipts",
            "Found wallet black leather zipper",
            "Lost wallet brown leather with keys",
            "Found wallet gray leather slim",
            "Lost wallet black leather with cash",
            "Found wallet brown leather passport",
            "Lost wallet red leather with cards",
            "Found wallet blue leather compact"
        ],
        3: [  # Tumbler descriptions
            "Lost tumbler stainless steel silver",
            "Found water bottle blue plastic",
            "Lost tumbler black with handle",
            "Found coffee cup white ceramic",
            "Lost tumbler red with straw",
            "Found water bottle clear plastic",
            "Lost tumbler white with lid",
            "Found coffee mug black ceramic",
            "Lost tumbler blue with grip",
            "Found water bottle green plastic",
            "Lost tumbler silver with insulation",
            "Found coffee cup blue ceramic",
            "Lost tumbler black with logo",
            "Found water bottle pink plastic",
            "Lost tumbler white with measurement",
            "Found coffee mug white ceramic",
            "Lost tumbler red with spout",
            "Found water bottle clear with filter",
            "Lost tumbler blue with carabiner",
            "Found coffee cup black with handle"
        ]
    }
    
    dataset = []
    
    for i in range(num_samples):
        # Select random category (0-3 for phone, mouse, wallet, tumbler)
        category = random.randint(0, 3)
        
        # Select random description from that category
        description = random.choice(descriptions[category])
        
        # Add some variation to descriptions
        variations = [
            f"Lost {description.lower()}",
            f"Found {description.lower()}",
            f"Missing {description.lower()}",
            f"Looking for {description.lower()}",
            f"Need to find {description.lower()}",
            f"Seeking {description.lower()}",
            f"Searching for {description.lower()}",
            f"Can't find {description.lower()}",
            f"Looking to recover {description.lower()}",
            f"Trying to locate {description.lower()}"
        ]
        
        final_description = random.choice(variations)
        
        dataset.append({
            'text': final_description,
            'label': category,
            'category': categories[category],
            'original_description': description
        })
    
    return dataset

def create_temporal_dataset(num_samples=600):
    """
    Create training dataset for TemporalPatternRNN
    Features: [hour, day, month, season, location_type, item_frequency, weather_factor, holiday_factor]
    Labels: [morning, afternoon, evening, night]
    """
    
    # Time periods and their characteristics
    time_periods = {
        0: 'morning',    # 6-11 AM
        1: 'afternoon',  # 12-17 PM
        2: 'evening',    # 18-21 PM
        3: 'night'       # 22-5 AM
    }
    
    # Item types that are commonly lost/found
    item_types = ['phone', 'wallet', 'keys', 'laptop', 'glasses', 'watch', 'bag', 'book', 'charger', 'headphones']
    
    # Location types
    locations = ['office', 'home', 'public', 'transport']
    
    dataset = []
    
    for i in range(num_samples):
        # Generate realistic temporal features
        hour = random.randint(0, 23)
        day = random.randint(1, 31)
        month = random.randint(1, 12)
        
        # Determine season
        if month in [12, 1, 2]:
            season = 0  # Winter
        elif month in [3, 4, 5]:
            season = 1  # Spring
        elif month in [6, 7, 8]:
            season = 2  # Summer
        else:
            season = 3  # Fall
        
        # Determine time period based on hour
        if 6 <= hour < 12:
            time_period = 0  # morning
        elif 12 <= hour < 18:
            time_period = 1  # afternoon
        elif 18 <= hour < 22:
            time_period = 2  # evening
        else:
            time_period = 3  # night
        
        # Select item type and location
        item_type = random.choice(item_types)
        location = random.choice(locations)
        
        # Create features
        features = [0.0] * 8
        
        features[0] = hour / 24.0
        features[1] = day / 31.0
        features[2] = month / 12.0
        features[3] = season / 4.0
        
        # Location type encoding
        location_encoding = {'office': 1.0, 'home': 2.0, 'public': 3.0, 'transport': 4.0}
        features[4] = location_encoding[location] / 4.0
        
        # Item frequency (simulate based on item type)
        item_frequency = {
            'phone': 0.8, 'wallet': 0.7, 'keys': 0.6, 'laptop': 0.4,
            'glasses': 0.5, 'watch': 0.3, 'bag': 0.6, 'book': 0.4,
            'charger': 0.5, 'headphones': 0.4
        }
        features[5] = item_frequency.get(item_type, 0.5)
        
        # Weather factor (simplified)
        weather_factors = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]  # 0.2 = bad weather, 0.8 = good weather
        features[6] = random.choice(weather_factors)
        
        # Holiday factor
        holiday_months = [12, 1, 7, 11]  # December, January, July, November
        features[7] = 0.8 if month in holiday_months else 0.2
        
        dataset.append({
            'features': features,
            'label': time_period,
            'time_period': time_periods[time_period],
            'item_type': item_type,
            'location': location,
            'hour': hour,
            'month': month,
            'season': season
        })
    
    return dataset

def save_datasets():
    """Save all datasets to files"""
    
    print("Creating RNN training datasets...")
    
    # Create datasets
    user_behavior_data = create_user_behavior_dataset(1000)
    description_data = create_description_dataset(800)
    temporal_data = create_temporal_dataset(600)
    
    # Save user behavior dataset
    with open('rnn_user_behavior_dataset.json', 'w') as f:
        json.dump(user_behavior_data, f, indent=2, default=str)
    
    # Save description dataset as CSV
    with open('rnn_description_dataset.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label', 'category', 'original_description'])
        for item in description_data:
            writer.writerow([item['text'], item['label'], item['category'], item['original_description']])
    
    # Save temporal dataset
    with open('rnn_temporal_dataset.json', 'w') as f:
        json.dump(temporal_data, f, indent=2, default=str)
    
    print(f"✅ Created user behavior dataset: {len(user_behavior_data)} samples")
    print(f"✅ Created description dataset: {len(description_data)} samples")
    print(f"✅ Created temporal dataset: {len(temporal_data)} samples")
    
    # Print sample data
    print("\n=== SAMPLE USER BEHAVIOR DATA ===")
    for i, sample in enumerate(user_behavior_data[:3]):
        print(f"Sample {i+1}:")
        print(f"  Features: {[f'{x:.2f}' for x in sample['features']]}")
        print(f"  Action: {sample['action']} -> Next: {sample['next_action']}")
        print(f"  Label: {sample['label']}")
    
    print("\n=== SAMPLE DESCRIPTION DATA ===")
    for i, sample in enumerate(description_data[:3]):
        print(f"Sample {i+1}:")
        print(f"  Text: {sample['text']}")
        print(f"  Category: {sample['category']} (Label: {sample['label']})")
    
    print("\n=== SAMPLE TEMPORAL DATA ===")
    for i, sample in enumerate(temporal_data[:3]):
        print(f"Sample {i+1}:")
        print(f"  Features: {[f'{x:.2f}' for x in sample['features']]}")
        print(f"  Time Period: {sample['time_period']} (Label: {sample['label']})")
        print(f"  Item: {sample['item_type']}, Location: {sample['location']}")

if __name__ == "__main__":
    save_datasets()
