#!/usr/bin/env python3
"""
Complete Data Cleanup Script
Clears all data from database, JSON files, and training data directories.
"""

import os
import shutil
import glob
from pathlib import Path

def clear_database():
    """Clear all data from the database"""
    print("🗄️  Clearing database...")
    try:
        from app import app, db, Item, Match, User, ModelMetrics, Feedback
        from flask_login import current_user
        
        with app.app_context():
            # Clear all tables
            print("  - Clearing Item table...")
            Item.query.delete()
            
            print("  - Clearing Match table...")
            Match.query.delete()
            
            print("  - Clearing ModelMetrics table...")
            ModelMetrics.query.delete()
            
            print("  - Clearing Feedback table...")
            Feedback.query.delete()
            
            # Note: Not clearing User table to preserve user accounts
            print("  - Preserving User table (keeping user accounts)")
            
            # Commit changes
            db.session.commit()
            print("  ✅ Database cleared successfully")
            
    except Exception as e:
        print(f"  ❌ Error clearing database: {e}")

def clear_json_files():
    """Clear all JSON files in the project"""
    print("📄 Clearing JSON files...")
    
    # Directories to search for JSON files
    json_directories = [
        '.',
        'training_data',
        'static',
        'instance'
    ]
    
    json_files_cleared = 0
    
    for directory in json_directories:
        if os.path.exists(directory):
            # Find all JSON files
            json_pattern = os.path.join(directory, '**', '*.json')
            json_files = glob.glob(json_pattern, recursive=True)
            
            for json_file in json_files:
                try:
                    os.remove(json_file)
                    json_files_cleared += 1
                    print(f"  - Removed: {json_file}")
                except Exception as e:
                    print(f"  - Error removing {json_file}: {e}")
    
    print(f"  ✅ Cleared {json_files_cleared} JSON files")

def clear_training_data():
    """Clear all training data directories"""
    print("🧠 Clearing training data...")
    
    training_directories = [
        'training_data',
        'training_images',
        'training_labels',
        'models',
        'static/uploads'
    ]
    
    for directory in training_directories:
        if os.path.exists(directory):
            try:
                # Remove all files in directory
                for file_path in glob.glob(os.path.join(directory, '*')):
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"  - Removed file: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        print(f"  - Removed directory: {file_path}")
                
                print(f"  ✅ Cleared {directory}")
            except Exception as e:
                print(f"  ❌ Error clearing {directory}: {e}")
        else:
            print(f"  - Directory {directory} does not exist, skipping")

def clear_cache_files():
    """Clear cache and temporary files"""
    print("🗑️  Clearing cache files...")
    
    cache_patterns = [
        '*.pyc',
        '__pycache__',
        '*.log',
        '*.tmp',
        '.DS_Store',
        'Thumbs.db'
    ]
    
    files_cleared = 0
    
    for pattern in cache_patterns:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    files_cleared += 1
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    files_cleared += 1
            except Exception as e:
                print(f"  - Error removing {file_path}: {e}")
    
    print(f"  ✅ Cleared {files_cleared} cache files")

def clear_model_files():
    """Clear model files (but keep the main best_model1.pth)"""
    print("🤖 Clearing model files...")
    
    model_files_to_remove = [
        'best_model.pth',
        'checkpoint.pt',
        'checkpoint.pth',
        'unified_model.pth',
        'unified_confidence_adjuster.pkl',
        'unified_similarity_adjuster.pkl'
    ]
    
    for model_file in model_files_to_remove:
        if os.path.exists(model_file):
            try:
                os.remove(model_file)
                print(f"  - Removed: {model_file}")
            except Exception as e:
                print(f"  - Error removing {model_file}: {e}")
    
    # Clear RNN model directory
    rnn_models_dir = 'models/rnn_models'
    if os.path.exists(rnn_models_dir):
        try:
            shutil.rmtree(rnn_models_dir)
            os.makedirs(rnn_models_dir, exist_ok=True)
            print(f"  - Cleared RNN models directory")
        except Exception as e:
            print(f"  - Error clearing RNN models: {e}")
    
    print("  ✅ Model files cleared (kept best_model1.pth)")

def main():
    """Main cleanup function"""
    print("🧹 Starting Complete Data Cleanup")
    print("=" * 50)
    
    # Confirm before proceeding
    response = input("⚠️  This will delete ALL data. Are you sure? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cleanup cancelled")
        return
    
    print("\n🚀 Proceeding with cleanup...\n")
    
    # Clear all data
    clear_database()
    print()
    
    clear_json_files()
    print()
    
    clear_training_data()
    print()
    
    clear_cache_files()
    print()
    
    clear_model_files()
    print()
    
    print("🎉 Cleanup completed successfully!")
    print("\n📊 Summary:")
    print("  - Database: All tables cleared (users preserved)")
    print("  - JSON files: All removed")
    print("  - Training data: All directories cleared")
    print("  - Cache files: All removed")
    print("  - Model files: Cleared (kept best_model1.pth)")
    print("\n✨ Your system is now clean and ready for fresh data!")

if __name__ == "__main__":
    main()
