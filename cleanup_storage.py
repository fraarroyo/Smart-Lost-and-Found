#!/usr/bin/env python3
"""
Storage cleanup script for PythonAnywhere deployment
This script helps free up space by removing unnecessary files
"""

import os
import shutil
import glob
import sys

def cleanup_storage():
    """Clean up storage by removing unnecessary files"""
    print("🧹 Starting storage cleanup...")
    
    total_freed = 0
    
    # 1. Remove __pycache__ directories
    print("Removing __pycache__ directories...")
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                pycache_path = os.path.join(root, dir_name)
                try:
                    size = get_dir_size(pycache_path)
                    shutil.rmtree(pycache_path)
                    total_freed += size
                    print(f"  Removed: {pycache_path} ({format_size(size)})")
                except Exception as e:
                    print(f"  Could not remove {pycache_path}: {e}")
    
    # 2. Remove .pyc files
    print("Removing .pyc files...")
    for pyc_file in glob.glob('**/*.pyc', recursive=True):
        try:
            size = os.path.getsize(pyc_file)
            os.remove(pyc_file)
            total_freed += size
            print(f"  Removed: {pyc_file} ({format_size(size)})")
        except Exception as e:
            print(f"  Could not remove {pyc_file}: {e}")
    
    # 3. Remove temporary files
    print("Removing temporary files...")
    temp_patterns = [
        '*.tmp', '*.temp', '*.log', '*.cache',
        'temp_*.jpg', 'temp_*.png', 'temp_*.jpeg'
    ]
    
    for pattern in temp_patterns:
        for temp_file in glob.glob(pattern):
            try:
                size = os.path.getsize(temp_file)
                os.remove(temp_file)
                total_freed += size
                print(f"  Removed: {temp_file} ({format_size(size)})")
            except Exception as e:
                print(f"  Could not remove {temp_file}: {e}")
    
    # 4. Remove duplicate model files (keep only the best one)
    print("Checking for duplicate model files...")
    model_files = [
        'best_model1.pth',
        'best_model1.2.pth'
    ]
    
    # Keep only the latest model file
    if os.path.exists('best_model1.2.pth') and os.path.exists('best_model1.pth'):
        try:
            size = os.path.getsize('best_model1.pth')
            os.remove('best_model1.pth')
            total_freed += size
            print(f"  Removed duplicate: best_model1.pth ({format_size(size)})")
        except Exception as e:
            print(f"  Could not remove best_model1.pth: {e}")
    
    # 5. Clean up pip cache
    print("Cleaning pip cache...")
    try:
        import subprocess
        result = subprocess.run(['pip', 'cache', 'purge'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("  Pip cache cleaned successfully")
        else:
            print(f"  Pip cache cleanup failed: {result.stderr}")
    except Exception as e:
        print(f"  Could not clean pip cache: {e}")
    
    print(f"\n✅ Cleanup complete! Total space freed: {format_size(total_freed)}")
    
    # Show current directory size
    current_size = get_dir_size('.')
    print(f"📊 Current directory size: {format_size(current_size)}")

def get_dir_size(path):
    """Get the size of a directory in bytes"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception:
        pass
    return total_size

def format_size(size_bytes):
    """Format size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def check_storage_usage():
    """Check current storage usage"""
    print("📊 Storage Usage Report:")
    print("=" * 50)
    
    # Check main directories
    directories = [
        '.',
        'static',
        'templates',
        'models',
        'training_data',
        'instance'
    ]
    
    for directory in directories:
        if os.path.exists(directory):
            size = get_dir_size(directory)
            print(f"{directory:15}: {format_size(size)}")
    
    print("=" * 50)

if __name__ == "__main__":
    print("🚀 PythonAnywhere Storage Cleanup Tool")
    print("=" * 50)
    
    # Show current usage
    check_storage_usage()
    print()
    
    # Ask for confirmation
    response = input("Do you want to proceed with cleanup? (y/N): ")
    if response.lower() in ['y', 'yes']:
        cleanup_storage()
    else:
        print("Cleanup cancelled.")
