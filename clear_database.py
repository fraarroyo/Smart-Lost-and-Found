#!/usr/bin/env python3
"""
Database Clear Script
Clears only the database tables, preserves files.
"""

import os
import sys

# Add current directory to path to import app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def clear_database_only():
    """Clear only the database tables"""
    print("🗄️  Clearing Database Only")
    
    try:
        from app import app, db, Item, Match, User, ModelMetrics, Feedback
        
        with app.app_context():
            print("  - Clearing Item table...")
            Item.query.delete()
            
            print("  - Clearing Match table...")
            Match.query.delete()
            
            print("  - Clearing ModelMetrics table...")
            ModelMetrics.query.delete()
            
            print("  - Clearing Feedback table...")
            Feedback.query.delete()
            
            print("  - Preserving User table...")
            
            # Commit changes
            db.session.commit()
            print("  ✅ Database cleared successfully")
            
    except Exception as e:
        print(f"  ❌ Error clearing database: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if clear_database_only():
        print("\n🎉 Database cleared successfully!")
        print("📁 Files and training data preserved")
    else:
        print("\n❌ Database clear failed")
