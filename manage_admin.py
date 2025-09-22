#!/usr/bin/env python3
"""
Admin user management script.
"""

import os
import sys

# Add current directory to path to import app
sys.path.append('.')

def list_users():
    """List all users in the database."""
    try:
        from app import app, db, User
        
        with app.app_context():
            users = User.query.all()
            print(f"=== Current Users ({len(users)}) ===")
            for user in users:
                print(f"ID: {user.id}")
                print(f"  Username: {user.username}")
                print(f"  Email: {user.email}")
                print(f"  Admin: {user.is_admin}")
                print(f"  Active: {user.is_active}")
                print(f"  Date Joined: {user.date_joined}")
                print()
            
    except Exception as e:
        print(f"❌ Error listing users: {e}")

def create_admin_user(username, email, password):
    """Create a new admin user."""
    try:
        from app import app, db, User
        
        with app.app_context():
            # Check if user already exists
            existing_user = User.query.filter_by(username=username).first()
            if existing_user:
                print(f"❌ User '{username}' already exists!")
                return False
            
            # Create admin user
            admin_user = User(
                username=username,
                email=email,
                is_admin=True,
                is_active=True
            )
            admin_user.set_password(password)
            
            db.session.add(admin_user)
            db.session.commit()
            
            print(f"✓ Admin user '{username}' created successfully!")
            return True
            
    except Exception as e:
        print(f"❌ Error creating admin user: {e}")
        return False

def make_user_admin(username):
    """Make an existing user an admin."""
    try:
        from app import app, db, User
        
        with app.app_context():
            user = User.query.filter_by(username=username).first()
            if not user:
                print(f"❌ User '{username}' not found!")
                return False
            
            user.is_admin = True
            db.session.commit()
            
            print(f"✓ User '{username}' is now an admin!")
            return True
            
    except Exception as e:
        print(f"❌ Error making user admin: {e}")
        return False

def change_password(username, new_password):
    """Change a user's password."""
    try:
        from app import app, db, User
        
        with app.app_context():
            user = User.query.filter_by(username=username).first()
            if not user:
                print(f"❌ User '{username}' not found!")
                return False
            
            user.set_password(new_password)
            db.session.commit()
            
            print(f"✓ Password changed for user '{username}'!")
            return True
            
    except Exception as e:
        print(f"❌ Error changing password: {e}")
        return False

def main():
    """Main function with interactive menu."""
    print("=== Admin User Management ===\n")
    
    while True:
        print("1. List all users")
        print("2. Create new admin user")
        print("3. Make existing user admin")
        print("4. Change user password")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            list_users()
        elif choice == '2':
            username = input("Enter username: ").strip()
            email = input("Enter email: ").strip()
            password = input("Enter password: ").strip()
            create_admin_user(username, email, password)
        elif choice == '3':
            username = input("Enter username to make admin: ").strip()
            make_user_admin(username)
        elif choice == '4':
            username = input("Enter username: ").strip()
            password = input("Enter new password: ").strip()
            change_password(username, password)
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()
