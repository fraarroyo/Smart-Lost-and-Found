# Lost and Found Management System

A Flask-based web application for managing lost and found items. Users can post items they've lost or found, search for items, and manage their listings.

## Features

- User authentication (register, login, logout)
- Post lost or found items with images
- Search and filter items by category and status
- Responsive design using Bootstrap 5
- Secure file uploads
- SQLite database for data storage

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd lost-found-system
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
flask run
```
The database will be automatically created when you first run the application.

5. Run the application:
```bash
flask run
```

The application will be available at `http://localhost:5000`

## Usage

1. Register a new account or login with existing credentials
2. Browse the home page to see all items
3. Use the search page to find specific items
4. Add new items using the "Add Item" button
5. View item details by clicking on any item

## Security Features

- Password hashing using Werkzeug
- Secure file uploads with size limits and file type validation
- CSRF protection
- Session management
- Input validation and sanitization

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 