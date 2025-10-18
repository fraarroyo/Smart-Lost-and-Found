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

### Testing & Evaluation

#### Text model evaluation (RNN and BERT)

We provide two standalone evaluators:

1) BidirectionalDescriptionRNN classifier

Run evaluation on a labeled text dataset (JSONL or CSV) with fields `text` and `label` (int):

```bash
python evaluate_rnn_text.py --data path/to/dataset.jsonl --format jsonl --batch-size 64
# or CSV
python evaluate_rnn_text.py --data path/to/dataset.csv --format csv
```

Notes:
- The evaluator will load weights from `models/rnn_models/description_birnn.pth` if present.
- If no vocabulary has been saved, a simple whitespace vocabulary is built from the dataset on-the-fly.
- Outputs accuracy and a classification report (requires scikit-learn).

2) BERT similarity evaluator

Evaluate BERT-based similarity on pairs of texts. Dataset should contain `text1`, `text2`, and `label` (float similarity in [0,1] or 0/1):

```bash
# Regression metrics (Pearson/Spearman/MSE)
python evaluate_bert_text.py --data pairs.jsonl --format jsonl --mode regression

# Classification metrics (Accuracy, Precision, Recall, F1 at threshold)
python evaluate_bert_text.py --data pairs.csv --format csv --mode classification --threshold 0.5
```

Notes:
- Uses the unified model's BERT components from `ml_models.UnifiedModel` (lazy-loaded).
- Requires `transformers` and optionally `scipy`/`scikit-learn` for full metrics.

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