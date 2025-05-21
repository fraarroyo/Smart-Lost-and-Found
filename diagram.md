# Lost & Found Application Architecture

## System Overview

```mermaid
graph TB
    subgraph Frontend
        UI[User Interface]
        Templates[HTML Templates]
        Static[Static Files]
    end

    subgraph Backend
        Flask[Flask Application]
        Routes[Route Handlers]
        Auth[Authentication]
        ML[ML Models]
    end

    subgraph Database
        SQLite[(SQLite Database)]
        Users[(Users Table)]
        Items[(Items Table)]
        Settings[(Settings Table)]
    end

    UI --> Templates
    Templates --> Flask
    Flask --> Routes
    Routes --> Auth
    Routes --> ML
    Auth --> SQLite
    ML --> SQLite
    SQLite --> Users
    SQLite --> Items
    SQLite --> Settings
```

## User Flow

```mermaid
sequenceDiagram
    participant User
    participant Auth
    participant App
    participant DB
    participant ML

    User->>Auth: Login/Register
    Auth->>DB: Verify Credentials
    DB-->>Auth: Auth Result
    Auth-->>User: Session Token

    User->>App: Add Item
    App->>ML: Process Image
    ML-->>App: Detected Objects
    App->>DB: Save Item
    App->>ML: Find Matches
    ML-->>App: Potential Matches
    App-->>User: Show Matches

    User->>App: Search Items
    App->>DB: Query Items
    DB-->>App: Search Results
    App-->>User: Display Results
```

## Database Schema

```mermaid
erDiagram
    User {
        int id PK
        string username
        string email
        string password_hash
        string reset_token
        datetime reset_token_expiry
        boolean is_admin
        boolean is_active
        datetime date_joined
        datetime last_login
    }

    Item {
        int id PK
        string title
        text description
        string category
        string status
        string location
        datetime date
        string image_path
        int user_id FK
        text detected_objects
        text text_embedding
        string color
        string size
    }

    Settings {
        int id PK
        int item_expiry_days
        int max_image_size
        int matching_threshold
        boolean enable_email_notifications
    }

    User ||--o{ Item : "owns"
```

## Admin Dashboard Structure

```mermaid
graph TB
    subgraph Admin Dashboard
        Stats[Statistics]
        Users[User Management]
        Items[Item Management]
        Settings[System Settings]
        Export[Data Export]
    end

    Stats --> Users
    Stats --> Items
    Users --> Settings
    Items --> Settings
    Settings --> Export
```

## ML Processing Pipeline

```mermaid
graph LR
    subgraph Image Processing
        Upload[Image Upload]
        Detect[Object Detection]
        Color[Color Analysis]
        Size[Size Estimation]
    end

    subgraph Text Processing
        Title[Title Analysis]
        Desc[Description Analysis]
        Match[Matching Algorithm]
    end

    Upload --> Detect
    Detect --> Color
    Detect --> Size
    Title --> Match
    Desc --> Match
    Color --> Match
    Size --> Match
```

## ML Models Architecture

```mermaid
graph TB
    subgraph Text Processing
        BERT[BERT Model]
        RNN[RNN Model]
        TextEmbed[Text Embedding]
        TextMatch[Text Matching]
    end

    subgraph Image Processing
        ObjectDet[Object Detection]
        ColorDet[Color Detection]
        SizeEst[Size Estimation]
    end

    subgraph Integration
        MatchAlgo[Matching Algorithm]
        ScoreCalc[Score Calculation]
    end

    BERT --> TextEmbed
    RNN --> TextMatch
    TextEmbed --> MatchAlgo
    TextMatch --> MatchAlgo
    ObjectDet --> MatchAlgo
    ColorDet --> ScoreCalc
    SizeEst --> ScoreCalc
    ScoreCalc --> MatchAlgo
```

## ML Model References

### BERT (Bidirectional Encoder Representations from Transformers)
- **Purpose**: Text understanding and semantic matching
- **Implementation**: Using pre-trained BERT model for:
  - Text embedding generation
  - Semantic similarity calculation
  - Item description understanding
- **Key Features**:
  - Bidirectional context understanding
  - Pre-trained on large text corpus
  - Fine-tuned for item matching

### RNN (Recurrent Neural Network)
- **Purpose**: Sequence processing and pattern recognition
- **Implementation**: Using RNN for:
  - Text sequence analysis
  - Pattern matching in item descriptions
  - Temporal feature extraction
- **Key Features**:
  - Sequential data processing
  - Long-term dependency learning
  - Pattern recognition in text

### Integration Points
1. **Text Processing Pipeline**:
   - BERT for initial text embedding
   - RNN for sequence analysis
   - Combined features for matching

2. **Matching Algorithm**:
   - BERT embeddings for semantic similarity
   - RNN patterns for sequence matching
   - Combined scoring system

3. **Performance Optimization**:
   - Batch processing for BERT
   - Sequence batching for RNN
   - Caching of embeddings

### Model Usage Examples
```python
# BERT Usage
text_embedding = text_analyzer.analyze_text(f"{title} {description}")

# RNN Usage
sequence_features = sequence_processor.process(description)

# Combined Matching
similarity = text_analyzer.compute_similarity(query, item_text)
```

## Security Flow

```mermaid
sequenceDiagram
    participant User
    participant App
    participant Auth
    participant DB

    User->>App: Request Protected Resource
    App->>Auth: Check Session
    Auth->>DB: Verify User
    DB-->>Auth: User Status
    Auth-->>App: Auth Result
    App-->>User: Grant/Deny Access
```

## Search Implementation

```mermaid
graph TB
    subgraph Search Flow
        Query[Search Query]
        TextSearch[Text Search]
        ImageSearch[Image Search]
        Filter[Filters]
        Results[Search Results]
    end

    subgraph Search Components
        BERT[BERT Embedding]
        RNN[RNN Processing]
        ObjectDet[Object Detection]
        ColorMatch[Color Matching]
        SizeMatch[Size Matching]
    end

    subgraph Search Filters
        Category[Category Filter]
        Status[Status Filter]
        Date[Date Filter]
        Location[Location Filter]
    end

    Query --> TextSearch
    Query --> ImageSearch
    TextSearch --> BERT
    TextSearch --> RNN
    ImageSearch --> ObjectDet
    ImageSearch --> ColorMatch
    ImageSearch --> SizeMatch
    Filter --> Category
    Filter --> Status
    Filter --> Date
    Filter --> Location
    BERT --> Results
    RNN --> Results
    ObjectDet --> Results
    ColorMatch --> Results
    SizeMatch --> Results
    Category --> Results
    Status --> Results
    Date --> Results
    Location --> Results
```

## Search Implementation Details

### Search Components
1. **Text Search**
   - BERT for semantic understanding
   - RNN for pattern matching
   - Keyword matching
   - Category matching

2. **Image Search**
   - Object detection
   - Color analysis
   - Size estimation
   - Visual similarity

3. **Filters**
   - Category filtering
   - Status filtering (lost/found)
   - Date range filtering
   - Location-based filtering

### Search API Endpoints
```python
# Search endpoint
@app.route('/search')
def search():
    query = request.args.get('q', '')
    category = request.args.get('category', '')
    status = request.args.get('status', '')
    
    # Implementation details...
```

### Search Features
1. **Text-based Search**
   - Semantic search using BERT
   - Pattern matching using RNN
   - Category-based filtering
   - Status-based filtering

2. **Image-based Search**
   - Object detection
   - Color matching
   - Size comparison
   - Visual similarity

3. **Combined Search**
   - Text + Image search
   - Multiple filter combinations
   - Relevance scoring
   - Result ranking

### Search Results
- Sorted by relevance
- Filtered by category
- Filtered by status
- Filtered by date
- Filtered by location 