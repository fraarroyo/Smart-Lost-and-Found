# Lost and Found System - Visual Architecture Diagram

## High-Level System Flow

```mermaid
graph TD
    A[User Uploads Image] --> B[Image Preprocessing]
    B --> C{Processing Mode}
    
    C -->|Ultra-Fast| D[Faster R-CNN Only]
    C -->|Fast| E[Faster R-CNN + Basic RNN]
    C -->|Comprehensive| F[Faster R-CNN + RNN + BERT]
    
    D --> G[Basic Object Detection]
    E --> H[Object Detection + Basic Details]
    F --> I[Full Multi-Modal Analysis]
    
    G --> J[Item Storage]
    H --> J
    I --> J
    
    J --> K[Database Storage]
    K --> L[Search & Matching]
    L --> M[Results to User]
```

## Detailed Component Interaction

```mermaid
graph TB
    subgraph "Image Processing Pipeline"
        A[Image Input] --> B[Preprocessing]
        B --> C[Format Conversion]
        C --> D[Resizing]
        D --> E[Cache Check]
    end
    
    subgraph "Faster R-CNN Module"
        F[Image Tensor] --> G[ResNet50 Backbone]
        G --> H[Region Proposal Network]
        H --> I[ROI Head]
        I --> J[Object Classification]
        J --> K[Bounding Box Regression]
        K --> L[Object Detection Results]
    end
    
    subgraph "RNN Analysis Module"
        M[Image Features] --> N[ResNet18 Feature Extractor]
        N --> O[Color RNN]
        N --> P[Material RNN]
        N --> Q[Size RNN]
        N --> R[Condition RNN]
        O --> S[Color Analysis]
        P --> T[Material Analysis]
        Q --> U[Size Analysis]
        R --> V[Condition Analysis]
        S --> W[RNN Results]
        T --> W
        U --> W
        V --> W
    end
    
    subgraph "BERT Text Module"
        X[Image Description] --> Y[BERT Tokenizer]
        Y --> Z[BERT Encoder]
        Z --> AA[Text Embeddings]
        AA --> BB[Semantic Analysis]
        BB --> CC[Keyword Extraction]
        CC --> DD[BERT Results]
    end
    
    subgraph "Fusion & Matching"
        EE[R-CNN Results] --> FF[Multi-Modal Fusion]
        GG[RNN Results] --> FF
        HH[BERT Results] --> FF
        FF --> II[Enhanced Description]
        II --> JJ[Item Storage]
        JJ --> KK[Similarity Matching]
        KK --> LL[Ranked Results]
    end
    
    E --> F
    E --> M
    E --> X
    L --> EE
    W --> GG
    DD --> HH
```

## Data Flow Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant W as Web App
    participant R as Faster R-CNN
    participant N as RNN
    participant B as BERT
    participant F as Fusion Engine
    participant D as Database
    participant S as Search Engine
    
    U->>W: Upload Image
    W->>W: Preprocess Image
    W->>R: Send Image Tensor
    R->>R: Object Detection
    R->>W: Return Objects + Bounding Boxes
    
    W->>N: Send Image Features
    N->>N: Extract Detailed Features
    N->>W: Return Colors, Materials, Size, Condition
    
    W->>B: Send Image Description
    B->>B: Generate Text Embeddings
    B->>W: Return Semantic Analysis
    
    W->>F: Send All Results
    F->>F: Fuse Multi-Modal Data
    F->>W: Return Enhanced Analysis
    
    W->>D: Store Item with Analysis
    W->>S: Search for Matches
    S->>S: Calculate Similarity Scores
    S->>W: Return Ranked Matches
    W->>U: Display Results
```

## Component Weight Distribution

```mermaid
pie title Component Contribution to Final Analysis
    "Faster R-CNN" : 40
    "RNN Analysis" : 35
    "BERT Text" : 25
```

## Processing Time Breakdown

```mermaid
gantt
    title Processing Time Distribution (Comprehensive Mode)
    dateFormat X
    axisFormat %L ms
    
    section Faster R-CNN
    Image Preprocessing    :0, 200
    Object Detection      :200, 1200
    Post-processing       :1200, 1500
    
    section RNN Analysis
    Feature Extraction    :1500, 2000
    Color Analysis        :2000, 2500
    Material Analysis     :2500, 3000
    Size Analysis         :3000, 3500
    Condition Analysis    :3500, 4000
    
    section BERT Text
    Description Generation :4000, 4200
    Tokenization         :4200, 4400
    Encoding             :4400, 5000
    Semantic Analysis    :5000, 5200
    
    section Fusion
    Multi-modal Fusion   :5200, 5500
    Description Generation :5500, 5700
    Final Scoring        :5700, 6000
```

## Error Handling and Fallbacks

```mermaid
graph TD
    A[Image Processing Start] --> B{Processing Mode}
    
    B -->|Ultra-Fast| C[Faster R-CNN Only]
    B -->|Fast| D[Faster R-CNN + RNN]
    B -->|Comprehensive| E[All Three Models]
    
    C --> F{Success?}
    D --> G{Success?}
    E --> H{Success?}
    
    F -->|Yes| I[Return Results]
    F -->|No| J[Return Error]
    
    G -->|Yes| I
    G -->|No| K[Fallback to R-CNN Only]
    K --> I
    
    H -->|Yes| I
    H -->|No| L[Fallback to Fast Mode]
    L --> M{Success?}
    M -->|Yes| I
    M -->|No| K
```

## Memory and Resource Management

```mermaid
graph LR
    subgraph "Model Loading Strategy"
        A[Application Start] --> B[Load Faster R-CNN]
        B --> C[Keep in Memory]
        D[First BERT Request] --> E[Lazy Load BERT]
        E --> F[Keep in Memory]
        G[First RNN Request] --> H[Lazy Load RNN]
        H --> I[Keep in Memory]
    end
    
    subgraph "Resource Cleanup"
        J[Memory Pressure] --> K[Unload BERT]
        K --> L[Unload RNN]
        L --> M[Keep R-CNN]
    end
    
    subgraph "Caching Strategy"
        N[Image Hash] --> O{Cache Hit?}
        O -->|Yes| P[Return Cached Results]
        O -->|No| Q[Process Image]
        Q --> R[Store in Cache]
        R --> S[Return Results]
    end
```

## Performance Metrics

| Component | Processing Time | Memory Usage | Accuracy | Reliability |
|-----------|----------------|--------------|----------|-------------|
| Faster R-CNN | 1-2 seconds | 2-3 GB | 85-90% | High |
| RNN Analysis | 2-3 seconds | 1-2 GB | 75-85% | Medium |
| BERT Text | 1-2 seconds | 1-2 GB | 80-90% | High |
| Fusion Engine | 0.5 seconds | 0.5 GB | 90-95% | High |

## Integration Points

1. **Image Processing Pipeline**: Seamless handoff between preprocessing and model inference
2. **Multi-Modal Fusion**: Intelligent combination of all three model outputs
3. **Caching Layer**: Transparent caching for performance optimization
4. **Error Handling**: Graceful degradation and fallback mechanisms
5. **Database Integration**: Efficient storage and retrieval of analysis results
6. **Search Engine**: Real-time similarity matching using all model outputs
