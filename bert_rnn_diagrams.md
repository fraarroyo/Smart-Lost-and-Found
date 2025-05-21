# BERT and RNN Architecture Diagrams

## BERT Architecture
```mermaid
graph TD
    A[Input Text] --> B[Tokenization]
    B --> C[Input Embeddings]
    C --> D[Positional Embeddings]
    D --> E[Sum + LayerNorm + Dropout]
    E --> F[Transformer Encoder Layer 1]
    F --> G[Transformer Encoder Layer 2]
    G --> H[... N Layers ...]
    H --> I[Final Encoder Output]
    I --> J[Task-specific Head (e.g., Classification, QA, etc.)]
```

## RNN Architecture
```mermaid
graph TD
    A[Input Sequence] --> B[RNN/LSTM Cell t=1]
    B --> C[RNN/LSTM Cell t=2]
    C --> D[RNN/LSTM Cell t=3]
    D --> E[... t=T ...]
    E --> F[Final Output / Sequence Output]
    subgraph Hidden State Flow
        B1[Hidden State t=0] --> B
        B --> C1[Hidden State t=1]
        C1 --> C
        C --> D1[Hidden State t=2]
        D1 --> D
        D --> E1[Hidden State t=3]
        E1 --> E
    end
```

## Detailed BERT Architecture
```mermaid
graph TB
    subgraph BERT Components
        Input[Input Text]
        Tokenizer[Tokenizer]
        Embeddings[Embeddings Layer]
        Encoder[Transformer Encoder]
        Output[Output Layer]
    end

    subgraph Transformer Encoder
        SelfAttn[Self-Attention]
        FFN[Feed Forward Network]
        Norm1[Layer Norm]
        Norm2[Layer Norm]
    end

    Input --> Tokenizer
    Tokenizer --> Embeddings
    Embeddings --> Encoder
    Encoder --> Output

    SelfAttn --> Norm1
    Norm1 --> FFN
    FFN --> Norm2
```

## Detailed RNN Architecture
```mermaid
graph TB
    subgraph RNN Components
        Input[Input Sequence]
        Hidden[Hidden State]
        Output[Output Layer]
    end

    subgraph RNN Cell
        Xt[Input at time t]
        Ht1[Hidden State t-1]
        Ht[Hidden State t]
        Ot[Output at time t]
    end

    Input --> Xt
    Ht1 --> Ht
    Xt --> Ht
    Ht --> Ot
    Ot --> Output
``` 