# Native Go Transformers (GPT/BERT)

A comprehensive native Go implementation of transformer architectures including GPT (decoder-only) and BERT (encoder-only).

## Features

### Core (`core/`)
- **Tensor Operations**: Multi-dimensional tensor support (1D, 2D, 3D)
- **Math Operations**: Add, Mul, Softmax, GELU, Tanh, LayerNorm, etc.
- **Attention**: Scaled dot-product attention, causal masking
- **Initialization**: Xavier/Gaussian initialization support

### GPT (`gpt/`)
- **GPT-2 Model**: 105M+ parameters
- **Decoder Architecture**: Causal (unidirectional) self-attention
- **Components**: MultiHeadAttention, TransformerBlock, MLP, LayerNorm
- **Positional Encodings**: Sinusoidal, Learned, Rotary, ALiBi

### BERT (`bert/`)
- **BERT Models**: Base (110M), Large (340M)
- **Encoder Architecture**: Bidirectional self-attention
- **Components**: BERTAttention, BERTLayer, BERTEncoder, BERTPooler
- **Pre-training Heads**: MLM, NSP, Sequence Classification

### Tokenizer (`tokenizer/`)
- **BasicTokenizer**: Word-level tokenization
- **BPETokenizer**: Byte Pair Encoding
- **WordPieceTokenizer**: WordPiece subword tokenization

### Training (`training/`)
- **Optimizers**: AdamW, Adam, SGD with momentum
- **LR Schedulers**: Linear, Cosine, Constant with warmup
- **Utilities**: Metrics tracking, DataLoader, Trainer

## Installation

```bash
go get ./...
```

## Quick Start

```go
package main

import (
    "transformers/gpt"
    "transformers/bert"
    "transformers/tokenizer"
)

func main() {
    // Create GPT-2 model
    gptConfig := gpt.GPT2Config()
    gptModel := gpt.NewGPTModel(gptConfig)
    println("GPT-2 parameters:", gptModel.NumParameters())

    // Create BERT model
    bertConfig := bert.DefaultBERTConfig()
    bertModel := bert.NewBERTModel(bertConfig)

    // Use tokenizer
    tok := tokenizer.NewBasicTokenizer(1000)
    ids := tok.Encode("Hello world")
    text := tok.Decode(ids)
}
```

## Architecture

### GPT (Generative Pre-trained Transformer)

```
GPTModel
├── Token Embedding (Wte)
├── Positional Encoding (Wpe)
├── TransformerBlocks[] × N
│   ├── MultiHeadAttention
│   │   ├── Query (Wq)
│   │   ├── Key (Wk)
│   │   └── Value (Wv)
│   ├── LayerNorm
│   ├── MLP (GELU)
│   └── LayerNorm
├── Final LayerNorm
└── Language Model Head (LmHead)
```

### BERT (Bidirectional Encoder Representations)

```
BERTModel
├── BERTEmbeddings
│   ├── Word Embedding
│   ├── Position Embedding
│   ├── Token Type Embedding
│   └── LayerNorm
├── BERTEncoder
│   └── BERTLayers[] × N
│       ├── BERTAttention
│       │   ├── Self Attention
│       │   └── Self Output
│       ├── Intermediate (GELU)
│       └── Output
└── BERTPooler
```

## Configuration

### GPT-2 Variants

| Model | Hidden | Layers | Heads | Parameters |
|-------|--------|--------|-------|------------|
| GPT-2 | 768 | 12 | 12 | 105M |
| GPT-2 Medium | 1024 | 24 | 16 | 345M |
| GPT-Neo | 2048 | 32 | 16 | 2.7B |

### BERT Variants

| Model | Hidden | Layers | Heads | Parameters |
|-------|--------|--------|-------|------------|
| BERT-Base | 768 | 12 | 12 | 110M |
| BERT-Large | 1024 | 24 | 16 | 340M |

## Example: Training Loop

```go
// Create optimizer
params := []*training.AdamParam{
    {Data: weights, Grad: gradients},
}
optimizer := training.NewAdamW(params, 0.001, 0.9, 0.999, 1e-8, 0.01)

// Create scheduler
scheduler := training.NewScheduler(
    optimizer,
    training.SchedulerCosine,
    100,      // warmup steps
    10000,    // total steps
    0.001,    // base LR
)

// Training loop
for step := 0; step < maxSteps; step++ {
    // Forward pass
    logits := model.Forward(input)
    
    // Compute loss
    loss := computeLoss(logits, targets)
    
    // Backward pass (placeholder)
    // optimizer.Backward(loss)
    
    // Update
    optimizer.Step()
    scheduler.Step()
    
    // Log
    if step % logInterval == 0 {
        println("Step:", step, "Loss:", loss)
    }
}
```

## Demo

```bash
cd transformers && GOWORK=off go run main.go
```

## Performance

- Tensor operations: Optimized for CPU
- Memory efficient: Lazy tensor creation
- Configurable: Easy to customize model architectures

## Roadmap

- [x] Core tensor operations
- [x] GPT model
- [x] BERT model
- [x] Tokenizers
- [x] Training utilities
- [x] Full backward pass
- [ ] GPU support (requires CUDA/OpenCL bindings)
- [x] Quantization (INT8/INT4)
- [x] Model serialization (JSON/binary)

## References

- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et al., 2019)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)

## License

MIT
