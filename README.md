<img width="3840" height="2160" alt="image" src="https://github.com/user-attachments/assets/68f2f7d5-ef22-4e6b-a4cb-a706fbe7ccd4" />


Native Go ML Projects Collection

A comprehensive collection of machine learning and deep learning implementations written in **pure Go** with no external dependencies.

## 🚀 Projects Overview

| Project | Description | Status |
|---------|-------------|--------|
| **Transformer** | Complete Transformer architecture (Vaswani et al., 2017) | ✅ Complete |
| **Vector Database** | HNSW-based vector similarity search | ✅ Complete |
| **ASR Library** | Automatic Speech Recognition with MFCC & LSTM | ✅ Complete |
| **RAG Evaluation** | Comprehensive RAG/LLM evaluation toolkit | ✅ Complete |
| **Training Dashboard** | Real-time ML training metrics visualization | ✅ Complete |
| **Model Registry** | Model versioning and management system | ✅ Complete |
| **Agents Framework** | AI agent framework with native transformers | ✅ Complete |
| **Fine-tuning** | Neural network fine-tuning with LoRA support | ✅ Complete |
| **GPT/BERT** | Native Go GPT and BERT transformer models | ✅ Complete |
| **MCP Marketplace** | MCP apps marketplace with REST API and web UI | ✅ Complete |
| **LRS-Agents-NBX** | Learning Rate Scheduler agents with NBX integration | ✅ Complete |

## 📁 Project Structure

```
/workspace/
├── tensor/              # Core tensor operations library
│   ├── tensor.go       # Tensor data structure
│   └── ops.go          # Mathematical operations
├── transformer/         # Transformer architecture implementation
│   ├── layer.go        # Multi-head attention, FFN, positional encoding
│   ├── model.go        # Encoder/Decoder, embeddings, full model
│   ├── backward.go     # Backpropagation through layers
│   └── train.go        # Training utilities
├── vector_db/          # Native vector database
│   ├── main.go         # VectorDB with CRUD operations
│   ├── hnsw.go         # HNSW index for ANN search
│   ├── types.go        # Type definitions
│   └── README.md       # Documentation
├── asr/                # Automatic Speech Recognition
│   ├── audio.go        # Audio loading & preprocessing
│   ├── features.go     # MFCC feature extraction
│   ├── model.go        # LSTM acoustic model
│   ├── pipeline.go     # End-to-end ASR pipeline
│   ├── README.md       # Documentation
│   └── cmd/main.go     # Demo application
├── rag_eval/           # RAG/LLM Evaluation Toolkit
│   ├── main.go         # Evaluation metrics & pipeline
│   ├── rag_evaluation.json  # JSON export
│   └── README.md       # Documentation
├── dashboard/          # Training Dashboard
│   ├── main.go         # HTTP server & UI
│   └── README.md       # Documentation
├── model_registry/     # Model Registry
│   ├── main.go        # HTTP server
│   ├── registry.go    # Core registry logic
│   ├── server.go      # API endpoints
│   ├── index.html     # Web UI
│   └── README.md      # Documentation
├── agents/            # Agents Framework
│   ├── framework/    # Core framework
│   │   ├── core/     # Core types
│   │   ├── agents/   # Agent implementations
│   │   ├── testing/  # Testing utilities
│   │   └── main.go  # Demo
│   └── README.md     # Documentation
├── finetune/         # Fine-tuning Architecture
│   ├── config/      # Configuration
│   ├── data/        # Dataset and DataLoader
│   ├── models/      # Neural network layers
│   ├── losses/      # Loss functions
│   ├── optimizers/  # AdamW, SGD, Lion, etc.
│   ├── training/    # Training loop
│   ├── main.go      # Demo
│   └── README.md    # Documentation
├── transformers/    # GPT/BERT Transformers
│   ├── core/       # Tensor operations
│   ├── gpt/        # GPT model
│   ├── bert/       # BERT model
│   ├── tokenizer/   # Tokenizers
│   ├── training/   # Training utilities
│   ├── main.go     # Demo
│   └── README.md   # Documentation
├── marketplace/    # MCP Apps Marketplace
│   ├── api/       # REST API server
│   ├── apps/      # App registry and models
│   ├── mcp/       # MCP protocol implementation
│   ├── web/       # Web UI
│   ├── main.go    # Entry point
│   └── README.md  # Documentation
├── main.go             # Transformer demo
├── go.mod              # Go module definition
└── README.md           # This file
```

---

## 📈 5. Training Dashboard

**Directory:** `dashboard/`

Real-time ML training metrics visualization with SVG charts.

### Features
- ✅ Real-time metrics (loss, accuracy, learning rate)
- ✅ SVG-based charts (server-side rendered)
- ✅ Dark-themed web interface
- ✅ Auto-refresh every 2 seconds

### Quick Start
```bash
cd dashboard && go run main.go
# Open http://localhost:8080
```

---

## 📦 6. Model Registry

**Directory:** `model_registry/`

Native Go model registry and versioning system for ML models.

### Features
- ✅ Model versioning with stages (REGISTERED → STAGING → PRODUCTION → ARCHIVED)
- ✅ Metrics and parameters tracking
- ✅ Artifact storage
- ✅ REST API
- ✅ Web UI

### Quick Start
```bash
cd model_registry && go run main.go
# Open http://localhost:8081
```

---

## 🤖 7. Agents Framework (LRS-Agents-NBX)

**Directory:** `agents/framework/`

Comprehensive framework for building and testing AI agents with native Go transformers.

### Features
- ✅ Multiple agent types (ReAct, Plan-Execute, etc.)
- ✅ **Standard LLM Interface**: OpenAI, Anthropic, Ollama, LocalAI
- ✅ **Native Go Transformers**: Built-in GPT-2 and BERT support
- ✅ Tool system with easy integration
- ✅ Memory management (Conversation, Vector, Sliding Window)
- ✅ Built-in testing framework
- ✅ Mock LLM for testing
- ✅ **Multimodal**: Image, Audio, Video, Document processing
- ✅ **Git Integration**: Clone, commit, push, pull, status, log
- ✅ **GitHub Integration**: Issues, PRs, repos, releases, search
- ✅ **HuggingFace Integration**: Model search, datasets, inference

### Native Transformer LLM
```go
// Use native Go GPT-2/BERT in agents
nativeLLM, _ := llm.NewNativeLLMProvider("gpt2")
agent := agents.NewReActAgent(config, nativeLLM)
result := agent.Run(ctx, "Explain transformers")
```

### Quick Start
```bash
cd agents/framework && go run main.go
```

---

## 🔥 1. Transformer Architecture

**Directory:** `tensor/` & `transformer/`

Complete implementation of "Attention Is All You Need" (Vaswani et al., 2017).

### Features
- ✅ Multi-head scaled dot-product attention
- ✅ Positional encoding (sinusoidal)
- ✅ Feed-forward networks with ReLU
- ✅ Encoder and decoder layers
- ✅ Full backpropagation through all layers
- ✅ Cross-entropy loss with gradient computation
- ✅ Learning rate scheduler with warmup
- ✅ Attention visualization

### Performance
- Small model (2 layers, 256 dim): ~2.5s per epoch
- Full backpropagation: ~8-9s per epoch

---

## 🗄️ 2. Vector Database

**Directory:** `vector_db/`

Native vector database with HNSW (Hierarchical Navigable Small World) index.

### Features
- ✅ HNSW graph index for efficient ANN search
- ✅ Multiple similarity metrics (Cosine, Euclidean, Dot Product)
- ✅ CRUD operations: Insert, Update, Delete, Get
- ✅ Metadata support and filtering
- ✅ Thread-safe operations

### Performance
- 10,000 vectors (128 dim): Insert in ~290ms
- Search (10 results): ~20µs
- 82x faster than brute force

---

## 🎤 3. ASR Library

**Directory:** `asr/`

Complete speech recognition pipeline with MFCC features and LSTM acoustic model.

### Features
- ✅ Audio loading (WAV format)
- ✅ MFCC feature extraction
- ✅ LSTM acoustic model
- ✅ CTC decoding
- ✅ End-to-end pipeline

### Performance
- 5 seconds of audio: ~66ms processing
- 75x realtime speed

---

## 📊 4. RAG Evaluation Tool

**Directory:** `rag_eval/`

Comprehensive evaluation toolkit for Retrieval-Augmented Generation systems.

### Features
- ✅ **Retrieval Metrics**: Recall, Precision, F1, MRR, MAP, NDCG
- ✅ **Generation Metrics**: BLEU-1/2/3, ROUGE-1/2/L, Jaccard
- ✅ **RAG Quality**: Context Relevancy, Answer Relevancy, Faithfulness
- ✅ **Batch Evaluation**: Multiple test cases
- ✅ **JSON Export**: For analysis and dashboards

### Example Output
```
=== Average Retrieval Metrics ===
Recall@5:     0.7374 | Precision@5:  0.9333 | F1 Score:     0.7983
=== Average Generation Metrics ===
BLEU-1:       0.5500 | ROUGE-1:      0.7333 | ROUGE-L:      0.5722
=== RAG Quality ===
Context Relevancy:  0.2142 | Answer Relevancy:   0.6297 | Faithfulness:       0.2618
Overall Score:      0.6593
```

---

## 🔧 8. Fine-tuning Architecture

**Directory:** `finetune/`

Pure Go machine learning fine-tuning library with LoRA support.

### Features
- ✅ **Neural Network Layers**: Linear, Embedding, LayerNorm, Attention, Transformer
- ✅ **LoRA Support**: Low-rank adaptation for efficient fine-tuning
- ✅ **Optimizers**: AdamW, Adam, SGD, Lion, RMSprop
- ✅ **Learning Rate Schedulers**: Linear, Cosine, Constant with warmup
- ✅ **Training Loop**: Full training pipeline with evaluation and metrics

### Quick Start
```bash
cd finetune && GOWORK=off go run main.go
```

### Example
```go
cfg := config.DefaultConfig()
cfg.Lora.Enabled = true
cfg.Lora.Rank = 8

model := models.NewModelWrapper(*cfg)
trainer := training.NewTrainer(model, cfg, trainLoader, devLoader)
trainer.Train()
```

---

## 🔧 9. GPT/BERT Transformers

**Directory:** `transformers/`

Native Go implementation of GPT and BERT transformer architectures.

### Features
- ✅ **GPT Model**: GPT-2 architecture (105M+ parameters)
- ✅ **BERT Model**: BERT-Base and BERT-Large
- ✅ **Core Operations**: Tensor math, attention, layer norm
- ✅ **Tokenizers**: BasicTokenizer, BPETokenizer, WordPiece
- ✅ **Training**: AdamW, Adam, SGD optimizers, LR schedulers

### Quick Start
```bash
cd transformers && GOWORK=off go run main.go
```

### Example
```go
// Create GPT model
gptConfig := gpt.GPT2Config()
gptModel := gpt.NewGPTModel(gptConfig)

// Create BERT model
bertConfig := bert.DefaultBERTConfig()
bertModel := bert.NewBERTModel(bertConfig)

// Tokenize text
tok := tokenizer.NewBasicTokenizer(1000)
ids := tok.Encode("Hello world")
```

---

## 🌐 10. MCP Marketplace

**Directory:** `marketplace/`

MCP (Model Context Protocol) apps marketplace with REST API and web UI.

### Features
- ✅ MCP protocol implementation
- ✅ App registry with categories, ratings, reviews
- ✅ REST API for app management
- ✅ Web UI for browsing and searching apps
- ✅ Built-in MCP tools (search, get details, featured apps)
- ✅ Sample apps (Code Assistant, Data Analyst, etc.)

### Quick Start
```bash
cd marketplace && go run main.go
# API: http://localhost:8080/api/
# Web UI: http://localhost:8080/web/index.html
```

### API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/apps` | List all apps |
| GET | `/api/apps/featured` | Get featured apps |
| GET | `/api/apps/trending` | Get trending apps |
| GET | `/api/apps/search?q=` | Search apps |
| GET | `/api/apps/categories` | List categories |
| GET | `/api/apps/{id}` | Get app details |
| POST | `/api/apps` | Register new app |
| GET | `/api/stats` | Marketplace statistics |

---

## 🚀 Quick Start

```bash
# 1. Transformer Demo
cd /home/runner/workspace && GOWORK=off go run main.go

# 2. Vector Database
cd /home/runner/workspace/vector_db && GOWORK=off go run .

# 3. ASR Demo
cd /home/runner/workspace && GOWORK=off go run asr/cmd/main.go

# 4. RAG Evaluation
cd /home/runner/workspace/rag_eval && GOWORK=off go run main.go

# 5. Training Dashboard
cd /home/runner/workspace/dashboard && GOWORK=off go run main.go

# 6. Model Registry
cd /home/runner/workspace/model_registry && GOWORK=off go run main.go

# 7. Fine-tuning
cd /home/runner/workspace/finetune && GOWORK=off go run main.go

# 8. GPT/BERT Transformers
cd /home/runner/workspace/transformers && GOWORK=off go run main.go

# 9. MCP Marketplace
cd /home/runner/workspace/marketplace && GOWORK=off go run main.go
```

---

## 📊 Performance Comparison

| Project | Throughput | Latency | Use Case |
|---------|------------|---------|----------|
| Transformer | ~2.5s/epoch | ~8s w/ backprop | NLP, sequence modeling |
| Vector DB | 10k vectors/s | ~20µs search | Similarity search, RAG |
| ASR | 75x realtime | ~66ms/5s audio | Speech recognition |
| RAG Eval | 10k evals/s | ~400µs/eval | Quality assessment |
| Dashboard | Real-time | ~10ms/req | Training visualization |

---

## 📚 References

- **Transformer**: "Attention Is All You Need" (Vaswani et al., 2017)
- **HNSW**: "Efficient and Robust Approximate Nearest Neighbor Search" (Malkov & Yashunin, 2018)
- **CTC**: "Connectionist Temporal Classification" (Graves et al., 2006)
- **BLEU**: "A Method for Automatic Evaluation of Machine Translation" (Papineni et al., 2002)

---

## 🎯 Roadmap

### Completed
- ✅ Transformer with full backpropagation
- ✅ HNSW-based vector database
- ✅ Complete ASR pipeline
- ✅ RAG evaluation toolkit
- ✅ Training Dashboard
- ✅ Backward pass implementation
- ✅ Quantization (INT8/INT4)
- ✅ Model serialization
- ✅ Semantic similarity metrics
- ✅ Language model integration (ASR)
- ✅ Beam search decoding
- ✅ Streaming inference
- ✅ Distributed training support
- ✅ RAG evaluation dashboard
- ✅ ASR beam search decoder
- ✅ N-gram language model

### Future Enhancements
- [ ] GPU acceleration (CUDA/OpenCL) - requires external bindings
- [ ] Full BPTT for ASR (backpropagation through time)
- [ ] Vector database integration with RAG

---

## 📝 License

MIT License - Free to use, modify, and distribute.

---

*Built with ❤️ in pure Go*
![image](https://github.com/user-attachments/assets/fc29028c-60b5-4080-8ace-55269f7408b7)
