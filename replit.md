# Goainglys — Pure Go ML Platform

> **Zero external dependencies.** Every line of code is Go. Every algorithm is hand-rolled.

---

## What is Goainglys?

Goainglys is a **production-grade ML platform written entirely in Go 1.21+**. It provides everything you need to train transformers, run vector search, fine-tune models, and deploy AI agents — without touching a single Python dependency.

Built for developers who want:
- **Full control** — no black-box tensor libraries, no opaque GPU drivers
- **Easy integration** — import any module independently; no monolithic framework
- **Real implementations** — backprop actually computes gradients, transformers actually train

---

## Quick Start

```bash
# Run the unified ML portal (dashboard + model registry + marketplace)
cd portal && GOWORK=off go run .   # → http://localhost:5000

# Train a character language model
cd portal && GOWORK=off go run . -mode=train -corpus=shakespeare -epochs=10

# Run the vector database with persistence
cd vector_db && GOWORK=off go run . -dim=128 -max=10000

# Explore the agent framework
cd agents/framework && GOWORK=off go run examples/chat.go

# Run benchmarks
cd benchmarks && GOWORK=off go run cmd/main.go -bench=tensor -bench=transformer

# Self-improving agent demo
cd self_improving_agents && GOWORK=off go run .
```

---

## Core Modules

### `tensor/` — Foundation Layer

| Feature | Description |
|---------|-------------|
| `MatMul` | Naive O(n^3) and blocked cache-aware multiplication |
| `Softmax` | Stable log-softmax with in-place support |
| `LayerNorm` | Mean-variance normalization with learnable gamma/beta |
| `GELU` | Gaussian Error Linear Unit approximation |
| `Dropout` | Stochastic masking for training |
| `Tensor` | 1D/2D float32/float64 with views and reshape |

### `transformer/` — Transformer Architecture

| Feature | Description |
|---------|-------------|
| **Forward pass** | Encoder-decoder with multi-head attention, FFN, layer norm, positional encoding |
| **Backward pass** | Full gradient computation through all layers (softmax, attention, FFN, layernorm) |
| **Training** | Built-in Adam optimizer with bias correction |
| **Caching** | `ActivationCache` stores intermediates during forward for backward pass |
| **Loss** | Cross-entropy with label smoothing |

Architecture: `src embeddings → reshape (batch*seq, dModel) → DecoderLayer x N → FFN → output projection`

### `asr/` — Speech Recognition

| Feature | Description |
|---------|-------------|
| `LSTMLayer` | Forward + backward pass with forget/input/output/gate gates |
| `DenseLayer` | Fully-connected with gradient computation |
| `CTCLoss` | Connectionist Temporal Classification loss |
| `StreamingDecoder` | Chunked CTC decoding for real-time inference |
| `MFCC` | Mel-frequency cepstral coefficients |
| `BeamSearchDecode` | Beam search with blank collapsing |

### `vector_db/` — Vector Similarity Search

| Feature | Description |
|---------|-------------|
| **HNSW** | Hierarchical Navigable Small World graphs for approximate nearest neighbors |
| **Persistence** | Binary WAL + snapshots with crash recovery |
| **WAL format** | `[1 byte op][8 byte seq][id][vector][metadata]` |
| **Snapshots** | Magic header ("VDBS") + version + sequence + all vectors |
| **Recovery** | Load latest snapshot, replay WAL entries where seq > snapshot.seq |

### `nn/` — Unified Building Blocks

| Feature | Description |
|---------|-------------|
| **Optimizers** | Adam, AdamW, SGD, RMSprop, Lion (float64) |
| **Schedulers** | Linear, Cosine, WarmupCosine, OneCycle, Cyclic, Polynomial |
| **Layers** | Linear, Embedding, LayerNorm, GELU, Dropout, Sequential |
| **Transformer LR** | `dModel^(-0.5) * min(step^(-0.5), step*warmup^(-1.5))` |
| **OptimizerManager** | Unified interface across all optimizers |

This module is the **canonical implementation** — other packages (`finetune`, `transformers`, `portal`) include adapters to wire in `nn.Optimizer`.

---

## Agent Framework (`agents/framework/`)

### Agent Types

| Agent | Strategy |
|-------|----------|
| **ReAct** | Think → Act → Observe loop with tool use |
| **PlanExecute** | Decompose task into numbered steps, execute sequentially |
| **Supervisor** | Decompose into subtasks, delegate to workers round-robin |
| **Crew** | Sequential collaboration with shared context accumulation |
| **Handoff** | Conditional agent transfer with max-handoff loop protection |

### Core Components

- `core/types.go` — `Agent`, `Tool`, `Message`, `BuildToolsDescription`
- `core/llm.go` — LLM interface with `Generate()`, `GenerateStructured()`
- `core/registry.go` — Tool registration and lookup
- `framework/` — Agent framework core with planning, handoff, execution tracing

---

## Fine-tuning (`finetune/`)

| Feature | Description |
|---------|-------------|
| **LoRA** | Low-Rank Adaptation with configurable rank, alpha, dropout |
| **Full fine-tuning** | Full parameter update path |
| **Distributed** | Ring all-reduce for multi-GPU training |
| **Optimizer adapters** | Wires into `nn/` optimizers |

---

## Research Modules (`Advanced-Research/`)

| Module | Description |
|--------|-------------|
| **Mixture of Experts** | Top-K routing, load balancing, parallel dispatch, backward pass |
| **Diffusion (DDPM)** | Forward noise schedule, DDIM sampling, UNet denoiser, time embedding |
| **RL Agent** | REINFORCE, PPO with clipped surrogate, ReplayBuffer, GAE |
| **GraphNN** | GraphSAGE, GAT (graph attention), message passing, readout pooling |

---

## Self-Improving Agents (`self_improving_agents/`)

Recursive improvement loop that detects weaknesses, generates targeted training data, fine-tunes, and repeats.

| Component | Description |
|-----------|-------------|
| **WeaknessDetector** | Error pattern analysis by task type (>20% error rate triggers detection) |
| **DataGenerator** | Math problem generation: arithmetic, algebra, reasoning, word problems, fractions, geometry |
| **MetaLearner** | Strategy effectiveness tracking, learning rate adjustment, diminishing returns detection |
| **ExperienceStore** | Replay buffer for successful improvement cycles |
| **ImprovementLoop** | Orchestrates the full cycle |

---

## Distributed Training (`opencode-lrs-agents-nbx/`)

| Component | Description |
|-----------|-------------|
| **LR Schedulers** | OneCycleLR, CyclicLR, CosineAnnealingWarmRestarts, PolynomialLR, ReduceLROnPlateau, AdaptiveLR |
| **Optimizers** | Prodigy, AdamScale, LARS, LAMB with trust-based merging |
| **NBX Protocol** | Neural Block Exchange with ring/star/mesh topologies |
| **CollaborativeOptimizer** | Trust-weighted gradient merging across workers |

---

## Benchmarks (`benchmarks/`)

| Suite | Metrics |
|-------|---------|
| **Tensor** | MatMul (naive/blocked), Softmax, LayerNorm, MLP backward — ops/sec, latency p50/p95/p99 |
| **Transformer** | Forward/backward/training, scaling (dModel, layers, batch) |
| **ASR** | MFCC extraction, LSTM forward, CTC greedy + beam decoding |
| **Vector DB** | HNSW insert QPS, search QPS, recall vs brute force |
| **Fine-tuning** | LoRA forward/backward at ranks 4/8/16/32, ring all-reduce |

Run: `go run cmd/main.go -bench=tensor -bench=transformer -report=markdown`

---

## Architecture Notes

### Module Independence

Every subdirectory is an **independent Go module** (`go.mod` + `go.sum`). Import only what you need:

```go
import "github.com/goainglys/tensor"
import "github.com/goainglys/transformer"
```

No circular dependencies. No monolithic import graph.

### Precision

| Module | Precision |
|--------|-----------|
| `tensor/` | float32 and float64 variants |
| `transformer/` | float32 (batch-major 2D reshape) |
| `nn/` | float64 (canonical implementations) |
| `asr/` | float64 |
| `portal/` | float64 |

### Gradient Flow (Transformer Example)

```
Forward: src/tgt embeddings → reshape(batch*seq, dModel)
              ↓
         DecoderLayer.Forward3D (passes 2D through)
              ↓
         SubLayer.ForwardCross (Q/K/V projections + attention)
              ↓
         MultiHeadAttention.ForwardCross → softmax → output
              ↓
         Cache stores: SrcEmbed, TgtEmbed, EncOutputs, DecOutputs, etc.

Backward: loss gradient → softmax backward → attention backward
              → cross-attention backward (routes to decoder input)
              → FFN backward → layer norm backward
              → residual gradient accumulation → param updates
```

### Persistence Format (Vector DB)

```
WAL entry:  [1 byte op][8 byte seq][4 byte idLen][id][4 byte dim][8*dim floats][4 byte metaLen][meta bytes]
Snapshot:   [4 byte magic "VDBS"][4 byte version][8 byte seqNum][4 byte count][vectors...]
Recovery:   Load latest snapshot → replay WAL entries where seq > snapshot.seq
```

---

## File Structure

```
/home/runner/workspace/
├── tensor/                   # Core tensor ops (no deps)
├── transformer/               # Transformer forward + backward + train
├── asr/                      # LSTM + CTC speech recognition
├── nn/                       # Unified layers, optimizers, schedulers
├── vector_db/                 # HNSW vector DB with WAL + snapshots
├── finetune/                  # LoRA + full fine-tuning
├── transformers/              # GPT/BERT implementations
├── portal/                   # Web dashboard + model registry + marketplace
├── agents/framework/          # ReAct, PlanExecute, Supervisor, Crew, Handoff
├── self_improving_agents/     # Weakness detection + data gen + meta-learning
├── Advanced-Research/         # MoE, Diffusion, RL, GraphNN
├── opencode-lrs-agents-nbx/   # LR schedulers + NBX distributed training
├── benchmarks/                # Performance benchmark suites
├── rag_eval/                  # RAG/LLM evaluation
├── dashboard/                 # Standalone training dashboard
├── model_registry/            # Standalone model registry
└── marketplace/               # Standalone MCP marketplace
```

---

## Development

```bash
# Build all modules
for dir in tensor transformer asr nn finetune transformers portal vector_db; do
  (cd $dir && GOWORK=off go build ./...)
done

# Run tests (where they exist)
cd transformer && GOWORK=off go test ./...

# Check for LSP errors
GOWORK=off go build ./...
```

---

## Contributing

All implementations must:

- Compile with `GOWORK=off go build ./...`
- Use only Go standard library (no external dependencies)
- Include meaningful gradient computation (no stub-only returns)
- Follow existing code conventions in the module

When adding new features to existing modules, maintain backward compatibility. When unifying duplicate implementations, preserve the interface contracts.
