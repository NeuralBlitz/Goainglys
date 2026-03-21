# Goainglys — Native Go ML Platform

## Overview
A comprehensive collection of machine learning and deep learning implementations written in pure Go with no external dependencies.

## Web Application
The main web application is a unified portal located in `portal/` that runs on **port 5000** and combines:

- **Training Dashboard** — Real-time ML training metrics with live Chart.js charts (loss, accuracy)
- **Model Registry** — CRUD interface for managing ML models through lifecycle stages (REGISTERED → STAGING → PRODUCTION → ARCHIVED)
- **MCP Marketplace** — Browsable grid of AI agents/tools with search and category filtering

### Running the App
```bash
cd portal && GOWORK=off go run .
```

### Architecture
- `portal/main.go` — Go HTTP server with all REST APIs and business logic
- `portal/static/index.html` — Single-page application (vanilla JS + Chart.js)
- Server runs on `0.0.0.0:5000`

### API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dashboard/metrics` | Live training metrics (polled every 2s) |
| GET | `/api/models` | List all registered models |
| POST | `/api/models` | Register a new model |
| PUT | `/api/models/{id}/stage` | Promote/demote model stage |
| DELETE | `/api/models/{id}` | Delete a model |
| GET | `/api/marketplace/apps` | List marketplace apps (supports `?category=` and `?q=`) |
| GET | `/api/marketplace/stats` | Marketplace statistics |

## ML Library Modules
| Directory | Description |
|-----------|-------------|
| `tensor/` | Core tensor operations |
| `transformer/` | Full Transformer architecture (Attention Is All You Need) |
| `vector_db/` | HNSW-based vector similarity search |
| `asr/` | Automatic Speech Recognition with MFCC + LSTM |
| `rag_eval/` | RAG/LLM evaluation toolkit |
| `dashboard/` | Standalone training dashboard (port 8080) |
| `model_registry/` | Standalone model registry (port 8081) |
| `marketplace/` | Standalone MCP marketplace (port 8080) |
| `agents/` | AI agent framework |
| `finetune/` | Fine-tuning with LoRA support |
| `transformers/` | GPT/BERT implementations |

## Tech Stack
- **Backend**: Pure Go (no external dependencies)
- **Frontend**: Vanilla JS + Chart.js 4.x
- **Module**: Each subdirectory is an independent Go module
