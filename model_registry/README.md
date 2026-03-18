# Model Registry

Native Go model registry and versioning system for ML models.

## Features

- **Model Management**: Create, list, search, and delete models
- **Version Control**: Register multiple versions per model
- **Stage Tracking**: Track models through REGISTERED → STAGING → PRODUCTION → ARCHIVED
- **Metadata Storage**: Store metrics, parameters, and custom metadata
- **Artifact Storage**: Store model files with versioning
- **REST API**: Full HTTP API for integration
- **Web UI**: Modern browser-based interface

## Quick Start

```bash
go run main.go
```

Open http://localhost:8081 in your browser.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List all models |
| POST | `/api/models` | Create model |
| GET | `/api/models/:id` | Get model details |
| DELETE | `/api/models/:id` | Delete model |
| GET | `/api/models/:id/versions` | List versions |
| POST | `/api/models/:id/versions` | Register new version |
| GET | `/api/models/:id/versions/:v` | Get version details |
| PUT | `/api/models/:id/versions/:v` | Set version stage |
| GET | `/api/models/:id/artifact/:v` | Download artifact |
| GET | `/api/search?q=query` | Search models |
| GET | `/api/stages?stage=STAGE` | Get models by stage |
| GET | `/api/summary` | Registry statistics |

## Example Usage

```bash
# Create a model
curl -X POST http://localhost:8081/api/models \
  -H "Content-Type: application/json" \
  -d '{"name": "transformer-base", "framework": "pytorch", "description": "Base transformer model"}'

# Register a version
curl -X POST http://localhost:8081/api/models/model_xxx/versions \
  -H "Content-Type: application/json" \
  -d '{
    "version": "1.0.0",
    "description": "Initial release",
    "created_by": "user@example.com",
    "metrics": {"accuracy": 0.92, "loss": 0.08},
    "params": {"lr": 0.001, "batch_size": 32}
  }'

# Set stage to production
curl -X PUT http://localhost:8081/api/models/model_xxx/versions/v_xxx \
  -H "Content-Type: application/json" \
  -d '{"stage": "PRODUCTION"}'

# List models
curl http://localhost:8081/api/models

# Search models
curl "http://localhost:8081/api/search?q=transformer"
```

## Architecture

- **Model**: Represents an ML model with metadata
- **ModelVersion**: Represents a specific version with metrics, params, and artifact
- **ModelRegistry**: Core service managing all models and versions
- **APIServer**: HTTP REST API
- **Storage**: File-based persistence in `data/` directory

## Model I/O Utilities

The `model_io.go` file provides utilities for saving and loading models:

```go
import "path/to/model_registry"

// Save model weights
weights := map[string][]float32{
    "encoder.layer0.weight": encoderWeights,
    "decoder.layer0.weight": decoderWeights,
}
params := map[string]any{"lr": 0.001, "batch_size": 32}
ExportToRegistry("./models", "transformer-v1", weights, params, metrics)

// Load model
model, metadata, _ := ImportFromRegistry("./models", "transformer-v1")

// List available models
models, _ := ListModels("./models")
```

## Integration with ML Projects

Connect to the registry server:

```go
client := NewRegistryClient("http://localhost:8081")

// Register a trained model
client.RegisterModel(ModelMetadata{
    Name:        "transformer-base",
    Framework:   "go",
    Description: "Transformer model v1",
})

client.RegisterVersion("model_id", "1.0.0", weights, metrics, params)
```
