# Go Fine-tuning Architecture

Pure Go machine learning fine-tuning library with LoRA support.

## Features

- **Neural Network Layers**: Linear, Embedding, LayerNorm, Attention, Transformer
- **LoRA Support**: Low-rank adaptation for efficient fine-tuning
- **Optimizers**: AdamW, Adam, SGD, Lion, RMSprop
- **Learning Rate Schedulers**: Linear, Cosine, Constant with warmup
- **Training Loop**: Full training pipeline with evaluation, logging, and checkpoints

## Installation

```bash
go get ./...
```

## Quick Start

```go
cfg := config.DefaultConfig()
cfg.Model.VocabSize = 10000
cfg.Lora.Enabled = true
cfg.Lora.Rank = 8
cfg.Training.NumEpochs = 3
cfg.Training.BatchSize = 8

model := models.NewModelWrapper(*cfg)
trainer := training.NewTrainer(model, cfg, trainLoader, devLoader)
trainer.Train()
```

## Architecture

### Models (`models/`)

- `Tensor`: Multi-dimensional array with automatic differentiation support
- `Linear`: Fully connected layer
- `Embedding`: Token embedding layer
- `LayerNorm`: Layer normalization
- `Attention`: Multi-head self-attention
- `Transformer`: Transformer encoder/decoder
- `LoRA`: Low-rank adaptation layers
- `ModelWrapper`: Unified interface for base and LoRA models

### Optimizers (`optimizers/`)

| Optimizer | Description |
|-----------|-------------|
| AdamW | Adam with weight decay |
| Adam | Adam optimizer |
| SGD | Stochastic Gradient Descent with momentum |
| Lion | Linear Oracle oracles with exponential moving average |
| RMSprop | Root Mean Square Propagation |

### Training (`training/`)

- `Trainer`: Main training loop
- `Metrics`: Tracks loss, learning rate, timing
- `Scheduler`: Learning rate schedulers
- `Evaluator`: Model evaluation utilities

### Configuration (`config/`)

```go
type FineTuneConfig struct {
    Model        ModelConfig
    Training     TrainingConfig
    Data         DataConfig
    LoRA         LoRAConfig
    Optimization OptimizationConfig
}
```

### Data Handling (`data/`)

- `Dataset`: Tokenized training/evaluation data
- `DataLoader`: Batching and shuffling
- `Tokenizer`: Text tokenization interface

## LoRA Configuration

```go
cfg.LoRA = LoRAConfig{
    Enabled:       true,
    Rank:          8,
    Alpha:         16,
    Dropout:       0.1,
    TargetModules: []string{"q_proj", "v_proj"},
}
```

LoRA reduces trainable parameters by factor of `(r * 2 * hidden_size) / original_params`.

## Training Options

### Learning Rate Schedulers

```go
// Linear decay
cfg.Training.LRScheduler = config.SchedulerLinear

// Cosine annealing
cfg.Training.LRScheduler = config.SchedulerCosine

// Constant
cfg.Training.LRScheduler = config.SchedulerConstant
```

### Optimizers

```go
cfg.Optimization = OptimizationConfig{
    Optimizer:   config.OptimizerAdamW,
    Beta1:       0.9,
    Beta2:       0.999,
    Epsilon:     1e-8,
    Momentum:    0.9,
}
```

## Example

```go
package main

import (
    "fmt"
    "time"

    "finetune/config"
    "finetune/models"
    "finetune/optimizers"
    "finetune/training"
)

func main() {
    cfg := config.DefaultConfig()
    cfg.Lora.Enabled = true
    cfg.Lora.Rank = 4

    model := models.NewModelWrapper(*cfg)
    
    optimizer := optimizers.NewAdamW(1e-3, 0.9, 0.999, 1e-8, 0.01)
    for _, p := range model.Parameters() {
        optimizer.AddParams([]*optimizers.Tensor{{Data: p.Data, Shape: p.Shape}})
    }
    
    metrics := training.NewMetrics()
    metrics.AddTrainLoss(2.5)
    metrics.AddStepTime(time.Duration(100) * time.Millisecond)
    
    fmt.Printf("Average loss: %.4f\n", metrics.AverageTrainLoss())
}
```

## License

MIT
