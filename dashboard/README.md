# Training Dashboard

Real-time ML training monitoring dashboard built in pure Go.

## Features

- Real-time training metrics visualization
- SVG-based charts for loss, accuracy, and learning rate
- Auto-refreshing web interface (2-second refresh)
- Dark-themed modern UI
- No external dependencies (pure Go HTTP server)

## Quick Start

```bash
go run main.go
```

Then open http://localhost:8080 in your browser.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | Dashboard web interface |
| `/metrics` | JSON metrics data |
| `/api/data` | Full dashboard data as JSON |

## Architecture

- **MetricsCollector**: Thread-safe storage for training metrics
- **DashboardServer**: HTTP server with embedded HTML/CSS/SVG generation
- **Server-side rendering**: Charts are generated in Go, no JavaScript needed

## Integration

To integrate with your own ML projects, use the `MetricsCollector`:

```go
collector := NewMetricsCollector(1000)

// Add metrics during training
collector.AddMetric(TrainingMetrics{
    Timestamp:    time.Now(),
    Epoch:        epoch,
    Step:         step,
    Loss:         loss,
    Accuracy:     accuracy,
    LearningRate: lr,
    BatchSize:    32,
    Duration:     elapsed,
})

// Expose via HTTP
server := NewDashboardServer(collector)
http.ListenAndServe(":8080", server)
```
