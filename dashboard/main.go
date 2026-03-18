package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

type TrainingMetrics struct {
	Timestamp    time.Time `json:"timestamp"`
	Epoch        int       `json:"epoch"`
	Step         int       `json:"step"`
	Loss         float64   `json:"loss"`
	Accuracy     float64   `json:"accuracy"`
	LearningRate float64   `json:"learning_rate"`
	BatchSize    int       `json:"batch_size"`
	Duration     float64   `json:"duration"`
}

type ModelInfo struct {
	Name        string    `json:"name"`
	Type        string    `json:"type"`
	Params      int       `json:"params"`
	Status      string    `json:"status"`
	StartTime   time.Time `json:"start_time"`
	ElapsedTime float64   `json:"elapsed_time"`
}

type DashboardData struct {
	ModelInfo        ModelInfo         `json:"model_info"`
	Metrics          []TrainingMetrics `json:"metrics"`
	CurrentLoss      float64           `json:"current_loss"`
	CurrentAcc       float64           `json:"current_accuracy"`
	MinLoss          float64           `json:"min_loss"`
	MaxLoss          float64           `json:"max_loss"`
	AvgLoss          float64           `json:"avg_loss"`
	EstTimeRemaining float64           `json:"est_time_remaining"`
}

type MetricsCollector struct {
	metrics    []TrainingMetrics
	modelInfo  ModelInfo
	mu         sync.RWMutex
	maxMetrics int
}

func NewMetricsCollector(maxMetrics int) *MetricsCollector {
	return &MetricsCollector{
		metrics:    make([]TrainingMetrics, 0, maxMetrics),
		maxMetrics: maxMetrics,
		modelInfo: ModelInfo{
			Name:      "Transformer Model",
			Type:      "Transformer",
			Params:    12500000,
			Status:    "Training",
			StartTime: time.Now(),
		},
	}
}

func (mc *MetricsCollector) AddMetric(metric TrainingMetrics) {
	mc.mu.Lock()
	defer mc.mu.Unlock()
	mc.metrics = append(mc.metrics, metric)
	if len(mc.metrics) > mc.maxMetrics {
		mc.metrics = mc.metrics[1:]
	}
}

func (mc *MetricsCollector) GetMetrics() []TrainingMetrics {
	mc.mu.RLock()
	defer mc.mu.RUnlock()
	return append([]TrainingMetrics{}, mc.metrics...)
}

func (mc *MetricsCollector) GetDashboardData() DashboardData {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	data := DashboardData{
		ModelInfo: mc.modelInfo,
		Metrics:   mc.metrics,
	}

	if len(mc.metrics) > 0 {
		latest := mc.metrics[len(mc.metrics)-1]
		data.CurrentLoss = latest.Loss
		data.CurrentAcc = latest.Accuracy

		minLoss := math.MaxFloat64
		maxLoss := -math.MaxFloat64
		sumLoss := 0.0

		for _, m := range mc.metrics {
			if m.Loss < minLoss {
				minLoss = m.Loss
			}
			if m.Loss > maxLoss {
				maxLoss = m.Loss
			}
			sumLoss += m.Loss
		}

		data.MinLoss = minLoss
		data.MaxLoss = maxLoss
		data.AvgLoss = sumLoss / float64(len(mc.metrics))

		avgStepTime := 0.0
		if len(mc.metrics) > 1 {
			last := mc.metrics[len(mc.metrics)-1]
			first := mc.metrics[0]
			totalTime := last.Timestamp.Sub(first.Timestamp).Seconds()
			avgStepTime = totalTime / float64(len(mc.metrics)-1)
		}
		data.EstTimeRemaining = avgStepTime * 100
	}

	data.ModelInfo.ElapsedTime = time.Since(mc.modelInfo.StartTime).Seconds()

	return data
}

type DashboardServer struct {
	collector *MetricsCollector
}

func NewDashboardServer(collector *MetricsCollector) *DashboardServer {
	return &DashboardServer{collector: collector}
}

func (ds *DashboardServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	switch r.URL.Path {
	case "/", "/dashboard":
		ds.handleDashboard(w, r)
	case "/metrics":
		ds.handleMetrics(w, r)
	case "/api/data":
		ds.handleAPIData(w, r)
	default:
		http.NotFound(w, r)
	}
}

func generateLossChart(metrics []TrainingMetrics) string {
	if len(metrics) == 0 {
		return `<svg viewBox="0 0 600 250"><text x="300" y="125" text-anchor="middle" fill="#888">No data yet</text></svg>`
	}

	width, height := 600, 250
	padding := 50
	plotW := width - 2*padding
	plotH := height - 2*padding

	losses := make([]float64, len(metrics))
	for i, m := range metrics {
		losses[i] = m.Loss
	}
	minLoss := math.MaxFloat64
	maxLoss := -math.MaxFloat64
	for _, l := range losses {
		if l < minLoss {
			minLoss = l
		}
		if l > maxLoss {
			maxLoss = l
		}
	}
	if maxLoss == minLoss {
		maxLoss = minLoss + 1
	}

	path := ""
	step := float64(plotW) / float64(len(metrics)-1)
	if len(metrics) == 1 {
		step = 0
	}
	for i, m := range losses {
		x := float64(padding) + float64(i)*step
		y := float64(padding+plotH) - ((m-minLoss)/(maxLoss-minLoss))*float64(plotH)
		if i == 0 {
			path = fmt.Sprintf("M %.1f %.1f", x, y)
		} else {
			path += fmt.Sprintf(" L %.1f %.1f", x, y)
		}
	}

	svg := fmt.Sprintf(`<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">`, width, height)
	svg += fmt.Sprintf(`<rect width="100%%" height="100%%" fill="transparent"/>`)

	for i := 0; i <= 4; i++ {
		y := padding + (plotH*i)/4
		val := maxLoss - (maxLoss-minLoss)*float64(i)/4
		svg += fmt.Sprintf(`<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`, padding, y, width-padding, y)
		svg += fmt.Sprintf(`<text x="%d" y="%d" text-anchor="end" fill="#888" font-size="12">%.3f</text>`, padding-5, y+4, val)
	}

	svg += fmt.Sprintf(`<path d="%s" fill="none" stroke="#ff6b6b" stroke-width="2"/>`, path)
	svg += `</svg>`
	return svg
}

func generateAccChart(metrics []TrainingMetrics) string {
	if len(metrics) == 0 {
		return `<svg viewBox="0 0 600 250"><text x="300" y="125" text-anchor="middle" fill="#888">No data yet</text></svg>`
	}

	width, height := 600, 250
	padding := 50
	plotW := width - 2*padding
	plotH := height - 2*padding

	path := ""
	step := float64(plotW) / float64(len(metrics)-1)
	if len(metrics) == 1 {
		step = 0
	}
	for i, m := range metrics {
		x := float64(padding) + float64(i)*step
		y := float64(padding+plotH) - m.Accuracy*float64(plotH)
		if i == 0 {
			path = fmt.Sprintf("M %.1f %.1f", x, y)
		} else {
			path += fmt.Sprintf(" L %.1f %.1f", x, y)
		}
	}

	svg := fmt.Sprintf(`<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">`, width, height)
	svg += fmt.Sprintf(`<rect width="100%%" height="100%%" fill="transparent"/>`)

	for i := 0; i <= 4; i++ {
		y := padding + (plotH*i)/4
		val := 1.0 - float64(i)/4.0
		svg += fmt.Sprintf(`<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`, padding, y, width-padding, y)
		svg += fmt.Sprintf(`<text x="%d" y="%d" text-anchor="end" fill="#888" font-size="12">%.0f%%</text>`, padding-5, y+4, val*100)
	}

	svg += fmt.Sprintf(`<path d="%s" fill="none" stroke="#51cf66" stroke-width="2"/>`, path)
	svg += `</svg>`
	return svg
}

func generateLRChart() string {
	width, height := 1200, 150
	padding := 50
	plotW := width - 2*padding
	plotH := height - 2*padding

	warmupSteps := 4000.0
	maxLR := 0.0001

	path := ""
	for i := 0; i <= 100; i++ {
		x := float64(padding) + float64(i)/100.0*float64(plotW)
		step := float64(i) / 100.0 * 10000.0
		var lr float64
		if step < warmupSteps {
			lr = maxLR * (step / warmupSteps)
		} else {
			lr = maxLR * math.Sqrt(warmupSteps/step)
		}
		y := float64(padding+plotH) - (lr/maxLR)*float64(plotH)
		if i == 0 {
			path = fmt.Sprintf("M %.1f %.1f", x, y)
		} else {
			path += fmt.Sprintf(" L %.1f %.1f", x, y)
		}
	}

	svg := fmt.Sprintf(`<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg">`, width, height)
	svg += fmt.Sprintf(`<rect width="100%%" height="100%%" fill="transparent"/>`)

	for i := 0; i <= 4; i++ {
		y := padding + (plotH*i)/4
		svg += fmt.Sprintf(`<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="rgba(255,255,255,0.1)" stroke-width="1"/>`, padding, y, width-padding, y)
	}

	svg += fmt.Sprintf(`<path d="%s" fill="none" stroke="#00d4ff" stroke-width="2"/>`, path)
	svg += `</svg>`
	return svg
}

func (ds *DashboardServer) handleDashboard(w http.ResponseWriter, r *http.Request) {
	data := ds.collector.GetDashboardData()

	html := `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Training Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .header h1 {
            color: #00d4ff;
            font-size: 2.5em;
            text-shadow: 0 0 10px rgba(0,212,255,0.5);
        }
        .header .subtitle { color: #888; margin-top: 5px; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 15px 20px;
            text-align: center;
        }
        .stat-card .label { color: #888; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
        .stat-card .value { color: #00d4ff; font-size: 1.8em; font-weight: bold; }
        .chart-container {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .chart-title { color: #00d4ff; font-size: 1.2em; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid rgba(0,212,255,0.3); }
        .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .charts-row { grid-template-columns: 1fr; } }
        svg { width: 100%; height: 250px; }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; background: #51cf66; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
    <meta http-equiv="refresh" content="2">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Training Dashboard</h1>
            <div class="subtitle">Real-time ML Training Monitor</div>
        </div>

        <div class="status-bar">
            <div style="display:flex;align-items:center;gap:8px;">
                <div class="status-dot"></div>
                <span>Training in progress...</span>
            </div>
            <div>Elapsed: ` + fmt.Sprintf("%.1fs", data.ModelInfo.ElapsedTime) + `</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="label">Model</div>
                <div class="value" style="font-size: 1.2em">` + data.ModelInfo.Name + `</div>
            </div>
            <div class="stat-card">
                <div class="label">Params</div>
                <div class="value">` + fmt.Sprintf("%.1fM", float64(data.ModelInfo.Params)/1e6) + `</div>
            </div>
            <div class="stat-card">
                <div class="label">Current Loss</div>
                <div class="value">` + fmt.Sprintf("%.4f", data.CurrentLoss) + `</div>
            </div>
            <div class="stat-card">
                <div class="label">Best Loss</div>
                <div class="value">` + fmt.Sprintf("%.4f", data.MinLoss) + `</div>
            </div>
            <div class="stat-card">
                <div class="label">Accuracy</div>
                <div class="value">` + fmt.Sprintf("%.1f%%", data.CurrentAcc*100) + `</div>
            </div>
            <div class="stat-card">
                <div class="label">Epoch</div>
                <div class="value">` + data.ModelInfo.Status + `</div>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-container">
                <div class="chart-title">Loss Over Time</div>
                ` + generateLossChart(data.Metrics) + `
            </div>
            <div class="chart-container">
                <div class="chart-title">Accuracy Over Time</div>
                ` + generateAccChart(data.Metrics) + `
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-title">Learning Rate Schedule</div>
            ` + generateLRChart() + `
        </div>
    </div>
</body>
</html>`

	w.Header().Set("Content-Type", "text/html")
	w.Write([]byte(html))
}

func (ds *DashboardServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	metrics := ds.collector.GetMetrics()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

func (ds *DashboardServer) handleAPIData(w http.ResponseWriter, r *http.Request) {
	data := ds.collector.GetDashboardData()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(data)
}

func main() {
	collector := NewMetricsCollector(1000)
	server := NewDashboardServer(collector)
	fmt.Println("Dashboard server starting on http://localhost:8080")
	fmt.Println("Open your browser and navigate to the URL above")

	go func() {
		if err := http.ListenAndServe(":8080", server); err != nil {
			fmt.Printf("Server error: %v\n", err)
		}
	}()

	go func() {
		epoch := 1
		step := 0
		baseLoss := 2.0

		for {
			time.Sleep(100 * time.Millisecond)

			step++
			if step%100 == 0 {
				epoch++
				baseLoss *= 0.9
			}

			loss := baseLoss + (rand.Float64() * 0.1)
			accuracy := 0.1 + (2.0-loss)*0.4 + rand.Float64()*0.05

			metric := TrainingMetrics{
				Timestamp:    time.Now(),
				Epoch:        epoch,
				Step:         step,
				Loss:         math.Max(0, loss),
				Accuracy:     math.Min(1, math.Max(0, accuracy)),
				LearningRate: 0.0001,
				BatchSize:    32,
				Duration:     0.001,
			}

			collector.AddMetric(metric)
		}
	}()

	select {}
}
