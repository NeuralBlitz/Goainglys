package main

import (
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"
)

type RAGEvaluationDashboard struct {
	mu          sync.RWMutex
	evaluations []RAGEvaluationResult
	server      *http.Server
	port        int
}

type RAGEvaluationResult struct {
	ID           string                 `json:"id"`
	Timestamp    time.Time              `json:"timestamp"`
	Question     string                 `json:"question"`
	Retrieval    DashRetrievalMetrics   `json:"retrieval"`
	Generation   DashGenerationMetrics  `json:"generation"`
	Quality      DashQualityMetrics     `json:"quality"`
	Semantic     DashSemanticMetrics    `json:"semantic,omitempty"`
	OverallScore float64                `json:"overall_score"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

type DashRetrievalMetrics struct {
	Recall    float64 `json:"recall"`
	Precision float64 `json:"precision"`
	F1        float64 `json:"f1"`
	MRR       float64 `json:"mrr"`
	MAP       float64 `json:"map"`
	NDCG      float64 `json:"ndcg"`
}

type DashGenerationMetrics struct {
	BLEU1   float64 `json:"bleu1"`
	BLEU2   float64 `json:"bleu2"`
	BLEU3   float64 `json:"bleu3"`
	ROUGE1  float64 `json:"rouge1"`
	ROUGE2  float64 `json:"rouge2"`
	ROUGEL  float64 `json:"rougel"`
	Jaccard float64 `json:"jaccard"`
}

type DashQualityMetrics struct {
	ContextRelevancy float64 `json:"context_relevancy"`
	AnswerRelevancy  float64 `json:"answer_relevancy"`
	Faithfulness     float64 `json:"faithfulness"`
}

type DashSemanticMetrics struct {
	EmbeddingSimilarity float64 `json:"embedding_similarity"`
	HallucinationScore  float64 `json:"hallucination_score"`
	BiasScore           float64 `json:"bias_score"`
	FairnessScore       float64 `json:"fairness_score"`
}

type DashboardGenerationMetrics struct {
	BLEU1   float64 `json:"bleu1"`
	BLEU2   float64 `json:"bleu2"`
	BLEU3   float64 `json:"bleu3"`
	ROUGE1  float64 `json:"rouge1"`
	ROUGE2  float64 `json:"rouge2"`
	ROUGEL  float64 `json:"rougel"`
	Jaccard float64 `json:"jaccard"`
}

type DashboardRAGQualityMetrics struct {
	ContextRelevancy float64 `json:"context_relevancy"`
	AnswerRelevancy  float64 `json:"answer_relevancy"`
	Faithfulness     float64 `json:"faithfulness"`
}

type DashboardSemanticMetrics struct {
	EmbeddingSimilarity float64 `json:"embedding_similarity"`
	HallucinationScore  float64 `json:"hallucination_score"`
	BiasScore           float64 `json:"bias_score"`
	FairnessScore       float64 `json:"fairness_score"`
}

func NewRAGEvaluationDashboard(port int) *RAGEvaluationDashboard {
	return &RAGEvaluationDashboard{
		evaluations: make([]RAGEvaluationResult, 0),
		port:        port,
	}
}

func (d *RAGEvaluationDashboard) AddEvaluation(result RAGEvaluationResult) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if result.ID == "" {
		result.ID = fmt.Sprintf("eval_%d", len(d.evaluations))
	}
	if result.Timestamp.IsZero() {
		result.Timestamp = time.Now()
	}

	d.evaluations = append(d.evaluations, result)

	if len(d.evaluations) > 1000 {
		d.evaluations = d.evaluations[len(d.evaluations)-1000:]
	}
}

func (d *RAGEvaluationDashboard) GetSummary() map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if len(d.evaluations) == 0 {
		return map[string]interface{}{
			"total_evaluations": 0,
			"status":            "no_data",
		}
	}

	var totalRecall, totalPrecision, totalF1, totalMRR, totalMAP, totalNDCG float64
	var totalBLEU1, totalBLEU2, totalBLEU3, totalROUGE1, totalROUGE2, totalROUGEL, totalJaccard float64
	var totalContext, totalAnswer, totalFaithfulness float64
	var totalOverall float64

	for _, e := range d.evaluations {
		totalRecall += e.Retrieval.Recall
		totalPrecision += e.Retrieval.Precision
		totalF1 += e.Retrieval.F1
		totalMRR += e.Retrieval.MRR
		totalMAP += e.Retrieval.MAP
		totalNDCG += e.Retrieval.NDCG

		totalBLEU1 += e.Generation.BLEU1
		totalBLEU2 += e.Generation.BLEU2
		totalBLEU3 += e.Generation.BLEU3
		totalROUGE1 += e.Generation.ROUGE1
		totalROUGE2 += e.Generation.ROUGE2
		totalROUGEL += e.Generation.ROUGEL
		totalJaccard += e.Generation.Jaccard

		totalContext += e.Quality.ContextRelevancy
		totalAnswer += e.Quality.AnswerRelevancy
		totalFaithfulness += e.Quality.Faithfulness

		totalOverall += e.OverallScore
	}

	n := float64(len(d.evaluations))

	return map[string]interface{}{
		"total_evaluations": len(d.evaluations),
		"retrieval": map[string]float64{
			"avg_recall":    totalRecall / n,
			"avg_precision": totalPrecision / n,
			"avg_f1":        totalF1 / n,
			"avg_mrr":       totalMRR / n,
			"avg_map":       totalMAP / n,
			"avg_ndcg":      totalNDCG / n,
		},
		"generation": map[string]float64{
			"avg_bleu1":   totalBLEU1 / n,
			"avg_bleu2":   totalBLEU2 / n,
			"avg_bleu3":   totalBLEU3 / n,
			"avg_rouge1":  totalROUGE1 / n,
			"avg_rouge2":  totalROUGE2 / n,
			"avg_rougel":  totalROUGEL / n,
			"avg_jaccard": totalJaccard / n,
		},
		"quality": map[string]float64{
			"avg_context_relevancy": totalContext / n,
			"avg_answer_relevancy":  totalAnswer / n,
			"avg_faithfulness":      totalFaithfulness / n,
		},
		"overall_score":   totalOverall / n,
		"last_evaluation": d.evaluations[len(d.evaluations)-1].Timestamp,
	}
}

func (d *RAGEvaluationDashboard) GetRecentEvaluations(n int) []RAGEvaluationResult {
	d.mu.RLock()
	defer d.mu.RUnlock()

	if n > len(d.evaluations) {
		n = len(d.evaluations)
	}

	result := make([]RAGEvaluationResult, n)
	copy(result, d.evaluations[len(d.evaluations)-n:])
	return result
}

func (d *RAGEvaluationDashboard) GetTrendData(metric string, windowSize int) []float64 {
	d.mu.RLock()
	defer d.mu.RUnlock()

	evaluations := d.evaluations
	if len(evaluations) > 100 {
		evaluations = evaluations[len(evaluations)-100:]
	}

	data := make([]float64, 0, len(evaluations))

	for _, e := range evaluations {
		var value float64
		switch metric {
		case "recall":
			value = e.Retrieval.Recall
		case "precision":
			value = e.Retrieval.Precision
		case "f1":
			value = e.Retrieval.F1
		case "bleu1":
			value = e.Generation.BLEU1
		case "rouge1":
			value = e.Generation.ROUGE1
		case "overall":
			value = e.OverallScore
		default:
			value = e.OverallScore
		}
		data = append(data, value)
	}

	if windowSize <= 1 || len(data) <= windowSize {
		return data
	}

	smoothed := make([]float64, len(data))
	for i := range data {
		start := i - windowSize/2
		if start < 0 {
			start = 0
		}
		end := i + windowSize/2
		if end > len(data) {
			end = len(data)
		}

		var sum float64
		for j := start; j < end; j++ {
			sum += data[j]
		}
		smoothed[i] = sum / float64(end-start)
	}

	return smoothed
}

func (d *RAGEvaluationDashboard) GetHistogramData(metric string, bins int) []map[string]interface{} {
	d.mu.RLock()
	defer d.mu.RUnlock()

	values := make([]float64, len(d.evaluations))
	for i, e := range d.evaluations {
		switch metric {
		case "f1":
			values[i] = e.Retrieval.F1
		case "bleu1":
			values[i] = e.Generation.BLEU1
		case "rouge1":
			values[i] = e.Generation.ROUGE1
		default:
			values[i] = e.OverallScore
		}
	}

	if len(values) == 0 {
		return nil
	}

	minVal := values[0]
	maxVal := values[0]
	for _, v := range values[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	if maxVal == minVal {
		return []map[string]interface{}{
			{"range": fmt.Sprintf("%.2f", minVal), "count": len(values)},
		}
	}

	binWidth := (maxVal - minVal) / float64(bins)
	buckets := make([]int, bins)

	for _, v := range values {
		binIdx := int((v - minVal) / binWidth)
		if binIdx >= bins {
			binIdx = bins - 1
		}
		buckets[binIdx]++
	}

	result := make([]map[string]interface{}, bins)
	for i := range buckets {
		rangeStart := minVal + float64(i)*binWidth
		result[i] = map[string]interface{}{
			"range": fmt.Sprintf("%.2f-%.2f", rangeStart, rangeStart+binWidth),
			"count": buckets[i],
		}
	}

	return result
}

func (d *RAGEvaluationDashboard) GetPercentiles(metric string) map[string]float64 {
	d.mu.RLock()
	defer d.mu.RUnlock()

	values := make([]float64, len(d.evaluations))
	for i, e := range d.evaluations {
		switch metric {
		case "f1":
			values[i] = e.Retrieval.F1
		case "bleu1":
			values[i] = e.Generation.BLEU1
		default:
			values[i] = e.OverallScore
		}
	}

	sort.Float64s(values)

	if len(values) == 0 {
		return nil
	}

	getPercentile := func(p float64) float64 {
		idx := int(float64(len(values)-1) * p)
		if idx >= len(values) {
			idx = len(values) - 1
		}
		return values[idx]
	}

	return map[string]float64{
		"p50": getPercentile(0.50),
		"p75": getPercentile(0.75),
		"p90": getPercentile(0.90),
		"p95": getPercentile(0.95),
		"p99": getPercentile(0.99),
		"min": values[0],
		"max": values[len(values)-1],
		"mean": func() float64 {
			var sum float64
			for _, v := range values {
				sum += v
			}
			return sum / float64(len(values))
		}(),
	}
}

func (d *RAGEvaluationDashboard) Start() error {
	mux := http.NewServeMux()

	mux.HandleFunc("/api/summary", d.handleSummary)
	mux.HandleFunc("/api/evaluations", d.handleEvaluations)
	mux.HandleFunc("/api/trend", d.handleTrend)
	mux.HandleFunc("/api/histogram", d.handleHistogram)
	mux.HandleFunc("/api/percentiles", d.handlePercentiles)
	mux.HandleFunc("/api/add", d.handleAdd)
	mux.HandleFunc("/", d.handleIndex)

	d.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", d.port),
		Handler: mux,
	}

	return d.server.ListenAndServe()
}

func (d *RAGEvaluationDashboard) handleSummary(w http.ResponseWriter, r *http.Request) {
	summary := d.GetSummary()
	json.NewEncoder(w).Encode(summary)
}

func (d *RAGEvaluationDashboard) handleEvaluations(w http.ResponseWriter, r *http.Request) {
	evaluations := d.GetRecentEvaluations(50)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"evaluations": evaluations,
		"total":       len(d.evaluations),
	})
}

func (d *RAGEvaluationDashboard) handleTrend(w http.ResponseWriter, r *http.Request) {
	metric := r.URL.Query().Get("metric")
	if metric == "" {
		metric = "overall"
	}

	window := 5
	fmt.Sscanf(r.URL.Query().Get("window"), "%d", &window)

	data := d.GetTrendData(metric, window)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"metric": metric,
		"data":   data,
	})
}

func (d *RAGEvaluationDashboard) handleHistogram(w http.ResponseWriter, r *http.Request) {
	metric := r.URL.Query().Get("metric")
	if metric == "" {
		metric = "overall"
	}

	bins := 10
	fmt.Sscanf(r.URL.Query().Get("bins"), "%d", &bins)

	data := d.GetHistogramData(metric, bins)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"metric": metric,
		"bins":   bins,
		"data":   data,
	})
}

func (d *RAGEvaluationDashboard) handlePercentiles(w http.ResponseWriter, r *http.Request) {
	metric := r.URL.Query().Get("metric")
	if metric == "" {
		metric = "overall"
	}

	percentiles := d.GetPercentiles(metric)
	json.NewEncoder(w).Encode(percentiles)
}

func (d *RAGEvaluationDashboard) handleAdd(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var result RAGEvaluationResult
	if err := json.NewDecoder(r.Body).Decode(&result); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	d.AddEvaluation(result)
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (d *RAGEvaluationDashboard) handleIndex(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html")
	fmt.Fprint(w, getDashboardHTML())
}

func (d *RAGEvaluationDashboard) Stop() error {
	if d.server != nil {
		return d.server.Close()
	}
	return nil
}

func getDashboardHTML() string {
	return `<!DOCTYPE html>
<html>
<head>
    <title>RAG Evaluation Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { color: #38bdf8; margin-bottom: 20px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }
        .stat-card { background: #1e293b; border-radius: 12px; padding: 20px; }
        .stat-card h3 { font-size: 14px; color: #94a3b8; margin-bottom: 8px; }
        .stat-card .value { font-size: 28px; font-weight: bold; color: #38bdf8; }
        .charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .chart { background: #1e293b; border-radius: 12px; padding: 20px; }
        .chart h3 { margin-bottom: 15px; color: #94a3b8; }
        canvas { width: 100% !important; height: 200px !important; }
        table { width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 12px; overflow: hidden; }
        th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #334155; }
        th { background: #334155; color: #94a3b8; font-weight: 600; }
        tr:hover { background: #334155; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
        .badge-good { background: #166534; color: #4ade80; }
        .badge-warning { background: #854d0e; color: #facc15; }
        .badge-bad { background: #991b1b; color: #f87171; }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Evaluation Dashboard</h1>
        <div class="stats-grid" id="stats"></div>
        <div class="charts">
            <div class="chart"><h3>Overall Score Trend</h3><canvas id="trendChart"></canvas></div>
            <div class="chart"><h3>Score Distribution</h3><canvas id="histogramChart"></canvas></div>
        </div>
        <div class="chart"><h3>Recent Evaluations</h3><table id="evaluations"><thead><tr><th>ID</th><th>Question</th><th>F1</th><th>BLEU-1</th><th>ROUGE-1</th><th>Overall</th><th>Time</th></tr></thead><tbody></tbody></table></div>
    </div>
    <script>
        async function fetchSummary() {
            const res = await fetch('/api/summary');
            return res.json();
        }
        async function fetchTrend() {
            const res = await fetch('/api/trend?metric=overall&window=5');
            return res.json();
        }
        async function fetchHistogram() {
            const res = await fetch('/api/histogram?metric=overall&bins=10');
            return res.json();
        }
        async function fetchEvaluations() {
            const res = await fetch('/api/evaluations');
            return res.json();
        }
        function init() {
            fetchSummary().then(data => {
                document.getElementById('stats').innerHTML = ' +
                    '"<div class=\"stat-card\"><h3>Total Evaluations</h3><div class=\"value\">' + data.total_evaluations + '</div></div>' +
                    '<div class=\"stat-card\"><h3>Avg F1 Score</h3><div class=\"value\">' + (data.retrieval && data.retrieval.avg_f1 ? data.retrieval.avg_f1.toFixed(4) : 'N/A') + '</div></div>' +
                    '<div class=\"stat-card\"><h3>Avg BLEU-1</h3><div class=\"value\">' + (data.generation && data.generation.avg_bleu1 ? data.generation.avg_bleu1.toFixed(4) : 'N/A') + '</div></div>' +
                    '<div class=\"stat-card\"><h3>Avg ROUGE-1</h3><div class=\"value\">' + (data.generation && data.generation.avg_rouge1 ? data.generation.avg_rouge1.toFixed(4) : 'N/A') + '</div></div>' +
                    '<div class=\"stat-card\"><h3>Overall Score</h3><div class=\"value\">' + (data.overall_score ? data.overall_score.toFixed(4) : 'N/A') + '</div></div>";
            });
            setTimeout(init, 5000);
        }
        init();
    </script>
</body>
</html>`
}

func GenerateSampleEvaluations(n int) []RAGEvaluationResult {
	results := make([]RAGEvaluationResult, n)
	questions := []string{
		"What is machine learning?",
		"How does neural network work?",
		"What is transformers architecture?",
		"Explain backpropagation",
		"What is gradient descent?",
	}

	for i := 0; i < n; i++ {
		base := float64(i) / float64(n)

		results[i] = RAGEvaluationResult{
			ID:        fmt.Sprintf("eval_%d", i),
			Timestamp: time.Now().Add(-time.Duration(n-i) * time.Minute),
			Question:  questions[i%len(questions)],
			Retrieval: DashRetrievalMetrics{
				Recall:    0.6 + base*0.3 + math.Mod(float64(i)*0.01, 0.1),
				Precision: 0.7 + base*0.2 + math.Mod(float64(i)*0.01, 0.1),
				F1:        0.65 + base*0.25 + math.Mod(float64(i)*0.01, 0.1),
				MRR:       0.8 + base*0.15,
				MAP:       0.75 + base*0.2,
				NDCG:      0.7 + base*0.25,
			},
			Generation: DashGenerationMetrics{
				BLEU1:   0.4 + base*0.3 + math.Mod(float64(i)*0.02, 0.2),
				BLEU2:   0.3 + base*0.25 + math.Mod(float64(i)*0.02, 0.15),
				BLEU3:   0.2 + base*0.2 + math.Mod(float64(i)*0.02, 0.1),
				ROUGE1:  0.5 + base*0.3 + math.Mod(float64(i)*0.02, 0.15),
				ROUGE2:  0.3 + base*0.25 + math.Mod(float64(i)*0.02, 0.1),
				ROUGEL:  0.45 + base*0.25 + math.Mod(float64(i)*0.02, 0.1),
				Jaccard: 0.35 + base*0.3,
			},
			Quality: DashQualityMetrics{
				ContextRelevancy: 0.5 + base*0.4,
				AnswerRelevancy:  0.6 + base*0.3,
				Faithfulness:     0.4 + base*0.4,
			},
			OverallScore: 0.5 + base*0.35 + math.Mod(float64(i)*0.02, 0.15),
		}
	}

	return results
}
