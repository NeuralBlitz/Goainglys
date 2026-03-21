package main

import (
        "embed"
        "encoding/json"
        "fmt"
        "io/fs"
        "math"
        "math/rand"
        "net/http"
        "sort"
        "strings"
        "sync"
        "time"
)

//go:embed static
var staticFiles embed.FS

// ── Training Dashboard ──────────────────────────────────────────────

type TrainingMetric struct {
        Step     int     `json:"step"`
        Epoch    int     `json:"epoch"`
        Loss     float64 `json:"loss"`
        Accuracy float64 `json:"accuracy"`
        LR       float64 `json:"lr"`
        Ts       int64   `json:"ts"`
}

type MetricsStore struct {
        mu      sync.RWMutex
        metrics []TrainingMetric
        max     int
}

func NewMetricsStore() *MetricsStore { return &MetricsStore{max: 120} }

func (s *MetricsStore) Add(m TrainingMetric) {
        s.mu.Lock()
        defer s.mu.Unlock()
        s.metrics = append(s.metrics, m)
        if len(s.metrics) > s.max {
                s.metrics = s.metrics[1:]
        }
}

func (s *MetricsStore) GetAll() []TrainingMetric {
        s.mu.RLock()
        defer s.mu.RUnlock()
        out := make([]TrainingMetric, len(s.metrics))
        copy(out, s.metrics)
        return out
}

// ── Model Registry ──────────────────────────────────────────────────

type RegistryModel struct {
        ID          string    `json:"id"`
        Name        string    `json:"name"`
        Description string    `json:"description"`
        Framework   string    `json:"framework"`
        Stage       string    `json:"stage"`
        Accuracy    float64   `json:"accuracy"`
        Loss        float64   `json:"loss"`
        Parameters  int64     `json:"parameters"`
        CreatedAt   time.Time `json:"created_at"`
        UpdatedAt   time.Time `json:"updated_at"`
}

type ModelStore struct {
        mu     sync.RWMutex
        models map[string]*RegistryModel
        seq    int
}

func NewModelStore() *ModelStore {
        s := &ModelStore{models: make(map[string]*RegistryModel)}
        seeds := []RegistryModel{
                {Name: "Transformer-Base", Description: "Complete Transformer (Attention Is All You Need) architecture", Framework: "Go-Native", Stage: "PRODUCTION", Accuracy: 0.912, Loss: 0.241, Parameters: 12500000},
                {Name: "BERT-Large", Description: "BERT-Large bidirectional encoder for NLP classification tasks", Framework: "Go-Native", Stage: "STAGING", Accuracy: 0.883, Loss: 0.318, Parameters: 340000000},
                {Name: "GPT-Mini", Description: "Lightweight GPT-2 variant optimized for low-resource inference", Framework: "Go-Native", Stage: "REGISTERED", Accuracy: 0.791, Loss: 0.523, Parameters: 6000000},
                {Name: "LSTM-ASR", Description: "LSTM acoustic model for automatic speech recognition pipeline", Framework: "Go-Native", Stage: "PRODUCTION", Accuracy: 0.944, Loss: 0.178, Parameters: 2100000},
                {Name: "LoRA-FT-v2", Description: "Fine-tuned transformer with LoRA adapters for domain adaptation", Framework: "Go-Native", Stage: "STAGING", Accuracy: 0.921, Loss: 0.209, Parameters: 7800000},
        }
        for _, m := range seeds {
                s.seq++
                m.ID = fmt.Sprintf("model-%d", s.seq)
                m.CreatedAt = time.Now().Add(-time.Duration(rand.Intn(180*24)) * time.Hour)
                m.UpdatedAt = time.Now().Add(-time.Duration(rand.Intn(7*24)) * time.Hour)
                mc := m
                s.models[m.ID] = &mc
        }
        return s
}

func (s *ModelStore) List() []*RegistryModel {
        s.mu.RLock()
        defer s.mu.RUnlock()
        out := make([]*RegistryModel, 0, len(s.models))
        for _, m := range s.models {
                out = append(out, m)
        }
        sort.Slice(out, func(i, j int) bool { return out[i].UpdatedAt.After(out[j].UpdatedAt) })
        return out
}

func (s *ModelStore) Create(m *RegistryModel) *RegistryModel {
        s.mu.Lock()
        defer s.mu.Unlock()
        s.seq++
        m.ID = fmt.Sprintf("model-%d", s.seq)
        m.CreatedAt = time.Now()
        m.UpdatedAt = time.Now()
        if m.Stage == "" {
                m.Stage = "REGISTERED"
        }
        s.models[m.ID] = m
        return m
}

func (s *ModelStore) Delete(id string) bool {
        s.mu.Lock()
        defer s.mu.Unlock()
        if _, ok := s.models[id]; !ok {
                return false
        }
        delete(s.models, id)
        return true
}

func (s *ModelStore) UpdateStage(id, stage string) (*RegistryModel, bool) {
        s.mu.Lock()
        defer s.mu.Unlock()
        m, ok := s.models[id]
        if !ok {
                return nil, false
        }
        m.Stage = stage
        m.UpdatedAt = time.Now()
        return m, true
}

func (s *ModelStore) Summary() map[string]any {
        s.mu.RLock()
        defer s.mu.RUnlock()
        stages := map[string]int{}
        for _, m := range s.models {
                stages[m.Stage]++
        }
        return map[string]any{
                "total":      len(s.models),
                "by_stage":   stages,
        }
}

// ── Marketplace ─────────────────────────────────────────────────────

type MarketApp struct {
        ID          string   `json:"id"`
        Name        string   `json:"name"`
        Description string   `json:"description"`
        Author      string   `json:"author"`
        Category    string   `json:"category"`
        Tags        []string `json:"tags"`
        Rating      float64  `json:"rating"`
        Downloads   int64    `json:"downloads"`
        Version     string   `json:"version"`
        Icon        string   `json:"icon"`
}

type AppStore struct {
        mu   sync.RWMutex
        apps map[string]*MarketApp
}

func NewAppStore() *AppStore {
        s := &AppStore{apps: make(map[string]*MarketApp)}
        seeds := []MarketApp{
                {ID: "react-agent", Name: "ReAct Code Agent", Description: "Intelligent code generation using ReAct reasoning with multi-step planning and self-correction", Author: "AI Studio", Category: "Development", Tags: []string{"coding", "agent", "llm", "planning"}, Rating: 4.8, Downloads: 15240, Version: "1.2.0", Icon: "🤖"},
                {ID: "data-analyzer", Name: "Data Analyzer", Description: "Automated data analysis and visualization powered by native ML models with chart generation", Author: "DataLabs", Category: "Data Analysis", Tags: []string{"data", "analysis", "visualization", "statistics"}, Rating: 4.6, Downloads: 8530, Version: "2.1.0", Icon: "📊"},
                {ID: "research-asst", Name: "Research Assistant", Description: "AI-powered research for academic papers, literature review, citation extraction and summarization", Author: "ScholarAI", Category: "Research", Tags: []string{"research", "papers", "academic", "nlp"}, Rating: 4.9, Downloads: 22100, Version: "1.5.0", Icon: "🔬"},
                {ID: "email-asst", Name: "Email Assistant", Description: "Smart email management with automated drafting, triage, scheduling and context-aware replies", Author: "ProductivityPro", Category: "Productivity", Tags: []string{"email", "automation", "communication"}, Rating: 4.4, Downloads: 12080, Version: "1.0.0", Icon: "📧"},
                {ID: "git-agent", Name: "Git Agent", Description: "Automated Git operations, commit analysis, PR review assistance and repository management", Author: "DevTools Inc", Category: "Development", Tags: []string{"git", "devops", "automation", "vcs"}, Rating: 4.7, Downloads: 18340, Version: "1.3.0", Icon: "🌿"},
                {ID: "creative-writer", Name: "Creative Writer", Description: "AI creative writing for stories, articles, scripts and long-form content with style transfer", Author: "CreativeAI", Category: "Creative", Tags: []string{"writing", "creative", "content", "stories"}, Rating: 4.5, Downloads: 9520, Version: "2.0.0", Icon: "✍️"},
                {ID: "sql-agent", Name: "SQL Query Agent", Description: "Natural language to SQL conversion, query optimization and database schema exploration", Author: "DBPro", Category: "Data Analysis", Tags: []string{"sql", "database", "nlp", "query"}, Rating: 4.7, Downloads: 11040, Version: "1.1.0", Icon: "🗄️"},
                {ID: "doc-agent", Name: "Documentation Agent", Description: "Auto-generate READMEs, API docs and technical specifications directly from source code", Author: "DocuBot", Category: "Development", Tags: []string{"docs", "readme", "api", "codegen"}, Rating: 4.3, Downloads: 6810, Version: "1.0.2", Icon: "📝"},
                {ID: "test-agent", Name: "Test Generator", Description: "Automated unit and integration test generation with coverage analysis and edge case detection", Author: "QualityFirst", Category: "Development", Tags: []string{"testing", "qa", "automation", "coverage"}, Rating: 4.6, Downloads: 9200, Version: "1.4.0", Icon: "🧪"},
                {ID: "rag-eval", Name: "RAG Evaluator", Description: "Comprehensive RAG system evaluation with BLEU, ROUGE, faithfulness and relevancy metrics", Author: "EvalLabs", Category: "Research", Tags: []string{"rag", "evaluation", "metrics", "llm"}, Rating: 4.8, Downloads: 7650, Version: "1.0.0", Icon: "📐"},
        }
        for _, a := range seeds {
                ac := a
                s.apps[a.ID] = &ac
        }
        return s
}

func (s *AppStore) List(category, search string) []*MarketApp {
        s.mu.RLock()
        defer s.mu.RUnlock()
        out := make([]*MarketApp, 0, len(s.apps))
        for _, a := range s.apps {
                if category != "" && a.Category != category {
                        continue
                }
                if search != "" {
                        q := strings.ToLower(search)
                        if !strings.Contains(strings.ToLower(a.Name), q) &&
                                !strings.Contains(strings.ToLower(a.Description), q) &&
                                !strings.Contains(strings.ToLower(strings.Join(a.Tags, " ")), q) {
                                continue
                        }
                }
                out = append(out, a)
        }
        sort.Slice(out, func(i, j int) bool { return out[i].Downloads > out[j].Downloads })
        return out
}

func (s *AppStore) Stats() map[string]any {
        s.mu.RLock()
        defer s.mu.RUnlock()
        var totalDownloads int64
        var totalRating float64
        cats := map[string]int{}
        for _, a := range s.apps {
                totalDownloads += a.Downloads
                totalRating += a.Rating
                cats[a.Category]++
        }
        avg := 0.0
        if len(s.apps) > 0 {
                avg = totalRating / float64(len(s.apps))
        }
        return map[string]any{
                "total_apps":      len(s.apps),
                "total_downloads": totalDownloads,
                "average_rating":  math.Round(avg*100) / 100,
                "categories":      cats,
        }
}

// ── HTTP Server ─────────────────────────────────────────────────────

type Server struct {
        metrics *MetricsStore
        models  *ModelStore
        apps    *AppStore
}

func cors(h http.HandlerFunc) http.HandlerFunc {
        return func(w http.ResponseWriter, r *http.Request) {
                w.Header().Set("Access-Control-Allow-Origin", "*")
                w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
                w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
                if r.Method == http.MethodOptions {
                        w.WriteHeader(http.StatusNoContent)
                        return
                }
                h(w, r)
        }
}

func jsonResp(w http.ResponseWriter, status int, data any) {
        w.Header().Set("Content-Type", "application/json")
        w.WriteHeader(status)
        json.NewEncoder(w).Encode(data)
}

func (s *Server) routes() http.Handler {
        mux := http.NewServeMux()

        staticFS, _ := fs.Sub(staticFiles, "static")
        mux.Handle("/static/", http.StripPrefix("/static/", http.FileServer(http.FS(staticFS))))
        mux.HandleFunc("/", s.serveHTML)

        mux.HandleFunc("/api/dashboard/metrics", cors(s.handleMetrics))
        mux.HandleFunc("/api/models", cors(s.handleModels))
        mux.HandleFunc("/api/models/", cors(s.handleModel))
        mux.HandleFunc("/api/marketplace/apps", cors(s.handleApps))
        mux.HandleFunc("/api/marketplace/stats", cors(s.handleMarketStats))
        return mux
}

func (s *Server) serveHTML(w http.ResponseWriter, r *http.Request) {
        data, _ := staticFiles.ReadFile("static/index.html")
        w.Header().Set("Content-Type", "text/html; charset=utf-8")
        w.Write(data)
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
        jsonResp(w, http.StatusOK, s.metrics.GetAll())
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
        switch r.Method {
        case http.MethodGet:
                jsonResp(w, http.StatusOK, s.models.List())
        case http.MethodPost:
                var m RegistryModel
                if err := json.NewDecoder(r.Body).Decode(&m); err != nil {
                        jsonResp(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
                        return
                }
                jsonResp(w, http.StatusCreated, s.models.Create(&m))
        default:
                jsonResp(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
        }
}

func (s *Server) handleModel(w http.ResponseWriter, r *http.Request) {
        path := strings.TrimPrefix(r.URL.Path, "/api/models/")
        parts := strings.SplitN(path, "/", 2)
        id := parts[0]

        if len(parts) == 2 && parts[1] == "stage" && r.Method == http.MethodPut {
                var body struct {
                        Stage string `json:"stage"`
                }
                json.NewDecoder(r.Body).Decode(&body)
                m, ok := s.models.UpdateStage(id, body.Stage)
                if !ok {
                        jsonResp(w, http.StatusNotFound, map[string]string{"error": "not found"})
                        return
                }
                jsonResp(w, http.StatusOK, m)
                return
        }

        if r.Method == http.MethodDelete {
                if s.models.Delete(id) {
                        jsonResp(w, http.StatusOK, map[string]string{"status": "deleted"})
                } else {
                        jsonResp(w, http.StatusNotFound, map[string]string{"error": "not found"})
                }
                return
        }

        jsonResp(w, http.StatusMethodNotAllowed, map[string]string{"error": "method not allowed"})
}

func (s *Server) handleApps(w http.ResponseWriter, r *http.Request) {
        category := r.URL.Query().Get("category")
        search := r.URL.Query().Get("q")
        jsonResp(w, http.StatusOK, s.apps.List(category, search))
}

func (s *Server) handleMarketStats(w http.ResponseWriter, r *http.Request) {
        jsonResp(w, http.StatusOK, s.apps.Stats())
}

// ── Main ────────────────────────────────────────────────────────────

func main() {
        metricsStore := NewMetricsStore()

        go func() {
                step := 0
                epoch := 1
                baseLoss := 2.5
                for {
                        time.Sleep(150 * time.Millisecond)
                        step++
                        if step%80 == 0 {
                                epoch++
                                baseLoss *= 0.88
                        }
                        noise := (rand.Float64() - 0.5) * 0.18
                        loss := math.Max(0.05, baseLoss+noise)
                        acc := math.Min(0.99, math.Max(0, 0.38+(2.5-loss)*0.24+rand.Float64()*0.03))
                        metricsStore.Add(TrainingMetric{
                                Step:     step,
                                Epoch:    epoch,
                                Loss:     math.Round(loss*10000) / 10000,
                                Accuracy: math.Round(acc*10000) / 10000,
                                LR:       0.0001,
                                Ts:       time.Now().UnixMilli(),
                        })
                }
        }()

        srv := &Server{
                metrics: metricsStore,
                models:  NewModelStore(),
                apps:    NewAppStore(),
        }

        handler := srv.routes()

        go func() {
                fmt.Println("Goainglys ML Platform also listening on :8080")
                _ = http.ListenAndServe("0.0.0.0:8080", handler)
        }()

        fmt.Println("Goainglys ML Platform running on http://0.0.0.0:5000")
        if err := http.ListenAndServe("0.0.0.0:5000", handler); err != nil {
                fmt.Printf("Server error: %v\n", err)
        }
}
