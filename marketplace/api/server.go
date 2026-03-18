package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"marketplace/apps"
)

type MarketplaceAPI struct {
	registry *apps.AppRegistry
	mux      *http.ServeMux
}

func NewMarketplaceAPI(registry *apps.AppRegistry) *MarketplaceAPI {
	api := &MarketplaceAPI{
		registry: registry,
		mux:      http.NewServeMux(),
	}

	api.setupRoutes()

	return api
}

func (a *MarketplaceAPI) setupRoutes() {
	a.mux.HandleFunc("GET /api/apps", a.handleListApps)
	a.mux.HandleFunc("GET /api/apps/featured", a.handleFeaturedApps)
	a.mux.HandleFunc("GET /api/apps/trending", a.handleTrendingApps)
	a.mux.HandleFunc("GET /api/apps/search", a.handleSearchApps)
	a.mux.HandleFunc("GET /api/apps/categories", a.handleCategories)
	a.mux.HandleFunc("GET /api/apps/{id}", a.handleGetApp)
	a.mux.HandleFunc("POST /api/apps", a.handleRegisterApp)
	a.mux.HandleFunc("PUT /api/apps/{id}", a.handleUpdateApp)
	a.mux.HandleFunc("DELETE /api/apps/{id}", a.handleDeleteApp)
	a.mux.HandleFunc("GET /api/apps/{id}/reviews", a.handleGetReviews)
	a.mux.HandleFunc("POST /api/apps/{id}/reviews", a.handleAddReview)
	a.mux.HandleFunc("GET /api/stats", a.handleGetStats)
	a.mux.HandleFunc("GET /health", a.handleHealth)
}

func (a *MarketplaceAPI) Handler() http.Handler {
	return a.mux
}

func (a *MarketplaceAPI) handleListApps(w http.ResponseWriter, r *http.Request) {
	category := r.URL.Query().Get("category")
	author := r.URL.Query().Get("author")

	filter := &apps.AppFilter{
		Category: category,
		Author:   author,
	}

	appList := a.registry.ListApps(filter)

	sendJSON(w, map[string]any{
		"apps":  appList,
		"total": len(appList),
	})
}

func (a *MarketplaceAPI) handleFeaturedApps(w http.ResponseWriter, r *http.Request) {
	appList := a.registry.GetFeaturedApps(10)

	sendJSON(w, map[string]any{
		"apps": appList,
	})
}

func (a *MarketplaceAPI) handleTrendingApps(w http.ResponseWriter, r *http.Request) {
	appList := a.registry.GetTrendingApps(10)

	sendJSON(w, map[string]any{
		"apps": appList,
	})
}

func (a *MarketplaceAPI) handleSearchApps(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		http.Error(w, "query parameter 'q' is required", http.StatusBadRequest)
		return
	}

	appList := a.registry.SearchApps(query)

	sendJSON(w, map[string]any{
		"apps":  appList,
		"query": query,
		"total": len(appList),
	})
}

func (a *MarketplaceAPI) handleCategories(w http.ResponseWriter, r *http.Request) {
	categories := a.registry.GetCategories()

	sendJSON(w, map[string]any{
		"categories": categories,
	})
}

func (a *MarketplaceAPI) handleGetApp(w http.ResponseWriter, r *http.Request) {
	id := extractPathVar(r.URL.Path, "/api/apps/")

	app, ok := a.registry.GetApp(id)
	if !ok {
		http.Error(w, "app not found", http.StatusNotFound)
		return
	}

	sendJSON(w, app)
}

func (a *MarketplaceAPI) handleRegisterApp(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	app, err := apps.AppFromJSON(body)
	if err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if err := a.registry.RegisterApp(app); err != nil {
		http.Error(w, err.Error(), http.StatusConflict)
		return
	}

	w.WriteHeader(http.StatusCreated)
	sendJSON(w, app)
}

func (a *MarketplaceAPI) handleUpdateApp(w http.ResponseWriter, r *http.Request) {
	id := extractPathVar(r.URL.Path, "/api/apps/")

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	app, err := apps.AppFromJSON(body)
	if err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	if app.ID != id {
		http.Error(w, "app ID mismatch", http.StatusBadRequest)
		return
	}

	if err := a.registry.UpdateApp(app); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	sendJSON(w, app)
}

func (a *MarketplaceAPI) handleDeleteApp(w http.ResponseWriter, r *http.Request) {
	id := extractPathVar(r.URL.Path, "/api/apps/")

	if err := a.registry.DeleteApp(id); err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

func (a *MarketplaceAPI) handleGetReviews(w http.ResponseWriter, r *http.Request) {
	id := extractPathVar(r.URL.Path, "/api/apps/")

	reviews := a.registry.GetReviews(id)

	sendJSON(w, map[string]any{
		"reviews": reviews,
		"total":   len(reviews),
	})
}

func (a *MarketplaceAPI) handleAddReview(w http.ResponseWriter, r *http.Request) {
	id := extractPathVar(r.URL.Path, "/api/apps/")

	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}

	var review struct {
		Author  string  `json:"author"`
		Rating  float64 `json:"rating"`
		Title   string  `json:"title"`
		Content string  `json:"content"`
	}

	if err := json.Unmarshal(body, &review); err != nil {
		http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
		return
	}

	appReview := &apps.AppReview{
		AppID:   id,
		Author:  review.Author,
		Rating:  review.Rating,
		Title:   review.Title,
		Content: review.Content,
	}

	if err := a.registry.AddReview(appReview); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	w.WriteHeader(http.StatusCreated)
	sendJSON(w, appReview)
}

func (a *MarketplaceAPI) handleGetStats(w http.ResponseWriter, r *http.Request) {
	stats := a.registry.GetStats()

	sendJSON(w, stats)
}

func (a *MarketplaceAPI) handleHealth(w http.ResponseWriter, r *http.Request) {
	sendJSON(w, map[string]string{
		"status":  "healthy",
		"service": "marketplace-api",
	})
}

func sendJSON(w http.ResponseWriter, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

	if err := json.NewEncoder(w).Encode(data); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}

func extractPathVar(path, prefix string) string {
	var id string
	rest := strings.TrimPrefix(path, prefix)
	if idx := strings.Index(rest, "/"); idx > 0 {
		id = rest[:idx]
	} else {
		id = rest
	}
	return id
}

type ErrorResponse struct {
	Error   string `json:"error"`
	Code    int    `json:"code"`
	Message string `json:"message,omitempty"`
}

func sendError(w http.ResponseWriter, code int, message string) {
	sendJSON(w, ErrorResponse{
		Error:   http.StatusText(code),
		Code:    code,
		Message: message,
	})
}

type PaginatedResponse struct {
	Data       any `json:"data"`
	Page       int `json:"page"`
	PageSize   int `json:"pageSize"`
	TotalItems int `json:"totalItems"`
	TotalPages int `json:"totalPages"`
}

func sendPaginated(w http.ResponseWriter, data any, page, pageSize, total int) {
	totalPages := total / pageSize
	if total%pageSize > 0 {
		totalPages++
	}

	sendJSON(w, PaginatedResponse{
		Data:       data,
		Page:       page,
		PageSize:   pageSize,
		TotalItems: total,
		TotalPages: totalPages,
	})
}

type MarketplaceServer struct {
	api      *MarketplaceAPI
	server   *http.Server
	registry *apps.AppRegistry
}

func NewMarketplaceServer(addr string) *MarketplaceServer {
	registry := apps.NewAppRegistry()
	api := NewMarketplaceAPI(registry)

	return &MarketplaceServer{
		api:      api,
		registry: registry,
		server: &http.Server{
			Addr:    addr,
			Handler: api.Handler(),
		},
	}
}

func (s *MarketplaceServer) Start() error {
	fmt.Printf("Starting Marketplace API server on %s\n", s.server.Addr)
	return s.server.ListenAndServe()
}

func (s *MarketplaceServer) Stop(ctx context.Context) error {
	return s.server.Shutdown(ctx)
}

func (s *MarketplaceServer) GetRegistry() *apps.AppRegistry {
	return s.registry
}
