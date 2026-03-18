package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"path"
	"strings"
)

type APIServer struct {
	registry *ModelRegistry
}

func NewAPIServer(registry *ModelRegistry) *APIServer {
	return &APIServer{registry: registry}
}

func (s *APIServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/api/models":
		s.listModels(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/api/models":
		s.createModel(w, r)
	case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/api/models/"):
		id := strings.TrimPrefix(r.URL.Path, "/api/models/")
		s.getModel(w, r, id)
	case r.Method == http.MethodDelete && strings.HasPrefix(r.URL.Path, "/api/models/"):
		id := strings.TrimPrefix(r.URL.Path, "/api/models/")
		s.deleteModel(w, r, id)
	case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/api/models/") && strings.Contains(r.URL.Path, "/versions"):
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/models/"), "/versions")
		modelID := parts[0]
		s.listVersions(w, r, modelID)
	case r.Method == http.MethodPost && strings.HasPrefix(r.URL.Path, "/api/models/") && strings.Contains(r.URL.Path, "/versions"):
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/models/"), "/versions")
		modelID := parts[0]
		s.registerVersion(w, r, modelID)
	case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/api/models/") && strings.Contains(r.URL.Path, "/versions/"):
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/models/"), "/versions/")
		modelID := parts[0]
		versionID := parts[1]
		s.getVersion(w, r, modelID, versionID)
	case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/api/models/") && strings.Contains(r.URL.Path, "/versions/"):
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/models/"), "/versions/")
		modelID := parts[0]
		versionID := parts[1]
		s.setStage(w, r, modelID, versionID)
	case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/api/models/") && strings.Contains(r.URL.Path, "/artifact"):
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/models/"), "/artifact")
		modelID := parts[0]
		versionID := strings.TrimPrefix(parts[1], "/")
		s.getArtifact(w, r, modelID, versionID)
	case r.Method == http.MethodGet && r.URL.Path == "/api/search":
		s.searchModels(w, r)
	case r.Method == http.MethodGet && r.URL.Path == "/api/stages/":
		stage := r.URL.Query().Get("stage")
		s.getByStage(w, r, stage)
	case r.Method == http.MethodGet && r.URL.Path == "/api/summary":
		s.getSummary(w, r)
	default:
		http.NotFound(w, r)
	}
}

func (s *APIServer) listModels(w http.ResponseWriter, r *http.Request) {
	models := s.registry.ListModels()
	json.NewEncoder(w).Encode(map[string]any{
		"models": models,
		"count":  len(models),
	})
}

func (s *APIServer) createModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name        string         `json:"name"`
		Description string         `json:"description"`
		Framework   string         `json:"framework"`
		Metadata    map[string]any `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	model, err := s.registry.CreateModel(req.Name, req.Description, req.Framework, req.Metadata)
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	json.NewEncoder(w).Encode(model)
}

func (s *APIServer) getModel(w http.ResponseWriter, r *http.Request, id string) {
	model, err := s.registry.GetModel(id)
	if err != nil {
		http.Error(w, err.Error(), 404)
		return
	}

	json.NewEncoder(w).Encode(model)
}

func (s *APIServer) deleteModel(w http.ResponseWriter, r *http.Request, id string) {
	if err := s.registry.DeleteModel(id); err != nil {
		http.Error(w, err.Error(), 404)
		return
	}

	json.NewEncoder(w).Encode(map[string]string{"status": "deleted"})
}

func (s *APIServer) listVersions(w http.ResponseWriter, r *http.Request, modelID string) {
	versions := s.registry.ListVersions(modelID)
	json.NewEncoder(w).Encode(map[string]any{
		"versions": versions,
		"count":    len(versions),
	})
}

func (s *APIServer) registerVersion(w http.ResponseWriter, r *http.Request, modelID string) {
	var req struct {
		Version       string             `json:"version"`
		Description   string             `json:"description"`
		CreatedBy     string             `json:"created_by"`
		RunID         string             `json:"run_id"`
		ParentVersion string             `json:"parent_version"`
		Metrics       map[string]float64 `json:"metrics"`
		Params        map[string]any     `json:"params"`
		Artifact      io.Reader          `json:"-"`
	}

	contentType := r.Header.Get("Content-Type")
	var artifactData []byte
	var err error

	if strings.Contains(contentType, "multipart/form-data") {
		file, _, err := r.FormFile("artifact")
		if err == nil {
			artifactData, _ = io.ReadAll(file)
			req.Artifact = file
		}
	}

	if artifactData == nil {
		artifactData, _ = io.ReadAll(r.Body)
	}

	if err := json.Unmarshal(artifactData, &req); err != nil {
		// Try as plain binary
		req.Artifact = strings.NewReader(string(artifactData))
		artifactData = []byte(fmt.Sprintf("model_data_%s", req.Version))
	}

	version, err := s.registry.RegisterVersion(
		modelID,
		req.Version,
		req.Description,
		req.CreatedBy,
		req.RunID,
		req.ParentVersion,
		artifactData,
		req.Metrics,
		req.Params,
	)

	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	json.NewEncoder(w).Encode(version)
}

func (s *APIServer) getVersion(w http.ResponseWriter, r *http.Request, modelID, versionID string) {
	version, err := s.registry.GetVersion(modelID, versionID)
	if err != nil {
		http.Error(w, err.Error(), 404)
		return
	}

	json.NewEncoder(w).Encode(version)
}

func (s *APIServer) setStage(w http.ResponseWriter, r *http.Request, modelID, versionID string) {
	var req struct {
		Stage string `json:"stage"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}

	version, err := s.registry.SetStage(modelID, versionID, ModelStage(req.Stage))
	if err != nil {
		http.Error(w, err.Error(), 500)
		return
	}

	json.NewEncoder(w).Encode(version)
}

func (s *APIServer) getArtifact(w http.ResponseWriter, r *http.Request, modelID, versionID string) {
	artifact, err := s.registry.GetArtifact(modelID, versionID)
	if err != nil {
		http.Error(w, err.Error(), 404)
		return
	}

	w.Header().Set("Content-Type", "application/octet-stream")
	w.Write(artifact)
}

func (s *APIServer) searchModels(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	models := s.registry.SearchModels(query)

	json.NewEncoder(w).Encode(map[string]any{
		"models": models,
		"query":  query,
		"count":  len(models),
	})
}

func (s *APIServer) getByStage(w http.ResponseWriter, r *http.Request, stage string) {
	versions := s.registry.GetByStage(ModelStage(stage))

	json.NewEncoder(w).Encode(map[string]any{
		"versions": versions,
		"stage":    stage,
		"count":    len(versions),
	})
}

func (s *APIServer) getSummary(w http.ResponseWriter, r *http.Request) {
	summary := s.registry.GetSummary()
	json.NewEncoder(w).Encode(summary)
}

func staticFileHandler(w http.ResponseWriter, r *http.Request) {
	p := path.Join("static", r.URL.Path)
	if p == "static" {
		p = "static/index.html"
	}
	http.ServeFile(w, r, p)
}
