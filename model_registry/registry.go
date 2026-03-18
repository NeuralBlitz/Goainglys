package main

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type ModelStage string

const (
	StageNone       ModelStage = ""
	StageRegistered ModelStage = "REGISTERED"
	StageStaging    ModelStage = "STAGING"
	StageProduction ModelStage = "PRODUCTION"
	StageArchived   ModelStage = "ARCHIVED"
	StageFailed     ModelStage = "FAILED"
)

type Model struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Framework   string         `json:"framework"`
	Metadata    map[string]any `json:"metadata"`
	CreatedAt   time.Time      `json:"created_at"`
	UpdatedAt   time.Time      `json:"updated_at"`
	Versions    []ModelVersion `json:"versions,omitempty"`
}

type ModelVersion struct {
	ID            string             `json:"id"`
	Version       string             `json:"version"`
	ModelID       string             `json:"model_id"`
	ModelName     string             `json:"model_name"`
	Stage         ModelStage         `json:"stage"`
	Metrics       map[string]float64 `json:"metrics,omitempty"`
	Params        map[string]any     `json:"params,omitempty"`
	ArtifactURI   string             `json:"artifact_uri"`
	FileSize      int64              `json:"file_size"`
	Status        string             `json:"status"`
	Description   string             `json:"description"`
	CreatedAt     time.Time          `json:"created_at"`
	UpdatedAt     time.Time          `json:"updated_at"`
	CreatedBy     string             `json:"created_by"`
	RunID         string             `json:"run_id,omitempty"`
	ParentVersion string             `json:"parent_version,omitempty"`
}

type ModelRegistry struct {
	models      map[string]*Model
	versions    map[string][]ModelVersion
	storagePath string
	mu          sync.RWMutex
}

func NewModelRegistry(storagePath string) *ModelRegistry {
	_ = os.MkdirAll(storagePath, 0755)
	_ = os.MkdirAll(filepath.Join(storagePath, "artifacts"), 0755)
	return &ModelRegistry{
		models:      make(map[string]*Model),
		versions:    make(map[string][]ModelVersion),
		storagePath: storagePath,
	}
}

func (r *ModelRegistry) CreateModel(name, description, framework string, metadata map[string]any) (*Model, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	id := generateID("model")
	model := &Model{
		ID:          id,
		Name:        name,
		Description: description,
		Framework:   framework,
		Metadata:    metadata,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	r.models[id] = model
	r.versions[id] = []ModelVersion{}

	r.saveToDisk(model)
	return model, nil
}

func (r *ModelRegistry) GetModel(id string) (*Model, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	model, ok := r.models[id]
	if !ok {
		for _, m := range r.models {
			if m.Name == id {
				return m, nil
			}
		}
		return nil, fmt.Errorf("model not found: %s", id)
	}

	result := *model
	result.Versions = r.versions[id]
	return &result, nil
}

func (r *ModelRegistry) ListModels() []Model {
	r.mu.RLock()
	defer r.mu.RUnlock()

	models := make([]Model, 0, len(r.models))
	for _, m := range r.models {
		mCopy := *m
		mCopy.Versions = nil
		models = append(models, mCopy)
	}

	sort.Slice(models, func(i, j int) bool {
		return models[i].UpdatedAt.After(models[j].UpdatedAt)
	})

	return models
}

func (r *ModelRegistry) RegisterVersion(
	modelID, version, description, createdBy, runID, parentVersion string,
	artifactData []byte,
	metrics map[string]float64,
	params map[string]any,
) (*ModelVersion, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	model, ok := r.models[modelID]
	if !ok {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	for _, v := range r.versions[modelID] {
		if v.Version == version {
			return nil, fmt.Errorf("version already exists: %s", version)
		}
	}

	versionID := generateID("v")
	artifactPath := filepath.Join(r.storagePath, "artifacts", fmt.Sprintf("%s_%s.bin", modelID, versionID))

	if err := os.WriteFile(artifactPath, artifactData, 0644); err != nil {
		return nil, fmt.Errorf("failed to save artifact: %w", err)
	}

	mv := ModelVersion{
		ID:            versionID,
		Version:       version,
		ModelID:       modelID,
		ModelName:     model.Name,
		Stage:         StageRegistered,
		Metrics:       metrics,
		Params:        params,
		ArtifactURI:   artifactPath,
		FileSize:      int64(len(artifactData)),
		Status:        "READY",
		Description:   description,
		CreatedAt:     time.Now(),
		UpdatedAt:     time.Now(),
		CreatedBy:     createdBy,
		RunID:         runID,
		ParentVersion: parentVersion,
	}

	r.versions[modelID] = append(r.versions[modelID], mv)
	model.UpdatedAt = time.Now()

	r.saveVersionToDisk(modelID, &mv)
	r.saveToDisk(model)

	return &mv, nil
}

func (r *ModelRegistry) GetVersion(modelID, versionID string) (*ModelVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, ok := r.versions[modelID]
	if !ok {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	for i, v := range versions {
		if v.ID == versionID || v.Version == versionID {
			return &versions[i], nil
		}
	}

	return nil, fmt.Errorf("version not found: %s", versionID)
}

func (r *ModelRegistry) ListVersions(modelID string) []ModelVersion {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions := make([]ModelVersion, len(r.versions[modelID]))
	copy(versions, r.versions[modelID])

	sort.Slice(versions, func(i, j int) bool {
		return versions[i].CreatedAt.After(versions[j].CreatedAt)
	})

	return versions
}

func (r *ModelRegistry) SetStage(modelID, versionID string, stage ModelStage) (*ModelVersion, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	versions, ok := r.versions[modelID]
	if !ok {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	for i, v := range versions {
		if v.ID == versionID || v.Version == versionID {
			versions[i].Stage = stage
			versions[i].UpdatedAt = time.Now()

			r.saveVersionToDisk(modelID, &versions[i])
			return &versions[i], nil
		}
	}

	return nil, fmt.Errorf("version not found: %s", versionID)
}

func (r *ModelRegistry) GetArtifact(modelID, versionID string) ([]byte, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, ok := r.versions[modelID]
	if !ok {
		return nil, fmt.Errorf("model not found: %s", modelID)
	}

	for _, v := range versions {
		if v.ID == versionID || v.Version == versionID {
			return os.ReadFile(v.ArtifactURI)
		}
	}

	return nil, fmt.Errorf("version not found: %s", versionID)
}

func (r *ModelRegistry) DeleteModel(modelID string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	model, ok := r.models[modelID]
	if !ok {
		return fmt.Errorf("model not found: %s", modelID)
	}

	for _, v := range r.versions[modelID] {
		os.Remove(v.ArtifactURI)
	}

	os.RemoveAll(filepath.Join(r.storagePath, "models", modelID))
	delete(r.versions, modelID)
	delete(r.models, modelID)

	_ = model
	return nil
}

func (r *ModelRegistry) SearchModels(query string) []Model {
	r.mu.RLock()
	defer r.mu.RUnlock()

	query = strings.ToLower(query)
	var results []Model

	for _, m := range r.models {
		if strings.Contains(strings.ToLower(m.Name), query) ||
			strings.Contains(strings.ToLower(m.Description), query) ||
			strings.Contains(strings.ToLower(m.Framework), query) {
			mCopy := *m
			mCopy.Versions = nil
			results = append(results, mCopy)
		}
	}

	return results
}

func (r *ModelRegistry) GetLatestVersion(modelID string) (*ModelVersion, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	versions, ok := r.versions[modelID]
	if !ok || len(versions) == 0 {
		return nil, fmt.Errorf("no versions found for model: %s", modelID)
	}

	latest := &versions[0]
	for i := range versions {
		if versions[i].CreatedAt.After(latest.CreatedAt) {
			latest = &versions[i]
		}
	}

	return latest, nil
}

func (r *ModelRegistry) GetByStage(stage ModelStage) []ModelVersion {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var results []ModelVersion
	for _, versions := range r.versions {
		for _, v := range versions {
			if v.Stage == stage {
				results = append(results, v)
			}
		}
	}

	return results
}

func (r *ModelRegistry) saveToDisk(model *Model) {
	path := filepath.Join(r.storagePath, "models", model.ID)
	_ = os.MkdirAll(path, 0755)

	data, _ := json.MarshalIndent(model, "", "  ")
	_ = os.WriteFile(filepath.Join(path, "model.json"), data, 0644)
}

func (r *ModelRegistry) saveVersionToDisk(modelID string, version *ModelVersion) {
	path := filepath.Join(r.storagePath, "models", modelID, "versions")
	_ = os.MkdirAll(path, 0755)

	data, _ := json.MarshalIndent(version, "", "  ")
	_ = os.WriteFile(filepath.Join(path, version.ID+".json"), data, 0644)
}

func (r *ModelRegistry) LoadFromDisk() error {
	modelsPath := filepath.Join(r.storagePath, "models")
	entries, err := os.ReadDir(modelsPath)
	if err != nil {
		return nil
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelFile := filepath.Join(modelsPath, entry.Name(), "model.json")
		data, err := os.ReadFile(modelFile)
		if err != nil {
			continue
		}

		var model Model
		if err := json.Unmarshal(data, &model); err != nil {
			continue
		}

		r.models[model.ID] = &model

		versionsPath := filepath.Join(modelsPath, entry.Name(), "versions")
		versionEntries, _ := os.ReadDir(versionsPath)
		var versions []ModelVersion
		for _, v := range versionEntries {
			vData, _ := os.ReadFile(filepath.Join(versionsPath, v.Name()))
			var mv ModelVersion
			if err := json.Unmarshal(vData, &mv); err == nil {
				versions = append(versions, mv)
			}
		}
		r.versions[model.ID] = versions
	}

	return nil
}

func generateID(prefix string) string {
	return fmt.Sprintf("%s_%d%d", prefix, time.Now().UnixNano(), rand.Intn(1000))
}

type ModelSummary struct {
	TotalModels    int            `json:"total_models"`
	TotalVersions  int            `json:"total_versions"`
	ByStage        map[string]int `json:"by_stage"`
	ByFramework    map[string]int `json:"by_framework"`
	TotalArtifacts int64          `json:"total_artifacts_bytes"`
}

func (r *ModelRegistry) GetSummary() ModelSummary {
	r.mu.RLock()
	defer r.mu.RUnlock()

	summary := ModelSummary{
		TotalModels: len(r.models),
		ByStage:     make(map[string]int),
		ByFramework: make(map[string]int),
	}

	for _, model := range r.models {
		summary.ByFramework[model.Framework]++
	}

	var totalVersions int
	for _, versions := range r.versions {
		totalVersions += len(versions)
		for _, v := range versions {
			summary.ByStage[string(v.Stage)]++
			summary.TotalArtifacts += v.FileSize
		}
	}
	summary.TotalVersions = totalVersions

	return summary
}
