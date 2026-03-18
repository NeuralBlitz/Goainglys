package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

type RegistryClient struct {
	baseURL string
}

func NewRegistryClient(baseURL string) *RegistryClient {
	if !strings.HasSuffix(baseURL, "/") {
		baseURL += "/"
	}
	return &RegistryClient{baseURL: baseURL}
}

type ModelMetadata struct {
	Name        string             `json:"name"`
	Framework   string             `json:"framework"`
	Description string             `json:"description"`
	Metrics     map[string]float64 `json:"metrics"`
	Params      map[string]any     `json:"params"`
}

func (c *RegistryClient) RegisterModel(metadata ModelMetadata) (string, error) {
	model := map[string]any{
		"name":        metadata.Name,
		"framework":   metadata.Framework,
		"description": metadata.Description,
	}

	data, _ := json.Marshal(model)
	fmt.Printf("POST %sapi/models: %s\n", c.baseURL, string(data))
	return "model_id_placeholder", nil
}

func (c *RegistryClient) RegisterVersion(modelID string, version string, weights []byte, metrics, params map[string]any) error {
	ver := map[string]any{
		"version":    version,
		"metrics":    metrics,
		"params":     params,
		"created_by": "ml-project",
	}
	data, _ := json.Marshal(ver)
	fmt.Printf("POST %sapi/models/%s/versions: %s\n", c.baseURL, modelID, string(data))
	fmt.Printf("   Artifact size: %d bytes\n", len(weights))
	return nil
}

func (c *RegistryClient) GetLatestVersion(modelName string) error {
	fmt.Printf("GET %sapi/models/%s/versions (latest)\n", c.baseURL, modelName)
	return nil
}

func (c *RegistryClient) SetStage(modelID, versionID, stage string) error {
	fmt.Printf("PUT %sapi/models/%s/versions/%s: stage=%s\n", c.baseURL, modelID, versionID, stage)
	return nil
}
