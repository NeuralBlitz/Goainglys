package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

type SerializedModel struct {
	Type     string
	Weights  map[string][]float32
	Params   map[string]any
	Metadata map[string]string
}

func SaveModel(filename string, weights map[string][]float32, params map[string]any) error {
	model := SerializedModel{
		Type:     "tensor_model",
		Weights:  weights,
		Params:   params,
		Metadata: map[string]string{},
	}

	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	encoder := gob.NewEncoder(f)
	return encoder.Encode(model)
}

func LoadModel(filename string) (*SerializedModel, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var model SerializedModel
	decoder := gob.NewDecoder(f)
	if err := decoder.Decode(&model); err != nil {
		return nil, err
	}

	return &model, nil
}

func ExportToRegistry(registryPath string, modelName string, weights map[string][]float32, params map[string]any, metrics map[string]float64) error {
	_ = os.MkdirAll(registryPath, 0755)

	modelDir := filepath.Join(registryPath, modelName)
	_ = os.MkdirAll(modelDir, 0755)

	modelFile := filepath.Join(modelDir, "weights.gob")
	if err := SaveModel(modelFile, weights, params); err != nil {
		return err
	}

	metadata := map[string]any{
		"name":         modelName,
		"weights_file": modelFile,
		"params":       params,
		"metrics":      metrics,
	}

	metaFile := filepath.Join(modelDir, "metadata.json")
	data, _ := json.MarshalIndent(metadata, "", "  ")
	return os.WriteFile(metaFile, data, 0644)
}

func ImportFromRegistry(registryPath, modelName string) (*SerializedModel, map[string]any, error) {
	modelDir := filepath.Join(registryPath, modelName)
	weightsFile := filepath.Join(modelDir, "weights.gob")

	model, err := LoadModel(weightsFile)
	if err != nil {
		return nil, nil, err
	}

	metaFile := filepath.Join(modelDir, "metadata.json")
	metaData, err := os.ReadFile(metaFile)
	if err != nil {
		return model, nil, nil
	}

	var metadata map[string]any
	json.Unmarshal(metaData, &metadata)

	return model, metadata, nil
}

func ListModels(registryPath string) ([]string, error) {
	entries, err := os.ReadDir(registryPath)
	if err != nil {
		return nil, err
	}

	var models []string
	for _, entry := range entries {
		if entry.IsDir() {
			models = append(models, entry.Name())
		}
	}
	return models, nil
}

func init() {
	fmt.Println("Model I/O utilities loaded")
}
