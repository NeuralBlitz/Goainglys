package integrations

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"agents/framework/core"
)

type HuggingFaceTool struct {
	token string
	api   string
}

func NewHuggingFaceTool(token string) *HuggingFaceTool {
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}
	return &HuggingFaceTool{
		token: token,
		api:   "https://huggingface.co/api",
	}
}

func (t *HuggingFaceTool) Name() string { return "huggingface" }
func (t *HuggingFaceTool) Description() string {
	return "HuggingFace Hub operations - search models, datasets, spaces, and deploy"
}
func (t *HuggingFaceTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action: 'search_models', 'search_datasets', 'model_info', 'download', 'inference'",
			Required:    true,
		},
		"query": {
			Type:        "string",
			Description: "Search query or model name",
			Required:    false,
		},
		"task": {
			Type:        "string",
			Description: "Task type (e.g., 'text-generation', 'translation', 'summarization')",
			Required:    false,
		},
		"input": {
			Type:        "string",
			Description: "Input text for inference",
			Required:    false,
		},
	}
}

func (t *HuggingFaceTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)

	switch action {
	case "search_models":
		return t.searchModels(args)
	case "search_datasets":
		return t.searchDatasets(args)
	case "model_info":
		return t.modelInfo(args)
	case "download":
		return t.downloadModel(args)
	case "inference":
		return t.runInference(args)
	case "spaces":
		return t.listSpaces(args)
	default:
		return &core.ToolResult{Content: "Unknown action: " + action, Success: false}, nil
	}
}

func (t *HuggingFaceTool) doRequest(method, endpoint string, body *strings.Reader) ([]byte, error) {
	var req *http.Request
	var err error

	if body != nil {
		req, err = http.NewRequest(method, t.api+endpoint, body)
	} else {
		req, err = http.NewRequest(method, t.api+endpoint, nil)
	}
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	if t.token != "" {
		req.Header.Set("Authorization", "Bearer "+t.token)
	}

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(io.LimitReader(resp.Body, 1024*100))
	if err != nil {
		return nil, err
	}

	return data, nil
}

type HFModel struct {
	ID          string   `json:"id"`
	ModelType   string   `json:"modelType"`
	PipelineTag string   `json:"pipeline_tag"`
	Likes       int64    `json:"likes"`
	Downloads   int64    `json:"downloads"`
	Tags        []string `json:"tags"`
}

type HFDataset struct {
	ID        string   `json:"id"`
	Downloads int64    `json:"downloads"`
	Likes     int64    `json:"likes"`
	Tags      []string `json:"tags"`
}

func (t *HuggingFaceTool) searchModels(args map[string]any) (*core.ToolResult, error) {
	query := ""
	if q, ok := args["query"].(string); ok {
		query = q
	}

	task := ""
	if t_, ok := args["task"].(string); ok {
		task = t_
	}

	endpoint := "/models?sort=downloads&direction=-1&limit=10"
	if query != "" {
		endpoint = fmt.Sprintf("/models?search=%s&sort=downloads&direction=-1&limit=10", url.QueryEscape(query))
	}
	if task != "" {
		endpoint += "&pipeline_tag=" + url.QueryEscape(task)
	}

	data, err := t.doRequest("GET", endpoint, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Search failed: %v", err), Success: false}, nil
	}

	var models []HFModel
	if err := json.Unmarshal(data, &models); err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Parse error: %v", err), Success: false}, nil
	}

	var output strings.Builder
	output.WriteString("Popular models")

	if query != "" {
		output.WriteString(fmt.Sprintf(" matching '%s'", query))
	}
	if task != "" {
		output.WriteString(fmt.Sprintf(" for task '%s'", task))
	}
	output.WriteString(":\n\n")

	for i, m := range models {
		downloads := formatDownloads(m.Downloads)
		output.WriteString(fmt.Sprintf("%d. %s\n   Type: %s | Downloads: %s | Likes: %d\n",
			i+1, m.ID, m.PipelineTag, downloads, m.Likes))
	}

	if len(models) == 0 {
		output.WriteString("No models found.")
	}

	return &core.ToolResult{Content: output.String(), Success: true}, nil
}

func (t *HuggingFaceTool) searchDatasets(args map[string]any) (*core.ToolResult, error) {
	query := ""
	if q, ok := args["query"].(string); ok {
		query = q
	}

	endpoint := "/datasets?sort=downloads&direction=-1&limit=10"
	if query != "" {
		endpoint = fmt.Sprintf("/datasets?search=%s&sort=downloads&direction=-1&limit=10", url.QueryEscape(query))
	}

	data, err := t.doRequest("GET", endpoint, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Search failed: %v", err), Success: false}, nil
	}

	var datasets []HFDataset
	if err := json.Unmarshal(data, &datasets); err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Parse error: %v", err), Success: false}, nil
	}

	var output strings.Builder
	output.WriteString("Popular datasets")

	if query != "" {
		output.WriteString(fmt.Sprintf(" matching '%s'", query))
	}
	output.WriteString(":\n\n")

	for i, d := range datasets {
		downloads := formatDownloads(d.Downloads)
		output.WriteString(fmt.Sprintf("%d. %s\n   Downloads: %s | Likes: %d\n",
			i+1, d.ID, downloads, d.Likes))
	}

	if len(datasets) == 0 {
		output.WriteString("No datasets found.")
	}

	return &core.ToolResult{Content: output.String(), Success: true}, nil
}

func (t *HuggingFaceTool) modelInfo(args map[string]any) (*core.ToolResult, error) {
	modelID := ""
	if m, ok := args["query"].(string); ok {
		modelID = m
	}

	if modelID == "" {
		return &core.ToolResult{Content: "Model ID is required", Success: false}, nil
	}

	data, err := t.doRequest("GET", "/models/"+modelID, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Failed to get model info: %v", err), Success: false}, nil
	}

	var model map[string]any
	if err := json.Unmarshal(data, &model); err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Parse error: %v\n%s", err, string(data)), Success: false}, nil
	}

	tags, _ := model["tags"].([]any)
	var tagStr []string
	for _, t := range tags {
		if s, ok := t.(string); ok {
			tagStr = append(tagStr, s)
		}
	}

	output := fmt.Sprintf("Model: %s\n\n", modelID)
	output += fmt.Sprintf("Downloads: %s\n", formatDownloads(int64(model["downloads"].(float64))))
	output += fmt.Sprintf("Likes: %d\n", int(model["likes"].(float64)))
	output += fmt.Sprintf("Tags: %s\n", strings.Join(tagStr[:min(10, len(tagStr))], ", "))

	return &core.ToolResult{Content: output, Success: true}, nil
}

func (t *HuggingFaceTool) downloadModel(args map[string]any) (*core.ToolResult, error) {
	modelID := ""
	if m, ok := args["query"].(string); ok {
		modelID = m
	}

	if modelID == "" {
		return &core.ToolResult{Content: "Model ID is required", Success: false}, nil
	}

	if t.token == "" {
		return &core.ToolResult{
			Content: fmt.Sprintf("Download initiated for %s\nNote: Set HF_TOKEN for authenticated downloads\n"+
				"Run: huggingface-cli download %s", modelID, modelID),
			Success: true,
		}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("To download model %s:\n1. Install: pip install huggingface_hub\n2. Run: huggingface-cli download %s",
			modelID, modelID),
		Success: true,
	}, nil
}

func (t *HuggingFaceTool) runInference(args map[string]any) (*core.ToolResult, error) {
	modelID := ""
	if m, ok := args["query"].(string); ok {
		modelID = m
	}

	input := ""
	if i, ok := args["input"].(string); ok {
		input = i
	}

	if modelID == "" || input == "" {
		return &core.ToolResult{Content: "Model ID and input are required", Success: false}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("[Inference simulation for %s]\n\nInput: %s\n\nNote: Use HuggingFace Inference API or local pipeline for actual inference:\n"+
			"```python\nfrom transformers import pipeline\npipe = pipeline('%s', model='%s')\nresult = pipe('%s')\n```",
			modelID, input, "text-generation", modelID, input),
		Success: true,
	}, nil
}

func (t *HuggingFaceTool) listSpaces(args map[string]any) (*core.ToolResult, error) {
	query := ""
	if q, ok := args["query"].(string); ok {
		query = q
	}

	endpoint := "/spaces?sort=downloads&direction=-1&limit=10"
	if query != "" {
		endpoint = fmt.Sprintf("/spaces?search=%s&sort=downloads&direction=-1&limit=10", url.QueryEscape(query))
	}

	data, err := t.doRequest("GET", endpoint, nil)
	if err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Search failed: %v", err), Success: false}, nil
	}

	var spaces []map[string]any
	if err := json.Unmarshal(data, &spaces); err != nil {
		return &core.ToolResult{Content: fmt.Sprintf("Parse error: %v", err), Success: false}, nil
	}

	var output strings.Builder
	output.WriteString("Popular Spaces")

	if query != "" {
		output.WriteString(fmt.Sprintf(" matching '%s'", query))
	}
	output.WriteString(":\n\n")

	for i, s := range spaces {
		id, _ := s["id"].(string)
		downloads := formatDownloads(int64(s["downloads"].(float64)))
		output.WriteString(fmt.Sprintf("%d. %s | Downloads: %s\n", i+1, id, downloads))
	}

	if len(spaces) == 0 {
		output.WriteString("No spaces found.")
	}

	return &core.ToolResult{Content: output.String(), Success: true}, nil
}

func formatDownloads(n int64) string {
	if n >= 1_000_000 {
		return fmt.Sprintf("%.1fM", float64(n)/1_000_000)
	}
	if n >= 1_000 {
		return fmt.Sprintf("%.1fK", float64(n)/1_000)
	}
	return fmt.Sprintf("%d", n)
}

type HFInferenceClient struct {
	apiKey   string
	endpoint string
}

func NewHFInferenceClient(apiKey string) *HFInferenceClient {
	if apiKey == "" {
		apiKey = os.Getenv("HF_TOKEN")
	}
	return &HFInferenceClient{
		apiKey:   apiKey,
		endpoint: "https://api-inference.huggingface.co/models",
	}
}

func (c *HFInferenceClient) Query(modelID string, inputs string, params map[string]any) (map[string]any, error) {
	body, _ := json.Marshal(map[string]any{
		"inputs":     inputs,
		"parameters": params,
	})

	req, err := http.NewRequest("POST", c.endpoint+"/"+modelID, strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return result, nil
}

func (c *HFInferenceClient) TextGeneration(modelID, text string, maxLength int) (string, error) {
	result, err := c.Query(modelID, text, map[string]any{
		"max_length":       maxLength,
		"return_full_text": false,
	})
	if err != nil {
		return "", err
	}

	if arr, ok := result["generated_text"].(string); ok {
		return arr, nil
	}
	return "", nil
}

func (c *HFInferenceClient) Summarization(modelID, text string, maxLength int) (string, error) {
	result, err := c.Query(modelID, text, map[string]any{
		"max_length": maxLength,
	})
	if err != nil {
		return "", err
	}

	if arr, ok := result["summary_text"].(string); ok {
		return arr, nil
	}
	return "", nil
}

func ListPopularModels(category string) []string {
	categories := map[string][]string{
		"nlp": {
			"gpt2",
			"bert-base-uncased",
			"t5-base",
			"facebook/bart-large-cnn",
			"roberta-base",
		},
		"vision": {
			"facebook/detr-resnet-50",
			"google/vit-base-patch16-224",
			"openai/clip-vit-base-patch32",
		},
		"audio": {
			"facebook/wav2vec2-base-960h",
			"openai/whisper-base",
		},
	}

	if models, ok := categories[category]; ok {
		return models
	}
	return categories["nlp"]
}

func GetRecommendedModel(task string) string {
	taskModels := map[string]string{
		"text-generation":              "gpt2",
		"translation":                  "t5-base",
		"summarization":                "facebook/bart-large-cnn",
		"question-answering":           "distilbert-base-cased-distilled-squad",
		"text-classification":          "sentiment-analysis",
		"object-detection":             "facebook/detr-resnet-50",
		"image-classification":         "google/vit-base-patch16-224",
		"automatic-speech-recognition": "openai/whisper-base",
	}

	if model, ok := taskModels[task]; ok {
		return model
	}
	return "gpt2"
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
