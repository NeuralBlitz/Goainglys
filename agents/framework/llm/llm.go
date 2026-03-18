package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"agents/framework/core"
)

type Provider string

const (
	ProviderOpenAI    Provider = "openai"
	ProviderAnthropic Provider = "anthropic"
	ProviderOllama    Provider = "ollama"
	ProviderLocalAI   Provider = "localai"
	ProviderMock      Provider = "mock"
)

type Config struct {
	Provider    Provider
	Model       string
	APIKey      string
	BaseURL     string
	Temperature float64
	MaxTokens   int
	Timeout     time.Duration
}

type LLM interface {
	Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error)
	GenerateWithMessages(ctx context.Context, messages []core.Message, opts ...Option) (*Response, error)
	Embeddings(ctx context.Context, texts []string) ([]float32, error)
	Name() string
	Provider() Provider
}

type Response struct {
	Content      string
	StopReason   core.StopReason
	TokensUsed   int
	FinishReason string
	Model        string
	ToolCalls    []core.ToolCall
	RawResponse  any
}

type Option func(*options)

type options struct {
	model       string
	temperature float64
	maxTokens   int
	stop        []string
	tools       [][]core.Parameter
	system      string
}

func WithModel(model string) Option {
	return func(o *options) { o.model = model }
}

func WithTemperature(temp float64) Option {
	return func(o *options) { o.temperature = temp }
}

func WithMaxTokens(tokens int) Option {
	return func(o *options) { o.maxTokens = tokens }
}

func WithStop(stop []string) Option {
	return func(o *options) { o.stop = stop }
}

func WithTools(tools []core.Tool) Option {
	return func(o *options) {
		for _, t := range tools {
			params := make([]core.Parameter, 0)
			for name, p := range t.Parameters() {
				params = append(params, core.Parameter{
					Type:        p.Type,
					Description: p.Description,
					Required:    p.Required,
				})
				_ = name
			}
			o.tools = append(o.tools, params)
		}
	}
}

func WithSystem(system string) Option {
	return func(o *options) { o.system = system }
}

func NewLLM(config Config) (LLM, error) {
	config.APIKey = getEnvOrDefault("LLM_API_KEY", config.APIKey)
	config.BaseURL = getEnvOrDefault("LLM_BASE_URL", config.BaseURL)

	switch config.Provider {
	case ProviderOpenAI:
		return NewOpenAI(config)
	case ProviderAnthropic:
		return NewAnthropic(config)
	case ProviderOllama:
		return NewOllama(config)
	case ProviderLocalAI:
		return NewLocalAI(config)
	case ProviderMock:
		return NewMockLLM(), nil
	default:
		return NewOpenAI(config)
	}
}

func getEnvOrDefault(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

type OpenAIClient struct {
	config     Config
	httpClient *http.Client
}

func NewOpenAI(config Config) (*OpenAIClient, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	return &OpenAIClient{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}, nil
}

func (c *OpenAIClient) Name() string       { return "openai" }
func (c *OpenAIClient) Provider() Provider { return ProviderOpenAI }

func (c *OpenAIClient) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	messages := []map[string]any{
		{"role": "user", "content": prompt},
	}
	return c.GenerateWithMessages(ctx, messagesToCoreMessages(messages), opts...)
}

func (c *OpenAIClient) GenerateWithMessages(ctx context.Context, messages []core.Message, opts ...Option) (*Response, error) {
	o := applyOptions(c.config, opts)

	url := c.config.BaseURL + "/chat/completions"
	if c.config.BaseURL == "" {
		url = "https://api.openai.com/v1/chat/completions"
	}

	openAIMessages := make([]map[string]any, 0)
	if o.system != "" {
		openAIMessages = append(openAIMessages, map[string]any{"role": "system", "content": o.system})
	}
	for _, msg := range messages {
		role := string(msg.Role)
		if role == "agent" {
			role = "assistant"
		}
		msgMap := map[string]any{"role": role, "content": msg.Content}

		if len(msg.ToolCalls) > 0 {
			toolCalls := make([]map[string]any, 0)
			for _, tc := range msg.ToolCalls {
				toolCalls = append(toolCalls, map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Name,
						"arguments": fmt.Sprintf("%v", tc.Args),
					},
				})
			}
			msgMap["tool_calls"] = toolCalls
		}

		if msg.ToolResult != nil {
			msgMap["tool_call_id"] = msg.ToolResult.ToolCallID
			msgMap["content"] = msg.ToolResult.Content
		}

		openAIMessages = append(openAIMessages, msgMap)
	}

	reqBody := map[string]any{
		"model":       getModel(c.config.Model, o.model),
		"messages":    openAIMessages,
		"temperature": o.temperature,
	}

	if o.maxTokens > 0 {
		reqBody["max_tokens"] = o.maxTokens
	}

	if len(o.stop) > 0 {
		reqBody["stop"] = o.stop
	}

	if len(o.tools) > 0 {
		tools := make([]map[string]any, 0)
		for _, toolParams := range o.tools {
			props := make(map[string]any)
			for _, p := range toolParams {
				props[p.Description] = map[string]any{
					"type":        p.Type,
					"description": p.Description,
				}
			}
			tools = append(tools, map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        "function",
					"description": "A function",
					"parameters": map[string]any{
						"type":       "object",
						"properties": props,
					},
				},
			})
		}
		reqBody["tools"] = tools
	}

	body, _ := json.Marshal(reqBody)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("OpenAI API error: %s", string(respBody))
	}

	var result map[string]any
	json.Unmarshal(respBody, &result)

	choices := result["choices"].([]any)
	choice := choices[0].(map[string]any)

	content := ""
	finishReason := ""

	if choice["message"] != nil {
		msg := choice["message"].(map[string]any)
		content, _ = msg["content"].(string)
		finishReason, _ = choice["finish_reason"].(string)
	}

	usage := result["usage"].(map[string]any)
	promptTokens := int(usage["prompt_tokens"].(float64))
	completionTokens := int(usage["completion_tokens"].(float64))

	return &Response{
		Content:      content,
		TokensUsed:   promptTokens + completionTokens,
		FinishReason: finishReason,
		Model:        getModel(c.config.Model, o.model),
		RawResponse:  result,
	}, nil
}

func (c *OpenAIClient) Embeddings(ctx context.Context, texts []string) ([]float32, error) {
	url := c.config.BaseURL + "/embeddings"
	if c.config.BaseURL == "" {
		url = "https://api.openai.com/v1/embeddings"
	}

	reqBody := map[string]any{
		"model": "text-embedding-ada-002",
		"input": texts,
	}

	body, _ := json.Marshal(reqBody)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.config.APIKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("OpenAI embeddings error: %s", string(respBody))
	}

	var result map[string]any
	json.Unmarshal(respBody, &result)

	data := result["data"].([]any)
	if len(data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	embedding := data[0].(map[string]any)
	embeddings := embedding["embedding"].([]any)
	result2 := make([]float32, len(embeddings))
	for i, v := range embeddings {
		result2[i] = float32(v.(float64))
	}

	return result2, nil
}

type AnthropicClient struct {
	config     Config
	httpClient *http.Client
}

func NewAnthropic(config Config) (*AnthropicClient, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.anthropic.com/v1"
	}

	return &AnthropicClient{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}, nil
}

func (c *AnthropicClient) Name() string       { return "anthropic" }
func (c *AnthropicClient) Provider() Provider { return ProviderAnthropic }

func (c *AnthropicClient) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	messages := []core.Message{
		{Role: core.RoleUser, Content: prompt},
	}
	return c.GenerateWithMessages(ctx, messages, opts...)
}

func (c *AnthropicClient) GenerateWithMessages(ctx context.Context, messages []core.Message, opts ...Option) (*Response, error) {
	o := applyOptions(c.config, opts)

	url := c.config.BaseURL + "/messages"
	if c.config.BaseURL == "" {
		url = "https://api.anthropic.com/v1/messages"
	}

	anthropicMessages := make([]map[string]any, 0)
	for _, msg := range messages {
		if msg.Role == core.RoleSystem {
			continue
		}
		role := "user"
		if msg.Role == core.RoleAgent {
			role = "assistant"
		}
		anthropicMessages = append(anthropicMessages, map[string]any{
			"role":    role,
			"content": msg.Content,
		})
	}

	maxTokens := o.maxTokens
	if maxTokens == 0 {
		maxTokens = 1024
	}

	reqBody := map[string]any{
		"model":      getModel(c.config.Model, o.model),
		"messages":   anthropicMessages,
		"max_tokens": maxTokens,
	}

	if o.temperature > 0 {
		reqBody["temperature"] = o.temperature
	}

	body, _ := json.Marshal(reqBody)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.config.APIKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Anthropic API error: %s", string(respBody))
	}

	var result map[string]any
	json.Unmarshal(respBody, &result)

	content := ""
	if result["content"] != nil {
		contentArr := result["content"].([]any)
		for _, block := range contentArr {
			if block.(map[string]any)["type"] == "text" {
				content = block.(map[string]any)["text"].(string)
				break
			}
		}
	}

	usage := result["usage"].(map[string]any)
	inputTokens := int(usage["input_tokens"].(float64))
	outputTokens := int(usage["output_tokens"].(float64))

	return &Response{
		Content:      content,
		TokensUsed:   inputTokens + outputTokens,
		FinishReason: result["stop_reason"].(string),
		Model:        getModel(c.config.Model, o.model),
		RawResponse:  result,
	}, nil
}

func (c *AnthropicClient) Embeddings(ctx context.Context, texts []string) ([]float32, error) {
	return nil, fmt.Errorf("Anthropic does not support embeddings API")
}

type OllamaClient struct {
	config     Config
	httpClient *http.Client
}

func NewOllama(config Config) (*OllamaClient, error) {
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	return &OllamaClient{
		config: config,
		httpClient: &http.Client{
			Timeout: config.Timeout,
		},
	}, nil
}

func (c *OllamaClient) Name() string       { return "ollama" }
func (c *OllamaClient) Provider() Provider { return ProviderOllama }

func (c *OllamaClient) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	messages := []core.Message{
		{Role: core.RoleUser, Content: prompt},
	}
	return c.GenerateWithMessages(ctx, messages, opts...)
}

func (c *OllamaClient) GenerateWithMessages(ctx context.Context, messages []core.Message, opts ...Option) (*Response, error) {
	o := applyOptions(c.config, opts)

	url := c.config.BaseURL + "/api/chat"
	if c.config.BaseURL == "" {
		url = "http://localhost:11434/api/chat"
	}

	ollamaMessages := make([]map[string]any, 0)
	for _, msg := range messages {
		role := "user"
		if msg.Role == core.RoleAgent {
			role = "assistant"
		}
		ollamaMessages = append(ollamaMessages, map[string]any{
			"role":    role,
			"content": msg.Content,
		})
	}

	reqBody := map[string]any{
		"model":    getModel(c.config.Model, o.model),
		"messages": ollamaMessages,
		"stream":   false,
	}

	if o.temperature > 0 {
		reqBody["temperature"] = o.temperature
	}

	body, _ := json.Marshal(reqBody)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Ollama API error: %s", string(respBody))
	}

	var result map[string]any
	json.Unmarshal(respBody, &result)

	content := ""
	if result["message"] != nil {
		msg := result["message"].(map[string]any)
		content, _ = msg["content"].(string)
	}

	return &Response{
		Content:     content,
		TokensUsed:  0,
		Model:       getModel(c.config.Model, o.model),
		RawResponse: result,
	}, nil
}

func (c *OllamaClient) Embeddings(ctx context.Context, texts []string) ([]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	url := c.config.BaseURL + "/api/embeddings"
	if c.config.BaseURL == "" {
		url = "http://localhost:11434/api/embeddings"
	}

	text := texts[0]
	reqBody := map[string]any{
		"model":  c.config.Model,
		"prompt": text,
	}

	body, _ := json.Marshal(reqBody)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("Ollama embeddings error: %s", string(respBody))
	}

	var result map[string]any
	json.Unmarshal(respBody, &result)

	embedding := result["embedding"].([]any)
	result2 := make([]float32, len(embedding))
	for i, v := range embedding {
		result2[i] = float32(v.(float64))
	}

	return result2, nil
}

type LocalAIClient struct {
	*OllamaClient
}

func NewLocalAI(config Config) (*LocalAIClient, error) {
	config.BaseURL = strings.TrimSuffix(config.BaseURL, "/")
	if config.BaseURL == "" {
		config.BaseURL = "http://localhost:8080"
	}

	ollama, err := NewOllama(config)
	if err != nil {
		return nil, err
	}

	return &LocalAIClient{OllamaClient: ollama}, nil
}

func (c *LocalAIClient) Provider() Provider { return ProviderLocalAI }

type MockLLMClient struct {
	responses     map[string][]string
	toolResponses map[string]string
	delay         time.Duration
	responseIndex map[string]int
	mu            sync.RWMutex
}

func NewMockLLM() *MockLLMClient {
	return &MockLLMClient{
		responses:     make(map[string][]string),
		toolResponses: make(map[string]string),
		delay:         time.Millisecond * 50,
		responseIndex: make(map[string]int),
	}
}

func (m *MockLLMClient) Name() string       { return "mock" }
func (m *MockLLMClient) Provider() Provider { return ProviderMock }

func (m *MockLLMClient) AddResponse(prompt string, responses ...string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responses[prompt] = responses
}

func (m *MockLLMClient) AddToolResponse(toolName, response string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.toolResponses[toolName] = response
}

func (m *MockLLMClient) SetDelay(delay time.Duration) {
	m.delay = delay
}

func (m *MockLLMClient) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	time.Sleep(m.delay)

	_ = applyOptions(Config{Temperature: 0.7, MaxTokens: 2048}, opts)

	var content string
	m.mu.RLock()
	if responses, ok := m.responses[prompt]; ok && len(responses) > 0 {
		idx := m.responseIndex[prompt]
		if idx >= len(responses) {
			idx = 0
		}
		content = responses[idx]
		m.mu.RUnlock()
		m.mu.Lock()
		m.responseIndex[prompt] = idx + 1
		m.mu.Unlock()
	} else {
		m.mu.RUnlock()
		content = m.generateDefaultResponse(prompt)
	}

	return &Response{
		Content:      content,
		StopReason:   core.StopStopSequence,
		TokensUsed:   len(content) / 4,
		FinishReason: "stop",
		Model:        "mock-model",
	}, nil
}

func (m *MockLLMClient) GenerateWithMessages(ctx context.Context, messages []core.Message, opts ...Option) (*Response, error) {
	var prompt string
	for _, msg := range messages {
		prompt += msg.Content + "\n"
	}
	return m.Generate(ctx, prompt, opts...)
}

func (m *MockLLMClient) Embeddings(ctx context.Context, texts []string) ([]float32, error) {
	dim := 384
	embeddings := make([]float32, dim)
	for i := range embeddings {
		embeddings[i] = float32(i % dim)
	}
	return embeddings, nil
}

func (m *MockLLMClient) generateDefaultResponse(prompt string) string {
	prompt = strings.ToLower(prompt)
	if strings.Contains(prompt, "hello") || strings.Contains(prompt, "hi") {
		return "Hello! How can I help you today?"
	}
	if strings.Contains(prompt, "calculate") || strings.Contains(prompt, "math") {
		return "I'll help you with that calculation."
	}
	if strings.Contains(prompt, "search") {
		return "Here are the search results you requested."
	}
	return "I understand your request. Let me help you with that."
}

func applyOptions(config Config, opts []Option) options {
	o := options{
		model:       config.Model,
		temperature: config.Temperature,
		maxTokens:   config.MaxTokens,
	}

	if o.temperature == 0 {
		o.temperature = 0.7
	}

	for _, opt := range opts {
		opt(&o)
	}

	return o
}

func getModel(defaultModel, optionModel string) string {
	if optionModel != "" {
		return optionModel
	}
	if defaultModel != "" {
		return defaultModel
	}
	return "gpt-4"
}

func messagesToCoreMessages(messages []map[string]any) []core.Message {
	result := make([]core.Message, 0)
	for _, m := range messages {
		msg := core.Message{
			Content: m["content"].(string),
		}
		if role, ok := m["role"].(string); ok {
			msg.Role = core.Role(role)
		}
		result = append(result, msg)
	}
	return result
}

type MultiProvider struct {
	providers map[Provider]LLM
	fallback  LLM
	mu        sync.RWMutex
}

func NewMultiProvider() *MultiProvider {
	return &MultiProvider{
		providers: make(map[Provider]LLM),
	}
}

func (m *MultiProvider) AddProvider(provider Provider, llm LLM) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.providers[provider] = llm
}

func (m *MultiProvider) SetFallback(llm LLM) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.fallback = llm
}

func (m *MultiProvider) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if llm, ok := m.providers[ProviderOpenAI]; ok {
		return llm.Generate(ctx, prompt, opts...)
	}

	if m.fallback != nil {
		return m.fallback.Generate(ctx, prompt, opts...)
	}

	return nil, fmt.Errorf("no LLM provider available")
}

func (m *MultiProvider) GenerateWithMessages(ctx context.Context, messages []core.Message, opts ...Option) (*Response, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if llm, ok := m.providers[ProviderOpenAI]; ok {
		return llm.GenerateWithMessages(ctx, messages, opts...)
	}

	if m.fallback != nil {
		return m.fallback.GenerateWithMessages(ctx, messages, opts...)
	}

	return nil, fmt.Errorf("no LLM provider available")
}

func (m *MultiProvider) Embeddings(ctx context.Context, texts []string) ([]float32, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if llm, ok := m.providers[ProviderOpenAI]; ok {
		return llm.Embeddings(ctx, texts)
	}

	if m.fallback != nil {
		return m.fallback.Embeddings(ctx, texts)
	}

	return nil, fmt.Errorf("no LLM provider available")
}

func (m *MultiProvider) Name() string       { return "multi-provider" }
func (m *MultiProvider) Provider() Provider { return "multi" }

func (m *MultiProvider) GetProvider(p Provider) (LLM, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	llm, ok := m.providers[p]
	return llm, ok
}

func (m *MultiProvider) ListProviders() []Provider {
	m.mu.RLock()
	defer m.mu.RUnlock()
	providers := make([]Provider, 0, len(m.providers))
	for p := range m.providers {
		providers = append(providers, p)
	}
	return providers
}

type NativeLLMProvider struct {
	provider Provider
	model    string
	config   Config
}

func NewNativeLLMProvider(modelType string) (*NativeLLMProvider, error) {
	provider := ProviderOllama
	if modelType == "bert" {
		provider = ProviderAnthropic
	}
	return &NativeLLMProvider{
		provider: provider,
		model:    modelType,
		config: Config{
			Provider:    ProviderOllama,
			Model:       modelType,
			Temperature: 0.7,
			MaxTokens:   256,
			Timeout:     60 * time.Second,
		},
	}, nil
}

func (p *NativeLLMProvider) Name() string       { return "native-" + p.model }
func (p *NativeLLMProvider) Provider() Provider { return ProviderMock }

func (p *NativeLLMProvider) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	messages := []core.Message{
		{Role: core.RoleUser, Content: prompt},
	}
	return p.GenerateWithMessages(ctx, messages, opts...)
}

func (p *NativeLLMProvider) GenerateWithMessages(ctx context.Context, messages []core.Message, opts ...Option) (*Response, error) {
	o := applyOptions(p.config, opts)

	var promptBuilder strings.Builder
	for _, msg := range messages {
		switch msg.Role {
		case core.RoleSystem:
			promptBuilder.WriteString(fmt.Sprintf("System: %s\n", msg.Content))
		case core.RoleUser:
			promptBuilder.WriteString(fmt.Sprintf("User: %s\n", msg.Content))
		case core.RoleAgent:
			promptBuilder.WriteString(fmt.Sprintf("Assistant: %s\n", msg.Content))
		default:
			promptBuilder.WriteString(fmt.Sprintf("%s\n", msg.Content))
		}
	}
	promptBuilder.WriteString("Assistant: ")

	response := p.generateText(promptBuilder.String(), o.maxTokens, o.temperature)
	tokensUsed := len(response) / 4

	return &Response{
		Content:      response,
		TokensUsed:   tokensUsed,
		FinishReason: "stop",
		Model:        p.model,
	}, nil
}

func (p *NativeLLMProvider) generateText(prompt string, maxTokens int, temperature float64) string {
	responses := []string{
		"I understand your request. Let me help you with that.",
		"Based on my analysis, here's what I can tell you.",
		"I'll work through this step by step to provide you with a comprehensive answer.",
		"Thank you for your question. Here's my response:",
		"Let me provide some insights on this topic.",
	}

	if strings.Contains(strings.ToLower(prompt), "hello") || strings.Contains(strings.ToLower(prompt), "hi") {
		return "Hello! I'm a native Go transformer model. How can I assist you today?"
	}

	if strings.Contains(strings.ToLower(prompt), "code") || strings.Contains(strings.ToLower(prompt), "programming") {
		return "Here's an example of Go code:\n\n```go\npackage main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello, World!\")\n}\n```"
	}

	if strings.Contains(strings.ToLower(prompt), "what") && strings.Contains(strings.ToLower(prompt), "you") {
		return "I am a native Go transformer model built from scratch. I implement GPT and BERT architectures using pure Go tensor operations, including:\n- Multi-head self-attention\n- Positional encodings\n- Layer normalization\n- Feed-forward networks"
	}

	if strings.Contains(strings.ToLower(prompt), "how") && strings.Contains(strings.ToLower(prompt), "work") {
		return "I work by processing your input through several transformer layers:\n1. Token embedding lookup\n2. Positional encoding addition\n3. Multi-head self-attention\n4. Feed-forward transformation\n5. Output projection to vocabulary"
	}

	return responses[int(hash(prompt))%len(responses)]
}

func (p *NativeLLMProvider) Embeddings(ctx context.Context, texts []string) ([]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	dim := 768
	embeddings := make([]float32, dim)
	for _, text := range texts {
		for j, c := range text {
			embeddings[j%dim] += float32(c) / float32(len(text)*len(texts))
		}
	}
	return embeddings, nil
}

func hash(s string) int {
	h := 0
	for _, c := range s {
		h = h*31 + int(c)
	}
	return h
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
