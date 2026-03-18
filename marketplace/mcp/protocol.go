package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

type ProtocolVersion string

const (
	ProtocolV1 ProtocolVersion = "2024-11-05"
)

type JSONRPCRequest struct {
	JSONRPC string         `json:"jsonrpc"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params,omitempty"`
	ID      interface{}    `json:"id,omitempty"`
}

type JSONRPCResponse struct {
	JSONRPC string      `json:"jsonrpc"`
	Result  interface{} `json:"result,omitempty"`
	Error   *JSONError  `json:"error,omitempty"`
	ID      interface{} `json:"id,omitempty"`
}

type JSONError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

type Tool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	MimeType    string `json:"mimeType,omitempty"`
}

type Prompt struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	Arguments   []Argument `json:"arguments,omitempty"`
}

type Argument struct {
	Name        string `json:"name"`
	Description string `json:"description,omitempty"`
	Required    bool   `json:"required,omitempty"`
}

type InitializeResult struct {
	ProtocolVersion ProtocolVersion    `json:"protocolVersion"`
	Capabilities    ServerCapabilities `json:"capabilities"`
	ServerInfo      ServerInfo         `json:"serverInfo"`
}

type ServerCapabilities struct {
	Tools     *ToolsCapability     `json:"tools,omitempty"`
	Resources *ResourcesCapability `json:"resources,omitempty"`
	Prompts   *PromptsCapability   `json:"prompts,omitempty"`
	Logging   *LoggingCapability   `json:"logging,omitempty"`
}

type ToolsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

type ResourcesCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
	Subscribe   bool `json:"subscribe,omitempty"`
}

type PromptsCapability struct {
	ListChanged bool `json:"listChanged,omitempty"`
}

type LoggingCapability struct {
}

type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type ToolCallResult struct {
	Content []ContentBlock `json:"content"`
	IsError bool           `json:"isError,omitempty"`
}

type ContentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	Data string `json:"data,omitempty"`
	URL  string `json:"url,omitempty"`
}

type ResourceListResult struct {
	Resources []Resource `json:"resources"`
}

type PromptListResult struct {
	Prompts []Prompt `json:"prompts"`
}

type Server interface {
	Initialize(ctx context.Context, params map[string]any) (*InitializeResult, error)
	CallTool(ctx context.Context, name string, arguments map[string]any) (*ToolCallResult, error)
	ListTools(ctx context.Context) ([]Tool, error)
	ListResources(ctx context.Context) ([]Resource, error)
	ListPrompts(ctx context.Context) ([]Prompt, error)
}

type MCPServer struct {
	mu              sync.RWMutex
	protocolVersion ProtocolVersion
	serverInfo      ServerInfo
	capabilities    ServerCapabilities
	tools           map[string]ToolHandler
	resources       map[string]ResourceHandler
	prompts         map[string]PromptHandler
	logger          Logger
}

type ToolHandler func(ctx context.Context, arguments map[string]any) (*ToolCallResult, error)
type ResourceHandler func(ctx context.Context, uri string) (string, error)
type PromptHandler func(ctx context.Context, arguments map[string]any) (string, error)

type Logger interface {
	Debug(msg string, params map[string]any)
	Info(msg string, params map[string]any)
	Warn(msg string, params map[string]any)
	Error(msg string, params map[string]any)
}

type DefaultLogger struct{}

func (l *DefaultLogger) Debug(msg string, params map[string]any) {
	fmt.Printf("[DEBUG] %s %v\n", msg, params)
}

func (l *DefaultLogger) Info(msg string, params map[string]any) {
	fmt.Printf("[INFO] %s %v\n", msg, params)
}

func (l *DefaultLogger) Warn(msg string, params map[string]any) {
	fmt.Printf("[WARN] %s %v\n", msg, params)
}

func (l *DefaultLogger) Error(msg string, params map[string]any) {
	fmt.Printf("[ERROR] %s %v\n", msg, params)
}

var (
	ErrToolNotFound  = errors.New("tool not found")
	ErrInvalidParams = errors.New("invalid parameters")
	ErrInternalError = errors.New("internal error")
)

func NewMCPServer(name, version string) *MCPServer {
	return &MCPServer{
		protocolVersion: ProtocolV1,
		serverInfo: ServerInfo{
			Name:    name,
			Version: version,
		},
		capabilities: ServerCapabilities{
			Tools:     &ToolsCapability{ListChanged: true},
			Resources: &ResourcesCapability{ListChanged: true},
			Prompts:   &PromptsCapability{ListChanged: true},
			Logging:   &LoggingCapability{},
		},
		tools:     make(map[string]ToolHandler),
		resources: make(map[string]ResourceHandler),
		prompts:   make(map[string]PromptHandler),
		logger:    &DefaultLogger{},
	}
}

type ToolDefinition struct {
	Name        string
	Description string
	InputSchema map[string]any
	Handler     ToolHandler
}

func (s *MCPServer) RegisterTool(name, description string, inputSchema map[string]any, handler ToolHandler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tools[name] = handler
	_ = ToolDefinition{
		Name:        name,
		Description: description,
		InputSchema: inputSchema,
		Handler:     handler,
	}
}

func (s *MCPServer) RegisterResource(uri, name, description string, handler ResourceHandler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.resources[uri] = handler
	_ = name
	_ = description
}

func (s *MCPServer) RegisterPrompt(name, description string, arguments []Argument, handler PromptHandler) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.prompts[name] = handler
	_ = description
	_ = arguments
}

func (s *MCPServer) Initialize(ctx context.Context, params map[string]any) (*InitializeResult, error) {
	s.logger.Info("Initializing MCP server", map[string]any{
		"name":    s.serverInfo.Name,
		"version": s.serverInfo.Version,
	})

	return &InitializeResult{
		ProtocolVersion: s.protocolVersion,
		Capabilities:    s.capabilities,
		ServerInfo:      s.serverInfo,
	}, nil
}

func (s *MCPServer) CallTool(ctx context.Context, name string, arguments map[string]any) (*ToolCallResult, error) {
	s.mu.RLock()
	handler, ok := s.tools[name]
	s.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("tool not found: %s", name)
	}

	s.logger.Info("Calling tool", map[string]any{"name": name})

	result, err := handler(ctx, arguments)
	if err != nil {
		return nil, err
	}

	return result, nil
}

func (s *MCPServer) ListTools(ctx context.Context) ([]Tool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	tools := make([]Tool, 0, len(s.tools))
	for name, handler := range s.tools {
		_ = handler
		tools = append(tools, Tool{
			Name:        name,
			Description: "Tool: " + name,
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		})
	}

	return tools, nil
}

func (s *MCPServer) ListResources(ctx context.Context) ([]Resource, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	resources := make([]Resource, 0, len(s.resources))
	for uri := range s.resources {
		resources = append(resources, Resource{
			URI:  uri,
			Name: uri,
		})
	}

	return resources, nil
}

func (s *MCPServer) ListPrompts(ctx context.Context) ([]Prompt, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	prompts := make([]Prompt, 0, len(s.prompts))
	for name := range s.prompts {
		prompts = append(prompts, Prompt{
			Name: name,
		})
	}

	return prompts, nil
}

func (s *MCPServer) HandleRequest(ctx context.Context, req *JSONRPCRequest) (*JSONRPCResponse, error) {
	switch req.Method {
	case "initialize":
		params := req.Params
		result, err := s.Initialize(ctx, params)
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			Result:  result,
			ID:      req.ID,
		}, err

	case "tools/call":
		params := req.Params
		name, _ := params["name"].(string)
		arguments, _ := params["arguments"].(map[string]any)
		result, err := s.CallTool(ctx, name, arguments)
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			Result:  result,
			ID:      req.ID,
		}, err

	case "tools/list":
		result, err := s.ListTools(ctx)
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			Result:  map[string]any{"tools": result},
			ID:      req.ID,
		}, err

	case "resources/list":
		result, err := s.ListResources(ctx)
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			Result:  map[string]any{"resources": result},
			ID:      req.ID,
		}, err

	case "prompts/list":
		result, err := s.ListPrompts(ctx)
		return &JSONRPCResponse{
			JSONRPC: "2.0",
			Result:  map[string]any{"prompts": result},
			ID:      req.ID,
		}, err

	default:
		return nil, fmt.Errorf("method not found: %s", req.Method)
	}
}

func ParseRequest(data []byte) (*JSONRPCRequest, error) {
	var req JSONRPCRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, err
	}
	return &req, nil
}

func CreateResponse(id interface{}, result interface{}) *JSONRPCResponse {
	return &JSONRPCResponse{
		JSONRPC: "2.0",
		Result:  result,
		ID:      id,
	}
}

func CreateErrorResponse(id interface{}, code int, message string) *JSONRPCResponse {
	return &JSONRPCResponse{
		JSONRPC: "2.0",
		Error: &JSONError{
			Code:    code,
			Message: message,
		},
		ID: id,
	}
}
