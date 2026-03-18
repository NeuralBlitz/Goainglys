package tools

import (
	"context"
	"fmt"
	"strings"
	"time"

	"agents/framework/core"
)

type CalculatorTool struct{}

func NewCalculatorTool() *CalculatorTool {
	return &CalculatorTool{}
}

func (t *CalculatorTool) Name() string { return "calculator" }
func (t *CalculatorTool) Description() string {
	return "Performs basic arithmetic operations: add, subtract, multiply, divide"
}
func (t *CalculatorTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"expression": {
			Type:        "string",
			Description: "The arithmetic expression to evaluate (e.g., '2+2', '10*5', '100/4')",
			Required:    true,
		},
	}
}

func (t *CalculatorTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	expr, ok := args["expression"].(string)
	if !ok {
		return &core.ToolResult{
			Content: "Invalid expression",
			Success: false,
		}, nil
	}

	result, err := evaluateExpression(expr)
	if err != nil {
		return &core.ToolResult{
			Content: fmt.Sprintf("Error: %v", err),
			Success: false,
		}, nil
	}

	return &core.ToolResult{
		Content: fmt.Sprintf("%.2f", result),
		Success: true,
	}, nil
}

func evaluateExpression(expr string) (float64, error) {
	expr = strings.ReplaceAll(expr, " ", "")

	var a, b float64
	var op rune
	var err error

	for i, c := range expr {
		if c == '+' || c == '-' || c == '*' || c == '/' {
			a, err = parseNumber(expr[:i])
			if err != nil {
				return 0, err
			}
			op = c
			b, err = parseNumber(expr[i+1:])
			if err != nil {
				return 0, err
			}
			break
		}
	}

	switch op {
	case '+':
		return a + b, nil
	case '-':
		return a - b, nil
	case '*':
		return a * b, nil
	case '/':
		if b == 0 {
			return 0, fmt.Errorf("division by zero")
		}
		return a / b, nil
	}

	return 0, fmt.Errorf("invalid expression")
}

func parseNumber(s string) (float64, error) {
	var n float64
	_, err := fmt.Sscanf(s, "%f", &n)
	return n, err
}

type SearchTool struct {
	knowledgeBase map[string]string
}

func NewSearchTool() *SearchTool {
	return &SearchTool{
		knowledgeBase: map[string]string{
			"go":               "Go is a programming language created by Google in 2009.",
			"golang":           "Go is often referred to as Golang.",
			"python":           "Python is a high-level programming language created by Guido van Rossum.",
			"rust":             "Rust is a systems programming language focused on safety and performance.",
			"machine learning": "Machine learning is a subset of artificial intelligence.",
			"transformer":      "Transformer is a neural network architecture introduced in 'Attention Is All You Need'.",
		},
	}
}

func (t *SearchTool) Name() string { return "search" }
func (t *SearchTool) Description() string {
	return "Searches the knowledge base for information about a topic"
}
func (t *SearchTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"query": {
			Type:        "string",
			Description: "The search query",
			Required:    true,
		},
	}
}

func (t *SearchTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	query, ok := args["query"].(string)
	if !ok {
		return &core.ToolResult{
			Content: "Invalid query",
			Success: false,
		}, nil
	}

	query = strings.ToLower(query)
	var results []string

	for key, value := range t.knowledgeBase {
		if strings.Contains(key, query) || strings.Contains(value, query) {
			results = append(results, fmt.Sprintf("%s: %s", key, value))
		}
	}

	if len(results) == 0 {
		return &core.ToolResult{
			Content: "No results found for: " + query,
			Success: true,
		}, nil
	}

	return &core.ToolResult{
		Content: strings.Join(results, "\n"),
		Success: true,
	}, nil
}

type TimeTool struct{}

func NewTimeTool() *TimeTool {
	return &TimeTool{}
}

func (t *TimeTool) Name() string        { return "time" }
func (t *TimeTool) Description() string { return "Returns the current time and date" }
func (t *TimeTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"format": {
			Type:        "string",
			Description: "Time format (default: 'short')",
			Required:    false,
		},
	}
}

func (t *TimeTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	now := time.Now()
	format := "short"

	if f, ok := args["format"].(string); ok && f != "" {
		format = f
	}

	var output string
	switch format {
	case "short":
		output = now.Format("15:04:05")
	case "full":
		output = now.Format("15:04:05 MST 2006-01-02")
	case "date":
		output = now.Format("2006-01-02")
	default:
		output = now.Format("15:04:05")
	}

	return &core.ToolResult{
		Content: output,
		Success: true,
	}, nil
}

type FileTool struct {
	files map[string]string
}

func NewFileTool() *FileTool {
	return &FileTool{
		files: make(map[string]string),
	}
}

func (t *FileTool) Name() string        { return "file" }
func (t *FileTool) Description() string { return "Creates or reads files in memory" }
func (t *FileTool) Parameters() map[string]core.Parameter {
	return map[string]core.Parameter{
		"action": {
			Type:        "string",
			Description: "Action to perform: 'read' or 'write'",
			Required:    true,
			Enum:        []string{"read", "write"},
		},
		"filename": {
			Type:        "string",
			Description: "Name of the file",
			Required:    true,
		},
		"content": {
			Type:        "string",
			Description: "Content to write (required for 'write' action)",
			Required:    false,
		},
	}
}

func (t *FileTool) Execute(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
	action, _ := args["action"].(string)
	filename, _ := args["filename"].(string)

	switch action {
	case "read":
		if content, ok := t.files[filename]; ok {
			return &core.ToolResult{
				Content: content,
				Success: true,
			}, nil
		}
		return &core.ToolResult{
			Content: "File not found: " + filename,
			Success: false,
		}, nil

	case "write":
		content, _ := args["content"].(string)
		t.files[filename] = content
		return &core.ToolResult{
			Content: fmt.Sprintf("File '%s' written successfully", filename),
			Success: true,
		}, nil

	default:
		return &core.ToolResult{
			Content: "Unknown action: " + action,
			Success: false,
		}, nil
	}
}

type ToolSet struct {
	tools map[string]core.Tool
}

func NewToolSet() *ToolSet {
	return &ToolSet{
		tools: make(map[string]core.Tool),
	}
}

func (ts *ToolSet) Add(tool core.Tool) {
	ts.tools[tool.Name()] = tool
}

func (ts *ToolSet) Get(name string) (core.Tool, bool) {
	tool, ok := ts.tools[name]
	return tool, ok
}

func (ts *ToolSet) List() []core.Tool {
	tools := make([]core.Tool, 0, len(ts.tools))
	for _, tool := range ts.tools {
		tools = append(tools, tool)
	}
	return tools
}

func (ts *ToolSet) Names() []string {
	names := make([]string, 0, len(ts.tools))
	for name := range ts.tools {
		names = append(names, name)
	}
	return names
}

func (ts *ToolSet) Remove(name string) {
	delete(ts.tools, name)
}

func DefaultToolSet() *ToolSet {
	ts := NewToolSet()
	ts.Add(NewCalculatorTool())
	ts.Add(NewSearchTool())
	ts.Add(NewTimeTool())
	ts.Add(NewFileTool())
	return ts
}
