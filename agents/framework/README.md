# Agents Framework

A comprehensive framework for building and testing AI agents in Go.

## Features

- **Multiple Agent Types**: ReAct, Plan-Execute, Supervisor, Crew, Handoff
- **Standard LLM Interface**: OpenAI, Anthropic, Ollama, LocalAI
- **Tool System**: Easy tool creation and integration
- **Memory Management**: Conversation history and context
- **Testing Framework**: Built-in test runner with benchmarks
- **Mock LLM**: For testing without API calls
- **Multimodal**: Image, Audio, Video, Document processing
- **Integrations**: Git, GitHub, HuggingFace

## Architecture

```
framework/
├── core/           # Core types and interfaces
├── agents/        # Agent implementations
├── llm/           # LLM providers (OpenAI, Anthropic, Ollama, etc.)
├── tools/         # Tool definitions
├── memory/        # Memory implementations
├── multimodal/    # Image, audio, video, document tools
├── integrations/   # Git and HuggingFace integrations
└── testing/       # Testing framework
```

## LLM Providers

```go
// OpenAI
llm, _ := llm.NewLLM(llm.Config{
    Provider: llm.ProviderOpenAI,
    Model:    "gpt-4",
    APIKey:   "sk-...",
})

// Anthropic
llm, _ := llm.NewLLM(llm.Config{
    Provider: llm.ProviderAnthropic,
    Model:    "claude-3-sonnet",
    APIKey:   "sk-ant-...",
})

// Ollama (local)
llm, _ := llm.NewLLM(llm.Config{
    Provider: llm.ProviderOllama,
    Model:    "llama2",
    BaseURL:  "http://localhost:11434",
})

// Native Go Transformer
nativeLLM, _ := llm.NewNativeLLMProvider("gpt2")

// Multi-provider with fallback
multi := llm.NewMultiProvider()
multi.AddProvider(llm.ProviderOllama, ollama)
multi.SetFallback(mockLLM)
```

### Supported Providers

| Provider | Models | Embeddings | Tools |
|----------|--------|------------|-------|
| OpenAI | GPT-4, GPT-3.5 | ✅ | ✅ |
| Anthropic | Claude 3 | ❌ | ❌ |
| Ollama | Llama, Mistral, etc. | ✅ | ✅ |
| LocalAI | Any GGUF model | ✅ | ✅ |
| Native Go | GPT-2, BERT | ✅ | ❌ |
| Mock | Custom responses | ✅ | ✅ |

## Multimodal

```go
imageTool := multimodal.NewImageTool()
imageTool.Execute(ctx, map[string]any{"action": "download", "url": "..."})

audioTool := multimodal.NewAudioTool()
audioTool.Execute(ctx, map[string]any{"action": "transcribe", "path": "audio.wav"})

videoTool := multimodal.NewVideoTool()
videoTool.Execute(ctx, map[string]any{"action": "info", "path": "video.mp4"})

docTool := multimodal.NewDocumentTool()
docTool.Execute(ctx, map[string]any{"action": "read", "path": "file.txt"})
```

## Integrations

### Git

```go
gitTool := integrations.NewGitTool("/path/to/repo")
gitTool.Execute(ctx, map[string]any{"action": "status"})
gitTool.Execute(ctx, map[string]any{"action": "commit", "message": "Update"})
gitTool.Execute(ctx, map[string]any{"action": "push"})
```

### GitHub

```go
githubTool := integrations.NewGitHubTool("your-token")
githubTool.Execute(ctx, map[string]any{"action": "search", "query": "transformer"})
githubTool.Execute(ctx, map[string]any{"action": "issues", "owner": "owner", "repo": "repo"})
```

### HuggingFace

```go
hfTool := integrations.NewHuggingFaceTool("your-token")
hfTool.Execute(ctx, map[string]any{"action": "search_models", "task": "text-generation"})
hfTool.Execute(ctx, map[string]any{"action": "model_info", "query": "gpt2"})
hfTool.Execute(ctx, map[string]any{"action": "inference", "query": "gpt2", "input": "Hello"})
```

## Tools

```go
// Use built-in tools
toolSet := tools.DefaultToolSet()
calculator := tools.NewCalculatorTool()
search := tools.NewSearchTool()
timeTool := tools.NewTimeTool()

// Or create custom tools
myTool := core.NewFunctionTool("my_tool", "Description", func(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
    return &core.ToolResult{Content: "result", Success: true}, nil
})
```

## Memory

```go
// Conversation memory
mem := memory.NewConversationMemory(100)
mem.Add(core.Message{Role: core.RoleUser, Content: "Hello"})
msgs := mem.GetMessages(10)
mem.Search("query", 5)

// Vector memory
vecMem := memory.NewVectorMemory(1000)

// Sliding window
slideMem := memory.NewSlidingWindowMemory(5)
```

## Quick Start

```bash
go run main.go
```

## Agent Types

### ReAct Agent
Reasoning and Acting agent that iteratively calls tools.

```go
config := core.AgentConfig{
    Name:         "assistant",
    Type:         core.TypeReAct,
    Tools:        []core.Tool{tool1, tool2},
    SystemPrompt: "You are helpful.",
}

agent := agents.NewReActAgent(config, llm)
result, _ := agent.Run(ctx, "Your question")
```

## Testing

```go
runner := testing.NewTestRunner()
runner.AddSuite(testing.TestSuite{
    Name: "My Tests",
    AgentFn: func() core.Agent { return myAgent },
    Tests: []testing.TestCase{
        {
            Name:     "Test Name",
            Input:    "input",
            Expected: "expected output",
        },
    },
})

runner.Run()
runner.PrintReport()
```

## Tools

```go
// Built-in tools
calculator := tools.NewCalculatorTool()
search := tools.NewSearchTool()
timeTool := tools.NewTimeTool()
fileTool := tools.NewFileTool()

// Or create custom tools
myTool := core.NewFunctionTool("my_tool", "Description", func(ctx context.Context, args map[string]any) (*core.ToolResult, error) {
    // Tool logic
    return &core.ToolResult{Content: "result", Success: true}, nil
})

// ToolSet for managing multiple tools
ts := tools.NewToolSet()
ts.Add(myTool)
ts.List()  // Get all tools
ts.Names() // Get tool names
```

## Mock LLM

```go
llm := core.NewMockLLM()
llm.AddResponse("prompt", "response")
llm.AddToolResponse("tool_name", "tool result")
```
