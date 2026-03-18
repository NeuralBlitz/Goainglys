package main

import (
	"context"
	"fmt"
	"os"

	"agents/framework/agents"
	"agents/framework/core"
	"agents/framework/integrations"
	"agents/framework/llm"
	"agents/framework/memory"
	"agents/framework/multimodal"
	"agents/framework/testing"
	"agents/framework/tools"
)

func main() {
	fmt.Println("=== Agents Framework Demo ===\n")
	fmt.Println("Features: ReAct Agent | Tools | Memory | LLM Providers | Multimodal | Git | HuggingFace\n")

	demoLLMProviders()
	demoTools()
	demoMemory()
	demoMultimodal()
	demoGit()
	demoHuggingFace()
	demoAgent()
	demoTests()

	fmt.Println("\n=== Demo Complete ===")
}

func demoLLMProviders() {
	fmt.Println("--- LLM Providers ---")

	mockLLM := llm.NewMockLLM()
	mockLLM.AddResponse("Hello", "Hello! How are you?")
	mockLLM.AddResponse("What is Go?", "Go is a programming language by Google.")

	resp, err := mockLLM.Generate(context.Background(), "Hello")
	if err == nil {
		fmt.Printf("MockLLM: %s\n", resp.Content)
	}

	nativeLLM, err := llm.NewNativeLLMProvider("gpt2")
	if err == nil {
		resp, err := nativeLLM.Generate(context.Background(), "What is a transformer?")
		if err == nil {
			fmt.Printf("NativeLLM (GPT2): %s\n", truncate(resp.Content, 100))
		}
		fmt.Printf("Native LLM provider: %s (%s)\n", nativeLLM.Name(), nativeLLM.Provider())
	}

	bertLLM, err := llm.NewNativeLLMProvider("bert")
	if err == nil {
		embeddings, err := bertLLM.Embeddings(context.Background(), []string{"hello world"})
		if err == nil {
			fmt.Printf("BERT embeddings dim: %d\n", len(embeddings))
		}
	}

	multi := llm.NewMultiProvider()
	multi.AddProvider(llm.ProviderMock, mockLLM)

	providers := multi.ListProviders()
	fmt.Printf("Available providers: %v\n", providers)

	openai, err := llm.NewLLM(llm.Config{
		Provider: llm.ProviderOpenAI,
		Model:    "gpt-4",
		APIKey:   "demo-key",
	})
	if err == nil {
		fmt.Printf("OpenAI client: %s (%s)\n", openai.Name(), openai.Provider())
	}

	ollama, err := llm.NewLLM(llm.Config{
		Provider: llm.ProviderOllama,
		Model:    "llama2",
		BaseURL:  "http://localhost:11434",
	})
	if err == nil {
		fmt.Printf("Ollama client: %s (%s)\n", ollama.Name(), ollama.Provider())
	}

	embeddings, _ := mockLLM.Embeddings(context.Background(), []string{"hello world"})
	fmt.Printf("Embeddings dim: %d\n", len(embeddings))

	fmt.Println()
}

func demoTools() {
	fmt.Println("--- Tools ---")
	toolSet := tools.DefaultToolSet()
	fmt.Printf("Available tools: %v\n", toolSet.Names())

	calc := tools.NewCalculatorTool()
	result, _ := calc.Execute(context.Background(), map[string]any{"expression": "10+5"})
	fmt.Printf("Calculator: 10+5 = %s\n", result.Content)

	timeTool := tools.NewTimeTool()
	result, _ = timeTool.Execute(context.Background(), map[string]any{"format": "full"})
	fmt.Printf("Current time: %s\n", result.Content)

	search := tools.NewSearchTool()
	result, _ = search.Execute(context.Background(), map[string]any{"query": "go"})
	fmt.Printf("Search 'go': %s\n", result.Content)
	fmt.Println()
}

func demoMemory() {
	fmt.Println("--- Memory ---")
	mem := memory.NewConversationMemory(100)
	mem.Add(core.Message{ID: "1", Role: core.RoleUser, Content: "Hello"})
	mem.Add(core.Message{ID: "2", Role: core.RoleAgent, Content: "Hi there!"})
	mem.Add(core.Message{ID: "3", Role: core.RoleUser, Content: "How are you?"})
	fmt.Printf("ConversationMemory: %d messages\n", mem.Count())

	vecMem := memory.NewVectorMemory(100)
	vecMem.Add(memory.MemoryEntry{Content: "Python is great"})
	vecMem.Add(memory.MemoryEntry{Content: "Go is fast"})
	fmt.Printf("VectorMemory: %d entries\n", len(vecMem.GetRecent(10)))

	slideMem := memory.NewSlidingWindowMemory(5)
	slideMem.Add(core.Message{ID: "4", Role: core.RoleUser, Content: "msg1"})
	slideMem.Add(core.Message{ID: "5", Role: core.RoleUser, Content: "msg2"})
	fmt.Printf("SlidingWindowMemory: %d messages (window=5)\n", slideMem.Count())
	fmt.Println()
}

func demoMultimodal() {
	fmt.Println("--- Multimodal ---")
	imageTool := multimodal.NewImageTool()
	result, _ := imageTool.Execute(context.Background(), map[string]any{"action": "stats"})
	fmt.Printf("Image Tool: %s\n", result.Content)

	audioTool := multimodal.NewAudioTool()
	result, _ = audioTool.Execute(context.Background(), map[string]any{
		"action": "info",
		"path":   "example.wav",
	})
	fmt.Printf("Audio Tool: %s\n", result.Content)

	videoTool := multimodal.NewVideoTool()
	result, _ = videoTool.Execute(context.Background(), map[string]any{
		"action": "info",
		"path":   "example.mp4",
	})
	fmt.Printf("Video Tool: %s\n", result.Content)

	docTool := multimodal.NewDocumentTool()
	result, _ = docTool.Execute(context.Background(), map[string]any{
		"action":  "write",
		"path":    "test.txt",
		"content": "Hello from multimodal tool!",
	})
	fmt.Printf("Document Tool: %s\n", result.Content)
	fmt.Println()
}

func demoGit() {
	fmt.Println("--- Git Integration ---")
	gitTool := integrations.NewGitTool(".")

	if _, err := os.Stat(".git"); err == nil {
		result, err := gitTool.Execute(context.Background(), map[string]any{"action": "status"})
		if err == nil && result.Success {
			fmt.Printf("Git status: %s\n", result.Content)
		}
	} else {
		fmt.Println("Git: Not a git repository")
	}

	githubTool := integrations.NewGitHubTool("")
	result, err := githubTool.Execute(context.Background(), map[string]any{
		"action": "search",
		"query":  "transformer",
	})
	if err == nil && result.Success {
		fmt.Printf("GitHub search: %s\n", result.Content)
	} else {
		fmt.Println("GitHub: API request failed (may need token for higher limits)")
	}
	fmt.Println()
}

func demoHuggingFace() {
	fmt.Println("--- HuggingFace Integration ---")
	hfTool := integrations.NewHuggingFaceTool("")

	result, err := hfTool.Execute(context.Background(), map[string]any{
		"action": "search_models",
		"task":   "text-generation",
	})
	if err == nil && result.Success {
		fmt.Printf("Text-generation models:\n%s\n", truncate(result.Content, 300))
	} else {
		fmt.Println("HuggingFace: API request failed")
	}

	result, err = hfTool.Execute(context.Background(), map[string]any{
		"action": "inference",
		"query":  "gpt2",
		"input":  "Once upon a time",
	})
	if err == nil && result.Success {
		fmt.Printf("Inference demo:\n%s\n", truncate(result.Content, 200))
	}
	fmt.Println()
}

func truncate(s string, max int) string {
	if len(s) > max {
		return s[:max] + "..."
	}
	return s
}

func demoAgent() {
	fmt.Println("--- ReAct Agent ---")
	llm := core.NewMockLLM()
	llm.AddResponse("What's 5*5?", "Thought: I need to calculate 5*5\nAction: calculator\nAction Input: 5*5")
	llm.AddToolResponse("calculator", "25")
	llm.AddResponse("25", "5 * 5 = 25")

	calc := tools.NewCalculatorTool()
	config := core.AgentConfig{
		Name:          "assistant",
		Description:   "Helpful math assistant",
		Type:          core.TypeReAct,
		Model:         "mock-model",
		Temperature:   0.7,
		MaxTokens:     2048,
		MaxIterations: 5,
		Tools:         []core.Tool{calc},
		SystemPrompt:  "You are a helpful assistant. Use tools when needed.",
	}

	agent := agents.NewReActAgent(config, llm)
	result, _ := agent.Run(context.Background(), "What's 5*5?")

	fmt.Printf("Agent output: %s\n", result.Output)
	fmt.Printf("Iterations: %d | Duration: %v\n", result.Iterations, result.Duration)
	fmt.Println()
}

func demoTests() {
	fmt.Println("--- Testing Framework ---")
	runner := testing.NewTestRunner()

	mockAgent := testing.NewMockTestAgent(map[string]string{
		"Hello":     "Hello! How can I help?",
		"Calculate": "Calculation complete",
	})

	runner.AddSuite(testing.TestSuite{
		Name:      "Agent Tests",
		AgentType: core.TypeReAct,
		AgentFn: func() core.Agent {
			mockAgent.Reset()
			return mockAgent
		},
		Tests: []testing.TestCase{
			{Name: "Greeting", Input: "Hello", Expected: "hello"},
			{Name: "Calculation", Input: "Calculate", Expected: "Calculation"},
		},
	})

	runner.AddSuite(testing.TestSuite{
		Name:      "Tool Tests",
		AgentType: core.TypeReAct,
		AgentFn: func() core.Agent {
			mockAgent.Reset()
			return mockAgent
		},
		Tests: []testing.TestCase{
			{Name: "Tool Check", Input: "Test", Validate: func(r *core.AgentResult) bool {
				return len(r.Output) > 0
			}},
		},
	})

	report := runner.Run()
	runner.PrintReport()
	fmt.Printf("\n%d/%d tests passed\n", report.PassedTests, report.TotalTests)
}
