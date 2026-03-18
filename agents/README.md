# Agents

Collection of AI agent frameworks and implementations.

## Structure

```
agents/
└── framework/      # Core agent framework
    ├── core/       # Core types and interfaces
    ├── agents/     # Agent implementations
    ├── llm/        # LLM providers
    ├── tools/       # Tool definitions
    ├── memory/      # Memory implementations
    ├── multimodal/  # Multimodal tools
    ├── integrations/ # Git, HuggingFace
    ├── testing/    # Testing framework
    ├── main.go    # Demo
    └── README.md  # This file
```

## Run Demo

```bash
cd agents/framework && go run main.go
```

## Components

### Framework (`framework/`)

Complete agent development framework with:
- Multiple agent types (ReAct, etc.)
- Standard LLM interface (OpenAI, Anthropic, Ollama, LocalAI)
- Tool system
- Memory management
- Multimodal tools
- Git/HuggingFace integrations
- Testing utilities
- Mock LLM for testing

See `framework/README.md` for details.
