package agents

import (
	"context"
	"fmt"
	"strings"
	"time"

	"agents/framework/core"
)

type ReActAgent struct {
	core.BaseAgent
	llm core.LLM
}

func NewReActAgent(config core.AgentConfig, llm core.LLM) *ReActAgent {
	agent := &ReActAgent{
		llm: llm,
	}
	agent.BaseAgent.Config = config
	agent.BaseAgent.Init(config.Tools)
	return agent
}

func (a *ReActAgent) Run(ctx context.Context, input string) (*core.AgentResult, error) {
	startTime := time.Now()
	iterations := 0
	maxIterations := a.Ctx.Config.MaxIterations
	if maxIterations == 0 {
		maxIterations = 10
	}

	a.Ctx.AddMessage(core.Message{
		ID:        core.GenerateID(),
		Role:      core.RoleUser,
		Content:   input,
		Timestamp: time.Now(),
	})

	var output strings.Builder

	for iterations < maxIterations {
		iterations++

		messages := a.buildPrompt()
		resp, err := a.llm.GenerateWithMessages(ctx, messages,
			core.WithTemperature(a.Ctx.Config.Temperature),
			core.WithMaxTokens(a.Ctx.Config.MaxTokens),
		)

		if err != nil {
			return &core.AgentResult{
				Error:      err,
				Iterations: iterations,
				Duration:   time.Since(startTime),
			}, err
		}

		a.Ctx.AddEvent(core.AgentEvent{
			Type:      "llm_response",
			Timestamp: time.Now(),
			Data:      map[string]any{"content": resp.Content},
		})

		output.WriteString(resp.Content + "\n")

		toolCalls, hasToolCalls := parseToolCalls(resp.Content)
		if hasToolCalls && len(toolCalls) > 0 {
			for _, tc := range toolCalls {
				tool, exists := a.Ctx.Tools[tc.Name]
				if !exists {
					a.Ctx.AddMessage(core.Message{
						ID:        core.GenerateID(),
						Role:      core.RoleTool,
						Content:   fmt.Sprintf("Tool not found: %s", tc.Name),
						Timestamp: time.Now(),
					})
					continue
				}

				result, err := tool.Execute(ctx, tc.Args)
				if err != nil {
					result = &core.ToolResult{
						ToolCallID: tc.ID,
						Content:    "",
						Error:      err.Error(),
						Success:    false,
					}
				}

				result.ToolCallID = tc.ID
				result.Success = true

				a.Ctx.AddMessage(core.Message{
					ID:        core.GenerateID(),
					Role:      core.RoleTool,
					Content:   result.Content,
					ToolCalls: []core.ToolCall{tc},
					Timestamp: time.Now(),
				})

				a.Ctx.AddEvent(core.AgentEvent{
					Type:      "tool_execution",
					Timestamp: time.Now(),
					Data:      map[string]any{"tool": tc.Name, "result": result.Content},
				})
			}
		} else {
			break
		}
	}

	return &core.AgentResult{
		Output:     output.String(),
		Messages:   a.Ctx.Messages,
		Events:     a.Ctx.History,
		StopReason: core.StopCompleted,
		TokensUsed: 0,
		Duration:   time.Since(startTime),
		Iterations: iterations,
	}, nil
}

func (a *ReActAgent) Plan(ctx context.Context, input string) (string, error) {
	prompt := fmt.Sprintf(`%s

Task: %s

Break down this task into a step-by-step plan. Be specific and clear about each step.`, a.Ctx.Config.SystemPrompt, input)

	resp, err := a.llm.Generate(ctx, prompt, core.WithTemperature(0.3))
	if err != nil {
		return "", err
	}

	return resp.Content, nil
}

func (a *ReActAgent) buildPrompt() []core.Message {
	var messages []core.Message

	if a.Ctx.Config.SystemPrompt != "" {
		messages = append(messages, core.Message{
			ID:        core.GenerateID(),
			Role:      core.RoleSystem,
			Content:   a.Ctx.Config.SystemPrompt,
			Timestamp: time.Now(),
		})
	}

	toolsDescription := buildToolsDescription(a.Ctx.Tools)
	messages = append(messages, core.Message{
		ID:        core.GenerateID(),
		Role:      core.RoleSystem,
		Content:   "You have access to the following tools:\n" + toolsDescription,
		Timestamp: time.Now(),
	})

	messages = append(messages, a.Ctx.Messages...)

	return messages
}

func buildToolsDescription(tools map[string]core.Tool) string {
	if len(tools) == 0 {
		return "No tools available."
	}

	var desc strings.Builder
	for _, tool := range tools {
		desc.WriteString(fmt.Sprintf("## %s\n", tool.Name()))
		desc.WriteString(fmt.Sprintf("%s\n", tool.Description()))
		desc.WriteString("Parameters:\n")
		for paramName, param := range tool.Parameters() {
			required := ""
			if param.Required {
				required = " (required)"
			}
			desc.WriteString(fmt.Sprintf("  - %s: %s%s\n", paramName, param.Description, required))
		}
		desc.WriteString("\n")
	}

	return desc.String()
}

func parseToolCalls(content string) ([]core.ToolCall, bool) {
	var calls []core.ToolCall
	lines := strings.Split(content, "\n")

	var currentCall *core.ToolCall
	for _, line := range lines {
		line = strings.TrimSpace(line)

		if strings.HasPrefix(line, "Action:") {
			toolName := strings.TrimSpace(strings.TrimPrefix(line, "Action:"))
			if toolName != "None" && toolName != "" {
				currentCall = &core.ToolCall{
					ID:   core.GenerateID(),
					Name: toolName,
					Args: make(map[string]any),
				}
			}
		} else if strings.HasPrefix(line, "Action Input:") {
			if currentCall != nil {
				input := strings.TrimSpace(strings.TrimPrefix(line, "Action Input:"))
				currentCall.Args["input"] = input
			}
		} else if strings.HasPrefix(line, "Thought:") || strings.HasPrefix(line, "Observation:") {
			if currentCall != nil && currentCall.Name != "" {
				calls = append(calls, *currentCall)
				currentCall = nil
			}
		}
	}

	if currentCall != nil && currentCall.Name != "" {
		calls = append(calls, *currentCall)
	}

	return calls, len(calls) > 0
}
