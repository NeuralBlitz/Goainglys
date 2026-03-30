package agents

import (
	"context"
	"fmt"
	"strings"
	"time"

	"agents/framework/core"
)

// PlanExecuteAgent implements a planner-executor pattern where a planner creates
// a list of tasks and an executor executes them one by one.
type PlanExecuteAgent struct {
	core.BaseAgent
	llm core.LLM
}

// NewPlanExecuteAgent creates a new PlanExecute agent.
func NewPlanExecuteAgent(config core.AgentConfig, llm core.LLM) *PlanExecuteAgent {
	agent := &PlanExecuteAgent{
		llm: llm,
	}
	agent.BaseAgent.Config = config
	agent.BaseAgent.Init(config.Tools)
	return agent
}

// Run executes the plan-execute loop.
func (a *PlanExecuteAgent) Run(ctx context.Context, input string) (*core.AgentResult, error) {
	startTime := time.Now()
	iterations := 0
	maxIterations := a.Ctx.Config.MaxIterations
	if maxIterations == 0 {
		maxIterations = 10
	}

	// Add user input
	a.Ctx.AddMessage(core.Message{
		ID:        core.GenerateID(),
		Role:      core.RoleUser,
		Content:   input,
		Timestamp: time.Now(),
	})

	var plan []string
	var results []string
	var currentStep int
	var output strings.Builder

	// Phase 1: Planning
	planPrompt := fmt.Sprintf(`%s
	
Task: %s

Create a numbered list of specific, actionable steps to complete this task. 
Be concise and clear. If the task cannot be broken down, return a single step.
Format each step on its own line starting with a number and period.`, a.Ctx.Config.SystemPrompt, input)

	planResp, err := a.llm.Generate(ctx, planPrompt,
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

	// Parse plan
	planLines := strings.Split(planResp.Content, "\n")
	for _, line := range planLines {
		line = strings.TrimSpace(line)
		if line != "" && (strings.HasPrefix(line, "- ") || (len(line) > 1 && line[0] >= '0' && line[0] <= '9')) {
			// Remove leading number/dash and space
			if idx := strings.IndexAny(line, "-. "); idx >= 0 {
				step := strings.TrimSpace(line[idx+1:])
				if step != "" {
					plan = append(plan, step)
				}
			}
		}
	}
	if len(plan) == 0 {
		plan = []string{input} // fallback to single step
	}

	// Add plan to context
	a.Ctx.AddMessage(core.Message{
		ID:        core.GenerateID(),
		Role:      core.RoleAgent,
		Content:   fmt.Sprintf("Plan (%d steps):\n%s", len(plan), strings.Join(plan, "\n")),
		Timestamp: time.Now(),
	})

	// Phase 2: Execute each step
	for currentStep < len(plan) && iterations < maxIterations {
		iterations++
		step := plan[currentStep]

		// Execute step
		execPrompt := fmt.Sprintf(`%s
	
Current step (%d/%d): %s
Previous results: %s

Execute this step and provide the result. If you need to use tools, do so.
If the step is complete, state so clearly.`, a.Ctx.Config.SystemPrompt, currentStep+1, len(plan), step, strings.Join(results, "; "))

		execResp, err := a.llm.GenerateWithMessages(ctx, a.buildMessages(execPrompt),
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
			Data:      map[string]any{"content": execResp.Content},
		})

		output.WriteString(fmt.Sprintf("Step %d: %s\nResult: %s\n\n", currentStep+1, step, execResp.Content))

		// Check for tool calls in execution response
		toolCalls, hasToolCalls := parseToolCalls(execResp.Content)
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
		}

		results = append(results, execResp.Content)
		currentStep++
	}

	stopReason := core.StopCompleted
	if currentStep < len(plan) {
		stopReason = core.StopMaxIterations
	} else if iterations >= maxIterations {
		stopReason = core.StopMaxIterations
	}

	return &core.AgentResult{
		Output:     output.String(),
		Messages:   a.Ctx.Messages,
		Events:     a.Ctx.History,
		StopReason: stopReason,
		TokensUsed: 0,
		Duration:   time.Since(startTime),
		Iterations: iterations,
	}, nil
}

// Plan creates a plan without execution.
func (a *PlanExecuteAgent) Plan(ctx context.Context, input string) (string, error) {
	planPrompt := fmt.Sprintf(`%s
	
Task: %s

Create a numbered list of specific, actionable steps to complete this task. 
Be concise and clear.`, a.Ctx.Config.SystemPrompt, input)

	resp, err := a.llm.Generate(ctx, planPrompt,
		core.WithTemperature(a.Ctx.Config.Temperature),
		core.WithMaxTokens(a.Ctx.Config.MaxTokens),
	)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// buildMessages adds system and tools context to a user prompt.
func (a *PlanExecuteAgent) buildMessages(userPrompt string) []core.Message {
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
	if toolsDescription != "" && toolsDescription != "No tools available." {
		messages = append(messages, core.Message{
			ID:        core.GenerateID(),
			Role:      core.RoleSystem,
			Content:   "You have access to the following tools:\n" + toolsDescription,
			Timestamp: time.Now(),
		})
	}

	messages = append(messages, core.Message{
		ID:        core.GenerateID(),
		Role:      core.RoleUser,
		Content:   userPrompt,
		Timestamp: time.Now(),
	})

	return messages
}

// Reset resets the agent context.
func (a *PlanExecuteAgent) Reset() {
	a.BaseAgent.Reset()
}
