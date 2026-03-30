package agents

import (
	"context"
	"fmt"
	"strings"
	"time"

	"agents/framework/core"
)

// HandoffAgent represents an agent that can delegate control to another agent
// based on conditions or task completion.
type HandoffAgent struct {
	core.BaseAgent
	llm          core.LLM
	agents       map[string]core.Agent          // name -> agent
	currentAgent string                         // name of currently active agent
	handoffHooks map[string]func(string) string // condition -> next agent name
	maxHandoffs  int                            // prevent infinite handoff loops
}

// NewHandoffAgent creates a new Handoff agent.
func NewHandoffAgent(config core.AgentConfig, llm core.LLM, agents map[string]core.Agent, hooks map[string]func(string) string) *HandoffAgent {
	if len(agents) == 0 {
		panic("HandoffAgent requires at least one agent")
	}

	// Set first agent as default if not specified
	firstName := ""
	for name := range agents {
		if firstName == "" {
			firstName = name
			break
		}
	}

	handoff := &HandoffAgent{
		llm:          llm,
		agents:       agents,
		currentAgent: firstName,
		handoffHooks: hooks,
		maxHandoffs:  10,
	}
	handoff.BaseAgent.Config = config
	handoff.BaseAgent.Init(config.Tools)
	return handoff
}

// Run executes with potential handoffs between agents.
func (h *HandoffAgent) Run(ctx context.Context, input string) (*core.AgentResult, error) {
	startTime := time.Now()
	iterations := 0
	maxIterations := h.Ctx.Config.MaxIterations
	if maxIterations == 0 {
		maxIterations = 10
	}

	// Add user input
	h.Ctx.AddMessage(core.Message{
		ID:        core.GenerateID(),
		Role:      core.RoleUser,
		Content:   input,
		Timestamp: time.Now(),
	})

	var output strings.Builder
	var handoffCount int
	var agentTrace []string

	// Track the execution path
	currentInput := input

	for iterations < maxIterations && handoffCount < h.maxHandoffs {
		iterations++

		// Get current agent
		agent, exists := h.agents[h.currentAgent]
		if !exists {
			return &core.AgentResult{
				Error:      fmt.Errorf("agent '%s' not found", h.currentAgent),
				Iterations: iterations,
				Duration:   time.Since(startTime),
			}, nil
		}

		// Record agent in trace
		agentTrace = append(agentTrace, h.currentAgent)

		// Run current agent
		agentResult, err := agent.Run(ctx, currentInput)
		if err != nil {
			return &core.AgentResult{
				Error:      err,
				Iterations: iterations,
				Duration:   time.Since(startTime),
			}, err
		}

		// Add result to output
		if agentResult.Output != "" {
			output.WriteString(fmt.Sprintf("[Agent: %s]\n%s\n\n", h.currentAgent, agentResult.Output))
		}

		// Add agent messages to context
		h.Ctx.Messages = append(h.Ctx.Messages, agentResult.Messages...)
		h.Ctx.History = append(h.Ctx.History, agentResult.Events...)

		// Check for handoff conditions
		nextAgent := h.shouldHandoff(agentResult.Output)
		if nextAgent == "" || nextAgent == h.currentAgent {
			// No handoff, we're done
			break
		}

		// Prepare for handoff
		handoffCount++
		h.Ctx.AddMessage(core.Message{
			ID:   core.GenerateID(),
			Role: core.RoleAgent,
			Content: fmt.Sprintf("Handoff from %s to %s (reason: %s)",
				h.currentAgent, nextAgent, agentResult.Output),
			Timestamp: time.Now(),
		})

		// Update current agent and prepare next input
		h.currentAgent = nextAgent
		// Next input includes the previous result for context
		currentInput = fmt.Sprintf("Previous agent (%s) result:\n%s\n\nNow continue with: %s",
			agentTrace[len(agentTrace)-1], agentResult.Output, input)
	}

	// Determine stop reason
	stopReason := core.StopCompleted
	if iterations >= maxIterations {
		stopReason = core.StopMaxIterations
	} else if handoffCount >= h.maxHandoffs {
		stopReason = core.StopError // infinite loop protection
	}

	// Add trace to output for debugging
	if len(agentTrace) > 1 {
		output.WriteString(fmt.Sprintf("Agent execution path: %s\n", strings.Join(agentTrace, " -> ")))
	}

	return &core.AgentResult{
		Output:     output.String(),
		Messages:   h.Ctx.Messages,
		Events:     h.Ctx.History,
		StopReason: stopReason,
		TokensUsed: 0,
		Duration:   time.Since(startTime),
		Iterations: iterations,
	}, nil
}

// shouldHandoff checks if the result indicates a handoff should occur.
// Returns the name of the next agent, or empty string to stop.
func (h *HandoffAgent) shouldHandoff(result string) string {
	// Check registered handoff hooks
	for condition, hook := range h.handoffHooks {
		if strings.Contains(result, condition) {
			next := hook(result)
			if next != "" && h.agents[next] != nil {
				return next
			}
		}
	}

	// Default handoff patterns: look for explicit handoff statements
	lower := strings.ToLower(result)
	if strings.Contains(lower, "handoff to") || strings.Contains(lower, "transfer to") {
		// Simple extraction: look for agent names after keywords
		for name := range h.agents {
			if strings.Contains(lower, strings.ToLower(name)) {
				return name
			}
		}
	}

	return ""
}

// Plan describes how the handoff agent would approach the task.
func (h *HandoffAgent) Plan(ctx context.Context, input string) (string, error) {
	agentList := make([]string, 0, len(h.agents))
	for name := range h.agents {
		agentList = append(agentList, name)
	}

	planPrompt := fmt.Sprintf(`%s
	
Task: %s
Available agents: %s

Describe how you would approach this task, including potential handoffs between agents 
based on intermediate results or conditions.`,
		h.Ctx.Config.SystemPrompt, input, strings.Join(agentList, ", "))

	resp, err := h.llm.Generate(ctx, planPrompt,
		core.WithTemperature(h.Ctx.Config.Temperature),
		core.WithMaxTokens(h.Ctx.Config.MaxTokens),
	)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// SetCurrentAgent changes the active agent (for external control).
func (h *HandoffAgent) SetCurrentAgent(name string) {
	if _, exists := h.agents[name]; exists {
		h.currentAgent = name
	}
}

// GetCurrentAgent returns the name of the currently active agent.
func (h *HandoffAgent) GetCurrentAgent() string {
	return h.currentAgent
}

// AddHandoffHook adds a condition that triggers a handoff to the specified agent.
func (h *HandoffAgent) AddHandoffHook(condition string, nextAgent string, hookFunc func(string) string) {
	if hookFunc == nil {
		hookFunc = func(string) string { return nextAgent }
	}
	h.handoffHooks[condition] = hookFunc
}

// Reset resets the agent context.
func (h *HandoffAgent) Reset() {
	h.BaseAgent.Reset()
	h.currentAgent = ""
	// Reset to first agent
	for name := range h.agents {
		h.currentAgent = name
		break
	}
}
