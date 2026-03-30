package agents

import (
	"context"
	"fmt"
	"strings"
	"time"

	"agents/framework/core"
)

// CrewAgent represents a team of agents working collaboratively on a task.
// Agents can share context and build upon each other's work.
type CrewAgent struct {
	core.BaseAgent
	llm    core.LLM
	agents []core.Agent
	shared []string // shared context/context notes
}

// NewCrewAgent creates a new Crew agent.
func NewCrewAgent(config core.AgentConfig, llm core.LLM, agents []core.Agent) *CrewAgent {
	crew := &CrewAgent{
		llm:    llm,
		agents: agents,
		shared: make([]string, 0),
	}
	crew.BaseAgent.Config = config
	crew.BaseAgent.Init(config.Tools)
	return crew
}

// Run executes the crew collaboration process.
func (c *CrewAgent) Run(ctx context.Context, input string) (*core.AgentResult, error) {
	startTime := time.Now()
	iterations := 0
	maxIterations := c.Ctx.Config.MaxIterations
	if maxIterations == 0 {
		maxIterations = 10
	}

	// Add user input
	c.Ctx.AddMessage(core.Message{
		ID:        core.GenerateID(),
		Role:      core.RoleUser,
		Content:   input,
		Timestamp: time.Now(),
	})

	var output strings.Builder
	var agentResults []string

	// Phase 1: Context sharing
	if len(c.shared) > 0 {
		contextMsg := fmt.Sprintf("Shared context from previous work:\n%s", strings.Join(c.shared, "\n"))
		c.Ctx.AddMessage(core.Message{
			ID:        core.GenerateID(),
			Role:      core.RoleAgent,
			Content:   contextMsg,
			Timestamp: time.Now(),
		})
	}

	// Phase 2: Each agent works on the task (can build on previous results)
	for i, agent := range c.agents {
		if iterations >= maxIterations {
			break
		}
		iterations++

		// Build prompt with previous agent results
		var prevResults string
		if len(agentResults) > 0 {
			prevResults = strings.Join(agentResults, "\n\n")
		}

		agentPrompt := fmt.Sprintf(`%s
	
Task: %s
%s

You are agent %d/%d in a crew working on this task.
Review any previous results above, then provide your contribution.
If you're the first agent, start fresh.
If you're building on others, acknowledge and extend their work.`,
			c.Ctx.Config.SystemPrompt, input,
			func() string {
				if prevResults == "" {
					return ""
				}
				return "Previous agent results:\n" + prevResults
			}(), i+1, len(c.agents))

		agentResult, err := agent.Run(ctx, agentPrompt)
		if err != nil {
			agentResults = append(agentResults, fmt.Sprintf("Agent %d error: %v", i+1, err))
		} else {
			agentResults = append(agentResults, agentResult.Output)
			// Extract key insights for shared context
			if len(agentResult.Output) > 50 {
				c.shared = append(c.shared, fmt.Sprintf("Agent %d insight: %s", i+1,
					func() string {
						if len(agentResult.Output) > 100 {
							return agentResult.Output[:100] + "..."
						}
						return agentResult.Output
					}()))
			}
		}

		c.Ctx.AddMessage(core.Message{
			ID:        core.GenerateID(),
			Role:      core.RoleAgent,
			Content:   fmt.Sprintf("Agent %d completed work", i+1),
			Timestamp: time.Now(),
		})
	}

	// Phase 3: Synthesis/integration
	if len(agentResults) > 0 {
		synthPrompt := fmt.Sprintf(`%s
	
Task: %s
Crew member results:
%s

As the crew lead, synthesize these contributions into a unified final answer.
Resolve contradictions, fill gaps, and ensure completeness.`,
			c.Ctx.Config.SystemPrompt, input, strings.Join(agentResults, "\n\n---\n\n"))

		synthResp, err := c.llm.Generate(ctx, synthPrompt,
			core.WithTemperature(c.Ctx.Config.Temperature),
			core.WithMaxTokens(c.Ctx.Config.MaxTokens),
		)
		if err != nil {
			return &core.AgentResult{
				Error:      err,
				Iterations: iterations,
				Duration:   time.Since(startTime),
			}, err
		}

		output.WriteString(synthResp.Content)
	} else {
		output.WriteString("No agent results to synthesize.")
	}

	stopReason := core.StopCompleted
	if iterations >= maxIterations {
		stopReason = core.StopMaxIterations
	}

	return &core.AgentResult{
		Output:     output.String(),
		Messages:   c.Ctx.Messages,
		Events:     c.Ctx.History,
		StopReason: stopReason,
		TokensUsed: 0,
		Duration:   time.Since(startTime),
		Iterations: iterations,
	}, nil
}

// Plan describes how the crew would approach the task.
func (c *CrewAgent) Plan(ctx context.Context, input string) (string, error) {
	planPrompt := fmt.Sprintf(`%s
	
Task: %s
Crew size: %d agents

Describe how this crew would collaborate on the task. 
Include how agents would share context, divide work, and synthesize results.`,
		c.Ctx.Config.SystemPrompt, input, len(c.agents))

	resp, err := c.llm.Generate(ctx, planPrompt,
		core.WithTemperature(c.Ctx.Config.Temperature),
		core.WithMaxTokens(c.Ctx.Config.MaxTokens),
	)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// AddToSharedContext adds notes to the crew's shared context.
func (c *CrewAgent) AddToSharedContext(note string) {
	c.shared = append(c.shared, note)
}

// Reset resets the agent context.
func (c *CrewAgent) Reset() {
	c.BaseAgent.Reset()
	c.shared = make([]string, 0)
}
