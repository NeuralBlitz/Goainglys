package agents

import (
	"context"
	"fmt"
	"strings"
	"time"

	"agents/framework/core"
)

// SupervisorAgent manages worker agents, delegating tasks and synthesizing results.
type SupervisorAgent struct {
	core.BaseAgent
	llm     core.LLM
	workers map[string]core.Agent
	tasks   []string
	results map[string]string
}

// NewSupervisorAgent creates a new Supervisor agent with worker agents.
func NewSupervisorAgent(config core.AgentConfig, llm core.LLM, workers map[string]core.Agent) *SupervisorAgent {
	agent := &SupervisorAgent{
		llm:     llm,
		workers: workers,
		tasks:   make([]string, 0),
		results: make(map[string]string),
	}
	agent.BaseAgent.Config = config
	agent.BaseAgent.Init(config.Tools)
	return agent
}

// Run executes the supervision loop: delegate, collect, synthesize.
func (a *SupervisorAgent) Run(ctx context.Context, input string) (*core.AgentResult, error) {
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

	var output strings.Builder
	var taskList string

	// Phase 1: Task decomposition
	decompPrompt := fmt.Sprintf(`%s
	
Task: %s

Break this complex task into distinct subtasks that can be handled by specialized agents.
Available workers: %s

Return a numbered list of subtasks, one per line. Each should be clear and actionable.`,
		a.Ctx.Config.SystemPrompt, input, a.listWorkerNames())

	decompResp, err := a.llm.Generate(ctx, decompPrompt,
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

	// Parse tasks
	taskLines := strings.Split(decompResp.Content, "\n")
	for _, line := range taskLines {
		line = strings.TrimSpace(line)
		if line != "" && (strings.HasPrefix(line, "- ") || (len(line) > 1 && line[0] >= '0' && line[0] <= '9')) {
			if idx := strings.IndexAny(line, "-. "); idx >= 0 {
				task := strings.TrimSpace(line[idx+1:])
				if task != "" {
					a.tasks = append(a.tasks, task)
				}
			}
		}
	}
	if len(a.tasks) == 0 {
		a.tasks = []string{input} // fallback
	}

	taskList = strings.Join(a.tasks, "\n")
	a.Ctx.AddMessage(core.Message{
		ID:        core.GenerateID(),
		Role:      core.RoleAgent,
		Content:   fmt.Sprintf("Decomposed into %d tasks:\n%s", len(a.tasks), taskList),
		Timestamp: time.Now(),
	})

	// Phase 2: Delegate tasks to workers (round-robin for simplicity)
	var workerNames []string
	for name := range a.workers {
		workerNames = append(workerNames, name)
	}
	if len(workerNames) == 0 {
		return &core.AgentResult{
			Error:      fmt.Errorf("no worker agents available"),
			Iterations: iterations,
			Duration:   time.Since(startTime),
		}, err
	}

	for i, task := range a.tasks {
		if iterations >= maxIterations {
			break
		}
		iterations++

		workerName := workerNames[i%len(workerNames)]
		worker := a.workers[workerName]

		delegatePrompt := fmt.Sprintf(`%s
	
You are a %s agent. Your task: %s

Complete this task using your available tools and expertise.
Provide a clear, concise result.`, a.Ctx.Config.SystemPrompt, workerName, task)

		workerResult, err := worker.Run(ctx, delegatePrompt)
		if err != nil {
			a.results[task] = fmt.Sprintf("Error: %v", err)
		} else {
			a.results[task] = workerResult.Output
		}

		a.Ctx.AddMessage(core.Message{
			ID:        core.GenerateID(),
			Role:      core.RoleAgent,
			Content:   fmt.Sprintf("Delegated task '%s' to %s", task, workerName),
			Timestamp: time.Now(),
		})
	}

	// Phase 3: Synthesize results
	synthPrompt := fmt.Sprintf(`%s
	
Original task: %s
Completed subtasks:
%s

Synthesize these results into a coherent final answer. 
Identify any conflicts, gaps, or redundancies.`,
		a.Ctx.Config.SystemPrompt, input, a.formatResults())

	synthResp, err := a.llm.Generate(ctx, synthPrompt,
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

	output.WriteString(synthResp.Content)

	stopReason := core.StopCompleted
	if iterations >= maxIterations {
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

// Plan creates a supervision plan without execution.
func (a *SupervisorAgent) Plan(ctx context.Context, input string) (string, error) {
	planPrompt := fmt.Sprintf(`%s
	
Task: %s

As a supervisor, describe how you would decompose this task and delegate to your workers.
Available workers: %s`, a.Ctx.Config.SystemPrompt, input, a.listWorkerNames())

	resp, err := a.llm.Generate(ctx, planPrompt,
		core.WithTemperature(a.Ctx.Config.Temperature),
		core.WithMaxTokens(a.Ctx.Config.MaxTokens),
	)
	if err != nil {
		return "", err
	}
	return resp.Content, nil
}

// Helper methods
func (a *SupervisorAgent) listWorkerNames() string {
	if len(a.workers) == 0 {
		return "None"
	}
	names := make([]string, 0, len(a.workers))
	for name := range a.workers {
		names = append(names, name)
	}
	return strings.Join(names, ", ")
}

func (a *SupervisorAgent) formatResults() string {
	if len(a.results) == 0 {
		return "No results yet."
	}
	var lines []string
	for task, result := range a.results {
		lines = append(lines, fmt.Sprintf("- %s: %s", task, result))
	}
	return strings.Join(lines, "\n")
}

func (a *SupervisorAgent) Reset() {
	a.BaseAgent.Reset()
	a.tasks = make([]string, 0)
	a.results = make(map[string]string)
}
