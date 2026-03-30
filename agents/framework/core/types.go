package core

import (
	"context"
	"fmt"
	"strings"
	"time"
)

type AgentType string

const (
	TypeReAct       AgentType = "react"
	TypePlanExecute AgentType = "plan_execute"
	TypeSupervisor  AgentType = "supervisor"
	TypeCrew        AgentType = "crew"
	TypeHandoff     AgentType = "handoff"
)

type Role string

const (
	RoleUser   Role = "user"
	RoleAgent  Role = "agent"
	RoleSystem Role = "system"
	RoleTool   Role = "tool"
)

type Message struct {
	ID         string         `json:"id"`
	Role       Role           `json:"role"`
	Content    string         `json:"content"`
	Name       string         `json:"name,omitempty"`
	ToolCalls  []ToolCall     `json:"tool_calls,omitempty"`
	ToolResult *ToolResult    `json:"tool_result,omitempty"`
	Metadata   map[string]any `json:"metadata,omitempty"`
	Timestamp  time.Time      `json:"timestamp"`
}

type ToolCall struct {
	ID   string         `json:"id"`
	Name string         `json:"name"`
	Args map[string]any `json:"args"`
}

type ToolResult struct {
	ToolCallID string `json:"tool_call_id"`
	Content    string `json:"content"`
	Error      string `json:"error,omitempty"`
	Success    bool   `json:"success"`
}

type AgentConfig struct {
	Name          string
	Description   string
	Type          AgentType
	Model         string
	Temperature   float64
	MaxTokens     int
	Tools         []Tool
	SystemPrompt  string
	MaxIterations int
	Timeout       time.Duration
}

type State map[string]any

type Context struct {
	Messages   []Message
	State      State
	Memory     Memory
	Tools      map[string]Tool
	Config     AgentConfig
	History    []AgentEvent
	StopReason string
}

func NewContext(config AgentConfig) *Context {
	return &Context{
		Messages: make([]Message, 0),
		State:    make(State),
		Tools:    make(map[string]Tool),
		Config:   config,
		History:  make([]AgentEvent, 0),
	}
}

func (c *Context) AddMessage(msg Message) {
	c.Messages = append(c.Messages, msg)
}

func (c *Context) AddEvent(event AgentEvent) {
	c.History = append(c.History, event)
}

type AgentEvent struct {
	Type      string         `json:"type"`
	Timestamp time.Time      `json:"timestamp"`
	Data      map[string]any `json:"data"`
}

type StopReason string

const (
	StopTokenLimit    StopReason = "token_limit"
	StopToolCalls     StopReason = "tool_calls"
	StopStopSequence  StopReason = "stop_sequence"
	StopMaxIterations StopReason = "max_iterations"
	StopUserAbort     StopReason = "user_abort"
	StopCompleted     StopReason = "completed"
	StopError         StopReason = "error"
)

type AgentResult struct {
	Output     string
	Messages   []Message
	Events     []AgentEvent
	StopReason StopReason
	TokensUsed int
	Duration   time.Duration
	Iterations int
	Error      error
}

type Tool interface {
	Name() string
	Description() string
	Parameters() map[string]Parameter
	Execute(ctx context.Context, args map[string]any) (*ToolResult, error)
}

type Parameter struct {
	Type        string   `json:"type"`
	Description string   `json:"description"`
	Required    bool     `json:"required"`
	Default     any      `json:"default,omitempty"`
	Enum        []string `json:"enum,omitempty"`
}

type Agent interface {
	Name() string
	Description() string
	Type() AgentType
	Run(ctx context.Context, input string) (*AgentResult, error)
	Plan(ctx context.Context, input string) (string, error)
	Reset()
}

type BaseAgent struct {
	Config AgentConfig
	Ctx    *Context
}

func (b *BaseAgent) Name() string        { return b.Config.Name }
func (b *BaseAgent) Description() string { return b.Config.Description }
func (b *BaseAgent) Type() AgentType     { return b.Config.Type }

func (b *BaseAgent) Init(tools []Tool) {
	b.Ctx = NewContext(b.Config)
	for _, t := range tools {
		b.Ctx.Tools[t.Name()] = t
	}
}

func (b *BaseAgent) Reset() {
	b.Ctx = NewContext(b.Config)
	for name, tool := range b.Ctx.Tools {
		b.Ctx.Tools[name] = tool
	}
}

func (b *BaseAgent) Context() *Context {
	return b.Ctx
}

type FunctionTool struct {
	name        string
	description string
	params      map[string]Parameter
	fn          func(ctx context.Context, args map[string]any) (*ToolResult, error)
}

func NewFunctionTool(name, description string, fn func(ctx context.Context, args map[string]any) (*ToolResult, error)) *FunctionTool {
	return &FunctionTool{
		name:        name,
		description: description,
		params:      make(map[string]Parameter),
		fn:          fn,
	}
}

func (t *FunctionTool) WithParams(params map[string]Parameter) *FunctionTool {
	t.params = params
	return t
}

func (t *FunctionTool) Name() string                     { return t.name }
func (t *FunctionTool) Description() string              { return t.description }
func (t *FunctionTool) Parameters() map[string]Parameter { return t.params }

func (t *FunctionTool) Execute(ctx context.Context, args map[string]any) (*ToolResult, error) {
	return t.fn(ctx, args)
}

type Memory interface {
	Add(msg Message)
	GetMessages(n int) []Message
	Search(query string, limit int) []Message
	Clear()
	Summary() string
}

type AgentRegistry struct {
	agents map[string]Agent
}

func NewAgentRegistry() *AgentRegistry {
	return &AgentRegistry{agents: make(map[string]Agent)}
}

func (r *AgentRegistry) Register(name string, agent Agent) {
	r.agents[name] = agent
}

func (r *AgentRegistry) Get(name string) (Agent, bool) {
	agent, ok := r.agents[name]
	return agent, ok
}

func (r *AgentRegistry) List() []Agent {
	agents := make([]Agent, 0, len(r.agents))
	for _, a := range r.agents {
		agents = append(agents, a)
	}
	return agents
}

func RunAgent(agent Agent, input string) (*AgentResult, error) {
	ctx := context.Background()
	return agent.Run(ctx, input)
}

// buildToolsDescription formats tools for inclusion in agent prompts.
// Shared by ReAct, PlanExecute, and other agents.
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

func GenerateID() string {
	return fmt.Sprintf("msg_%d_%d", time.Now().UnixNano(), time.Now().UnixNano()%10000)
}
