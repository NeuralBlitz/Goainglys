package core

import (
	"context"
	"fmt"
	"math/rand"
	"time"
)

type LLM interface {
	Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error)
	GenerateWithMessages(ctx context.Context, messages []Message, opts ...Option) (*Response, error)
}

type Response struct {
	Content      string
	StopReason   StopReason
	TokensUsed   int
	FinishReason string
	Model        string
	ToolCalls    []ToolCall
}

type Option func(*options)

type options struct {
	model       string
	temperature float64
	maxTokens   int
	stop        []string
	tools       []Tool
}

func WithModel(model string) Option {
	return func(o *options) { o.model = model }
}

func WithTemperature(temp float64) Option {
	return func(o *options) { o.temperature = temp }
}

func WithMaxTokens(tokens int) Option {
	return func(o *options) { o.maxTokens = tokens }
}

func WithStop(stop []string) Option {
	return func(o *options) { o.stop = stop }
}

func WithTools(tools []Tool) Option {
	return func(o *options) { o.tools = tools }
}

type MockLLM struct {
	responses     map[string][]string
	toolResponses map[string]string
	delay         time.Duration
	responseIndex map[string]int
}

func NewMockLLM() *MockLLM {
	return &MockLLM{
		responses:     make(map[string][]string),
		toolResponses: make(map[string]string),
		delay:         time.Millisecond * 100,
		responseIndex: make(map[string]int),
	}
}

func (m *MockLLM) AddResponse(prompt string, responses ...string) {
	m.responses[prompt] = responses
}

func (m *MockLLM) AddToolResponse(toolName, response string) {
	m.toolResponses[toolName] = response
}

func (m *MockLLM) SetDelay(delay time.Duration) {
	m.delay = delay
}

func (m *MockLLM) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	time.Sleep(m.delay)

	o := &options{
		model:       "mock-model",
		temperature: 0.7,
		maxTokens:   2048,
	}
	for _, opt := range opts {
		opt(o)
	}

	var content string
	if responses, ok := m.responses[prompt]; ok && len(responses) > 0 {
		idx := m.responseIndex[prompt]
		if idx >= len(responses) {
			idx = 0
		}
		content = responses[idx]
		m.responseIndex[prompt] = idx + 1
	} else {
		content = m.generateDefaultResponse(prompt)
	}

	return &Response{
		Content:      content,
		StopReason:   StopStopSequence,
		TokensUsed:   len(content) / 4,
		Model:        o.model,
		FinishReason: "stop",
	}, nil
}

func (m *MockLLM) GenerateWithMessages(ctx context.Context, messages []Message, opts ...Option) (*Response, error) {
	var prompt string
	for _, msg := range messages {
		prompt += msg.Content + "\n"
	}
	return m.Generate(ctx, prompt, opts...)
}

func (m *MockLLM) generateDefaultResponse(prompt string) string {
	responses := []string{
		"I understand your request. Let me help you with that.",
		"I'll analyze this and provide a solution.",
		"Based on my understanding, here's what I can do.",
		"Let me think about this step by step.",
		"I'll help you accomplish this task.",
	}
	return responses[rand.Intn(len(responses))]
}

type StreamingLLM struct {
	delegate LLM
}

func NewStreamingLLM(delegate LLM) *StreamingLLM {
	return &StreamingLLM{delegate: delegate}
}

func (s *StreamingLLM) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	return s.delegate.Generate(ctx, prompt, opts...)
}

func (s *StreamingLLM) GenerateWithMessages(ctx context.Context, messages []Message, opts ...Option) (*Response, error) {
	return s.delegate.GenerateWithMessages(ctx, messages, opts...)
}

type CachedLLM struct {
	delegate LLM
	cache    map[string]*Response
	Hits     int
	Misses   int
}

func NewCachedLLM(delegate LLM) *CachedLLM {
	return &CachedLLM{
		delegate: delegate,
		cache:    make(map[string]*Response),
	}
}

func (c *CachedLLM) Generate(ctx context.Context, prompt string, opts ...Option) (*Response, error) {
	if resp, ok := c.cache[prompt]; ok {
		c.Hits++
		return resp, nil
	}
	c.Misses++
	resp, err := c.delegate.Generate(ctx, prompt, opts...)
	if err == nil {
		c.cache[prompt] = resp
	}
	return resp, err
}

func (c *CachedLLM) GenerateWithMessages(ctx context.Context, messages []Message, opts ...Option) (*Response, error) {
	var prompt string
	for _, msg := range messages {
		prompt += msg.Role.String() + ":" + msg.Content + "\n"
	}
	return c.Generate(ctx, prompt, opts...)
}

func (r Role) String() string {
	switch r {
	case RoleUser:
		return "user"
	case RoleAgent:
		return "assistant"
	case RoleSystem:
		return "system"
	case RoleTool:
		return "tool"
	default:
		return "user"
	}
}

func ParseRole(s string) Role {
	switch s {
	case "user":
		return RoleUser
	case "assistant", "agent":
		return RoleAgent
	case "system":
		return RoleSystem
	case "tool":
		return RoleTool
	default:
		return RoleUser
	}
}

func FormatMessages(messages []Message) string {
	var result string
	for _, msg := range messages {
		result += fmt.Sprintf("%s: %s\n", msg.Role.String(), msg.Content)
	}
	return result
}
