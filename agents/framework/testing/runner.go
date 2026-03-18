package testing

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"agents/framework/core"
)

type TestCase struct {
	Name        string
	Input       string
	Expected    string
	ExpectedErr bool
	Timeout     time.Duration
	Setup       func(*core.BaseAgent)
	Validate    func(*core.AgentResult) bool
}

type TestSuite struct {
	Name      string
	AgentType core.AgentType
	AgentFn   func() core.Agent
	Tests     []TestCase
}

type TestResult struct {
	Name     string
	Passed   bool
	Duration time.Duration
	Error    error
	Output   string
	Expected string
	Actual   string
}

type TestRunner struct {
	suites    []TestSuite
	results   []TestResult
	verbosity VerbosityLevel
	report    *TestReport
}

type VerbosityLevel int

const (
	VerbosityQuiet VerbosityLevel = iota
	VerbosityNormal
	VerbosityVerbose
	VerbosityDebug
)

type TestReport struct {
	TotalTests   int
	PassedTests  int
	FailedTests  int
	SkippedTests int
	Duration     time.Duration
	Results      []TestResult
	Failures     []TestResult
}

func NewTestRunner() *TestRunner {
	return &TestRunner{
		results:   make([]TestResult, 0),
		verbosity: VerbosityNormal,
		report:    &TestReport{},
	}
}

func (tr *TestRunner) AddSuite(suite TestSuite) {
	tr.suites = append(tr.suites, suite)
}

func (tr *TestRunner) Run() *TestReport {
	startTime := time.Now()

	for _, suite := range tr.suites {
		tr.runSuite(suite)
	}

	tr.report.Duration = time.Since(startTime)
	tr.report.TotalTests = len(tr.results)
	for _, r := range tr.results {
		if r.Passed {
			tr.report.PassedTests++
		} else {
			tr.report.FailedTests++
			tr.report.Failures = append(tr.report.Failures, r)
		}
	}

	return tr.report
}

func (tr *TestRunner) runSuite(suite TestSuite) {
	for _, tc := range suite.Tests {
		result := tr.runTest(suite, tc)
		tr.results = append(tr.results, result)
	}
}

func (tr *TestRunner) runTest(suite TestSuite, tc TestCase) TestResult {
	startTime := time.Now()

	agent := suite.AgentFn()

	if tc.Setup != nil {
		if ba, ok := any(agent).(interface{ Reset() }); ok {
			ba.Reset()
		}
	}

	timeout := tc.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	done := make(chan *core.AgentResult, 1)
	var result *core.AgentResult
	var runErr error

	go func() {
		result, runErr = agent.Run(ctx, tc.Input)
		done <- result
	}()

	select {
	case <-ctx.Done():
		return TestResult{
			Name:     tc.Name,
			Passed:   false,
			Duration: time.Since(startTime),
			Error:    fmt.Errorf("test timed out after %v", timeout),
		}
	case <-done:
		if runErr != nil {
			return TestResult{
				Name:     tc.Name,
				Passed:   tc.ExpectedErr,
				Duration: time.Since(startTime),
				Error:    runErr,
				Output:   runErr.Error(),
			}
		}
	}

	passed := tr.evaluateResult(result, tc)

	return TestResult{
		Name:     tc.Name,
		Passed:   passed,
		Duration: time.Since(startTime),
		Output:   result.Output,
		Expected: tc.Expected,
		Actual:   result.Output,
	}
}

func (tr *TestRunner) evaluateResult(result *core.AgentResult, tc TestCase) bool {
	if tc.Validate != nil {
		return tc.Validate(result)
	}

	if tc.ExpectedErr {
		return result.Error != nil
	}

	if tc.Expected == "" {
		return true
	}

	normalizedExpected := strings.ToLower(strings.TrimSpace(tc.Expected))
	normalizedOutput := strings.ToLower(strings.TrimSpace(result.Output))

	if tr.verbosity >= VerbosityDebug {
		fmt.Printf("  Expected: %s\n", normalizedExpected)
		fmt.Printf("  Actual:   %s\n", normalizedOutput)
	}

	return strings.Contains(normalizedOutput, normalizedExpected)
}

func (tr *TestRunner) PrintReport() {
	report := tr.report

	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Printf("TEST REPORT: %s\n", time.Now().Format("2006-01-02 15:04:05"))
	fmt.Println(strings.Repeat("=", 60))

	fmt.Printf("\nTotal:   %d tests\n", report.TotalTests)
	fmt.Printf("Passed:  %d\n", report.PassedTests)
	fmt.Printf("Failed:  %d\n", report.FailedTests)
	fmt.Printf("Duration: %v\n", report.Duration)

	if report.FailedTests > 0 {
		fmt.Println("\n" + strings.Repeat("-", 60))
		fmt.Println("FAILED TESTS:")
		fmt.Println(strings.Repeat("-", 60))

		for _, f := range report.Failures {
			fmt.Printf("\n  FAIL: %s\n", f.Name)
			fmt.Printf("  Duration: %v\n", f.Duration)
			if f.Error != nil {
				fmt.Printf("  Error: %v\n", f.Error)
			}
			if f.Expected != "" {
				fmt.Printf("  Expected: %s\n", f.Expected)
			}
			if f.Output != "" {
				fmt.Printf("  Output: %s\n", truncate(f.Output, 200))
			}
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 60))
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func (tr *TestRunner) Assert(t *testing.T) {
	report := tr.Run()
	tr.PrintReport()

	if report.FailedTests > 0 {
		t.Errorf("%d tests failed", report.FailedTests)
	}
}

func (tr *TestRunner) AssertAllPassed(t *testing.T) {
	report := tr.Run()

	if report.FailedTests > 0 {
		for _, f := range report.Failures {
			t.Errorf("Test '%s' failed: %v", f.Name, f.Error)
		}
	}
}

func CreateMockAgent(responses map[string]string) *MockTestAgent {
	return &MockTestAgent{
		responses: responses,
	}
}

type MockTestAgent struct {
	responses map[string]string
	calls     []string
}

func NewMockTestAgent(responses map[string]string) *MockTestAgent {
	return &MockTestAgent{
		responses: responses,
	}
}

func (m *MockTestAgent) Name() string         { return "MockAgent" }
func (m *MockTestAgent) Description() string  { return "Mock agent for testing" }
func (m *MockTestAgent) Type() core.AgentType { return core.TypeReAct }

func (m *MockTestAgent) Run(ctx context.Context, input string) (*core.AgentResult, error) {
	m.calls = append(m.calls, input)

	response := "Mock response"
	if r, ok := m.responses[input]; ok {
		response = r
	}

	return &core.AgentResult{
		Output:     response,
		Messages:   []core.Message{},
		Events:     []core.AgentEvent{},
		StopReason: core.StopCompleted,
		TokensUsed: len(response) / 4,
		Duration:   time.Millisecond * 10,
		Iterations: 1,
	}, nil
}

func (m *MockTestAgent) Plan(ctx context.Context, input string) (string, error) {
	return m.responses[input], nil
}

func (m *MockTestAgent) Reset() {
	m.calls = nil
}

func (m *MockTestAgent) GetCalls() []string {
	return m.calls
}

func RunBenchmark(agent core.Agent, inputs []string, iterations int) BenchmarkResult {
	results := make([]time.Duration, iterations*len(inputs))
	totalTokens := 0

	startTime := time.Now()

	for i, input := range inputs {
		for j := 0; j < iterations; j++ {
			idx := i*iterations + j
			iterStart := time.Now()

			result, _ := agent.Run(context.Background(), input)
			results[idx] = time.Since(iterStart)
			totalTokens += result.TokensUsed
		}
	}

	totalDuration := time.Since(startTime)

	var total time.Duration
	for _, d := range results {
		total += d
	}
	avgDuration := total / time.Duration(len(results))

	var minDuration, maxDuration time.Duration = results[0], results[0]
	for _, d := range results[1:] {
		if d < minDuration {
			minDuration = d
		}
		if d > maxDuration {
			maxDuration = d
		}
	}

	return BenchmarkResult{
		TotalDuration: totalDuration,
		AvgDuration:   avgDuration,
		MinDuration:   minDuration,
		MaxDuration:   maxDuration,
		TotalTokens:   totalTokens,
		Iterations:    iterations * len(inputs),
	}
}

type BenchmarkResult struct {
	TotalDuration time.Duration
	AvgDuration   time.Duration
	MinDuration   time.Duration
	MaxDuration   time.Duration
	TotalTokens   int
	Iterations    int
}

func (r BenchmarkResult) Print() {
	fmt.Println("\n" + strings.Repeat("-", 40))
	fmt.Println("BENCHMARK RESULTS")
	fmt.Println(strings.Repeat("-", 40))
	fmt.Printf("Total Duration: %v\n", r.TotalDuration)
	fmt.Printf("Avg Duration:   %v\n", r.AvgDuration)
	fmt.Printf("Min Duration:   %v\n", r.MinDuration)
	fmt.Printf("Max Duration:   %v\n", r.MaxDuration)
	fmt.Printf("Total Tokens:   %d\n", r.TotalTokens)
	fmt.Printf("Iterations:     %d\n", r.Iterations)
	fmt.Println(strings.Repeat("-", 40))
}
