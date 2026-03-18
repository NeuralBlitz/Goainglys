package core

import "time"

// AgentInterface defines the methods an agent must implement to participate in self-improvement
type AgentInterface interface {
	// ExecuteTask performs a task and returns the result and any error
	ExecuteTask(task string) (string, error)

	// GetCapabilities returns a description of the agent's current capabilities
	GetCapabilities() string

	// GetModel returns the underlying model that can be fine-tuned
	GetModel() ModelInterface

	// SaveCheckpoint saves the current state of the agent
	SaveCheckpoint(path string) error

	// LoadCheckpoint restores the agent from a saved state
	LoadCheckpoint(path string) error
}

// ModelInterface defines methods for models that can be fine-tuned
type ModelInterface interface {
	// Forward performs a forward pass
	Forward(input []float32) []float32

	// GetParameters returns the current model parameters
	GetParameters() []float32

	// ApplyUpdates applies parameter updates (used by fine-tuning)
	ApplyUpdates(updates []float32, learningRate float32)

	// GetArchitecture returns a description of the model architecture
	GetArchitecture() string
}

// Weakness represents a identified gap in agent performance
type Weakness struct {
	ID             string
	Description    string
	TaskType       string
	Difficulty     float64  // 0.0 to 1.0
	Frequency      int      // How often this weakness appears
	Examples       []string // Example tasks where this weakness occurred
	SuggestedFocus string   // What to focus on to improve
}

// ImprovementCycle represents one iteration of the self-improvement loop
type ImprovementCycle struct {
	CycleNumber         int
	StartTime           time.Time
	EndTime             time.Time
	WeaknessesAddressed []Weakness
	PerformanceBefore   float64
	PerformanceAfter    float64
	Improvement         float64
	DataGenerated       int
	TrainingSteps       int
	ParametersUpdated   int
}

// MetaLearningState tracks the meta-learning component's understanding of what works
type MetaLearningState struct {
	ImprovementHistory          []ImprovementCycle
	EffectiveStrategies         map[string]float64 // strategy -> effectiveness score
	DiminishingReturnsThreshold float64
	CurrentFocus                string
	LearningRate                float32
}

// Experience represents a successful improvement that can be replayed
type Experience struct {
	ID                string
	WeaknessType      string
	BeforePerformance float64
	AfterPerformance  float64
	Improvement       float64
	DataUsed          []string
	ParameterUpdates  []float32
	Context           map[string]string
	Timestamp         time.Time
}

// Configuration for the self-improving agent
type Config struct {
	// Improvement loop settings
	MaxCycles               int
	PerformanceTarget       float64
	MinImprovementThreshold float64

	// Data generation
	ExamplesPerWeakness   int
	DifficultyProgression float64

	// Fine-tuning
	LearningRate float32
	LoRARank     int
	LoRAAlpha    float32

	// Evaluation
	ValidationSetSize   int
	EvaluationFrequency int

	// Meta-learning
	MetaLearningRate    float32
	StrategyExploration float64

	// System
	CheckpointFrequency int
	ExperienceStoreSize int
	Device              string
}
