package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"time"

	"self_improving_agents/core"
	"self_improving_agents/data_generator"
	"self_improving_agents/experience_store"
	"self_improving_agents/improvement_loop"
	"self_improving_agents/integrations/agents"
	"self_improving_agents/integrations/dashboard"
	"self_improving_agents/integrations/finetune"
	"self_improving_agents/integrations/model_registry"
	"self_improving_agents/integrations/rag_eval"
	"self_improving_agents/integrations/transformers"
	"self_improving_agents/meta_learner"
	"self_improving_agents/weakness_detector"
)

// MathAgentAdapter adapts our agent interface to handle mathematical tasks
type MathAgentAdapter struct {
	baseAgent    core.AgentInterface
	difficulty   float64
	solvingSkill float64 // 0.0 to 1.0, represents current ability
}

// NewMathAgentAdapter creates a math-specific agent adapter
func NewMathAgentAdapter(baseAgent core.AgentInterface) *MathAgentAdapter {
	return &MathAgentAdapter{
		baseAgent:    baseAgent,
		difficulty:   0.5, // Start with medium difficulty
		solvingSkill: 0.3, // Start with low ability
	}
}

// ExecuteTask handles mathematical tasks
func (a *MathAgentAdapter) ExecuteTask(task string) (string, error) {
	// Simulate solving a mathematical problem
	// In a real implementation, this would use the underlying agent/model

	// Simple simulation: probability of success based on skill vs difficulty
	successProbability := a.solvingSkill / (a.difficulty + 0.1) // Add small constant to avoid division by zero
	if successProbability > 1.0 {
		successProbability = 1.0
	}

	if rand.Float64() < successProbability {
		return "Correct solution: x = 2", nil
	}
	return "", fmt.Errorf("failed to solve: %s", task)
}

// GetCapabilities returns a description of the agent's current capabilities
func (a *MathAgentAdapter) GetCapabilities() string {
	return fmt.Sprintf("Mathematical reasoning agent (skill: %.2f, handles difficulty up to: %.2f)",
		a.solvingSkill, a.difficulty)
}

// GetModel returns the underlying model (delegated to base agent)
func (a *MathAgentAdapter) GetModel() core.ModelInterface {
	return a.baseAgent.GetModel()
}

// SaveCheckpoint saves the current state of the agent
func (a *MathAgentAdapter) SaveCheckpoint(path string) error {
	return a.baseAgent.SaveCheckpoint(path)
}

// LoadCheckpoint restores the agent from a saved state
func (a *MathAgentAdapter) LoadCheckpoint(path string) error {
	return a.baseAgent.LoadCheckpoint(path)
}

// ImproveSkill increases the agent's skill level based on training
func (a *MathAgentAdapter) ImproveSkill(improvement float64) {
	a.solvingSkill += improvement
	if a.solvingSkill > 1.0 {
		a.solvingSkill = 1.0
	}
	// Optionally increase difficulty capacity as skill improves
	if a.solvingSkill > 0.8 {
		a.difficulty = 0.8
	}
}

// DecreaseSkill simulates skill degradation (for testing)
func (a *MathAgentAdapter) DecreaseSkill(amount float64) {
	a.solvingSkill -= amount
	if a.solvingSkill < 0.0 {
		a.solvingSkill = 0.0
	}
}

// SelfImprovingMathAgent orchestrates the self-improvement process for a math agent
type SelfImprovingMathAgent struct {
	config               core.Config
	agentAdapter         agents.Adapter
	modelAdapter         transformers.Adapter
	fineTuneAdapter      finetune.Adapter
	ragEvalAdapter       rag_eval.Adapter
	modelRegistryAdapter model_registry.Adapter
	dashboardAdapter     dashboard.Adapter

	weaknessDetector *weakness_detector.Detector
	dataGenerator    *data_generator.Generator
	improvementLoop  *improvement_loop.Loop
	metaLearner      *meta_learner.Learner
	experienceStore  *experience_store.Store

	currentAgent *MathAgentAdapter
	currentModel core.ModelInterface
}

// NewSelfImprovingMathAgent creates a new self-improving math agent
func NewSelfImprovingMathAgent(
	config core.Config,
	agentAdapter agents.Adapter,
	modelAdapter transformers.Adapter,
	fineTuneAdapter finetune.Adapter,
	ragEvalAdapter rag_eval.Adapter,
	modelRegistryAdapter model_registry.Adapter,
	dashboardAdapter dashboard.Adapter,
) *SelfImprovingMathAgent {
	// Initialize the base agent and model from adapters
	baseAgent := agentAdapter.GetAgent()
	model := modelAdapter.GetModel()

	// Create the math-specific agent adapter
	mathAgent := NewMathAgentAdapter(baseAgent)

	// Initialize components
	wd := weakness_detector.NewDetector(ragEvalAdapter)
	dg := data_generator.NewGenerator(modelAdapter)
	il := improvement_loop.NewLoop(fineTuneAdapter, ragEvalAdapter)
	ml := meta_learner.NewLearner()
	es := experience_store.NewStore(config.ExperienceStoreSize)

	return &SelfImprovingMathAgent{
		config:               config,
		agentAdapter:         agentAdapter,
		modelAdapter:         modelAdapter,
		fineTuneAdapter:      fineTuneAdapter,
		ragEvalAdapter:       ragEvalAdapter,
		modelRegistryAdapter: modelRegistryAdapter,
		dashboardAdapter:     dashboardAdapter,

		weaknessDetector: wd,
		dataGenerator:    dg,
		improvementLoop:  il,
		metaLearner:      ml,
		experienceStore:  es,

		currentAgent: mathAgent,
		currentModel: model,
	}
}

// Improve runs the recursive self-improvement process for the given task
func (sia *SelfImprovingMathAgent) Improve(task string) ([]core.ImprovementCycle, error) {
	var improvementHistory []core.ImprovementCycle

	// Register the agent with the model registry for versioning
	modelID, err := sia.modelRegistryAdapter.RegisterModel(
		sia.currentModel.GetArchitecture(),
		map[string]string{
			"task":                task,
			"initial_performance": "0.0",
		},
	)
	if err != nil {
		return nil, fmt.Errorf("failed to register model: %w", err)
	}
	_ = modelID // Suppress unused variable warning

	// Start the dashboard for monitoring
	if err := sia.dashboardAdapter.Start(); err != nil {
		return nil, fmt.Errorf("failed to start dashboard: %w", err)
	}
	defer sia.dashboardAdapter.Stop()

	// Initial evaluation
	initialPerformance, err := sia.ragEvalAdapter.Evaluate(sia.currentAgent, task, sia.config.ValidationSetSize)
	if err != nil {
		return nil, fmt.Errorf("failed to evaluate initial performance: %w", err)
	}

	for cycle := 0; cycle < sia.config.MaxCycles; cycle++ {
		fmt.Printf("\n--- Cycle %d/%d ---\n", cycle+1, sia.config.MaxCycles)

		// Step 1: Detect weaknesses from recent performance
		weaknesses, err := sia.weaknessDetector.Detect(sia.currentAgent, task, sia.config.ValidationSetSize)
		if err != nil {
			return nil, fmt.Errorf("weakness detection failed: %w", err)
		}
		if len(weaknesses) == 0 {
			fmt.Println("No weaknesses detected. Improvement may have plateaued.")
			break
		}
		fmt.Printf("Detected %d weaknesses\n", len(weaknesses))

		// Step 2: Generate synthetic training data for the weaknesses
		trainingData, err := sia.dataGenerator.Generate(weaknesses, sia.config.ExamplesPerWeakness, sia.config.DifficultyProgression)
		if err != nil {
			return nil, fmt.Errorf("data generation failed: %w", err)
		}
		fmt.Printf("Generated %d training examples\n", len(trainingData))

		// Step 3: Perform efficient fine-tuning (LoRA)
		startTime := time.Now()
		updatedModel, trainingStats, err := sia.improvementLoop.Improve(
			sia.currentModel,
			trainingData,
			sia.config.LearningRate,
			sia.config.LoRARank,
			sia.config.LoRAAlpha,
		)
		if err != nil {
			return nil, fmt.Errorf("improvement loop failed: %w", err)
		}
		trainingTime := time.Since(startTime)
		fmt.Printf("Fine-tuning completed in %v (%d steps)\n", trainingTime, trainingStats.Steps)

		// Step 4: Update the agent with the improved model
		sia.currentModel = updatedModel
		// For our math agent, we need to update the underlying model
		sia.currentAgent.baseAgent = sia.agentAdapter.WithModel(updatedModel)

		// Step 5: Evaluate the improved agent
		performanceAfter, err := sia.ragEvalAdapter.Evaluate(sia.currentAgent, task, sia.config.ValidationSetSize)
		if err != nil {
			return nil, fmt.Errorf("failed to evaluate improved performance: %w", err)
		}

		// Calculate improvement
		performanceBefore := initialPerformance
		if len(improvementHistory) > 0 {
			performanceBefore = improvementHistory[len(improvementHistory)-1].PerformanceAfter
		}
		improvement := performanceAfter - performanceBefore

		// Record the improvement cycle
		cycleRecord := core.ImprovementCycle{
			CycleNumber:         cycle + 1,
			StartTime:           time.Now().Add(-trainingTime),
			EndTime:             time.Now(),
			WeaknessesAddressed: weaknesses,
			PerformanceBefore:   performanceBefore,
			PerformanceAfter:    performanceAfter,
			Improvement:         improvement,
			DataGenerated:       len(trainingData),
			TrainingSteps:       trainingStats.Steps,
			ParametersUpdated:   trainingStats.ParametersUpdated,
		}
		improvementHistory = append(improvementHistory, cycleRecord)

		fmt.Printf("Performance: %.4f -> %.4f (+%.4f)\n",
			performanceBefore, performanceAfter, improvement)

		// Step 6: Store successful improvement in experience store
		if improvement > sia.config.MinImprovementThreshold {
			experience := core.Experience{
				ID:                fmt.Sprintf("exp_%d_%d", time.Now().Unix(), cycle),
				WeaknessType:      weaknesses[0].TaskType, // Simplified: use first weakness
				BeforePerformance: performanceBefore,
				AfterPerformance:  performanceAfter,
				Improvement:       improvement,
				DataUsed:          trainingData[:min(10, len(trainingData))], // Store first 10 examples
				ParameterUpdates:  make([]float32, 0),                        // In practice, we would store the actual updates
				Context: map[string]string{
					"task":  task,
					"cycle": fmt.Sprintf("%d", cycle),
				},
				Timestamp: time.Now(),
			}
			sia.experienceStore.Store(experience)
		}

		// Step 7: Update meta-learning strategy
		sia.metaLearner.Update(improvementHistory)

		// Step 8: Check if we've reached the performance target
		if performanceAfter >= sia.config.PerformanceTarget {
			fmt.Printf("Performance target (%.2f) reached!\n", sia.config.PerformanceTarget)
			break
		}

		// Step 9: Check for diminishing returns
		if len(improvementHistory) >= 3 {
			recentImprovements := make([]float64, 0, 3)
			for i := len(improvementHistory) - 3; i < len(improvementHistory); i++ {
				recentImprovements = append(recentImprovements, improvementHistory[i].Improvement)
			}
			avgRecent := 0.0
			for _, imp := range recentImprovements {
				avgRecent += imp
			}
			avgRecent /= float64(len(recentImprovements))

			if avgRecent < sia.config.MinImprovementThreshold {
				fmt.Println("Improvement has diminished below threshold. Stopping.")
				break
			}
		}

		// Step 10: Save checkpoint periodically
		if (cycle+1)%sia.config.CheckpointFrequency == 0 {
			checkpointPath := fmt.Sprintf("./checkpoints/cycle_%d", cycle+1)
			if err := sia.SaveCheckpoint(checkpointPath); err != nil {
				log.Printf("Warning: failed to save checkpoint at cycle %d: %v", cycle+1, err)
			} else {
				fmt.Printf("Checkpoint saved to %s\n", checkpointPath)
			}
		}

		// Update initial performance for next iteration
		initialPerformance = performanceAfter
	}

	return improvementHistory, nil
}

// SaveCheckpoint saves the current state of the self-improving agent
func (sia *SelfImprovingMathAgent) SaveCheckpoint(path string) error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(path, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Save the model
	if err := sia.modelAdapter.Save(sia.currentModel, path+"/model"); err != nil {
		return fmt.Errorf("failed to save model: %w", err)
	}

	// Save the experience store
	if err := sia.experienceStore.Save(path + "/experiences.json"); err != nil {
		return fmt.Errorf("failed to save experience store: %w", err)
	}

	// Save the meta-learner state
	if err := sia.metaLearner.Save(path + "/meta_learner.json"); err != nil {
		return fmt.Errorf("failed to save meta-learner: %w", err)
	}

	// Save the configuration
	configData := []byte(fmt.Sprintf("%+v", sia.config))
	if err := writeFile(path+"/config.json", configData, 0644); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}

	return nil
}

// LoadCheckpoint loads the state of the self-improving agent from a checkpoint
func (sia *SelfImprovingMathAgent) LoadCheckpoint(path string) error {
	// Load the model
	model, err := sia.modelAdapter.Load(path + "/model")
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}
	sia.currentModel = model
	// Update the underlying model in our math agent
	sia.currentAgent.baseAgent = sia.agentAdapter.WithModel(model)

	// Load the experience store
	if err := sia.experienceStore.Load(path + "/experiences.json"); err != nil {
		return fmt.Errorf("failed to load experience store: %w", err)
	}

	// Load the meta-learner state
	if err := sia.metaLearner.Load(path + "/meta_learner.json"); err != nil {
		return fmt.Errorf("failed to load meta-learner: %w", err)
	}

	// Note: In a real implementation, we would also load the config and restore the exact state
	// For simplicity, we're keeping the current config

	return nil
}

// writeFile is a helper function to write data to a file
func writeFile(filename string, data []byte, perm os.FileMode) error {
	return os.WriteFile(filename, data, perm)
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func main() {
	// Seed the random number generator for reproducible results
	rand.Seed(time.Now().UnixNano())

	fmt.Println("Starting Recursive Self-Improving AI Agent Framework (RSIAF)")
	fmt.Println("============================================================")

	// Initialize configuration
	config := core.Config{
		MaxCycles:               20,
		PerformanceTarget:       0.9,
		MinImprovementThreshold: 0.02,
		ExamplesPerWeakness:     50,
		DifficultyProgression:   0.05,
		LearningRate:            1e-4,
		LoRARank:                8,
		LoRAAlpha:               16.0,
		ValidationSetSize:       30,
		EvaluationFrequency:     1,
		MetaLearningRate:        0.01,
		StrategyExploration:     0.2,
		CheckpointFrequency:     5,
		ExperienceStoreSize:     1000,
		Device:                  "CPU",
	}

	// Initialize integrations with existing Go ML ecosystem
	agentAdapter := agents.NewAdapter()
	modelAdapter := transformers.NewAdapter()
	fineTuneAdapter := finetune.NewAdapter()
	ragEvalAdapter := rag_eval.NewAdapter()
	modelRegistryAdapter := model_registry.NewAdapter()
	dashboardAdapter := dashboard.NewAdapter()

	// Create the base agent and wrap it with our math-specific adapter
	baseAgent := agentAdapter.GetAgent()
	mathAgent := NewMathAgentAdapter(baseAgent)

	// Create the self-improving agent with our math agent
	siAgent := &SelfImprovingMathAgent{
		config:               config,
		agentAdapter:         *agentAdapter,
		modelAdapter:         *modelAdapter,
		fineTuneAdapter:      *fineTuneAdapter,
		ragEvalAdapter:       *ragEvalAdapter,
		modelRegistryAdapter: *modelRegistryAdapter,
		dashboardAdapter:     *dashboardAdapter,

		weaknessDetector: weakness_detector.NewDetector(*ragEvalAdapter),
		dataGenerator:    data_generator.NewGenerator(*modelAdapter),
		improvementLoop:  improvement_loop.NewLoop(*fineTuneAdapter, *ragEvalAdapter),
		metaLearner:      meta_learner.NewLearner(),
		experienceStore:  experience_store.NewStore(config.ExperienceStoreSize),

		currentAgent: mathAgent,
		currentModel: baseAgent.GetModel(),
	}

	// Define a task for the agent to improve on (mathematical reasoning)
	task := "Solve linear equations"

	// Start the self-improvement process
	startTime := time.Now()
	improvementHistory, err := siAgent.Improve(task)
	if err != nil {
		log.Fatalf("Self-improvement failed: %v", err)
	}
	endTime := time.Now()

	// Report results
	fmt.Printf("\nSelf-improvement completed in %v\n", endTime.Sub(startTime))
	fmt.Printf("Total cycles: %d\n", len(improvementHistory))

	bestPerformance := 0.0
	for _, cycle := range improvementHistory {
		if cycle.PerformanceAfter > bestPerformance {
			bestPerformance = cycle.PerformanceAfter
		}
	}
	fmt.Printf("Best performance achieved: %.4f\n", bestPerformance)

	// Save the final agent state
	if err := siAgent.SaveCheckpoint("./checkpoints/final_agent"); err != nil {
		log.Printf("Warning: failed to save checkpoint: %v", err)
	} else {
		fmt.Println("Final agent state saved to ./checkpoints/final_agent")
	}

	// Display improvement history
	fmt.Println("\nImprovement History:")
	fmt.Println("Cycle\tPerformance Before\tPerformance After\tImprovement")
	for i, cycle := range improvementHistory {
		fmt.Printf("%d\t%.4f\t\t%.4f\t\t%.4f\n",
			i+1, cycle.PerformanceBefore, cycle.PerformanceAfter, cycle.Improvement)
	}
}
