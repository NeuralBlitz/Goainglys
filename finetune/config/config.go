package config

import "fmt"

type FineTuneConfig struct {
	Model        ModelConfig
	Training     TrainingConfig
	Data         DataConfig
	Lora         LoRAConfig
	Optimization OptimizationConfig
}

type ModelConfig struct {
	BaseModel        string
	ModelType        ModelType
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	IntermediateSize int
	VocabSize        int
	MaxPosition      int
	Architecture     string
}

type ModelType string

const (
	TypeLLM     ModelType = "llm"
	TypeVision  ModelType = "vision"
	TypeEncoder ModelType = "encoder"
	TypeDecoder ModelType = "decoder"
)

type TrainingConfig struct {
	NumEpochs      int
	BatchSize      int
	SequenceLength int
	LearningRate   float64
	WarmupSteps    int
	WeightDecay    float64
	GradientClip   float64
	SaveSteps      int
	EvalSteps      int
	LogSteps       int
	Seed           int64
	Precision      Precision
	GradientAccum  int
	MaxSteps       int
	TrainRatio     float64
	LRScheduler    SchedulerType
}

type Precision string

const (
	PrecisionFP32 Precision = "fp32"
	PrecisionFP16 Precision = "fp16"
	PrecisionBF16 Precision = "bf16"
	PrecisionInt8 Precision = "int8"
	PrecisionInt4 Precision = "int4"
)

type DataConfig struct {
	TrainFile        string
	EvalFile         string
	TestFile         string
	TokenizerPath    string
	MaxLength        int
	Shuffle          bool
	NumWorkers       int
	PreprocessingNum int
}

type LoRAConfig struct {
	Enabled       bool
	Rank          int
	Alpha         int
	Dropout       float64
	TargetModules []string
	Bias          string
	TaskType      string
}

type OptimizationConfig struct {
	Optimizer   OptimizerType
	LRScheduler SchedulerType
	Beta1       float64
	Beta2       float64
	Epsilon     float64
	Momentum    float64
}

type OptimizerType string

const (
	OptimizerAdamW   OptimizerType = "adamw"
	OptimizerAdam    OptimizerType = "adam"
	OptimizerSGD     OptimizerType = "sgd"
	OptimizerLion    OptimizerType = "lion"
	OptimizerAdagrad OptimizerType = "adagrad"
)

type SchedulerType string

const (
	SchedulerLinear   SchedulerType = "linear"
	SchedulerCosine   SchedulerType = "cosine"
	SchedulerConstant SchedulerType = "constant"
	SchedulerWarmup   SchedulerType = "warmup"
)

func DefaultConfig() *FineTuneConfig {
	return &FineTuneConfig{
		Model: ModelConfig{
			BaseModel:        "base-model",
			ModelType:        TypeLLM,
			HiddenSize:       768,
			NumLayers:        12,
			NumHeads:         12,
			IntermediateSize: 3072,
			VocabSize:        50257,
			MaxPosition:      2048,
			Architecture:     "decoder-only",
		},
		Training: TrainingConfig{
			NumEpochs:      3,
			BatchSize:      8,
			SequenceLength: 512,
			LearningRate:   3e-4,
			WarmupSteps:    100,
			WeightDecay:    0.01,
			GradientClip:   1.0,
			SaveSteps:      500,
			EvalSteps:      100,
			LogSteps:       10,
			Seed:           42,
			Precision:      PrecisionFP32,
			GradientAccum:  1,
			MaxSteps:       0,
			TrainRatio:     0.9,
		},
		Data: DataConfig{
			TrainFile:        "train.jsonl",
			EvalFile:         "eval.jsonl",
			MaxLength:        512,
			Shuffle:          true,
			NumWorkers:       4,
			PreprocessingNum: 4,
		},
		Lora: LoRAConfig{
			Enabled:       false,
			Rank:          8,
			Alpha:         16,
			Dropout:       0.05,
			TargetModules: []string{"q_proj", "v_proj", "k_proj", "o_proj"},
			Bias:          "none",
			TaskType:      "CAUSAL_LM",
		},
		Optimization: OptimizationConfig{
			Optimizer:   OptimizerAdamW,
			LRScheduler: SchedulerLinear,
			Beta1:       0.9,
			Beta2:       0.999,
			Epsilon:     1e-8,
		},
	}
}

func (c *FineTuneConfig) String() string {
	return fmt.Sprintf(`FineTuneConfig:
  Model: %s (%s)
  Training: epochs=%d, batch=%d, lr=%.2e
  LoRA: enabled=%v, rank=%d, alpha=%d
  Optimizer: %s`, c.Model.BaseModel, c.Model.ModelType,
		c.Training.NumEpochs, c.Training.BatchSize, c.Training.LearningRate,
		c.Lora.Enabled, c.Lora.Rank, c.Lora.Alpha,
		c.Optimization.Optimizer)
}

func (c *FineTuneConfig) TotalBatchSize() int {
	return c.Training.BatchSize * c.Training.GradientAccum
}

func (c *FineTuneConfig) EffectiveBatchSize() int {
	return c.Training.BatchSize * c.Training.GradientAccum * c.Training.SequenceLength
}

func (c *FineTuneConfig) Validate() error {
	if c.Training.BatchSize <= 0 {
		return fmt.Errorf("batch size must be positive")
	}
	if c.Training.LearningRate <= 0 {
		return fmt.Errorf("learning rate must be positive")
	}
	if c.Lora.Enabled && c.Lora.Rank <= 0 {
		return fmt.Errorf("LoRA rank must be positive when enabled")
	}
	return nil
}
