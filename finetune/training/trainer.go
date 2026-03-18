package training

import (
	"fmt"
	"math"
	"time"

	"finetune/config"
	"finetune/data"
	"finetune/models"
	"finetune/optimizers"
)

type Trainer struct {
	model     *models.ModelWrapper
	config    *config.FineTuneConfig
	optimizer optimizers.Optimizer
	scheduler *Scheduler
	loader    *data.DataLoader
	devLoader *data.DataLoader
	metrics   *Metrics
	step      int
	epoch     int
}

type Metrics struct {
	TrainLoss    []float64
	EvalLoss     []float64
	LearningRate []float64
	StepTimes    []float64
}

func NewMetrics() *Metrics {
	return &Metrics{
		TrainLoss:    make([]float64, 0),
		EvalLoss:     make([]float64, 0),
		LearningRate: make([]float64, 0),
		StepTimes:    make([]float64, 0),
	}
}

func (m *Metrics) AddTrainLoss(loss float64) {
	m.TrainLoss = append(m.TrainLoss, loss)
}

func (m *Metrics) AddEvalLoss(loss float64) {
	m.EvalLoss = append(m.EvalLoss, loss)
}

func (m *Metrics) AddLearningRate(lr float64) {
	m.LearningRate = append(m.LearningRate, lr)
}

func (m *Metrics) AddStepTime(t time.Duration) {
	m.StepTimes = append(m.StepTimes, t.Seconds())
}

func (m *Metrics) AverageTrainLoss() float64 {
	if len(m.TrainLoss) == 0 {
		return 0
	}
	sum := 0.0
	for _, l := range m.TrainLoss {
		sum += l
	}
	return sum / float64(len(m.TrainLoss))
}

func (m *Metrics) AverageEvalLoss() float64 {
	if len(m.EvalLoss) == 0 {
		return 0
	}
	sum := 0.0
	for _, l := range m.EvalLoss {
		sum += l
	}
	return sum / float64(len(m.EvalLoss))
}

func (m *Metrics) AverageStepTime() float64 {
	if len(m.StepTimes) == 0 {
		return 0
	}
	sum := 0.0
	for _, t := range m.StepTimes {
		sum += t
	}
	return sum / float64(len(m.StepTimes))
}

func (t *Trainer) GetMetrics() *Metrics {
	return t.metrics
}

type Scheduler struct {
	optimizer     optimizers.Optimizer
	schedulerType config.SchedulerType
	warmupSteps   int
	totalSteps    int
	currentStep   int
	baseLR        float64
}

func NewScheduler(optimizer optimizers.Optimizer, cfg config.TrainingConfig, totalSteps int) *Scheduler {
	return &Scheduler{
		optimizer:     optimizer,
		schedulerType: cfg.LRScheduler,
		warmupSteps:   cfg.WarmupSteps,
		totalSteps:    totalSteps,
		currentStep:   0,
		baseLR:        cfg.LearningRate,
	}
}

func (s *Scheduler) Step() {
	s.currentStep++
	lr := s.getLR()
	s.optimizer.SetLearningRate(lr)
}

func (s *Scheduler) getLR() float64 {
	if s.currentStep < s.warmupSteps {
		return s.baseLR * float64(s.currentStep) / float64(s.warmupSteps)
	}

	switch s.schedulerType {
	case config.SchedulerLinear:
		progress := float64(s.currentStep-s.warmupSteps) / float64(s.totalSteps-s.warmupSteps)
		return s.baseLR * (1.0 - progress)

	case config.SchedulerCosine:
		progress := float64(s.currentStep-s.warmupSteps) / float64(s.totalSteps-s.warmupSteps)
		return s.baseLR * 0.5 * (1.0 + math.Cos(math.Pi*progress))

	case config.SchedulerConstant:
		return s.baseLR

	default:
		return s.baseLR
	}
}

func (s *Scheduler) GetCurrentLR() float64 {
	return s.getLR()
}

type TrainingState struct {
	Epoch      int
	Step       int
	GlobalStep int
	TokensSeen int64
	EpochStart time.Time
	StepStart  time.Time
	BatchSize  int
	SeqLength  int
}

func NewTrainingState(cfg *config.FineTuneConfig) *TrainingState {
	return &TrainingState{
		Epoch:      0,
		Step:       0,
		GlobalStep: 0,
		TokensSeen: 0,
		BatchSize:  cfg.Training.BatchSize,
		SeqLength:  cfg.Training.SequenceLength,
	}
}

func (s *TrainingState) AdvanceBatch() {
	s.Step++
	s.GlobalStep++
	s.TokensSeen += int64(s.BatchSize * s.SeqLength)
}

func (s *TrainingState) AdvanceEpoch() {
	s.Epoch++
	s.Step = 0
}

func createOptimizer(model *models.ModelWrapper, cfg *config.FineTuneConfig) optimizers.Optimizer {
	optCfg := cfg.Optimization
	params := model.Parameters()

	switch optCfg.Optimizer {
	case config.OptimizerAdamW:
		opt := optimizers.NewAdamW(
			cfg.Training.LearningRate,
			optCfg.Beta1,
			optCfg.Beta2,
			optCfg.Epsilon,
			cfg.Training.WeightDecay,
		)
		for _, p := range params {
			opt.AddParams([]*optimizers.Tensor{{Data: p.Data, Shape: p.Shape}})
		}
		return opt
	case config.OptimizerAdam:
		opt := optimizers.NewAdam(
			cfg.Training.LearningRate,
			optCfg.Beta1,
			optCfg.Beta2,
			optCfg.Epsilon,
		)
		for _, p := range params {
			opt.AddParams([]*optimizers.Tensor{{Data: p.Data, Shape: p.Shape}})
		}
		return opt
	case config.OptimizerSGD:
		opt := optimizers.NewSGD(cfg.Training.LearningRate, optCfg.Momentum)
		for _, p := range params {
			opt.AddParams([]*optimizers.Tensor{{Data: p.Data, Shape: p.Shape}})
		}
		return opt
	case config.OptimizerLion:
		opt := optimizers.NewLion(cfg.Training.LearningRate, optCfg.Beta1, optCfg.Beta2)
		for _, p := range params {
			opt.AddParams([]*optimizers.Tensor{{Data: p.Data, Shape: p.Shape}})
		}
		return opt
	default:
		opt := optimizers.NewAdamW(
			cfg.Training.LearningRate,
			optCfg.Beta1,
			optCfg.Beta2,
			optCfg.Epsilon,
			cfg.Training.WeightDecay,
		)
		for _, p := range params {
			opt.AddParams([]*optimizers.Tensor{{Data: p.Data, Shape: p.Shape}})
		}
		return opt
	}
}

func NewTrainer(model *models.ModelWrapper, cfg *config.FineTuneConfig, loader, devLoader *data.DataLoader) *Trainer {
	optimizer := createOptimizer(model, cfg)

	totalSteps := calculateTotalSteps(cfg, loader.Size())
	scheduler := NewScheduler(optimizer, cfg.Training, totalSteps)

	return &Trainer{
		model:     model,
		config:    cfg,
		optimizer: optimizer,
		scheduler: scheduler,
		loader:    loader,
		devLoader: devLoader,
		metrics:   NewMetrics(),
		step:      0,
		epoch:     0,
	}
}

func calculateTotalSteps(cfg *config.FineTuneConfig, datasetSize int) int {
	if cfg.Training.MaxSteps > 0 {
		return cfg.Training.MaxSteps
	}
	stepsPerEpoch := datasetSize / cfg.Training.BatchSize
	return stepsPerEpoch * cfg.Training.NumEpochs
}

func (t *Trainer) Train() error {
	cfg := t.config.Training
	state := NewTrainingState(t.config)

	fmt.Printf("Starting training with config:\n%s\n\n", t.config.String())
	fmt.Printf("Total parameters: %d\n", t.model.BaseModel.(*models.TransformerModel).NumParameters())

	if t.config.Lora.Enabled {
		fmt.Printf("LoRA enabled: rank=%d, alpha=%d\n", t.config.Lora.Rank, t.config.Lora.Alpha)
		fmt.Printf("Trainable parameters: ~%d\n", t.estimateLoRAParameters())
	}

	totalSteps := calculateTotalSteps(t.config, t.loader.Size())
	fmt.Printf("Total steps: %d\n\n", totalSteps)

	loader := t.loader
	loader.Init()

	for state.Epoch < cfg.NumEpochs {
		state.EpochStart = time.Now()
		state.AdvanceEpoch()

		fmt.Printf("\n=== Epoch %d/%d ===\n", state.Epoch, cfg.NumEpochs)

		batchCount := 0
		for {
			stepStart := time.Now()

			tokens, labels, hasMore := loader.Next()
			if !hasMore {
				break
			}

			loss := t.trainStep(tokens, labels)
			state.AdvanceBatch()
			t.scheduler.Step()

			t.metrics.AddTrainLoss(loss)
			t.metrics.AddStepTime(time.Since(stepStart))
			t.metrics.AddLearningRate(t.scheduler.GetCurrentLR())

			if state.Step%cfg.LogSteps == 0 {
				t.logStep(state, loss)
			}

			if state.Step%cfg.EvalSteps == 0 && t.devLoader != nil {
				evalLoss := t.evaluate()
				t.metrics.AddEvalLoss(evalLoss)
				fmt.Printf("Eval loss: %.4f\n", evalLoss)
			}

			if state.Step%cfg.SaveSteps == 0 {
				t.saveCheckpoint(state, fmt.Sprintf("checkpoint-%d-%d", state.Epoch, state.Step))
			}

			batchCount++
			if cfg.MaxSteps > 0 && state.GlobalStep >= cfg.MaxSteps {
				fmt.Printf("\nReached max steps: %d\n", cfg.MaxSteps)
				return nil
			}
		}
	}

	fmt.Printf("\n=== Training Complete ===\n")
	fmt.Printf("Final train loss: %.4f\n", t.metrics.AverageTrainLoss())
	fmt.Printf("Final eval loss: %.4f\n", t.metrics.AverageEvalLoss())
	fmt.Printf("Total time: %v\n", time.Since(state.EpochStart))

	return nil
}

func (t *Trainer) trainStep(tokens, labels []int) float64 {
	input := &models.Tensor{
		Data:  intToFloat32(tokens),
		Shape: []int{1, len(tokens)},
	}

	target := &models.Tensor{
		Data:  intToFloat32(labels),
		Shape: []int{1, len(labels)},
	}

	t.model.ZeroGrad()

	output := t.model.Forward(input)

	loss := CrossEntropyLoss(output, target)

	grad := ComputeGrad(output, target)

	backpropagate(t.model, grad)

	for _, p := range t.model.Parameters() {
		if p.Grad != nil {
			for i := range p.Grad.Data {
				p.Grad.Data[i] /= float32(t.config.Training.GradientAccum)
			}
		}
	}

	gradClip := t.config.Training.GradientClip
	if gradClip > 0 {
		for _, p := range t.model.Parameters() {
			if p.Grad != nil {
				norm := computeNorm(p.Grad)
				if norm > gradClip {
					scale := float32(gradClip / norm)
					for i := range p.Grad.Data {
						p.Grad.Data[i] *= scale
					}
				}
			}
		}
	}

	t.optimizer.Step()
	t.optimizer.ZeroGrad()

	return loss
}

func (t *Trainer) evaluate() float64 {
	if t.devLoader == nil {
		return 0
	}

	t.model.EvalMode()
	defer t.model.TrainMode()

	devLoader := t.devLoader
	devLoader.Init()

	var totalLoss float64
	var numBatches int

	for {
		tokens, labels, hasMore := devLoader.Next()
		if !hasMore {
			break
		}

		input := &models.Tensor{
			Data:  intToFloat32(tokens),
			Shape: []int{1, len(tokens)},
		}

		target := &models.Tensor{
			Data:  intToFloat32(labels),
			Shape: []int{1, len(labels)},
		}

		output := t.model.Forward(input)
		loss := CrossEntropyLoss(output, target)

		totalLoss += loss
		numBatches++
	}

	if numBatches == 0 {
		return 0
	}

	return totalLoss / float64(numBatches)
}

func (t *Trainer) logStep(state *TrainingState, loss float64) {
	elapsed := time.Since(state.EpochStart)
	avgStepTime := elapsed.Seconds() / float64(state.Step+1)
	tokensPerSec := float64(state.TokensSeen) / elapsed.Seconds()

	fmt.Printf("Step %d | Loss: %.4f | LR: %.2e | Time: %.2fs | Speed: %.0f tok/s | Epoch: %d/%d\n",
		state.GlobalStep,
		loss,
		t.scheduler.GetCurrentLR(),
		avgStepTime,
		tokensPerSec,
		state.Epoch,
		t.config.Training.NumEpochs)
}

func (t *Trainer) saveCheckpoint(state *TrainingState, name string) {
	path := fmt.Sprintf("./checkpoints/%s", name)
	fmt.Printf("Would save checkpoint to %s\n", path)
}

func (t *Trainer) estimateLoRAParameters() int {
	cfg := t.config.Lora
	hiddenSize := t.config.Model.HiddenSize

	totalParams := 0
	numModules := len(cfg.TargetModules)
	if numModules == 0 {
		numModules = 1
	}
	totalParams = 2 * hiddenSize * cfg.Rank * numModules

	return totalParams
}

func intToFloat32(ints []int) []float32 {
	floats := make([]float32, len(ints))
	for i, v := range ints {
		floats[i] = float32(v)
	}
	return floats
}

func CrossEntropyLoss(logits, targets *models.Tensor) float64 {
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	loss := float64(0)
	count := 0

	for i := 0; i < seqLen; i++ {
		targetIdx := int(targets.Data[i])
		if targetIdx < 0 || targetIdx >= vocabSize {
			continue
		}

		maxLogit := float32(-math.MaxFloat32)
		for j := 0; j < vocabSize; j++ {
			if logits.Data[i*vocabSize+j] > maxLogit {
				maxLogit = logits.Data[i*vocabSize+j]
			}
		}

		sum := float32(0)
		for j := 0; j < vocabSize; j++ {
			sum += float32(math.Exp(float64(logits.Data[i*vocabSize+j] - maxLogit)))
		}

		logProb := float64(logits.Data[i*vocabSize+targetIdx]-maxLogit) - math.Log(float64(sum))
		loss -= logProb
		count++
	}

	if count > 0 {
		loss /= float64(count)
	}

	return loss
}

func ComputeGrad(output, target *models.Tensor) *models.Tensor {
	grad := &models.Tensor{
		Data:  make([]float32, len(output.Data)),
		Shape: output.Shape,
	}

	seqLen := output.Shape[1]
	vocabSize := output.Shape[2]

	for i := 0; i < seqLen; i++ {
		targetIdx := int(target.Data[i])
		if targetIdx >= 0 && targetIdx < vocabSize {
			maxLogit := float32(-math.MaxFloat32)
			for j := 0; j < vocabSize; j++ {
				if output.Data[i*vocabSize+j] > maxLogit {
					maxLogit = output.Data[i*vocabSize+j]
				}
			}

			sum := float32(0)
			for j := 0; j < vocabSize; j++ {
				sum += float32(math.Exp(float64(output.Data[i*vocabSize+j] - maxLogit)))
			}

			for j := 0; j < vocabSize; j++ {
				prob := float32(math.Exp(float64(output.Data[i*vocabSize+j]-maxLogit))) / sum
				if j == targetIdx {
					grad.Data[i*vocabSize+j] = (prob - 1) / float32(seqLen)
				} else {
					grad.Data[i*vocabSize+j] = prob / float32(seqLen)
				}
			}
		}
	}

	return grad
}

func backpropagate(model *models.ModelWrapper, grad *models.Tensor) {
	for _, p := range model.Parameters() {
		if p.RequiresGrad {
			p.Grad = &models.Tensor{
				Data:  make([]float32, p.Numel()),
				Shape: p.Shape,
			}
		}
	}
}

func computeNorm(t *models.Tensor) float64 {
	if t == nil || len(t.Data) == 0 {
		return 0
	}

	sum := float64(0)
	for _, v := range t.Data {
		sum += float64(v * v)
	}
	return math.Sqrt(sum)
}

type Evaluator struct {
	model   *models.ModelWrapper
	loader  *data.DataLoader
	metrics map[string]float64
}

func NewEvaluator(model *models.ModelWrapper, loader *data.DataLoader) *Evaluator {
	return &Evaluator{
		model:   model,
		loader:  loader,
		metrics: make(map[string]float64),
	}
}

func (e *Evaluator) Evaluate() map[string]float64 {
	e.model.EvalMode()
	defer e.model.TrainMode()

	e.metrics["loss"] = 0
	e.metrics["accuracy"] = 0
	e.metrics["perplexity"] = 0

	var totalLoss float64
	var correct int
	var total int

	loader := e.loader
	loader.Init()

	batchCount := 0
	for {
		tokens, labels, hasMore := loader.Next()
		if !hasMore {
			break
		}

		input := &models.Tensor{
			Data:  intToFloat32(tokens),
			Shape: []int{1, len(tokens)},
		}

		target := &models.Tensor{
			Data:  intToFloat32(labels),
			Shape: []int{1, len(labels)},
		}

		output := e.model.Forward(input)
		loss := CrossEntropyLoss(output, target)
		totalLoss += loss

		predictions := argmax(output)
		for i := range predictions {
			if predictions[i] == int(labels[i]) {
				correct++
			}
			total++
		}

		batchCount++
	}

	if batchCount > 0 {
		e.metrics["loss"] = totalLoss / float64(batchCount)
		e.metrics["accuracy"] = float64(correct) / float64(total)
		e.metrics["perplexity"] = math.Exp(e.metrics["loss"])
	}

	return e.metrics
}

func argmax(t *models.Tensor) []int {
	seqLen := t.Shape[1]
	vocabSize := t.Shape[2]

	result := make([]int, seqLen)
	for i := 0; i < seqLen; i++ {
		maxIdx := 0
		maxVal := t.Data[i*vocabSize]
		for j := 1; j < vocabSize; j++ {
			if t.Data[i*vocabSize+j] > maxVal {
				maxVal = t.Data[i*vocabSize+j]
				maxIdx = j
			}
		}
		result[i] = maxIdx
	}
	return result
}
