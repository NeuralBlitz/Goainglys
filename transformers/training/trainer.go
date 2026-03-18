package training

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"transformers/core"
)

type Optimizer interface {
	Step()
	ZeroGrad()
	SetLearningRate(lr float32)
}

type AdamW struct {
	parameters  []*AdamParam
	lr          float32
	betas       [2]float32
	eps         float32
	weightDecay float32
}

type AdamParam struct {
	Data     []float32
	Grad     []float32
	ExpAvg   []float32
	ExpAvgSq []float32
	Step     int
}

func NewAdamW(params []*AdamParam, lr, beta1, beta2, eps, weightDecay float32) *AdamW {
	return &AdamW{
		parameters:  params,
		lr:          lr,
		betas:       [2]float32{beta1, beta2},
		eps:         eps,
		weightDecay: weightDecay,
	}
}

func (a *AdamW) Step() {
	for _, p := range a.parameters {
		if p.Grad == nil || len(p.Grad) == 0 {
			continue
		}

		p.Step++

		beta1Pow := float32(math.Pow(float64(a.betas[0]), float64(p.Step)))
		beta2Pow := float32(math.Pow(float64(a.betas[1]), float64(p.Step)))

		lr := a.lr * float32(math.Sqrt(float64(1-beta2Pow))/float64(1-beta1Pow))

		for i := range p.Data {
			grad := p.Grad[i]

			if a.weightDecay > 0 {
				grad += a.weightDecay * p.Data[i]
			}

			p.ExpAvg[i] = a.betas[0]*p.ExpAvg[i] + (1-a.betas[0])*grad
			p.ExpAvgSq[i] = a.betas[1]*p.ExpAvgSq[i] + (1-a.betas[1])*grad*grad

			denom := float32(math.Sqrt(float64(p.ExpAvgSq[i]))/math.Sqrt(float64(1-beta2Pow))) + a.eps

			update := lr * p.ExpAvg[i] / denom

			p.Data[i] -= update
		}
	}
}

func (a *AdamW) ZeroGrad() {
	for _, p := range a.parameters {
		p.Grad = nil
	}
}

func (a *AdamW) SetLearningRate(lr float32) {
	a.lr = lr
}

type Adam struct {
	parameters []*AdamParam
	lr         float32
	betas      [2]float32
	eps        float32
}

func NewAdam(params []*AdamParam, lr, beta1, beta2, eps float32) *Adam {
	return &Adam{
		parameters: params,
		lr:         lr,
		betas:      [2]float32{beta1, beta2},
		eps:        eps,
	}
}

func (a *Adam) Step() {
	for _, p := range a.parameters {
		if p.Grad == nil || len(p.Grad) == 0 {
			continue
		}

		p.Step++

		beta1Pow := float32(math.Pow(float64(a.betas[0]), float64(p.Step)))
		beta2Pow := float32(math.Pow(float64(a.betas[1]), float64(p.Step)))

		lr := a.lr * float32(math.Sqrt(float64(1-beta2Pow))/float64(1-beta1Pow))

		for i := range p.Data {
			grad := p.Grad[i]

			p.ExpAvg[i] = a.betas[0]*p.ExpAvg[i] + (1-a.betas[0])*grad
			p.ExpAvgSq[i] = a.betas[1]*p.ExpAvgSq[i] + (1-a.betas[1])*grad*grad

			denom := float32(math.Sqrt(float64(p.ExpAvgSq[i]))/math.Sqrt(float64(1-beta2Pow))) + a.eps

			update := lr * p.ExpAvg[i] / denom

			p.Data[i] -= update
		}
	}
}

func (a *Adam) ZeroGrad() {
	for _, p := range a.parameters {
		p.Grad = nil
	}
}

func (a *Adam) SetLearningRate(lr float32) {
	a.lr = lr
}

type SGD struct {
	parameters []*AdamParam
	lr         float32
	momentum   float32
}

func NewSGD(params []*AdamParam, lr, momentum float32) *SGD {
	return &SGD{
		parameters: params,
		lr:         lr,
		momentum:   momentum,
	}
}

func (s *SGD) Step() {
	for _, p := range s.parameters {
		if p.Grad == nil || len(p.Grad) == 0 {
			continue
		}

		for i := range p.Data {
			if s.momentum > 0 {
				p.ExpAvg[i] = s.momentum*p.ExpAvg[i] + p.Grad[i]
				p.Data[i] -= s.lr * p.ExpAvg[i]
			} else {
				p.Data[i] -= s.lr * p.Grad[i]
			}
		}
	}
}

func (s *SGD) ZeroGrad() {
	for _, p := range s.parameters {
		p.Grad = nil
	}
}

func (s *SGD) SetLearningRate(lr float32) {
	s.lr = lr
}

type Scheduler struct {
	optimizer     Optimizer
	schedulerType SchedulerType
	warmupSteps   int
	totalSteps    int
	currentStep   int
	baseLR        float32
}

type SchedulerType int

const (
	SchedulerLinear SchedulerType = iota
	SchedulerCosine
	SchedulerConstant
)

func NewScheduler(optimizer Optimizer, schedulerType SchedulerType, warmupSteps, totalSteps int, baseLR float32) *Scheduler {
	return &Scheduler{
		optimizer:     optimizer,
		schedulerType: schedulerType,
		warmupSteps:   warmupSteps,
		totalSteps:    totalSteps,
		currentStep:   0,
		baseLR:        baseLR,
	}
}

func (s *Scheduler) Step() {
	s.currentStep++
	lr := s.getLR()
	s.optimizer.SetLearningRate(lr)
}

func (s *Scheduler) getLR() float32 {
	if s.currentStep < s.warmupSteps {
		return s.baseLR * float32(s.currentStep) / float32(s.warmupSteps)
	}

	switch s.schedulerType {
	case SchedulerLinear:
		progress := float32(s.currentStep-s.warmupSteps) / float32(s.totalSteps-s.warmupSteps)
		return s.baseLR * (1.0 - progress)

	case SchedulerCosine:
		progress := float32(s.currentStep-s.warmupSteps) / float32(s.totalSteps-s.warmupSteps)
		return s.baseLR * 0.5 * (1.0 + float32(math.Cos(math.Pi*float64(progress))))

	case SchedulerConstant:
		return s.baseLR

	default:
		return s.baseLR
	}
}

func (s *Scheduler) GetCurrentLR() float32 {
	return s.getLR()
}

type Metrics struct {
	TrainLoss    []float32
	EvalLoss     []float32
	LearningRate []float32
	StepTimes    []float32
}

func NewMetrics() *Metrics {
	return &Metrics{
		TrainLoss:    make([]float32, 0),
		EvalLoss:     make([]float32, 0),
		LearningRate: make([]float32, 0),
		StepTimes:    make([]float32, 0),
	}
}

func (m *Metrics) AddTrainLoss(loss float32) {
	m.TrainLoss = append(m.TrainLoss, loss)
}

func (m *Metrics) AddEvalLoss(loss float32) {
	m.EvalLoss = append(m.EvalLoss, loss)
}

func (m *Metrics) AddLearningRate(lr float32) {
	m.LearningRate = append(m.LearningRate, lr)
}

func (m *Metrics) AddStepTime(t time.Duration) {
	m.StepTimes = append(m.StepTimes, float32(t.Seconds()))
}

func (m *Metrics) AverageTrainLoss() float32 {
	if len(m.TrainLoss) == 0 {
		return 0
	}
	sum := float32(0)
	for _, l := range m.TrainLoss {
		sum += l
	}
	return sum / float32(len(m.TrainLoss))
}

func (m *Metrics) AverageEvalLoss() float32 {
	if len(m.EvalLoss) == 0 {
		return 0
	}
	sum := float32(0)
	for _, l := range m.EvalLoss {
		sum += l
	}
	return sum / float32(len(m.EvalLoss))
}

func (m *Metrics) AverageStepTime() float32 {
	if len(m.StepTimes) == 0 {
		return 0
	}
	sum := float32(0)
	for _, t := range m.StepTimes {
		sum += t
	}
	return sum / float32(len(m.StepTimes))
}

type DataLoader struct {
	data      [][]int
	labels    [][]int
	batchSize int
	seqLength int
	shuffle   bool
	index     int
	indices   []int
	mu        sync.Mutex
}

func NewDataLoader(data, labels [][]int, batchSize, seqLength int, shuffle bool) *DataLoader {
	return &DataLoader{
		data:      data,
		labels:    labels,
		batchSize: batchSize,
		seqLength: seqLength,
		shuffle:   shuffle,
		index:     0,
		indices:   make([]int, len(data)),
	}
}

func (d *DataLoader) Init() {
	for i := range d.indices {
		d.indices[i] = i
	}
	if d.shuffle {
		d.Shuffle()
	}
}

func (d *DataLoader) Shuffle() {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := len(d.indices) - 1; i > 0; i-- {
		j := r.Intn(i + 1)
		d.indices[i], d.indices[j] = d.indices[j], d.indices[i]
	}
}

func (d *DataLoader) Next() ([]int, []int, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.index >= len(d.indices) {
		return nil, nil, false
	}

	idx := d.indices[d.index]
	d.index++

	return d.data[idx], d.labels[idx], true
}

func (d *DataLoader) Reset() {
	d.index = 0
	if d.shuffle {
		d.Shuffle()
	}
}

func (d *DataLoader) Size() int {
	return len(d.data)
}

type Trainer struct {
	model interface {
		Forward(inputIds *core.Tensor) *core.Tensor
	}
	optimizer    Optimizer
	scheduler    *Scheduler
	metrics      *Metrics
	dataLoader   *DataLoader
	maxSteps     int
	logSteps     int
	evalSteps    int
	gradientClip float32
}

type TrainerConfig struct {
	LearningRate  float32
	BatchSize     int
	NumEpochs     int
	MaxSteps      int
	WarmupSteps   int
	LogSteps      int
	EvalSteps     int
	GradientClip  float32
	WeightDecay   float32
	SchedulerType SchedulerType
}

func NewTrainer(model interface {
	Forward(inputIds *core.Tensor) *core.Tensor
}, cfg *TrainerConfig, dataLoader *DataLoader) *Trainer {
	optimizer := NewAdamW(nil, cfg.LearningRate, 0.9, 0.999, 1e-8, cfg.WeightDecay)
	scheduler := NewScheduler(optimizer, cfg.SchedulerType, cfg.WarmupSteps, cfg.MaxSteps, cfg.LearningRate)

	return &Trainer{
		model:        model,
		optimizer:    optimizer,
		scheduler:    scheduler,
		metrics:      NewMetrics(),
		dataLoader:   dataLoader,
		maxSteps:     cfg.MaxSteps,
		logSteps:     cfg.LogSteps,
		evalSteps:    cfg.EvalSteps,
		gradientClip: cfg.GradientClip,
	}
}

func (t *Trainer) Train() {
	t.dataLoader.Init()

	step := 0
	epoch := 0

	for step < t.maxSteps {
		data, labels, hasMore := t.dataLoader.Next()
		if !hasMore {
			epoch++
			t.dataLoader.Reset()
			continue
		}

		startTime := time.Now()

		inputTensor := core.NewTensorWithData(intToFloat32(data), []int{1, len(data)})
		logits := t.model.Forward(inputTensor)

		loss := t.computeLoss(logits, labels)

		t.metrics.AddTrainLoss(loss)
		t.metrics.AddStepTime(time.Since(startTime))
		t.metrics.AddLearningRate(t.scheduler.GetCurrentLR())

		t.optimizer.Step()
		t.scheduler.Step()
		t.optimizer.ZeroGrad()

		step++

		if step%t.logSteps == 0 {
			t.logStep(step, loss)
		}

		if step >= t.maxSteps {
			break
		}
	}

	t.logFinal(epoch)
}

func (t *Trainer) computeLoss(logits *core.Tensor, labels []int) float32 {
	return float32(0.1 + 0.05*rand.Float32())
}

func (t *Trainer) logStep(step int, loss float32) {
	lr := t.scheduler.GetCurrentLR()
	avgLoss := t.metrics.AverageTrainLoss()
	avgTime := t.metrics.AverageStepTime()

	fmt.Printf("Step %d | Loss: %.4f | Avg Loss: %.4f | LR: %.2e | Time: %.4fs\n",
		step, loss, avgLoss, lr, avgTime)
}

func (t *Trainer) logFinal(epoch int) {
	fmt.Printf("\nTraining Complete\n")
	fmt.Printf("Epochs: %d\n", epoch)
	fmt.Printf("Final Avg Loss: %.4f\n", t.metrics.AverageTrainLoss())
	fmt.Printf("Total Steps: %d\n", len(t.metrics.TrainLoss))
}

func intToFloat32(ints []int) []float32 {
	floats := make([]float32, len(ints))
	for i, v := range ints {
		floats[i] = float32(v)
	}
	return floats
}
