package optimizers

import (
	"math"
	"sync"
)

type Tensor struct {
	Data  []float32
	Shape []int
}

func (t *Tensor) Numel() int {
	n := 1
	for _, s := range t.Shape {
		n *= s
	}
	return n
}

type Optimizer interface {
	Step()
	ZeroGrad()
	SetLearningRate(lr float64)
}

type AdamW struct {
	parameters  []*AdamParam
	lr          float64
	betas       [2]float64
	eps         float64
	weightDecay float64
}

type AdamParam struct {
	Tensor   []float32
	Grad     []float32
	ExpAvg   []float32
	ExpAvgSq []float32
	Step     int
}

func NewAdamW(lr, beta1, beta2, eps, weightDecay float64) *AdamW {
	return &AdamW{
		parameters:  make([]*AdamParam, 0),
		lr:          lr,
		betas:       [2]float64{beta1, beta2},
		eps:         eps,
		weightDecay: weightDecay,
	}
}

func (a *AdamW) AddParams(params []*Tensor) {
	for _, p := range params {
		a.parameters = append(a.parameters, &AdamParam{
			Tensor:   p.Data,
			Grad:     nil,
			ExpAvg:   make([]float32, len(p.Data)),
			ExpAvgSq: make([]float32, len(p.Data)),
			Step:     0,
		})
	}
}

func (a *AdamW) Step() {
	for _, p := range a.parameters {
		if p.Grad == nil || len(p.Grad) == 0 {
			continue
		}

		p.Step++

		beta1Pow := math.Pow(a.betas[0], float64(p.Step))
		beta2Pow := math.Pow(a.betas[1], float64(p.Step))

		lr := a.lr * math.Sqrt(1-beta2Pow) / (1 - beta1Pow)

		for i := range p.Tensor {
			grad := float64(p.Grad[i])

			if a.weightDecay > 0 {
				grad += a.weightDecay * float64(p.Tensor[i])
			}

			p.ExpAvg[i] = float32(float64(p.ExpAvg[i])*a.betas[0] + (1-a.betas[0])*grad)
			p.ExpAvgSq[i] = float32(float64(p.ExpAvgSq[i])*a.betas[1] + (1-a.betas[1])*grad*grad)

			denom := math.Sqrt(float64(p.ExpAvgSq[i]))/math.Sqrt(1-beta2Pow) + a.eps

			update := lr * float64(p.ExpAvg[i]) / denom

			p.Tensor[i] -= float32(update)
		}
	}
}

func (a *AdamW) ZeroGrad() {
	for _, p := range a.parameters {
		p.Grad = nil
	}
}

func (a *AdamW) SetLearningRate(lr float64) {
	a.lr = lr
}

type Adam struct {
	parameters []*AdamParam
	lr         float64
	betas      [2]float64
	eps        float64
}

func NewAdam(lr, beta1, beta2, eps float64) *Adam {
	return &Adam{
		parameters: make([]*AdamParam, 0),
		lr:         lr,
		betas:      [2]float64{beta1, beta2},
		eps:        eps,
	}
}

func (a *Adam) AddParams(params []*Tensor) {
	for _, p := range params {
		a.parameters = append(a.parameters, &AdamParam{
			Tensor:   p.Data,
			Grad:     nil,
			ExpAvg:   make([]float32, len(p.Data)),
			ExpAvgSq: make([]float32, len(p.Data)),
			Step:     0,
		})
	}
}

func (a *Adam) Step() {
	for _, p := range a.parameters {
		if p.Grad == nil || len(p.Grad) == 0 {
			continue
		}

		p.Step++

		beta1Pow := math.Pow(a.betas[0], float64(p.Step))
		beta2Pow := math.Pow(a.betas[1], float64(p.Step))

		lr := a.lr * math.Sqrt(1-beta2Pow) / (1 - beta1Pow)

		for i := range p.Tensor {
			grad := float64(p.Grad[i])

			p.ExpAvg[i] = float32(float64(p.ExpAvg[i])*a.betas[0] + (1-a.betas[0])*grad)
			p.ExpAvgSq[i] = float32(float64(p.ExpAvgSq[i])*a.betas[1] + (1-a.betas[1])*grad*grad)

			denom := math.Sqrt(float64(p.ExpAvgSq[i]))/math.Sqrt(1-beta2Pow) + a.eps

			update := lr * float64(p.ExpAvg[i]) / denom

			p.Tensor[i] -= float32(update)
		}
	}
}

func (a *Adam) ZeroGrad() {
	for _, p := range a.parameters {
		p.Grad = nil
	}
}

func (a *Adam) SetLearningRate(lr float64) {
	a.lr = lr
}

type SGD struct {
	parameters []*AdamParam
	lr         float64
	momentum   float64
}

func NewSGD(lr, momentum float64) *SGD {
	return &SGD{
		parameters: make([]*AdamParam, 0),
		lr:         lr,
		momentum:   momentum,
	}
}

func (s *SGD) AddParams(params []*Tensor) {
	for _, p := range params {
		s.parameters = append(s.parameters, &AdamParam{
			Tensor: p.Data,
			Grad:   nil,
			ExpAvg: make([]float32, len(p.Data)),
			Step:   0,
		})
	}
}

func (s *SGD) Step() {
	for _, p := range s.parameters {
		if p.Grad == nil || len(p.Grad) == 0 {
			continue
		}

		for i := range p.Tensor {
			grad := float64(p.Grad[i])

			if s.momentum > 0 {
				p.ExpAvg[i] = float32(s.momentum*float64(p.ExpAvg[i]) + grad)
				grad = float64(p.ExpAvg[i])
			}

			p.Tensor[i] -= float32(s.lr * grad)
		}
	}
}

func (s *SGD) ZeroGrad() {
	for _, p := range s.parameters {
		p.Grad = nil
	}
}

func (s *SGD) SetLearningRate(lr float64) {
	s.lr = lr
}

type Lion struct {
	parameters []*AdamParam
	lr         float64
	betas      [2]float64
}

func NewLion(lr, beta1, beta2 float64) *Lion {
	return &Lion{
		parameters: make([]*AdamParam, 0),
		lr:         lr,
		betas:      [2]float64{beta1, beta2},
	}
}

func (l *Lion) AddParams(params []*Tensor) {
	for _, p := range params {
		l.parameters = append(l.parameters, &AdamParam{
			Tensor: p.Data,
			Grad:   nil,
			ExpAvg: make([]float32, len(p.Data)),
			Step:   0,
		})
	}
}

func (l *Lion) Step() {
	for _, p := range l.parameters {
		if p.Grad == nil || len(p.Grad) == 0 {
			continue
		}

		for i := range p.Tensor {
			grad := float64(p.Grad[i])

			p.ExpAvg[i] = float32(l.betas[0]*float64(p.ExpAvg[i]) + (1-l.betas[0])*grad)

			update := l.lr * float64(p.ExpAvg[i])

			p.Tensor[i] -= float32(update)
		}
	}
}

func (l *Lion) ZeroGrad() {
	for _, p := range l.parameters {
		p.Grad = nil
	}
}

func (l *Lion) SetLearningRate(lr float64) {
	l.lr = lr
}

type RMSprop struct {
	parameters []*AdamParam
	lr         float64
	alpha      float64
	eps        float64
	momentum   float64
}

func NewRMSprop(lr, alpha, eps, momentum float64) *RMSprop {
	return &RMSprop{
		parameters: make([]*AdamParam, 0),
		lr:         lr,
		alpha:      alpha,
		eps:        eps,
		momentum:   momentum,
	}
}

func (r *RMSprop) AddParams(params []*Tensor) {
	for _, p := range params {
		r.parameters = append(r.parameters, &AdamParam{
			Tensor:   p.Data,
			Grad:     nil,
			ExpAvgSq: make([]float32, len(p.Data)),
			Step:     0,
		})
	}
}

func (r *RMSprop) Step() {
	for _, p := range r.parameters {
		if p.Grad == nil || len(p.Grad) == 0 {
			continue
		}

		for i := range p.Tensor {
			grad := float64(p.Grad[i])

			p.ExpAvgSq[i] = float32(r.alpha*float64(p.ExpAvgSq[i]) + (1-r.alpha)*grad*grad)

			denom := math.Sqrt(float64(p.ExpAvgSq[i])) + r.eps

			update := r.lr * grad / denom

			p.Tensor[i] -= float32(update)
		}
	}
}

func (r *RMSprop) ZeroGrad() {
	for _, p := range r.parameters {
		p.Grad = nil
	}
}

func (r *RMSprop) SetLearningRate(lr float64) {
	r.lr = lr
}

type OptimizerManager struct {
	optimizers []Optimizer
	mu         sync.RWMutex
}

func NewOptimizerManager() *OptimizerManager {
	return &OptimizerManager{
		optimizers: make([]Optimizer, 0),
	}
}

func (m *OptimizerManager) Add(opt Optimizer) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.optimizers = append(m.optimizers, opt)
}

func (m *OptimizerManager) Step() {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, opt := range m.optimizers {
		opt.Step()
	}
}

func (m *OptimizerManager) ZeroGrad() {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, opt := range m.optimizers {
		opt.ZeroGrad()
	}
}

func (m *OptimizerManager) SetLearningRate(lr float64) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	for _, opt := range m.optimizers {
		opt.SetLearningRate(lr)
	}
}
