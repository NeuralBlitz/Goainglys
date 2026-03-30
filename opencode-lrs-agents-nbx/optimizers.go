package lrs

import (
	"math"
	"math/rand"
)

// LRSOptimizer wraps an optimizer with per-step LR scheduling
// This wrapper applies LR from any scheduler on each step
type LRSOptimizer struct {
	optimizer     interface {
		Step()
		ZeroGrad()
		SetLearningRate(lr float64)
	}
	scheduler     LRScheduler
	gradScale     float64
	clipGradNorm  float64
	historyLR    []float64
	historyLoss  []float64
}

func NewLRSOptimizer(optimizer interface {
	Step()
	ZeroGrad()
	SetLearningRate(lr float64)
}, scheduler LRScheduler) *LRSOptimizer {
	return &LRSOptimizer{
		optimizer:    optimizer,
		scheduler:    scheduler,
		gradScale:    1.0,
		clipGradNorm: 0.0,
		historyLR:    make([]float64, 0),
		historyLoss:  make([]float64, 0),
	}
}

func (l *LRSOptimizer) Step() float64 {
	// Get current LR from scheduler
	lr := l.scheduler.Step()
	
	// Apply gradient clipping if enabled
	if l.clipGradNorm > 0 {
		l.applyGradientClipping()
	}
	
	// Set the learning rate on the optimizer
	l.optimizer.SetLearningRate(lr * l.gradScale)
	
	// Perform the optimization step
	l.optimizer.Step()
	
	// Record history
	l.historyLR = append(l.historyLR, l.optimizer.GetLR())
	
	return lr
}

func (l *LRSOptimizer) StepWithLoss(loss float64) float64 {
	l.historyLoss = append(l.historyLoss, loss)
	return l.Step()
}

func (l *LRSOptimizer) ZeroGrad() {
	l.optimizer.ZeroGrad()
}

func (l *LRSOptimizer) SetLearningRate(lr float64) {
	l.optimizer.SetLearningRate(lr)
}

func (l *LRSOptimizer) GetLR() float64 {
	return l.scheduler.GetLR()
}

func (l *LRSOptimizer) GetHistoryLR() []float64 {
	return l.historyLR
}

func (l *LRSOptimizer) GetHistoryLoss() []float64 {
	return l.historyLoss
}

// SetGradientScale sets a multiplier for gradient scaling
func (l *LRSOptimizer) SetGradientScale(scale float64) {
	l.gradScale = scale
}

// SetGradientClipping sets gradient norm clipping
func (l *LRSOptimizer) SetGradientClipping(norm float64) {
	l.clipGradNorm = norm
}

func (l *LRSOptimizer) applyGradientClipping() {
	// This would need access to the actual parameter gradients
	// For now, this is a placeholder - in practice would compute norm
	// and scale gradients if norm > clipGradNorm
}

// ============================================================
// Prodigy: Probabilistically Optimal Optimizer with LR Integration
// A deterministic gradient scaling method with LR awareness
// ============================================================

// Prodigy implements the Prodigy optimizer
type Prodigy struct {
	Slots       []*ParamSlot
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	d           float64 // Damping factor
	step        int
	gradAnisotropy float64
}

type ParamSlot struct {
	Data     []float64
	Grad     []float64
	M1       []float64 // First moment (for bias-corrected update)
	M2       []float64 // Second moment
	Precond  []float64 // Preconditioner (adaptive per-param LR)
}

func NewProdigy(lr, beta1, beta2, eps, weightDecay, d, gradAnisotropy float64) *Prodigy {
	return &Prodigy{
		Slots:          make([]*ParamSlot, 0),
		lr:             lr,
		beta1:          beta1,
		beta2:          beta2,
		eps:            eps,
		weightDecay:    weightDecay,
		d:              d,
		gradAnisotropy: gradAnisotropy,
	}
}

func (p *Prodigy) AddParams(params [][]float64) {
	for _, param := range params {
		slot := &ParamSlot{
			Data:    param,
			Grad:    make([]float64, len(param)),
			M1:      make([]float64, len(param)),
			M2:      make([]float64, len(param)),
			Precond: make([]float64, len(param)),
		}
		p.Slots = append(p.Slots, slot)
	}
}

func (p *Prodigy) Step() {
	p.step++
	bc1 := 1.0 - math.Pow(p.beta1, float64(p.step))
	bc2 := 1.0 - math.Pow(p.beta2, float64(p.step))

	// Compute gradient statistics for preconditioner
	gradNormSq := 0.0
	for _, s := range p.Slots {
		for _, g := range s.Grad {
			gradNormSq += g * g
		}
	}
	gradNorm := math.Sqrt(gradNormSq + p.eps)

	// Adaptive learning rate scaling based on gradient norm
	// This is the "Prodigy" innovation - scale LR based on gradient magnitude
	lrScale := p.d / (gradNorm + p.d)

	for _, s := range p.Slots {
		if len(s.Grad) == 0 {
			continue
		}

		// Compute per-parameter preconditioner
		for i := range s.Data {
			g := s.Grad[i]

			// Weight decay (AdamW style)
			if p.weightDecay > 0 {
				g += p.weightDecay * s.Data[i]
			}

			// Update biased first moment estimate
			s.M1[i] = p.beta1*s.M1[i] + (1.0-p.beta1)*g

			// Update biased second raw moment estimate
			s.M2[i] = p.beta2*s.M2[i] + (1.0-p.beta2)*g*g

			// Compute preconditioner based on gradient anisotropy
			precond := s.M2[i]/bc2 + p.eps
			s.Precond[i] = precond

			// Bias-corrected first moment
			mHat := s.M1[i] / bc1

			// Apply anisotropic scaling
			anisotropyFactor := 1.0 + p.gradAnisotropy*math.Tanh(g/precond)

			// Prodigy update with adaptive scaling
			update := p.lr * lrScale * anisotropyFactor * mHat / math.Sqrt(precond)
			s.Data[i] -= update
		}
	}
}

func (p *Prodigy) ZeroGrad() {
	for _, s := range p.Slots {
		for i := range s.Grad {
			s.Grad[i] = 0
		}
	}
}

func (p *Prodigy) SetLearningRate(lr float64) {
	p.lr = lr
}

func (p *Prodigy) GetLearningRate() float64 {
	return p.lr
}

// ============================================================
// AdamScale: Adam variant scaling updates based on gradient norm ratio
// ============================================================

// AdamScale implements Adam with gradient norm-based scaling
type AdamScale struct {
	Slots       []*ParamSlot
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	step        int
	scaleFactor float64
	prevGradNorm float64
}

func NewAdamScale(lr, beta1, beta2, eps, weightDecay float64) *AdamScale {
	return &AdamScale{
		Slots:         make([]*ParamSlot, 0),
		lr:            lr,
		beta1:         beta1,
		beta2:         beta2,
		eps:           eps,
		weightDecay:   weightDecay,
		scaleFactor:   1.0,
		prevGradNorm:  0.0,
	}
}

func (a *AdamScale) AddParams(params [][]float64) {
	for _, param := range params {
		slot := &ParamSlot{
			Data: param,
			Grad: make([]float64, len(param)),
			M1:   make([]float64, len(param)),
			M2:   make([]float64, len(param)),
		}
		a.Slots = append(a.Slots, slot)
	}
}

func (a *AdamScale) Step() {
	a.step++
	bc1 := 1.0 - math.Pow(a.beta1, float64(a.step))
	bc2 := 1.0 - math.Pow(a.beta2, float64(a.step))

	// Compute current gradient norm
	gradNormSq := 0.0
	for _, s := range a.Slots {
		for _, g := range s.Grad {
			gradNormSq += g * g
		}
	}
	currentGradNorm := math.Sqrt(gradNormSq)

	// Compute scale factor based on gradient norm ratio
	if a.prevGradNorm > 0 {
		ratio := currentGradNorm / (a.prevGradNorm + a.eps)
		// Smooth the ratio
		a.scaleFactor = 0.9*a.scaleFactor + 0.1*math.Min(ratio, 2.0)
	}
	a.prevGradNorm = currentGradNorm

	for _, s := range a.Slots {
		if len(s.Grad) == 0 {
			continue
		}
		for i := range s.Data {
			g := s.Grad[i]

			// Weight decay
			if a.weightDecay > 0 {
				g += a.weightDecay * s.Data[i]
			}

			// Update moments
			s.M1[i] = a.beta1*s.M1[i] + (1.0-a.beta1)*g
			s.M2[i] = a.beta2*s.M2[i] + (1.0-a.beta2)*g*g

			// Bias-corrected moments
			mHat := s.M1[i] / bc1
			vHat := s.M2[i] / bc2

			// Apply scale factor
			denom := math.Sqrt(vHat) + a.eps
			update := a.lr * mHat / denom * a.scaleFactor

			s.Data[i] -= update
		}
	}
}

func (a *AdamScale) ZeroGrad() {
	for _, s := range a.Slots {
		for i := range s.Grad {
			s.Grad[i] = 0
		}
	}
}

func (a *AdamScale) SetLearningRate(lr float64) {
	a.lr = lr
}

func (a *AdamScale) GetLearningRate() float64 {
	return a.lr
}

// GetScaleFactor returns the current gradient scale factor
func (a *AdamScale) GetScaleFactor() float64 {
	return a.scaleFactor
}

// ============================================================
// LARS: Layer-wise Adaptive Rate Scaling
// For large batch training
// ============================================================

// LARS implements Layer-wise Adaptive Rate Scaling
type LARS struct {
	Slots       []*ParamSlot
	lr          float64
	momentum    float64
	weightDecay float64
	eps         float64
	trustCoeff  float64 // Trust coefficient (typically 0.001-0.01)
	maxScale    float64 // Maximum local LR multiplier
	minScale    float64 // Minimum local LR multiplier
}

func NewLARS(lr, momentum, weightDecay, trustCoeff, maxScale, minScale float64) *LARS {
	return &LARS{
		Slots:        make([]*ParamSlot, 0),
		lr:           lr,
		momentum:     momentum,
		weightDecay:  weightDecay,
		eps:          1e-9,
		trustCoeff:   trustCoeff,
		maxScale:     maxScale,
		minScale:     minScale,
	}
}

func (l *LARS) AddParams(params [][]float64) {
	for _, param := range params {
		slot := &ParamSlot{
			Data: param,
			Grad: make([]float64, len(param)),
			M1:   make([]float64, len(param)), // Velocity
		}
		l.Slots = append(l.Slots, slot)
	}
}

func (l *LARS) Step() {
	for _, s := range l.Slots {
		if len(s.Grad) == 0 {
			continue
		}

		// Compute weight norm and gradient norm for this parameter group
		weightNormSq := 0.0
		gradNormSq := 0.0
		for i := range s.Data {
			weightNormSq += s.Data[i] * s.Data[i]
			gradNormSq += s.Grad[i] * s.Grad[i]
		}
		weightNorm := math.Sqrt(weightNormSq)
		gradNorm := math.Sqrt(gradNormSq)

		// Compute local learning rate (LARS formula)
		var localLR float64
		if weightNorm > 0 && gradNorm > 0 {
			localLR = l.trustCoeff * weightNorm / (gradNorm + l.weightDecay*weightNorm + l.eps)
		} else {
			localLR = l.lr
		}

		// Clamp local LR
		if localLR > l.maxScale {
			localLR = l.maxScale
		}
		if localLR < l.minScale {
			localLR = l.minScale
		}

		// Apply the update with local LR
		actualLR := l.lr * localLR

		for i := range s.Data {
			g := s.Grad[i]

			// Add weight decay
			if l.weightDecay > 0 {
				g += l.weightDecay * s.Data[i]
			}

			// Momentum
			s.M1[i] = l.momentum*s.M1[i] + g

			// Update with local LR
			s.Data[i] -= actualLR * s.M1[i]
		}
	}
}

func (l *LARS) ZeroGrad() {
	for _, s := range l.Slots {
		for i := range s.Grad {
			s.Grad[i] = 0
		}
	}
}

func (l *LARS) SetLearningRate(lr float64) {
	l.lr = lr
}

func (l *LARS) GetLearningRate() float64 {
	return l.lr
}

// ============================================================
// LAMB: Layer-wise Adaptive Moments for Batch training
// ============================================================

// LAMB implements LAMB optimizer
type LAMB struct {
	Slots    []*ParamSlot
	lr       float64
	beta1    float64
	beta2    float64
	eps      float64
	weightDecay float64
	step     int
}

func NewLAMB(lr, beta1, beta2, eps, weightDecay float64) *LAMB {
	return &LAMB{
		Slots:       make([]*ParamSlot, 0),
		lr:          lr,
		beta1:       beta1,
		beta2:       beta2,
		eps:         eps,
		weightDecay: weightDecay,
	}
}

func (l *LAMB) AddParams(params [][]float64) {
	for _, param := range params {
		slot := &ParamSlot{
			Data: param,
			Grad: make([]float64, len(param)),
			M1:   make([]float64, len(param)),
			M2:   make([]float64, len(param)),
		}
		l.Slots = append(l.Slots, slot)
	}
}

func (l *LAMB) Step() {
	l.step++
	bc1 := 1.0 - math.Pow(l.beta1, float64(l.step))
	bc2 := 1.0 - math.Pow(l.beta2, float64(l.step))

	for _, s := range l.Slots {
		if len(s.Grad) == 0 {
			continue
		}

		// Compute norms
		weightNormSq := 0.0
		gradNormSq := 0.0
		for i := range s.Data {
			weightNormSq += s.Data[i] * s.Data[i]
			gradNormSq += s.Grad[i] * s.Grad[i]
		}
		weightNorm := math.Sqrt(weightNormSq)
		gradNorm := math.Sqrt(gradNormSq)

		for i := range s.Data {
			g := s.Grad[i]

			// Weight decay (AdamW style)
			if l.weightDecay > 0 {
				g += l.weightDecay * s.Data[i]
			}

			// Update moments
			s.M1[i] = l.beta1*s.M1[i] + (1.0-l.beta1)*g
			s.M2[i] = l.beta2*s.M2[i] + (1.0-l.beta2)*g*g

			// Bias correction
			mHat := s.M1[i] / bc1
			vHat := s.M2[i] / bc2

			// Compute update
			update := mHat / (math.Sqrt(vHat) + l.eps)

			// LAMB: weight norm scaling
			var r float64
			if weightNorm > 0 && gradNorm > 0 {
				// r = weight_norm / ||update||
				updateNorm := math.Abs(update)
				if updateNorm > 0 {
					r = weightNorm / updateNorm
				} else {
					r = 1.0
				}
			} else {
				r = 1.0
			}

			// Apply with LAMB scaling
			s.Data[i] -= l.lr * r * update
		}
	}
}

func (l *LAMB) ZeroGrad() {
	for _, s := range l.Slots {
		for i := range s.Grad {
			s.Grad[i] = 0
		}
	}
}

func (l *LAMB) SetLearningRate(lr float64) {
	l.lr = lr
}

func (l *LAMB) GetLearningRate() float64 {
	return l.lr
}

// ============================================================
// Shampoo: Preconditioned Stochastic Gradient Descent
// ============================================================

// Shampoo implements the Shampoo optimizer
type Shampoo struct {
	Slots       []*ParamSlot
	lr          float64
	beta1       float64
	beta2       float64
	eps         float64
	weightDecay float64
	step        int
	graftBeta   float64 // Graft from SGD
}

func NewShampoo(lr, beta1, beta2, eps, weightDecay, graftBeta float64) *Shampoo {
	return &Shampoo{
		Slots:       make([]*ParamSlot, 0),
		lr:          lr,
		beta1:       beta1,
		beta2:       beta2,
		eps:         eps,
		weightDecay: weightDecay,
		graftBeta:   graftBeta,
	}
}

func (s *Shampoo) AddParams(params [][]float64) {
	for _, param := range params {
		slot := &ParamSlot{
			Data: param,
			Grad: make([]float64, len(param)),
			M1:   make([]float64, len(param)), // First moment
		}
		s.Slots = append(s.Slots, slot)
	}
}

func (s *Shampoo) Step() {
	s.step++

	for _, slot := range s.Slots {
		if len(slot.Grad) == 0 {
			continue
		}

		// Compute diagonal preconditioner (simplified Shampoo)
		// Full Shampoo uses Kronecker products per dimension
		gradNormSq := 0.0
		for _, g := range slot.Grad {
			gradNormSq += g * g
		}
		precond := math.Sqrt(gradNormSq + s.eps)

		for i := range slot.Data {
			g := slot.Grad[i]

			// Graft from SGD
			slot.M1[i] = s.graftBeta*slot.M1[i] + g

			// Shampoo update (diagonal approximation)
			update := g / precond

			// Apply update
			slot.Data[i] -= s.lr * update
		}
	}
}

func (s *Shampoo) ZeroGrad() {
	for _, slot := range s.Slots {
		for i := range slot.Grad {
			slot.Grad[i] = 0
		}
	}
}

func (s *Shampoo) SetLearningRate(lr float64) {
	s.lr = lr
}

func (s *Shampoo) GetLearningRate() float64 {
	return s.lr
}

// ============================================================
// Optimizer Factory Functions
// ============================================================

// CreateOptimizer creates an optimizer by name
func CreateOptimizer(name string, lr float64, params [][]float64) interface {
	Step()
	ZeroGrad()
	SetLearningRate(lr float64)
} {
	switch name {
	case "adam":
		opt := NewAdamScale(lr, 0.9, 0.999, 1e-8, 0.0)
		opt.AddParams(params)
		return opt
	case "prodigy":
		opt := NewProdigy(lr, 0.9, 0.999, 1e-8, 0.01, 1.0, 0.5)
		opt.AddParams(params)
		return opt
	case "lars":
		opt := NewLARS(lr, 0.9, 0.01, 0.001, 10.0, 0.001)
		opt.AddParams(params)
		return opt
	case "lamb":
		opt := NewLAMB(lr, 0.9, 0.999, 1e-6, 0.01)
		opt.AddParams(params)
		return opt
	default:
		// Default to AdamScale
		opt := NewAdamScale(lr, 0.9, 0.999, 1e-8, 0.0)
		opt.AddParams(params)
		return opt
	}
}
