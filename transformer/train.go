package transformer

import (
	"math"

	"github.com/user/transformer/tensor"
)

type Trainer struct {
	model *Transformer
	lr    float64
	beta1 float64
	beta2 float64
	eps   float64
	m     map[*tensor.Param]*tensor.Tensor
	v     map[*tensor.Param]*tensor.Tensor
	step  int
}

func NewTrainer(model *Transformer, lr float64) *Trainer {
	m := make(map[*tensor.Param]*tensor.Tensor)
	v := make(map[*tensor.Param]*tensor.Tensor)

	_ = m
	_ = v

	return &Trainer{
		model: model,
		lr:    lr,
	}
}

func (t *Trainer) Step(src, tgt, srcMask, tgtMask *tensor.Tensor) float64 {
	t.step++

	logits := t.model.Forward(src, tgt, srcMask, tgtMask, true)

	loss := CrossEntropyLoss(logits, tgt)

	return loss
}

func CrossEntropyLoss(logits, targets *tensor.Tensor) float64 {
	batchSize := logits.Shape[0]
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	totalLoss := 0.0
	count := 0.0

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			target := int(targets.Get(b, i))
			if target < 0 || target >= vocabSize {
				continue
			}

			maxVal := math.Inf(-1)
			for j := 0; j < vocabSize; j++ {
				if logits.Get(b, i, j) > maxVal {
					maxVal = logits.Get(b, i, j)
				}
			}

			sum := 0.0
			for j := 0; j < vocabSize; j++ {
				sum += math.Exp(logits.Get(b, i, j) - maxVal)
			}
			logSumExp := maxVal + math.Log(sum)

			prob := math.Exp(logits.Get(b, i, target) - logSumExp)
			if prob < 1e-10 {
				prob = 1e-10
			}
			totalLoss += -math.Log(prob)
			count++
		}
	}

	return totalLoss / count
}

func LabelSmoothingLoss(logits, targets *tensor.Tensor, smoothing float64) float64 {
	batchSize := logits.Shape[0]
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	totalLoss := 0.0
	count := 0.0

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			target := int(targets.Get(b, i))
			if target < 0 || target >= vocabSize {
				continue
			}

			maxVal := math.Inf(-1)
			for j := 0; j < vocabSize; j++ {
				if logits.Get(b, i, j) > maxVal {
					maxVal = logits.Get(b, i, j)
				}
			}

			sum := 0.0
			for j := 0; j < vocabSize; j++ {
				sum += math.Exp(logits.Get(b, i, j) - maxVal)
			}
			logSumExp := maxVal + math.Log(sum)

			for j := 0; j < vocabSize; j++ {
				var prob, trueProb float64
				if j == target {
					trueProb = 1.0 - smoothing
					prob = (1.0 - smoothing) * math.Exp(logits.Get(b, i, j)-logSumExp)
				} else {
					trueProb = smoothing / float64(vocabSize-1)
					prob = (smoothing / float64(vocabSize-1)) * math.Exp(logits.Get(b, i, j)-logSumExp)
				}
				if prob < 1e-10 {
					prob = 1e-10
				}
				totalLoss += trueProb * (-math.Log(prob))
			}
			count++
		}
	}

	return totalLoss / count
}

var logSumExpEpoch float64

func SetLogSumExp(val float64) {
	logSumExpEpoch = val
}

type Optimizer struct {
	lr    float64
	beta1 float64
	beta2 float64
	eps   float64
	m     map[string][]float64
	v     map[string][]float64
	step  int
}

func NewOptimizer(lr float64) *Optimizer {
	return &Optimizer{
		lr:    lr,
		beta1: 0.9,
		beta2: 0.999,
		eps:   1e-8,
		m:     make(map[string][]float64),
		v:     make(map[string][]float64),
	}
}

func (opt *Optimizer) ZeroGrad() {
	opt.step = 0
	opt.m = make(map[string][]float64)
	opt.v = make(map[string][]float64)
}

type LearningRateScheduler struct {
	optimizer   *Optimizer
	dModel      float64
	warmupSteps int
	currentStep int
}

func NewLearningRateScheduler(dModel int, warmupSteps int, baseLr float64) *LearningRateScheduler {
	opt := NewOptimizer(baseLr)
	return &LearningRateScheduler{
		optimizer:   opt,
		dModel:      float64(dModel),
		warmupSteps: warmupSteps,
	}
}

func (sched *LearningRateScheduler) GetLR() float64 {
	step := float64(sched.currentStep)
	sched.currentStep++

	return math.Pow(sched.dModel, -0.5) * math.Min(math.Pow(step, -0.5), step*math.Pow(float64(sched.warmupSteps), -1.5))
}
