package losses

import (
	"math"
)

func CrossEntropy(logits []float32, targets []int, vocabSize int) float64 {
	if len(logits) == 0 || len(targets) == 0 {
		return 0
	}

	seqLen := len(targets)
	loss := 0.0
	count := 0

	for i := 0; i < seqLen; i++ {
		targetIdx := targets[i]
		if targetIdx < 0 || targetIdx >= vocabSize {
			continue
		}

		maxLogit := float32(-math.MaxFloat32)
		offset := i * vocabSize
		for j := 0; j < vocabSize; j++ {
			if logits[offset+j] > maxLogit {
				maxLogit = logits[offset+j]
			}
		}

		sum := float32(0)
		for j := 0; j < vocabSize; j++ {
			sum += float32(math.Exp(float64(logits[offset+j] - maxLogit)))
		}

		logProb := float64(logits[offset+targetIdx]-maxLogit) - math.Log(float64(sum))
		loss -= logProb
		count++
	}

	if count > 0 {
		loss /= float64(count)
	}

	return loss
}

func CrossEntropyWithSoftmax(logits []float32, targets []float32, vocabSize int) float64 {
	if len(logits) == 0 || len(targets) == 0 {
		return 0
	}

	seqLen := len(targets)
	loss := 0.0

	for i := 0; i < seqLen; i++ {
		offset := i * vocabSize

		maxLogit := float32(-math.MaxFloat32)
		for j := 0; j < vocabSize; j++ {
			if logits[offset+j] > maxLogit {
				maxLogit = logits[offset+j]
			}
		}

		sum := float32(0)
		for j := 0; j < vocabSize; j++ {
			sum += float32(math.Exp(float64(logits[offset+j] - maxLogit)))
		}

		logSumExp := float64(maxLogit) + math.Log(float64(sum))

		targetVal := targets[i]
		if targetVal > 0 {
			loss += float64(targetVal) * (logSumExp - float64(logits[offset+i]))
		}
	}

	return loss / float64(seqLen)
}

func MSE(predictions, targets []float32) float64 {
	if len(predictions) != len(targets) {
		return 0
	}

	sum := 0.0
	for i := range predictions {
		diff := float64(predictions[i] - targets[i])
		sum += diff * diff
	}

	return sum / float64(len(predictions))
}

func BinaryCrossEntropy(logits, targets []float32) float64 {
	if len(logits) != len(targets) {
		return 0
	}

	loss := 0.0
	eps := 1e-7

	for i := range logits {
		p := float64(1.0 / (1.0 + math.Exp(-float64(logits[i]))))
		p = math.Max(eps, math.Min(1.0-eps, p))
		t := float64(targets[i])
		loss -= t*math.Log(p) + (1-t)*math.Log(1-p)
	}

	return loss / float64(len(logits))
}

type LossFunction func([]float32, []int, int) float64

var RegisteredLosses = map[string]LossFunction{
	"cross_entropy": CrossEntropy,
}

func GetLoss(name string) LossFunction {
	if loss, ok := RegisteredLosses[name]; ok {
		return loss
	}
	return CrossEntropy
}
