package bert

import (
	"math"

	"transformers/core"
)

type BERTPreTrainingHeads struct {
	Dense     *Linear
	LayerNorm *LayerNorm
	LmHead    *Linear
}

func NewBERTPreTrainingHeads(config *BERTConfig) *BERTPreTrainingHeads {
	return &BERTPreTrainingHeads{
		Dense:     NewLinear(config.HiddenSize, config.HiddenSize),
		LayerNorm: NewLayerNorm(config.HiddenSize, config.LayerNormEpsilon),
		LmHead:    NewLinear(config.HiddenSize, config.VocabSize),
	}
}

func (h *BERTPreTrainingHeads) Forward(sequence_output, pooled_output *core.Tensor) *PreTrainingOutput {
	pooled_output = h.Dense.Forward(pooled_output)
	pooled_output = Gelu(pooled_output)
	pooled_output = h.LayerNorm.Forward(pooled_output)
	sequence_output = h.LmHead.Forward(sequence_output)

	return &PreTrainingOutput{
		PredictionScores:     sequence_output,
		SeqRelationshipScore: pooled_output,
	}
}

type PreTrainingOutput struct {
	PredictionScores     *core.Tensor
	SeqRelationshipScore *core.Tensor
	HiddenStates         []*core.Tensor
	Attentions           []*core.Tensor
}

type MaskedLMOutput struct {
	Loss             float32
	PredictionScores *core.Tensor
	HiddenStates     []*core.Tensor
	Attentions       []*core.Tensor
}

type NextSentencePredictionOutput struct {
	Loss             float32
	SeqRelationScore *core.Tensor
	HiddenStates     []*core.Tensor
	Attentions       []*core.Tensor
}

type SequenceClassifierOutput struct {
	Logits       *core.Tensor
	HiddenStates []*core.Tensor
	Attentions   []*core.Tensor
}

type TokenClassifierOutput struct {
	Logits       *core.Tensor
	HiddenStates []*core.Tensor
	Attentions   []*core.Tensor
}

type QuestionAnsweringOutput struct {
	Start_logits *core.Tensor
	End_logits   *core.Tensor
	HiddenStates []*core.Tensor
	Attentions   []*core.Tensor
}

func MLMCrossEntropyLoss(predictionScores, maskedLabels *core.Tensor, ignoreIndex int) float32 {
	seqLen := predictionScores.Shape[1]
	vocabSize := predictionScores.Shape[2]

	loss := float32(0)
	count := 0

	batchSize := predictionScores.Shape[0]
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			label := int(maskedLabels.Data[b*seqLen+i])
			if label == ignoreIndex {
				continue
			}

			maxLogit := float32(-math.MaxFloat32)
			for k := 0; k < vocabSize; k++ {
				logit := predictionScores.Data[(b*seqLen+i)*vocabSize+k]
				if logit > maxLogit {
					maxLogit = logit
				}
			}

			sum := float32(0)
			for k := 0; k < vocabSize; k++ {
				sum += float32(math.Exp(float64(predictionScores.Data[(b*seqLen+i)*vocabSize+k] - maxLogit)))
			}

			logProb := float64(predictionScores.Data[(b*seqLen+i)*vocabSize+label]-maxLogit) - math.Log(float64(sum))
			loss -= float32(logProb)
			count++
		}
	}

	if count > 0 {
		loss /= float32(count)
	}
	return loss
}

func BinaryCrossEntropyLoss(logits, labels *core.Tensor) float32 {
	batchSize := logits.Shape[0]
	loss := float32(0)

	for i := 0; i < batchSize; i++ {
		logit := logits.Data[i]
		label := labels.Data[i]

		logit = max32(logit, 1e-7)
		logit = min32(logit, 1-1e-7)

		loss -= label * float32(math.Log(float64(logit)))
		loss -= (1 - label) * float32(math.Log(float64(1-logit)))
	}

	return loss / float32(batchSize)
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}

func min32(a, b float32) float32 {
	if a < b {
		return a
	}
	return b
}
