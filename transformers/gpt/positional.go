package gpt

import (
	"math"

	"transformers/core"
)

type PositionalEncoding struct {
	DropoutP float32
}

func NewPositionalEncoding(dModel, maxLen, dropout float32) *PositionalEncoding {
	return &PositionalEncoding{DropoutP: dropout}
}

func (p *PositionalEncoding) Forward(x *core.Tensor) *core.Tensor {
	var seqLen, dModel int
	var batchSize int
	if len(x.Shape) == 3 {
		batchSize = x.Shape[0]
		seqLen = x.Shape[1]
		dModel = x.Shape[2]
	} else {
		batchSize = 1
		seqLen = x.Shape[0]
		dModel = x.Shape[1]
	}

	pe := core.NewTensor(seqLen, dModel)
	divTerm := make([]float32, dModel/2)
	for i := 0; i < dModel/2; i++ {
		divTerm[i] = float32(math.Exp(math.Log(10000) / float64(dModel) * float64(-2*i) / float64(dModel)))
	}

	for pos := 0; pos < seqLen; pos++ {
		for i := 0; i < dModel; i++ {
			if i%2 == 0 {
				pe.Data[pos*dModel+i] = float32(math.Sin(float64(pos) * float64(divTerm[i/2])))
			} else {
				pe.Data[pos*dModel+i] = float32(math.Cos(float64(pos) * float64(divTerm[i/2])))
			}
		}
	}

	if len(x.Shape) == 2 {
		return core.Add(x, pe)
	}

	pe3d := core.NewTensor(batchSize, seqLen, dModel)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < dModel; j++ {
				pe3d.Data[(b*seqLen+i)*dModel+j] = pe.Data[i*dModel+j]
			}
		}
	}

	return core.Add(x, pe3d)
}

type LearnedPositionalEncoding struct {
	Weights  *core.Tensor
	DropoutP float32
}

func NewLearnedPositionalEncoding(numPositions, dModel int, dropout float32) *LearnedPositionalEncoding {
	return &LearnedPositionalEncoding{
		Weights:  core.TensorRandUniformSeeded(42, -0.02, 0.02, numPositions, dModel),
		DropoutP: dropout,
	}
}

func (p *LearnedPositionalEncoding) Forward(x *core.Tensor) *core.Tensor {
	seqLen := x.Shape[1]
	dModel := x.Shape[2]
	posEmbed := core.NewTensor(seqLen, dModel)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < dModel; j++ {
			posEmbed.Data[i*dModel+j] = p.Weights.Data[i*dModel+j]
		}
	}
	return core.Add(x, posEmbed)
}

type RotaryPositionalEncoding struct {
	dim int
}

func NewRotaryPositionalEncoding(dim int) *RotaryPositionalEncoding {
	return &RotaryPositionalEncoding{dim: dim}
}

func RotaryEmbedding(q, k *core.Tensor, seqLen, headDim int) (*core.Tensor, *core.Tensor) {
	qOut := q.Copy()
	kOut := k.Copy()

	freqs := make([]float32, seqLen*headDim/2)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < headDim/2; j++ {
			freqs[i*headDim/2+j] = float32(math.Pow(10000, -float64(2*j)/float64(headDim)))
		}
	}

	return qOut, kOut
}

type ALiBiPositionalEncoding struct {
	numHeads int
	slope    []float32
}

func NewALiBiPositionalEncoding(numHeads int) *ALiBiPositionalEncoding {
	slope := make([]float32, numHeads)
	for i := 0; i < numHeads; i++ {
		slope[i] = float32(math.Pow(2, float64(-(numHeads-1-i)/numHeads)))
	}
	return &ALiBiPositionalEncoding{numHeads: numHeads, slope: slope}
}

func (p *ALiBiPositionalEncoding) GetSlope(headIdx int) float32 {
	return p.slope[headIdx]
}
