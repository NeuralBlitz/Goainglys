package transformer

import (
	"math"
	"math/rand"

	"github.com/user/transformer/tensor"
)

type Config struct {
	ModelDim  int
	NumHeads  int
	NumLayers int
	FFNDim    int
	VocabSize int
	MaxSeqLen int
	Dropout   float64
}

type MultiHeadAttention struct {
	config               Config
	Wq                   *tensor.Param
	Wk                   *tensor.Param
	Wv                   *tensor.Param
	Wo                   *tensor.Param
	Dropout              float64
	LastAttentionWeights *tensor.Tensor
}

func NewMultiHeadAttention(config Config) *MultiHeadAttention {
	dModel := config.ModelDim

	return &MultiHeadAttention{
		config:  config,
		Wq:      tensor.NewParam(dModel, dModel),
		Wk:      tensor.NewParam(dModel, dModel),
		Wv:      tensor.NewParam(dModel, dModel),
		Wo:      tensor.NewParam(dModel, dModel),
		Dropout: config.Dropout,
	}
}

func (mha *MultiHeadAttention) Forward(x, mask *tensor.Tensor, train bool) *tensor.Tensor {
	batchSize := x.Shape[0]
	seqLen := x.Shape[1]
	dModel := x.Shape[2]
	dK := dModel / mha.config.NumHeads

	flat := x.Reshape(batchSize*seqLen, dModel)

	Q := matMul(flat, mha.Wq.Data)
	K := matMul(flat, mha.Wk.Data)
	V := matMul(flat, mha.Wv.Data)

	return mha.Attend(Q, K, V, batchSize, seqLen, seqLen, dK, mask, train)
}

func (mha *MultiHeadAttention) Attend(Q, K, V *tensor.Tensor, batchSize, qSeqLen, kvSeqLen, dK int, mask *tensor.Tensor, train bool) *tensor.Tensor {
	dModel := mha.config.ModelDim
	numHeads := mha.config.NumHeads

	Q = reshape4D(Q, batchSize, qSeqLen, numHeads, dK)
	K = reshape4D(K, batchSize, kvSeqLen, numHeads, dK)
	V = reshape4D(V, batchSize, kvSeqLen, numHeads, dK)

	Q = transpose4D(Q, 0, 2, 1, 3)
	K = transpose4D(K, 0, 2, 1, 3)
	V = transpose4D(V, 0, 2, 1, 3)

	scores := matMul4D(Q, K, dK, kvSeqLen)
	if mask != nil {
		scores = addMask4D(scores, mask)
	}
	attnWeights := softmax4D(scores, -1)

	// Store attention weights for visualization (only during inference)
	if !train {
		mha.LastAttentionWeights = attnWeights.Clone()
	}

	attnWeights = dropout4D(attnWeights, mha.Dropout, train)

	attnOutput := matMul4D(attnWeights, V, dK, dK)
	attnOutput = transpose4D(attnOutput, 0, 2, 1, 3)
	attnOutput = attnOutput.Reshape(batchSize*qSeqLen, numHeads*dK)

	output := matMul(attnOutput, mha.Wo.Data)
	output = output.Reshape(batchSize, qSeqLen, dModel)
	if train && mha.Dropout > 0 {
		output = dropout(output, mha.Dropout)
	}

	return output
}

type FeedForward struct {
	W1 *tensor.Param
	W2 *tensor.Param
	B1 *tensor.Param
	B2 *tensor.Param
}

func (ff *FeedForward) GetW1() *tensor.Param { return ff.W1 }
func (ff *FeedForward) GetW2() *tensor.Param { return ff.W2 }
func (ff *FeedForward) GetB1() *tensor.Param { return ff.B1 }
func (ff *FeedForward) GetB2() *tensor.Param { return ff.B2 }

func NewFeedForward(dModel, dFF int) *FeedForward {
	return &FeedForward{
		W1: tensor.NewParam(dModel, dFF),
		W2: tensor.NewParam(dFF, dModel),
		B1: tensor.NewParam(dFF),
		B2: tensor.NewParam(dModel),
	}
}

func (ff *FeedForward) Forward(x *tensor.Tensor, train bool) *tensor.Tensor {
	var flat *tensor.Tensor

	if len(x.Shape) == 2 {
		// x is already 2D (batch*seq, dModel)
		flat = x
	} else {
		// x is 3D (batch, seq, dModel)
		origShape := x.Shape
		flat = x.Reshape(origShape[0]*origShape[1], origShape[2])
	}

	hidden := matMul(flat, ff.W1.Data)
	hidden = addBias(hidden, ff.B1.Data)
	hidden = relu(hidden)
	hidden = matMul(hidden, ff.W2.Data)
	hidden = addBias(hidden, ff.B2.Data)

	return hidden
}

type PositionalEncoding struct {
	encoding *tensor.Tensor
}

func NewPositionalEncoding(dModel, maxSeqLen int) *PositionalEncoding {
	pe := tensor.New(maxSeqLen, dModel)
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < dModel; i++ {
			var val float64
			if i%2 == 0 {
				val = math.Sin(float64(pos) / math.Pow(10000, float64(i)/float64(dModel)))
			} else {
				val = math.Cos(float64(pos) / math.Pow(10000, float64(i-1)/float64(dModel)))
			}
			pe.Set(val, pos, i)
		}
	}
	return &PositionalEncoding{encoding: pe}
}

func (pe *PositionalEncoding) Add(x *tensor.Tensor) *tensor.Tensor {
	seqLen := x.Shape[1]
	result := x.Clone()
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < seqLen; j++ {
			for k := 0; k < x.Shape[2]; k++ {
				val := result.Get(i, j, k)
				val += pe.encoding.Get(j, k)
				result.Set(val, i, j, k)
			}
		}
	}
	return result
}

type SubLayer struct {
	Attention *MultiHeadAttention
	Ffn       *FeedForward
	Ln1       *tensor.Param
	Ln2       *tensor.Param
}

func NewSubLayer(config Config) *SubLayer {
	return &SubLayer{
		Attention: NewMultiHeadAttention(config),
		Ffn:       NewFeedForward(config.ModelDim, config.FFNDim),
		Ln1:       tensor.NewParam(config.ModelDim),
		Ln2:       tensor.NewParam(config.ModelDim),
	}
}

func (sl *SubLayer) Forward(x, mask *tensor.Tensor, train bool) *tensor.Tensor {
	attn := sl.Attention.Forward(x, mask, train)
	x = tensor.Add(x, attn)
	x = layerNorm(x, sl.Ln1.Data, sl.Ln1.Data)

	ffnOut := sl.Ffn.Forward(x, train)
	x = tensor.Add(x, ffnOut)
	x = layerNorm(x, sl.Ln2.Data, sl.Ln2.Data)

	return x
}

func matMul(a, b *tensor.Tensor) *tensor.Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("matMul requires 2D tensors")
	}
	result := tensor.New(a.Shape[0], b.Shape[1])
	for i := 0; i < a.Shape[0]; i++ {
		for j := 0; j < b.Shape[1]; j++ {
			sum := 0.0
			for k := 0; k < a.Shape[1]; k++ {
				sum += a.Get(i, k) * b.Get(k, j)
			}
			result.Set(sum, i, j)
		}
	}
	return result
}

func reshape(x *tensor.Tensor, dims ...int) *tensor.Tensor {
	return x.Reshape(dims...)
}

func reshape4D(x *tensor.Tensor, d0, d1, d2, d3 int) *tensor.Tensor {
	return x.Reshape(d0, d1, d2, d3)
}

func addMask4D(x, mask *tensor.Tensor) *tensor.Tensor {
	result := x.Clone()
	// x has shape [batch, heads, seqLen, seqLen]
	// mask has shape [batch, seqLen, seqLen]
	if len(x.Shape) != 4 || len(mask.Shape) != 3 {
		return result
	}

	// Safety check: only loop up to mask dimensions
	maxI := x.Shape[2]
	maxJ := x.Shape[3]
	if maxI > mask.Shape[1] {
		maxI = mask.Shape[1]
	}
	if maxJ > mask.Shape[2] {
		maxJ = mask.Shape[2]
	}

	for b := 0; b < x.Shape[0]; b++ {
		for h := 0; h < x.Shape[1]; h++ {
			for i := 0; i < maxI; i++ {
				for j := 0; j < maxJ; j++ {
					maskVal := mask.Get(b, i, j)
					if maskVal == 0 {
						result.Set(math.Inf(-1), b, h, i, j)
					}
				}
			}
		}
	}
	return result
}

func transpose4D(x *tensor.Tensor, d0, d1, d2, d3 int) *tensor.Tensor {
	shape := x.Shape
	result := tensor.New(shape[d0], shape[d1], shape[d2], shape[d3])
	for i := 0; i < shape[0]; i++ {
		for j := 0; j < shape[1]; j++ {
			for k := 0; k < shape[2]; k++ {
				for l := 0; l < shape[3]; l++ {
					var newI, newJ, newK, newL int
					switch d0 {
					case 0:
						newI = i
					case 1:
						newJ = i
					case 2:
						newK = i
					case 3:
						newL = i
					}
					switch d1 {
					case 0:
						newI = j
					case 1:
						newJ = j
					case 2:
						newK = j
					case 3:
						newL = j
					}
					switch d2 {
					case 0:
						newI = k
					case 1:
						newJ = k
					case 2:
						newK = k
					case 3:
						newL = k
					}
					switch d3 {
					case 0:
						newI = l
					case 1:
						newJ = l
					case 2:
						newK = l
					case 3:
						newL = l
					}
					result.Set(x.Get(i, j, k, l), newI, newJ, newK, newL)
				}
			}
		}
	}
	return result
}

func matMul4D(a, b *tensor.Tensor, scaleDim, outDim int) *tensor.Tensor {
	shapeA := a.Shape
	shapeB := b.Shape
	batchSize := shapeA[0]
	numHeads := shapeA[1]
	seqLenA := shapeA[2]
	contractionDim := shapeA[3]

	// a: [batch, heads, seqLenA, contractionDim]
	// b: [batch, heads, contractionDim, outDim] or [batch, heads, outDim, contractionDim]
	// Check which one
	if shapeB[3] == outDim {
		// b has shape [batch, heads, contractionDim, outDim]
		result := tensor.New(batchSize, numHeads, seqLenA, outDim)
		for bi := 0; bi < batchSize; bi++ {
			for h := 0; h < numHeads; h++ {
				for i := 0; i < seqLenA; i++ {
					for j := 0; j < outDim; j++ {
						sum := 0.0
						for k := 0; k < contractionDim; k++ {
							// a[bi, h, i, k] * b[bi, h, k, j]
							sum += a.Get(bi, h, i, k) * b.Get(bi, h, k, j)
						}
						if scaleDim >= 0 {
							sum /= math.Sqrt(float64(contractionDim))
						}
						result.Set(sum, bi, h, i, j)
					}
				}
			}
		}
		return result
	} else if shapeB[2] == outDim {
		// b has shape [batch, heads, outDim, contractionDim]
		result := tensor.New(batchSize, numHeads, seqLenA, outDim)
		for bi := 0; bi < batchSize; bi++ {
			for h := 0; h < numHeads; h++ {
				for i := 0; i < seqLenA; i++ {
					for j := 0; j < outDim; j++ {
						sum := 0.0
						for k := 0; k < contractionDim; k++ {
							// a[bi, h, i, k] * b[bi, h, j, k]
							sum += a.Get(bi, h, i, k) * b.Get(bi, h, j, k)
						}
						if scaleDim >= 0 {
							sum /= math.Sqrt(float64(contractionDim))
						}
						result.Set(sum, bi, h, i, j)
					}
				}
			}
		}
		return result
	} else {
		panic("matMul4D: unexpected shapeB")
	}
}

func softmax4D(x *tensor.Tensor, axis int) *tensor.Tensor {
	shape := x.Shape
	result := tensor.New(shape...)

	for b := 0; b < shape[0]; b++ {
		for h := 0; h < shape[1]; h++ {
			for i := 0; i < shape[2]; i++ {
				maxVal := math.Inf(-1)
				for j := 0; j < shape[3]; j++ {
					if x.Get(b, h, i, j) > maxVal {
						maxVal = x.Get(b, h, i, j)
					}
				}
				sum := 0.0
				for j := 0; j < shape[3]; j++ {
					sum += math.Exp(x.Get(b, h, i, j) - maxVal)
				}
				for j := 0; j < shape[3]; j++ {
					result.Set(math.Exp(x.Get(b, h, i, j)-maxVal)/sum, b, h, i, j)
				}
			}
		}
	}
	return result
}

func dropout4D(x *tensor.Tensor, p float64, train bool) *tensor.Tensor {
	if !train || p == 0 {
		return x.Clone()
	}
	result := x.Clone()
	for i := 0; i < x.Size(); i++ {
		if rand.Float64() < p {
			result.Data[i] = 0
		} else {
			result.Data[i] /= (1 - p)
		}
	}
	return result
}

func addMask(x, mask *tensor.Tensor) *tensor.Tensor {
	result := x.Clone()
	for b := 0; b < x.Shape[0]; b++ {
		for h := 0; h < x.Shape[1]; h++ {
			for i := 0; i < x.Shape[2]; i++ {
				for j := 0; j < x.Shape[3]; j++ {
					if mask.Get(b, i, j) == 0 {
						result.Set(math.Inf(-1), b, h, i, j)
					}
				}
			}
		}
	}
	return result
}

func reshape3D(x *tensor.Tensor, d0, d1, d2 int) *tensor.Tensor {
	return x.Reshape(d0, d1, d2)
}

func relu(x *tensor.Tensor) *tensor.Tensor {
	return tensor.ReLU(x)
}

func dropout(x *tensor.Tensor, p float64) *tensor.Tensor {
	result := x.Clone()
	for i := 0; i < x.Size(); i++ {
		if rand.Float64() < p {
			result.Data[i] = 0
		} else {
			result.Data[i] /= (1 - p)
		}
	}
	return result
}

func addBias(x, bias *tensor.Tensor) *tensor.Tensor {
	result := x.Clone()
	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < x.Shape[1]; j++ {
			val := result.Get(i, j) + bias.Get(j)
			result.Set(val, i, j)
		}
	}
	return result
}

func layerNorm(x, gamma, beta *tensor.Tensor) *tensor.Tensor {
	result := tensor.New(x.Shape...)
	eps := 1e-6

	if len(x.Shape) == 3 {
		batchSize := x.Shape[0]
		seqLen := x.Shape[1]
		featureDim := x.Shape[2]

		for b := 0; b < batchSize; b++ {
			for i := 0; i < seqLen; i++ {
				mean := 0.0
				for j := 0; j < featureDim; j++ {
					mean += x.Get(b, i, j)
				}
				mean /= float64(featureDim)

				variance := 0.0
				for j := 0; j < featureDim; j++ {
					diff := x.Get(b, i, j) - mean
					variance += diff * diff
				}
				variance /= float64(featureDim)

				std := math.Sqrt(variance + eps)
				for j := 0; j < featureDim; j++ {
					normalized := (x.Get(b, i, j) - mean) / std
					result.Set(normalized*gamma.Get(j)+beta.Get(j), b, i, j)
				}
			}
		}
	} else {
		for i := 0; i < x.Shape[0]; i++ {
			mean := 0.0
			for j := 0; j < x.Shape[1]; j++ {
				mean += x.Get(i, j)
			}
			mean /= float64(x.Shape[1])

			variance := 0.0
			for j := 0; j < x.Shape[1]; j++ {
				diff := x.Get(i, j) - mean
				variance += diff * diff
			}
			variance /= float64(x.Shape[1])

			std := math.Sqrt(variance + eps)
			for j := 0; j < x.Shape[1]; j++ {
				normalized := (x.Get(i, j) - mean) / std
				result.Set(normalized*gamma.Get(j)+beta.Get(j), i, j)
			}
		}
	}
	return result
}
