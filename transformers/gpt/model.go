package gpt

import (
	"math"
	"math/rand"

	"transformers/core"
)

type GPTConfig struct {
	VocabSize        int
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	IntermediateSize int
	MaxPositionEmbed int
	Dropout          float32
	LayerNormEpsilon float32
}

func DefaultGPTConfig() *GPTConfig {
	return &GPTConfig{
		VocabSize:        50257,
		HiddenSize:       768,
		NumLayers:        12,
		NumHeads:         12,
		IntermediateSize: 3072,
		MaxPositionEmbed: 1024,
		Dropout:          0.1,
		LayerNormEpsilon: 1e-12,
	}
}

func GPT2Config() *GPTConfig {
	return &GPTConfig{
		VocabSize:        50257,
		HiddenSize:       768,
		NumLayers:        12,
		NumHeads:         12,
		IntermediateSize: 3072,
		MaxPositionEmbed: 1024,
		Dropout:          0.1,
		LayerNormEpsilon: 1e-5,
	}
}

func GPT2MediumConfig() *GPTConfig {
	return &GPTConfig{
		VocabSize:        50257,
		HiddenSize:       1024,
		NumLayers:        24,
		NumHeads:         16,
		IntermediateSize: 4096,
		MaxPositionEmbed: 1024,
		Dropout:          0.1,
		LayerNormEpsilon: 1e-5,
	}
}

func GPTNeoConfig() *GPTConfig {
	return &GPTConfig{
		VocabSize:        50257,
		HiddenSize:       2048,
		NumLayers:        32,
		NumHeads:         16,
		IntermediateSize: 8192,
		MaxPositionEmbed: 2048,
		Dropout:          0.1,
		LayerNormEpsilon: 1e-5,
	}
}

type GPTModel struct {
	Config  *GPTConfig
	Wte     *Embedding
	Wpe     *PositionalEncoding
	Dropout float32
	Layers  []*TransformerBlock
	Fnorm   *LayerNorm
	LmHead  *Linear
}

func NewGPTModel(config *GPTConfig) *GPTModel {
	layers := make([]*TransformerBlock, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		layers[i] = NewTransformerBlock(config)
	}

	return &GPTModel{
		Config:  config,
		Wte:     NewEmbedding(config.VocabSize, config.HiddenSize),
		Wpe:     NewPositionalEncoding(float32(config.HiddenSize), float32(config.MaxPositionEmbed), config.Dropout),
		Dropout: config.Dropout,
		Layers:  layers,
		Fnorm:   NewLayerNorm(config.HiddenSize, config.LayerNormEpsilon),
		LmHead:  NewLinear(config.HiddenSize, config.VocabSize),
	}
}

func (m *GPTModel) Forward(inputIds *core.Tensor, attentionMask *core.Tensor) *core.Tensor {
	_ = inputIds.Shape[1]
	hiddenStates := m.Wte.Forward(inputIds)
	hiddenStates = m.Wpe.Forward(hiddenStates)

	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(hiddenStates)
	}

	hiddenStates = m.Fnorm.Forward(hiddenStates)
	logits := m.LmHead.Forward(hiddenStates)

	return logits
}

func (m *GPTModel) Generate(inputIds *core.Tensor, maxNewTokens, temperature float32, topK int) *core.Tensor {
	generated := make([]int, 0)
	for _, id := range inputIds.Data {
		generated = append(generated, int(id))
	}

	for i := 0; i < int(maxNewTokens); i++ {
		input := core.NewTensorWithData(intToFloat32(generated), []int{1, len(generated)})
		logits := m.Forward(input, nil)

		_ = temperature
		_ = logits

		probs := core.Softmax(logits, 2)

		nextToken := sampleTopK(probs, topK)
		generated = append(generated, nextToken)
	}

	return core.NewTensorWithData(intToFloat32(generated), []int{1, len(generated)})
}

func intToFloat32(ints []int) []float32 {
	floats := make([]float32, len(ints))
	for i, v := range ints {
		floats[i] = float32(v)
	}
	return floats
}

func sampleTopK(probs *core.Tensor, topK int) int {
	seqLen := len(probs.Data)
	lastTokenProbs := make([]float32, seqLen)
	copy(lastTokenProbs, probs.Data[seqLen-probs.Shape[1]:])

	k := topK
	if k <= 0 || k >= seqLen {
		k = seqLen
	}

	topKIndices := make([]int, k)
	for i := 0; i < k; i++ {
		topKIndices[i] = i
	}

	for i := 1; i < k; i++ {
		for j := i; j > 0 && lastTokenProbs[j] > lastTokenProbs[j-1]; j-- {
			lastTokenProbs[j], lastTokenProbs[j-1] = lastTokenProbs[j-1], lastTokenProbs[j]
			topKIndices[j], topKIndices[j-1] = topKIndices[j-1], topKIndices[j]
		}
	}

	total := float32(0)
	for i := 0; i < k; i++ {
		total += lastTokenProbs[i]
	}

	r := float32(rand.Float64()) * total
	cumsum := float32(0)
	for i := 0; i < k; i++ {
		cumsum += lastTokenProbs[i]
		if r < cumsum {
			return topKIndices[i]
		}
	}

	return topKIndices[k-1]
}

func (m *GPTModel) NumParameters() int {
	params := m.Wte.NumParameters()
	for _, layer := range m.Layers {
		params += layer.NumParameters()
	}
	params += m.Fnorm.NumParameters() + m.LmHead.NumParameters()
	return params
}

type TransformerBlock struct {
	Attention  *MultiHeadAttention
	LayerNorm1 *LayerNorm
	MLP        *MLP
	LayerNorm2 *LayerNorm
}

func NewTransformerBlock(config *GPTConfig) *TransformerBlock {
	return &TransformerBlock{
		Attention:  NewMultiHeadAttention(config.HiddenSize, config.NumHeads, config.Dropout),
		LayerNorm1: NewLayerNorm(config.HiddenSize, config.LayerNormEpsilon),
		MLP:        NewMLP(config.HiddenSize, config.IntermediateSize, config.Dropout),
		LayerNorm2: NewLayerNorm(config.HiddenSize, config.LayerNormEpsilon),
	}
}

func (b *TransformerBlock) Forward(hiddenStates *core.Tensor) *core.Tensor {
	residual := hiddenStates
	hiddenStates = b.LayerNorm1.Forward(hiddenStates)
	attnOutput := b.Attention.Forward(hiddenStates, hiddenStates, hiddenStates, nil)
	hiddenStates = core.Add(residual, attnOutput)

	residual = hiddenStates
	hiddenStates = b.LayerNorm2.Forward(hiddenStates)
	mlpOutput := b.MLP.Forward(hiddenStates)
	hiddenStates = core.Add(residual, mlpOutput)

	return hiddenStates
}

func (b *TransformerBlock) NumParameters() int {
	return b.Attention.NumParameters() + b.LayerNorm1.NumParameters() +
		b.LayerNorm2.NumParameters()
}

type MultiHeadAttention struct {
	HiddenSize int
	NumHeads   int
	HeadDim    int
	Wq         *Linear
	Wk         *Linear
	Wv         *Linear
	Wo         *Linear
	Dropout    float32
	Scale      float32
	CausalMask *core.Tensor
}

func NewMultiHeadAttention(hiddenSize, numHeads int, dropout float32) *MultiHeadAttention {
	headDim := hiddenSize / numHeads
	return &MultiHeadAttention{
		HiddenSize: hiddenSize,
		NumHeads:   numHeads,
		HeadDim:    headDim,
		Wq:         NewLinear(hiddenSize, hiddenSize),
		Wk:         NewLinear(hiddenSize, hiddenSize),
		Wv:         NewLinear(hiddenSize, hiddenSize),
		Wo:         NewLinear(hiddenSize, hiddenSize),
		Dropout:    dropout,
		Scale:      float32(1.0 / math.Sqrt(float64(headDim))),
	}
}

func (m *MultiHeadAttention) Forward(query, key, value *core.Tensor, mask *core.Tensor) *core.Tensor {
	batchSize := query.Shape[0]
	seqLen := query.Shape[1]

	q := core.Matmul(query, m.Wq.Weight)
	k := core.Matmul(key, m.Wk.Weight)
	v := core.Matmul(value, m.Wv.Weight)

	q = m.ReshapeToHeads(q)
	k = m.ReshapeToHeads(k)
	v = m.ReshapeToHeads(v)

	scores := core.NewTensor(batchSize, m.NumHeads, seqLen, seqLen)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < m.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					sum := float32(0)
					for d := 0; d < m.HeadDim; d++ {
						qIdx := ((b*m.NumHeads+h)*seqLen+i)*m.HeadDim + d
						kIdx := ((b*m.NumHeads+h)*seqLen+j)*m.HeadDim + d
						sum += q.Data[qIdx] * k.Data[kIdx]
					}
					scores.Data[(((b*m.NumHeads+h)*seqLen+i)*seqLen + j)] = sum * m.Scale
				}
			}
		}
	}

	if mask != nil {
		for i := 0; i < scores.Numel(); i++ {
			if mask.Data[i] == 0 {
				scores.Data[i] = float32(-1e9)
			}
		}
	}

	attnWeights := core.Softmax(scores, 3)

	context := core.NewTensor(batchSize, m.NumHeads, seqLen, m.HeadDim)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < m.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for d := 0; d < m.HeadDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						aIdx := ((b*m.NumHeads+h)*seqLen+i)*seqLen + j
						vIdx := ((b*m.NumHeads+h)*seqLen+j)*m.HeadDim + d
						sum += attnWeights.Data[aIdx] * v.Data[vIdx]
					}
					context.Data[(((b*m.NumHeads+h)*seqLen+i)*m.HeadDim + d)] = sum
				}
			}
		}
	}

	context = m.ReshapeFromHeads(context)

	output := core.Matmul(context, m.Wo.Weight)

	return output
}

func (m *MultiHeadAttention) ReshapeToHeads(t *core.Tensor) *core.Tensor {
	batchSize := t.Shape[0]
	seqLen := t.Shape[1]
	hiddenSize := t.Shape[2]

	out := core.NewTensor(batchSize, m.NumHeads, seqLen, m.HeadDim)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < m.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for d := 0; d < m.HeadDim; d++ {
					srcIdx := (b*seqLen+i)*hiddenSize + h*m.HeadDim + d
					dstIdx := (((b*m.NumHeads+h)*seqLen+i)*m.HeadDim + d)
					out.Data[dstIdx] = t.Data[srcIdx]
				}
			}
		}
	}
	return out
}

func (m *MultiHeadAttention) ReshapeFromHeads(t *core.Tensor) *core.Tensor {
	batchSize := t.Shape[0]
	numHeads := t.Shape[1]
	seqLen := t.Shape[2]
	headDim := t.Shape[3]
	hiddenSize := numHeads * headDim

	out := core.NewTensor(batchSize, seqLen, hiddenSize)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for h := 0; h < numHeads; h++ {
				for d := 0; d < headDim; d++ {
					srcIdx := (((b*numHeads+h)*seqLen+i)*headDim + d)
					dstIdx := (b*seqLen+i)*hiddenSize + h*headDim + d
					out.Data[dstIdx] = t.Data[srcIdx]
				}
			}
		}
	}
	return out
}

func (m *MultiHeadAttention) NumParameters() int {
	return m.Wq.NumParameters() + m.Wk.NumParameters() +
		m.Wv.NumParameters() + m.Wo.NumParameters()
}

type MLP struct {
	C_proj *Linear
	Act    func(*core.Tensor) *core.Tensor
}

func NewMLP(hiddenSize, intermediateSize int, dropout float32) *MLP {
	mlp := &MLP{
		C_proj: NewLinear(intermediateSize, hiddenSize),
	}
	mlp.Act = Gelu
	return mlp
}

func (m *MLP) Forward(x *core.Tensor) *core.Tensor {
	return x
}

func (m *MLP) NumParameters() int {
	return m.C_proj.NumParameters()
}

type LayerNorm struct {
	Weight *core.Tensor
	Bias   *core.Tensor
	Eps    float32
}

func NewLayerNorm(normalizedShape int, eps float32) *LayerNorm {
	return &LayerNorm{
		Weight: core.TensorRandUniformSeeded(42, -0.02, 0.02, normalizedShape),
		Bias:   core.TensorZeros(normalizedShape),
		Eps:    eps,
	}
}

func (ln *LayerNorm) Forward(x *core.Tensor) *core.Tensor {
	if len(x.Shape) == 2 {
		batchSize := x.Shape[0]
		seqLen := x.Shape[1]
		normShape := len(ln.Weight.Data)

		out := core.NewTensor(batchSize, seqLen)
		for b := 0; b < batchSize; b++ {
			var mean float32
			for i := 0; i < seqLen; i++ {
				mean += x.Data[b*seqLen+i]
			}
			mean /= float32(seqLen)

			var variance float32
			for i := 0; i < seqLen; i++ {
				diff := x.Data[b*seqLen+i] - mean
				variance += diff * diff
			}
			variance /= float32(seqLen)

			norm := float32(math.Sqrt(float64(variance) + float64(ln.Eps)))
			for i := 0; i < seqLen; i++ {
				idx := b*seqLen + i
				if i < normShape {
					out.Data[idx] = (x.Data[idx]-mean)/norm*ln.Weight.Data[i] + ln.Bias.Data[i]
				} else {
					out.Data[idx] = x.Data[idx]
				}
			}
		}
		return out
	}
	return x.Copy()
}

func (ln *LayerNorm) NumParameters() int {
	return len(ln.Weight.Data) + len(ln.Bias.Data)
}

type Embedding struct {
	Weight *core.Tensor
}

func NewEmbedding(numEmbeddings, embeddingDim int) *Embedding {
	return &Embedding{
		Weight: core.TensorRandUniformSeeded(42, -0.02, 0.02, numEmbeddings, embeddingDim),
	}
}

func (e *Embedding) Forward(input *core.Tensor) *core.Tensor {
	batchSize := input.Shape[0]
	seqLen := input.Shape[1]
	embDim := e.Weight.Shape[1]

	out := core.NewTensor(batchSize, seqLen, embDim)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			idx := int(input.Data[b*seqLen+i])
			if idx >= 0 && idx < e.Weight.Shape[0] {
				for j := 0; j < embDim; j++ {
					out.Data[(b*seqLen+i)*embDim+j] = e.Weight.Data[idx*embDim+j]
				}
			}
		}
	}
	return out
}

func (e *Embedding) NumParameters() int {
	return e.Weight.Numel()
}

type Linear struct {
	Weight *core.Tensor
	Bias   *core.Tensor
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	return &Linear{
		Weight: core.TensorRandUniformSeeded(42, -0.02, 0.02, inFeatures, outFeatures),
		Bias:   core.TensorZeros(outFeatures),
	}
}

func (l *Linear) Forward(x *core.Tensor) *core.Tensor {
	if len(x.Shape) == 2 {
		batchSize := x.Shape[0]
		seqLen := x.Shape[1]
		outFeatures := l.Bias.Shape[0]

		out := core.NewTensor(batchSize, seqLen, outFeatures)
		for b := 0; b < batchSize; b++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < outFeatures; j++ {
					sum := float32(0)
					for k := 0; k < x.Shape[2]; k++ {
						sum += x.Data[(b*seqLen+i)*x.Shape[2]+k] * l.Weight.Data[k*outFeatures+j]
					}
					out.Data[(b*seqLen+i)*outFeatures+j] = sum + l.Bias.Data[j]
				}
			}
		}
		return out
	}
	if len(x.Shape) == 3 {
		batchSize := x.Shape[0]
		seqLen := x.Shape[1]
		hiddenSize := x.Shape[2]
		outFeatures := l.Bias.Shape[0]

		out := core.NewTensor(batchSize, seqLen, outFeatures)
		for b := 0; b < batchSize; b++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < outFeatures; j++ {
					sum := float32(0)
					for k := 0; k < hiddenSize; k++ {
						sum += x.Data[(b*seqLen+i)*hiddenSize+k] * l.Weight.Data[k*outFeatures+j]
					}
					out.Data[(b*seqLen+i)*outFeatures+j] = sum + l.Bias.Data[j]
				}
			}
		}
		return out
	}
	return x.Copy()
}

func (l *Linear) NumParameters() int {
	return l.Weight.Numel() + l.Bias.Numel()
}

func Gelu(x *core.Tensor) *core.Tensor {
	out := core.NewTensor(x.Shape...)
	sqrt2pi := float32(math.Sqrt(2 / math.Pi))
	c1 := float32(0.044715)
	for i, v := range x.Data {
		v3 := v * v * v
		t := float32(math.Tanh(float64(sqrt2pi * (v + c1*v3))))
		out.Data[i] = 0.5 * v * (1 + t)
	}
	return out
}
