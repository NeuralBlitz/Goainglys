package bert

import (
	"math"

	"transformers/core"
)

type BERTConfig struct {
	VocabSize        int
	HiddenSize       int
	NumLayers        int
	NumHeads         int
	IntermediateSize int
	MaxPositionEmbed int
	TypeVocabSize    int
	HiddenAct        string
	Dropout          float32
	LayerNormEpsilon float32
}

func DefaultBERTConfig() *BERTConfig {
	return &BERTConfig{
		VocabSize:        30522,
		HiddenSize:       768,
		NumLayers:        12,
		NumHeads:         12,
		IntermediateSize: 3072,
		MaxPositionEmbed: 512,
		TypeVocabSize:    2,
		HiddenAct:        "gelu",
		Dropout:          0.1,
		LayerNormEpsilon: 1e-12,
	}
}

func BERTBaseConfig() *BERTConfig {
	return &BERTConfig{
		VocabSize:        30522,
		HiddenSize:       768,
		NumLayers:        12,
		NumHeads:         12,
		IntermediateSize: 3072,
		MaxPositionEmbed: 512,
		TypeVocabSize:    2,
		HiddenAct:        "gelu",
		Dropout:          0.1,
		LayerNormEpsilon: 1e-12,
	}
}

func BERTLargeConfig() *BERTConfig {
	return &BERTConfig{
		VocabSize:        30522,
		HiddenSize:       1024,
		NumLayers:        24,
		NumHeads:         16,
		IntermediateSize: 4096,
		MaxPositionEmbed: 512,
		TypeVocabSize:    2,
		HiddenAct:        "gelu",
		Dropout:          0.1,
		LayerNormEpsilon: 1e-12,
	}
}

type BERTModel struct {
	Config     *BERTConfig
	Embeddings *BERTEmbeddings
	Encoder    *BERTEncoder
	Pooler     *BERTPooler
}

func NewBERTModel(config *BERTConfig) *BERTModel {
	return &BERTModel{
		Embeddings: NewBERTEmbeddings(config),
		Encoder:    NewBERTEncoder(config),
		Pooler:     NewBERTPooler(config),
	}
}

func (m *BERTModel) Forward(inputIds, tokenTypeIds, attentionMask *core.Tensor) *BERTOutput {
	embeddingOutput := m.Embeddings.Forward(inputIds, tokenTypeIds)
	encoderOutput := m.Encoder.Forward(embeddingOutput, attentionMask)
	pooledOutput := m.Pooler.Forward(encoderOutput.LastHiddenState)

	return &BERTOutput{
		LastHiddenState: encoderOutput.LastHiddenState,
		PoolerOutput:    pooledOutput,
		HiddenStates:    encoderOutput.HiddenStates,
		Attentions:      encoderOutput.Attentions,
	}
}

type BERTOutput struct {
	LastHiddenState *core.Tensor
	PoolerOutput    *core.Tensor
	HiddenStates    []*core.Tensor
	Attentions      []*core.Tensor
}

type BERTEmbeddings struct {
	WordEmbedding      *Embedding
	PositionEmbedding  *Embedding
	TokenTypeEmbedding *Embedding
	LayerNorm          *LayerNorm
	Dropout            float32
}

func NewBERTEmbeddings(config *BERTConfig) *BERTEmbeddings {
	return &BERTEmbeddings{
		WordEmbedding:      NewEmbedding(config.VocabSize, config.HiddenSize),
		PositionEmbedding:  NewEmbedding(config.MaxPositionEmbed, config.HiddenSize),
		TokenTypeEmbedding: NewEmbedding(config.TypeVocabSize, config.HiddenSize),
		LayerNorm:          NewLayerNorm(config.HiddenSize, config.LayerNormEpsilon),
		Dropout:            config.Dropout,
	}
}

func (e *BERTEmbeddings) Forward(inputIds, tokenTypeIds *core.Tensor) *core.Tensor {
	batchSize := inputIds.Shape[0]
	seqLen := inputIds.Shape[1]

	wordEmbed := e.WordEmbedding.Forward(inputIds)

	positionIds := core.NewTensor(batchSize, seqLen)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			positionIds.Data[b*seqLen+i] = float32(i)
		}
	}
	positionEmbed := e.PositionEmbedding.Forward(positionIds)

	if tokenTypeIds != nil {
		tokenTypeEmbed := e.TokenTypeEmbedding.Forward(tokenTypeIds)
		wordEmbed = core.Add(core.Add(wordEmbed, positionEmbed), tokenTypeEmbed)
	} else {
		zeroTokenType := core.TensorZeros(batchSize, seqLen)
		tokenTypeEmbed := e.TokenTypeEmbedding.Forward(zeroTokenType)
		wordEmbed = core.Add(core.Add(wordEmbed, positionEmbed), tokenTypeEmbed)
	}

	output := e.LayerNorm.Forward(wordEmbed)
	return output
}

type BERTEncoder struct {
	Config *BERTConfig
	Layers []*BERTLayer
}

func NewBERTEncoder(config *BERTConfig) *BERTEncoder {
	layers := make([]*BERTLayer, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		layers[i] = NewBERTLayer(config)
	}
	return &BERTEncoder{Config: config, Layers: layers}
}

func (e *BERTEncoder) Forward(hiddenStates *core.Tensor, attentionMask *core.Tensor) *EncoderOutput {
	allHiddenStates := make([]*core.Tensor, e.Config.NumLayers+1)
	allAttentions := make([]*core.Tensor, e.Config.NumLayers)

	allHiddenStates[0] = hiddenStates

	for i, layer := range e.Layers {
		hiddenStates = layer.Forward(hiddenStates, attentionMask)
		allHiddenStates[i+1] = hiddenStates
	}

	return &EncoderOutput{
		LastHiddenState: hiddenStates,
		HiddenStates:    allHiddenStates,
		Attentions:      allAttentions,
	}
}

type EncoderOutput struct {
	LastHiddenState *core.Tensor
	HiddenStates    []*core.Tensor
	Attentions      []*core.Tensor
}

type BERTLayer struct {
	Attention    *BERTAttention
	LayerNorm1   *LayerNorm
	Intermediate *BERTIntermediate
	LayerNorm2   *LayerNorm
}

func NewBERTLayer(config *BERTConfig) *BERTLayer {
	return &BERTLayer{
		Attention:    NewBERTSelfAttention(config),
		LayerNorm1:   NewLayerNorm(config.HiddenSize, config.LayerNormEpsilon),
		Intermediate: NewBERTIntermediate(config),
		LayerNorm2:   NewLayerNorm(config.HiddenSize, config.LayerNormEpsilon),
	}
}

func (l *BERTLayer) Forward(hiddenStates, attentionMask *core.Tensor) *core.Tensor {
	selfAttention := l.Attention.Forward(hiddenStates, attentionMask)
	hiddenStates = core.Add(hiddenStates, selfAttention)
	hiddenStates = core.Add(hiddenStates, l.LayerNorm1.Forward(hiddenStates))

	intermediateOutput := l.Intermediate.Forward(hiddenStates)
	hiddenStates = core.Add(hiddenStates, intermediateOutput)
	hiddenStates = core.Add(hiddenStates, l.LayerNorm2.Forward(hiddenStates))

	return hiddenStates
}

type BERTAttention struct {
	Self   *BERTSelfAttention
	Output *BERTSelfOutput
}

func NewBERTSelfAttention(config *BERTConfig) *BERTAttention {
	return &BERTAttention{
		Self:   NewBERTSelfAttentionCore(config),
		Output: NewBERTSelfOutput(config),
	}
}

func (a *BERTAttention) Forward(hiddenStates, attentionMask *core.Tensor) *core.Tensor {
	selfOutput := a.Self.Forward(hiddenStates, attentionMask)
	return a.Output.Forward(selfOutput, hiddenStates)
}

type BERTSelfAttention struct {
	NumHeads int
	HeadDim  int
	Query    *Linear
	Key      *Linear
	Value    *Linear
	Dropout  float32
	Scale    float32
}

func NewBERTSelfAttentionCore(config *BERTConfig) *BERTSelfAttention {
	headDim := config.HiddenSize / config.NumHeads
	return &BERTSelfAttention{
		NumHeads: config.NumHeads,
		HeadDim:  headDim,
		Query:    NewLinear(config.HiddenSize, config.HiddenSize),
		Key:      NewLinear(config.HiddenSize, config.HiddenSize),
		Value:    NewLinear(config.HiddenSize, config.HiddenSize),
		Dropout:  config.Dropout,
		Scale:    float32(1.0 / math.Sqrt(float64(headDim))),
	}
}

func (a *BERTSelfAttention) Forward(hiddenStates, attentionMask *core.Tensor) *core.Tensor {
	batchSize := hiddenStates.Shape[0]
	seqLen := hiddenStates.Shape[1]

	queryLayer := core.Matmul(hiddenStates, a.Query.Weight)
	keyLayer := core.Matmul(hiddenStates, a.Key.Weight)
	valueLayer := core.Matmul(hiddenStates, a.Value.Weight)

	queryLayer = a.ReshapeToHeads(queryLayer)
	keyLayer = a.ReshapeToHeads(keyLayer)
	valueLayer = a.ReshapeToHeads(valueLayer)

	attentionScores := core.NewTensor(batchSize, a.NumHeads, seqLen, seqLen)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < a.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < seqLen; j++ {
					sum := float32(0)
					for d := 0; d < a.HeadDim; d++ {
						qIdx := ((b*a.NumHeads+h)*seqLen+i)*a.HeadDim + d
						kIdx := ((b*a.NumHeads+h)*seqLen+j)*a.HeadDim + d
						sum += queryLayer.Data[qIdx] * keyLayer.Data[kIdx]
					}
					attentionScores.Data[(((b*a.NumHeads+h)*seqLen+i)*seqLen + j)] = sum * a.Scale
				}
			}
		}
	}

	if attentionMask != nil {
		for i := 0; i < attentionScores.Numel(); i++ {
			if attentionMask.Data[i] == 0 {
				attentionScores.Data[i] = float32(-1e9)
			}
		}
	}

	attention_probs := core.Softmax(attentionScores, 3)

	contextLayer := core.NewTensor(batchSize, a.NumHeads, seqLen, a.HeadDim)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < a.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for d := 0; d < a.HeadDim; d++ {
					sum := float32(0)
					for j := 0; j < seqLen; j++ {
						aIdx := ((b*a.NumHeads+h)*seqLen+i)*seqLen + j
						vIdx := ((b*a.NumHeads+h)*seqLen+j)*a.HeadDim + d
						sum += attention_probs.Data[aIdx] * valueLayer.Data[vIdx]
					}
					contextLayer.Data[(((b*a.NumHeads+h)*seqLen+i)*a.HeadDim + d)] = sum
				}
			}
		}
	}

	return a.ReshapeFromHeads(contextLayer)
}

func (a *BERTSelfAttention) ReshapeToHeads(t *core.Tensor) *core.Tensor {
	batchSize := t.Shape[0]
	seqLen := t.Shape[1]
	hiddenSize := t.Shape[2]

	out := core.NewTensor(batchSize, a.NumHeads, seqLen, a.HeadDim)
	for b := 0; b < batchSize; b++ {
		for h := 0; h < a.NumHeads; h++ {
			for i := 0; i < seqLen; i++ {
				for d := 0; d < a.HeadDim; d++ {
					srcIdx := (b*seqLen+i)*hiddenSize + h*a.HeadDim + d
					dstIdx := (((b*a.NumHeads+h)*seqLen+i)*a.HeadDim + d)
					out.Data[dstIdx] = t.Data[srcIdx]
				}
			}
		}
	}
	return out
}

func (a *BERTSelfAttention) ReshapeFromHeads(t *core.Tensor) *core.Tensor {
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

type BERTSelfOutput struct {
	Dense     *Linear
	LayerNorm *LayerNorm
}

func NewBERTSelfOutput(config *BERTConfig) *BERTSelfOutput {
	return &BERTSelfOutput{
		Dense:     NewLinear(config.HiddenSize, config.HiddenSize),
		LayerNorm: NewLayerNorm(config.HiddenSize, config.LayerNormEpsilon),
	}
}

func (o *BERTSelfOutput) Forward(hiddenStates, inputTensor *core.Tensor) *core.Tensor {
	hiddenStates = o.Dense.Forward(hiddenStates)
	hiddenStates = core.Add(hiddenStates, inputTensor)
	return o.LayerNorm.Forward(hiddenStates)
}

type BERTIntermediate struct {
	Dense *Linear
	Act   func(*core.Tensor) *core.Tensor
}

func NewBERTIntermediate(config *BERTConfig) *BERTIntermediate {
	intermediate := &BERTIntermediate{
		Dense: NewLinear(config.HiddenSize, config.IntermediateSize),
	}
	intermediate.Act = Gelu
	return intermediate
}

func (i *BERTIntermediate) Forward(hiddenStates *core.Tensor) *core.Tensor {
	intermediate := i.Dense.Forward(hiddenStates)
	return i.Act(intermediate)
}

type BERTPooler struct {
	Dense *Linear
	Act   func(*core.Tensor) *core.Tensor
}

func NewBERTPooler(config *BERTConfig) *BERTPooler {
	return &BERTPooler{
		Dense: NewLinear(config.HiddenSize, config.HiddenSize),
		Act:   Tanh,
	}
}

func (p *BERTPooler) Forward(hiddenStates *core.Tensor) *core.Tensor {
	firstTokenTensor := hiddenStates.View(hiddenStates.Shape[0], 1, hiddenStates.Shape[2])
	pooled := p.Dense.Forward(firstTokenTensor)
	return p.Act(pooled)
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
	if len(x.Shape) == 3 {
		batchSize := x.Shape[0]
		seqLen := x.Shape[1]
		normShape := len(ln.Weight.Data)

		out := core.NewTensor(batchSize, seqLen, normShape)
		for b := 0; b < batchSize; b++ {
			for i := 0; i < seqLen; i++ {
				var mean float32
				for j := 0; j < normShape; j++ {
					mean += x.Data[(b*seqLen+i)*normShape+j]
				}
				mean /= float32(normShape)

				var variance float32
				for j := 0; j < normShape; j++ {
					diff := x.Data[(b*seqLen+i)*normShape+j] - mean
					variance += diff * diff
				}
				variance /= float32(normShape)

				norm := float32(math.Sqrt(float64(variance) + float64(ln.Eps)))
				for j := 0; j < normShape; j++ {
					idx := (b*seqLen+i)*normShape + j
					out.Data[idx] = (x.Data[idx]-mean)/norm*ln.Weight.Data[j] + ln.Bias.Data[j]
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
	if len(x.Shape) == 3 {
		batchSize := x.Shape[0]
		seqLen := x.Shape[1]
		inFeatures := x.Shape[2]
		outFeatures := l.Bias.Shape[0]

		out := core.NewTensor(batchSize, seqLen, outFeatures)
		for b := 0; b < batchSize; b++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < outFeatures; j++ {
					sum := float32(0)
					for k := 0; k < inFeatures; k++ {
						sum += x.Data[(b*seqLen+i)*inFeatures+k] * l.Weight.Data[k*outFeatures+j]
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

func Tanh(x *core.Tensor) *core.Tensor {
	out := core.NewTensor(x.Shape...)
	for i, v := range x.Data {
		out.Data[i] = float32(math.Tanh(float64(v)))
	}
	return out
}
