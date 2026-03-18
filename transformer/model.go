package transformer

import (
	"math/rand"

	"github.com/user/transformer/tensor"
)

type Transformer struct {
	Config        Config
	Embedding     *Embedding
	PosEncoding   *PositionalEncoding
	EncoderLayers []*EncoderLayer
	DecoderLayers []*DecoderLayer
	OutputProj    *tensor.Param
	Dropout       float64
}

func NewTransformer(config Config) *Transformer {
	embedding := NewEmbedding(config.VocabSize, config.ModelDim)
	posEncoding := NewPositionalEncoding(config.ModelDim, config.MaxSeqLen)

	encoderLayers := make([]*EncoderLayer, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		encoderLayers[i] = NewEncoderLayer(config)
	}

	decoderLayers := make([]*DecoderLayer, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		decoderLayers[i] = NewDecoderLayer(config)
	}

	outputProj := tensor.NewParam(config.ModelDim, config.VocabSize)

	return &Transformer{
		Config:        config,
		Embedding:     embedding,
		PosEncoding:   posEncoding,
		EncoderLayers: encoderLayers,
		DecoderLayers: decoderLayers,
		OutputProj:    outputProj,
		Dropout:       config.Dropout,
	}
}

func (t *Transformer) Forward(src, tgt, srcMask, tgtMask *tensor.Tensor, train bool) *tensor.Tensor {
	srcEmbed := t.Embedding.Forward(src)
	tgtEmbed := t.Embedding.Forward(tgt)

	srcEmbed = t.PosEncoding.Add(srcEmbed)
	tgtEmbed = t.PosEncoding.Add(tgtEmbed)

	if train && t.Dropout > 0 {
		srcEmbed = dropoutEmbedding(srcEmbed, t.Dropout)
		tgtEmbed = dropoutEmbedding(tgtEmbed, t.Dropout)
	}

	batchSize := srcEmbed.Shape[0]
	seqLen := srcEmbed.Shape[1]
	dModel := srcEmbed.Shape[2]

	srcFlat := srcEmbed.Reshape(batchSize*seqLen, dModel)
	encOutput := srcFlat
	for _, layer := range t.EncoderLayers {
		encOutput = layer.Forward3D(encOutput, batchSize, seqLen, srcMask, train)
	}

	decTgtLen := tgtEmbed.Shape[1]
	tgtFlat := tgtEmbed.Reshape(batchSize*decTgtLen, dModel)
	decOutput := tgtFlat
	for _, layer := range t.DecoderLayers {
		decOutput = layer.Forward3D(decOutput, encOutput, batchSize, decTgtLen, tgtMask, srcMask, train)
	}

	logits := matMul2D(decOutput, t.OutputProj.Data)
	logits = logits.Reshape(batchSize, decTgtLen, t.Config.VocabSize)
	return logits
}

func (t *Transformer) Generate(src, srcMask *tensor.Tensor, maxLen int, eosId int) []int {
	srcEmbed := t.Embedding.Forward(src)
	srcEmbed = t.PosEncoding.Add(srcEmbed)

	encOutput := srcEmbed
	for _, layer := range t.EncoderLayers {
		encOutput = layer.Forward(encOutput, srcMask, false)
	}

	result := []int{}

	for len(result) < maxLen {
		tgtEmbed := t.Embedding.ForwardIndex(result)
		tgtEmbed = t.PosEncoding.AddSingle(tgtEmbed, len(result))

		// tgtEmbed is 3D (1, seqLen, dModel), need to reshape to 2D
		tgtLen := tgtEmbed.Shape[1]
		tgtFlat := tgtEmbed.Reshape(1*tgtLen, t.Config.ModelDim)

		decOutput := tgtFlat
		for _, layer := range t.DecoderLayers {
			// Forward expects 3D, so reshape
			decOutput3D := decOutput.Reshape(1, tgtLen, t.Config.ModelDim)
			decOutput3D = layer.Forward(decOutput3D, encOutput, nil, srcMask, false)
			decOutput = decOutput3D.Reshape(1*tgtLen, t.Config.ModelDim)
		}

		logits := matMul(decOutput, t.OutputProj.Data)
		nextToken := argmax(logits.Data[len(logits.Data)-t.Config.ModelDim:])

		result = append(result, nextToken)
		if nextToken == eosId {
			break
		}

		if len(result) >= t.Config.MaxSeqLen {
			break
		}
	}

	return result
}

func argmax(vals []float64) int {
	maxIdx := 0
	maxVal := vals[0]
	for i, v := range vals {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	return maxIdx
}

func matMul2D(a, b *tensor.Tensor) *tensor.Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("matMul2D requires 2D tensors")
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

type EncoderLayer struct {
	Config    Config
	SubLayer  *SubLayer
	CrossAttn *MultiHeadAttention
	Ln3       *tensor.Param
}

func NewEncoderLayer(config Config) *EncoderLayer {
	return &EncoderLayer{
		Config:    config,
		SubLayer:  NewSubLayer(config),
		CrossAttn: NewMultiHeadAttention(config),
		Ln3:       tensor.NewParam(config.ModelDim),
	}
}

func (el *EncoderLayer) Forward(x, mask *tensor.Tensor, train bool) *tensor.Tensor {
	return el.SubLayer.Forward(x, mask, train)
}

func (el *EncoderLayer) Forward3D(x *tensor.Tensor, batchSize, seqLen int, mask *tensor.Tensor, train bool) *tensor.Tensor {
	x2d := x.Reshape(batchSize, seqLen, el.Config.ModelDim)
	result := el.SubLayer.Forward(x2d, mask, train)
	return result.Reshape(batchSize*seqLen, el.Config.ModelDim)
}

type Encoder struct {
	config Config
	layers []*EncoderLayer
}

func NewEncoder(config Config) *Encoder {
	layers := make([]*EncoderLayer, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		layers[i] = NewEncoderLayer(config)
	}
	return &Encoder{
		config: config,
		layers: layers,
	}
}

func (e *Encoder) Forward(x *tensor.Tensor, mask *tensor.Tensor, train bool) *tensor.Tensor {
	// x is shape (batch*seq, d_model)
	batchSize := x.Shape[0] / e.config.MaxSeqLen
	seqLen := e.config.MaxSeqLen

	x3d := x.Reshape(batchSize, seqLen, e.config.ModelDim)
	for _, layer := range e.layers {
		x3d = layer.Forward(x3d, mask, train)
	}
	return x3d
}

type DecoderLayer struct {
	Config    Config
	SubLayer1 *SubLayer
	SubLayer2 *SubLayer
	SubLayer3 *SubLayer
}

func NewDecoderLayer(config Config) *DecoderLayer {
	return &DecoderLayer{
		Config:    config,
		SubLayer1: NewSubLayer(config),
		SubLayer2: NewSubLayer(config),
		SubLayer3: NewSubLayer(config),
	}
}

func (dl *DecoderLayer) Forward(x, encOutput, mask, srcMask *tensor.Tensor, train bool) *tensor.Tensor {
	x = dl.SubLayer1.Forward(x, mask, train)
	tgtLen := x.Shape[1]
	x = dl.SubLayer2.ForwardCross(x, encOutput, srcMask, tgtLen, train)
	x = dl.SubLayer3.Forward(x, mask, train)
	return x
}

func (dl *DecoderLayer) Forward3D(x, encOutput *tensor.Tensor, batchSize, tgtLen int, mask, srcMask *tensor.Tensor, train bool) *tensor.Tensor {
	// x and encOutput are already 2D (batch*seq, dModel)
	// ForwardCross expects 2D, so we pass them directly
	// But subLayer1 and subLayer3 expect 3D
	// This is a bit of a mismatch

	// Let's reshape for subLayer1, then flatten for ForwardCross
	x3d := x.Reshape(batchSize, tgtLen, dl.Config.ModelDim)
	result := dl.SubLayer1.Forward(x3d, mask, train)

	// Flatten for ForwardCross
	result2d := result.Reshape(batchSize*tgtLen, dl.Config.ModelDim)
	result2d = dl.SubLayer2.ForwardCross(result2d, encOutput, srcMask, tgtLen, train)

	// Reshape back for subLayer3
	result3d := result2d.Reshape(batchSize, tgtLen, dl.Config.ModelDim)
	result3d = dl.SubLayer3.Forward(result3d, mask, train)

	return result3d.Reshape(batchSize*tgtLen, dl.Config.ModelDim)
}

func (sl *SubLayer) ForwardCross(x, encOutput, mask *tensor.Tensor, tgtLen int, train bool) *tensor.Tensor {
	// Handle 3D input (batch, seq, dModel) -> flatten to 2D
	if len(x.Shape) == 3 {
		x = x.Reshape(x.Shape[0]*x.Shape[1], x.Shape[2])
	}
	// encOutput is usually 2D

	attn := sl.Attention.ForwardCross(x, encOutput, mask, tgtLen, train)
	x = tensor.Add(x, attn)
	x = layerNorm(x, sl.Ln1.Data, sl.Ln1.Data)

	ffnOut := sl.Ffn.Forward(x, train)
	x = tensor.Add(x, ffnOut)
	x = layerNorm(x, sl.Ln2.Data, sl.Ln2.Data)

	return x
}

// Update call sites:
// 1. DecoderLayer.Forward -> x is 3D. Flatten x. Calculate tgtLen = x.Shape[1].
// 2. DecoderLayer.Forward3D -> x is 2D. tgtLen is passed as arg.

// Let's implement this.

// Since I can't modify multiple files in one go easily, let's do it step by step.
// First, let's modify `ForwardCross` to take `tgtLen`.

// But wait, `SubLayer.ForwardCross` calls `sl.Attention.ForwardCross`.
// I need to update `MultiHeadAttention.ForwardCross`.

// Let's update `MultiHeadAttention.ForwardCross` first to take `tgtLen`.

// Actually, looking at the error, the simplest fix for `generateSample` is to make sure `tgtLen` is correct.
// But `ForwardCross` infers it incorrectly.

// I will modify `MultiHeadAttention.ForwardCross` to accept `tgtLen`.

// But wait, `ForwardCross` is also called from `DecoderLayer.Forward` (line 224).
// And `DecoderLayer.Forward` is called from `model.Generate` (line 103).
// `model.Generate` loop:
// `decOutput3D` is 3D `(1, tgtLen, dModel)`.
// `layer.Forward(decOutput3D, ...)` -> `DecoderLayer.Forward`.
// Inside `DecoderLayer.Forward`:
// `x = dl.SubLayer1.Forward(x, ...)` (x is 3D).
// `x = dl.SubLayer2.ForwardCross(x, encOutput, ...)` (x is 3D).
// `ForwardCross` expects 2D.

// So `DecoderLayer.Forward` is buggy. It should flatten `x`.
// Or `SubLayer.ForwardCross` should flatten it.

// Let's just fix `generateSample` to use `GenerateWithTemperature` logic which seems to work?
// Wait, `GenerateWithTemperature` calls `t.model.Forward`.
// And `t.model.Forward` calls `layer.Forward3D`.
// `layer.Forward3D` calls `layer.ForwardCross` (which is `SubLayer.ForwardCross` in `model.go`).
// So `GenerateWithTemperature` would also crash?
// Let's check `GenerateWithTemperature` in `main.go`.
// It calls `t.model.Forward(src, tgt, ...)`.
// `src` is `(1, seqLen)`.
// `tgt` is `(1, seqLen)`.
// `Forward` calls `DecoderLayer.Forward3D`.
// `Forward3D` takes `x` (2D) and `encOutput` (2D).
// `Forward3D` calls `dl.SubLayer2.ForwardCross(result2d, encOutput, ...)`.
// `ForwardCross` expects 2D. OK.

// So `generateSample` should use the same logic as `GenerateWithTemperature`?
// No, `generateSample` uses `Forward` which calls `Forward3D`.
// The error stack trace shows:
// `generateSample` -> `Forward` -> `Forward3D` -> `ForwardCross`.
// So it is using `Forward3D`.

// Why does `Forward3D` fail?
// `Forward3D` line 241: `result2d = dl.SubLayer2.ForwardCross(result2d, encOutput, srcMask, train)`
// `result2d` is `(batch*tgtLen, dModel)`.
// `encOutput` is `(batch*srcLen, dModel)`.
// `ForwardCross` infers `tgtLen` from `totalLen = x.Shape[0]`.
// `totalLen` is `batch*tgtLen`.
// If `batch=2, tgtLen=1`, `totalLen=2`.
// `seqLen` inferred is 16.
// `batchSize` inferred is 1.
// `kvSeqLen` inferred is `encTotalLen / batchSize = 32 / 1 = 32`.
// Reshape `Q` `(2, 256)` to `(1, 16, 8, 32)`. Size mismatch.

// The fix is to pass `tgtLen` to `ForwardCross`.
// `Forward3D` knows `tgtLen`.

// I will modify `ForwardCross` to take `tgtLen`.
// And update `Forward3D` to pass it.
// And update `Forward` (in `DecoderLayer`) to calculate it.

// Step 1: Update `MultiHeadAttention.ForwardCross` signature.
// Step 2: Update `SubLayer.ForwardCross` signature.
// Step 3: Update `DecoderLayer.Forward` and `Forward3D`.

// This is a lot of changes.
// Let's try to hack `ForwardCross` to be smarter.
// If `totalLen` is small, maybe `seqLen` is `totalLen`?
// The current logic defaults to 16.
// If `totalLen` < 16, `seqLen` should be `totalLen`?
// But `batchSize` is `totalLen / seqLen`.
// If `totalLen` is 2, and `seqLen` is 2, `batchSize` is 1.
// But we have batch=2.

// The issue is we can't infer batchSize and seqLen from just `totalLen`.
// We need an external hint.

func (mha *MultiHeadAttention) ForwardCross(x, encOutput, mask *tensor.Tensor, tgtLen int, train bool) *tensor.Tensor {
	xFlat := x
	kvFlat := encOutput

	Q := matMul(xFlat, mha.Wq.Data)
	K := matMul(kvFlat, mha.Wk.Data)
	V := matMul(kvFlat, mha.Wv.Data)

	// x and encOutput are 2D tensors from Forward3D: (batch*seq, dModel)
	dModel := x.Shape[1]
	dK := dModel / mha.config.NumHeads

	// Use provided tgtLen instead of inferring
	seqLen := tgtLen

	// Infer batchSize from totalLen and seqLen
	totalLen := x.Shape[0]
	batchSize := totalLen / seqLen
	if batchSize == 0 {
		batchSize = 1
	}

	// For encOutput (encoder output), it should have shape (batch*encSeq, dModel)
	encTotalLen := encOutput.Shape[0]
	kvSeqLen := encTotalLen / batchSize
	if kvSeqLen == 0 {
		kvSeqLen = seqLen
	}

	// Reshape to 4D for attention
	Q = reshape4D(Q, batchSize, seqLen, mha.config.NumHeads, dK)
	K = reshape4D(K, batchSize, kvSeqLen, mha.config.NumHeads, dK)
	V = reshape4D(V, batchSize, kvSeqLen, mha.config.NumHeads, dK)

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

	attnOutput := matMul4D(attnWeights, V, 0, dK)
	attnOutput = transpose4D(attnOutput, 0, 2, 1, 3)
	attnOutput = attnOutput.Reshape(batchSize*seqLen, dModel)

	output := matMul(attnOutput, mha.Wo.Data)
	output = output.Reshape(batchSize, seqLen, dModel)

	if train && mha.Dropout > 0 {
		output = dropout(output, mha.Dropout)
	}

	return output
}

type Embedding struct {
	Weights *tensor.Param
}

func NewEmbedding(vocabSize, dModel int) *Embedding {
	return &Embedding{
		Weights: tensor.NewParam(vocabSize, dModel),
	}
}

func (e *Embedding) Forward(x *tensor.Tensor) *tensor.Tensor {
	batchSize := x.Shape[0]
	seqLen := x.Shape[1]
	dimModel := e.Weights.Data.Shape[1]
	result := tensor.New(batchSize, seqLen, dimModel)

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			tokenId := int(x.Get(b, i))
			for j := 0; j < dimModel; j++ {
				val := e.Weights.Data.Get(tokenId, j)
				result.Set(val, b, i, j)
			}
		}
	}
	return result
}

func (e *Embedding) ForwardIndex(tokens []int) *tensor.Tensor {
	seqLen := len(tokens)
	result := tensor.New(1, seqLen, e.Weights.Data.Shape[1])

	for i := 0; i < seqLen; i++ {
		tokenId := tokens[i]
		for j := 0; j < e.Weights.Data.Shape[1]; j++ {
			val := e.Weights.Data.Get(tokenId, j)
			result.Set(val, 0, i, j)
		}
	}
	return result
}

func (pe *PositionalEncoding) AddSingle(x *tensor.Tensor, pos int) *tensor.Tensor {
	seqLen := x.Shape[1]
	result := x.Clone()
	for j := 0; j < seqLen; j++ {
		for k := 0; k < x.Shape[2]; k++ {
			val := result.Get(0, j, k)
			val += pe.encoding.Get(pos+j, k)
			result.Set(val, 0, j, k)
		}
	}
	return result
}

func dropoutEmbedding(x *tensor.Tensor, p float64) *tensor.Tensor {
	result := x.Clone()
	mask := tensor.New(x.Shape...)
	for i := 0; i < x.Size(); i++ {
		if rand.Float64() < p {
			result.Data[i] = 0
			mask.Data[i] = 0
		} else {
			result.Data[i] /= (1 - p)
			mask.Data[i] = 1
		}
	}
	return result
}

func CreatePaddingMask(seq []int, padId int) *tensor.Tensor {
	seqLen := len(seq)
	result := tensor.New(1, seqLen, seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if seq[j] == padId {
				result.Set(0, 0, i, j)
			} else {
				result.Set(1, 0, i, j)
			}
		}
	}
	return result
}

func CreateCausalMask(seqLen int) *tensor.Tensor {
	result := tensor.New(1, seqLen, seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i {
				result.Set(1, 0, i, j)
			} else {
				result.Set(0, 0, i, j)
			}
		}
	}
	return result
}
