package models

import (
	"fmt"
	"math"

	"finetune/config"
)

type Tensor struct {
	Data         []float32
	Shape        []int
	Grad         *Tensor
	RequiresGrad bool
}

func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return &Tensor{
		Data:         make([]float32, size),
		Shape:        shape,
		RequiresGrad: true,
	}
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	if size != len(t.Data) {
		panic("reshape size mismatch")
	}
	return &Tensor{
		Data:         t.Data,
		Shape:        shape,
		Grad:         t.Grad,
		RequiresGrad: t.RequiresGrad,
	}
}

func (t *Tensor) Numel() int {
	n := 1
	for _, s := range t.Shape {
		n *= s
	}
	return n
}

func (t *Tensor) SetData(data []float32) {
	if len(data) != len(t.Data) {
		panic("data size mismatch")
	}
	copy(t.Data, data)
}

type Module interface {
	Forward(x *Tensor) *Tensor
	Parameters() []*Tensor
	ZeroGrad()
}

type Linear struct {
	Weight *Tensor
	Bias   *Tensor
}

func NewLinear(inFeatures, outFeatures int) *Linear {
	return &Linear{
		Weight: NewTensor(outFeatures, inFeatures),
		Bias:   NewTensor(outFeatures),
	}
}

func (l *Linear) Forward(x *Tensor) *Tensor {
	out := NewTensor(x.Shape[0], l.Weight.Shape[0])

	batchSize := x.Shape[0]
	outFeatures := l.Weight.Shape[0]
	inFeatures := l.Weight.Shape[1]

	for i := 0; i < batchSize; i++ {
		for j := 0; j < outFeatures; j++ {
			sum := float32(0)
			for k := 0; k < inFeatures; k++ {
				sum += x.Data[i*inFeatures+k] * l.Weight.Data[j*inFeatures+k]
			}
			out.Data[i*outFeatures+j] = sum + l.Bias.Data[j]
		}
	}

	return out
}

func (l *Linear) Parameters() []*Tensor {
	return []*Tensor{l.Weight, l.Bias}
}

func (l *Linear) ZeroGrad() {
	if l.Weight.Grad != nil {
		for i := range l.Weight.Grad.Data {
			l.Weight.Grad.Data[i] = 0
		}
	}
	if l.Bias.Grad != nil {
		for i := range l.Bias.Grad.Data {
			l.Bias.Grad.Data[i] = 0
		}
	}
}

type Embedding struct {
	Weight *Tensor
}

func NewEmbedding(numEmbeddings, embeddingDim int) *Embedding {
	return &Embedding{
		Weight: NewTensor(numEmbeddings, embeddingDim),
	}
}

func (e *Embedding) Forward(x *Tensor) *Tensor {
	var seqLen int
	if len(x.Shape) == 1 {
		seqLen = x.Shape[0]
	} else {
		seqLen = x.Shape[1]
	}
	embDim := e.Weight.Shape[1]
	out := NewTensor(seqLen, embDim)

	for i, idx := range x.Data {
		row := int(idx)
		if row >= 0 && row < e.Weight.Shape[0] {
			for j := 0; j < embDim; j++ {
				out.Data[i*embDim+j] = e.Weight.Data[row*embDim+j]
			}
		}
	}

	return out
}

func (e *Embedding) Parameters() []*Tensor {
	return []*Tensor{e.Weight}
}

func (e *Embedding) ZeroGrad() {
	if e.Weight.Grad != nil {
		for i := range e.Weight.Grad.Data {
			e.Weight.Grad.Data[i] = 0
		}
	}
}

type LayerNorm struct {
	Weight  *Tensor
	Bias    *Tensor
	Epsilon float64
}

func NewLayerNorm(normalizedShape int) *LayerNorm {
	return &LayerNorm{
		Weight:  NewTensor(normalizedShape),
		Bias:    NewTensor(normalizedShape),
		Epsilon: 1e-5,
	}
}

func (ln *LayerNorm) Forward(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)

	mean := float32(0)
	for _, v := range x.Data {
		mean += v
	}
	mean /= float32(len(x.Data))

	std := float32(0)
	for _, v := range x.Data {
		diff := v - mean
		std += diff * diff
	}
	std = float32(math.Sqrt(float64(std)/float64(len(x.Data)) + ln.Epsilon))

	for i, v := range x.Data {
		normalized := (v - mean) / std
		out.Data[i] = normalized*ln.Weight.Data[i%ln.Weight.Numel()] + ln.Bias.Data[i%ln.Bias.Numel()]
	}

	return out
}

func (ln *LayerNorm) Parameters() []*Tensor {
	return []*Tensor{ln.Weight, ln.Bias}
}

func (ln *LayerNorm) ZeroGrad() {
	if ln.Weight.Grad != nil {
		for i := range ln.Weight.Grad.Data {
			ln.Weight.Grad.Data[i] = 0
		}
	}
	if ln.Bias.Grad != nil {
		for i := range ln.Bias.Grad.Data {
			ln.Bias.Grad.Data[i] = 0
		}
	}
}

type LoRALayer struct {
	A           *Tensor
	B           *Tensor
	Alpha       float32
	Dropout     float32
	rank        int
	inFeatures  int
	outFeatures int
	enabled     bool
}

func NewLoRALayer(inFeatures, outFeatures, rank int, alpha float32, dropout float32) *LoRALayer {
	return &LoRALayer{
		A:           NewTensor(inFeatures, rank),
		B:           NewTensor(rank, outFeatures),
		Alpha:       alpha,
		Dropout:     dropout,
		rank:        rank,
		inFeatures:  inFeatures,
		outFeatures: outFeatures,
		enabled:     true,
	}
}

func (l *LoRALayer) Forward(x *Tensor) *Tensor {
	if !l.enabled {
		return NewTensor(x.Shape...)
	}

	loraOut := NewTensor(x.Shape[0], l.outFeatures)

	for i := 0; i < x.Shape[0]; i++ {
		for j := 0; j < l.outFeatures; j++ {
			sum := float32(0)
			for k := 0; k < l.rank; k++ {
				sum += x.Data[i*l.inFeatures+k] * l.A.Data[k*l.rank+j]
			}
			loraOut.Data[i*l.outFeatures+j] = sum * (l.Alpha / float32(l.rank))
		}
	}

	return loraOut
}

func (l *LoRALayer) Parameters() []*Tensor {
	return []*Tensor{l.A, l.B}
}

func (l *LoRALayer) ZeroGrad() {
	for i := range l.A.Data {
		l.A.Data[i] = 0
	}
	for i := range l.B.Data {
		l.B.Data[i] = 0
	}
}

func (l *LoRALayer) MergeWeights() {
}

func (l *LoRALayer) Enable()  { l.enabled = true }
func (l *LoRALayer) Disable() { l.enabled = false }

type LoRAParameter struct {
	Adapter *LoRALayer
	Target  Module
	Name    string
}

type LoRAModel struct {
	baseModel Module
	adapters  map[string]*LoRAParameter
	cfg       config.LoRAConfig
}

func NewLoRAModel(baseModel Module, cfg config.LoRAConfig) *LoRAModel {
	return &LoRAModel{
		baseModel: baseModel,
		adapters:  make(map[string]*LoRAParameter),
		cfg:       cfg,
	}
}

func (m *LoRAModel) AddAdapter(name string, target Module, inFeatures, outFeatures int) {
	if !m.cfg.Enabled {
		return
	}

	adapter := NewLoRALayer(
		inFeatures,
		outFeatures,
		m.cfg.Rank,
		float32(m.cfg.Alpha),
		float32(m.cfg.Dropout),
	)

	m.adapters[name] = &LoRAParameter{
		Adapter: adapter,
		Target:  target,
		Name:    name,
	}
}

func (m *LoRAModel) GetAdapter(name string) *LoRAParameter {
	return m.adapters[name]
}

func (m *LoRAModel) Parameters() []*Tensor {
	params := m.baseModel.Parameters()
	for _, adapter := range m.adapters {
		params = append(params, adapter.Adapter.Parameters()...)
	}
	return params
}

func (m *LoRAModel) MergeAllAdapters() {
	for _, adapter := range m.adapters {
		adapter.Adapter.MergeWeights()
	}
}

func (m *LoRAModel) DisableAllAdapters() {
	for _, adapter := range m.adapters {
		adapter.Adapter.Disable()
	}
}

func (m *LoRAModel) EnableAllAdapters() {
	for _, adapter := range m.adapters {
		adapter.Adapter.Enable()
	}
}

type TransformerLayer struct {
	Attention *Attention
	MLP       *MLP
	Norm1     *LayerNorm
	Norm2     *LayerNorm
}

func NewTransformerLayer(hiddenSize, numHeads, intermediateSize int) *TransformerLayer {
	headDim := hiddenSize / numHeads

	return &TransformerLayer{
		Attention: NewAttention(hiddenSize, numHeads, headDim),
		MLP:       NewMLP(hiddenSize, intermediateSize),
		Norm1:     NewLayerNorm(hiddenSize),
		Norm2:     NewLayerNorm(hiddenSize),
	}
}

func (t *TransformerLayer) Forward(x *Tensor) *Tensor {
	attnOut := t.Attention.Forward(t.Norm1.Forward(x))
	x = Add(x, attnOut)

	mlpOut := t.MLP.Forward(t.Norm2.Forward(x))
	x = Add(x, mlpOut)

	return x
}

func (t *TransformerLayer) Parameters() []*Tensor {
	return append(
		append(t.Attention.Parameters(), t.MLP.Parameters()...),
		append(t.Norm1.Parameters(), t.Norm2.Parameters()...)...,
	)
}

func (t *TransformerLayer) ZeroGrad() {
	t.Attention.ZeroGrad()
	t.MLP.ZeroGrad()
	t.Norm1.ZeroGrad()
	t.Norm2.ZeroGrad()
}

type Attention struct {
	Wq       *Linear
	Wk       *Linear
	Wv       *Linear
	Wo       *Linear
	NumHeads int
	HeadDim  int
}

func NewAttention(hiddenSize, numHeads, headDim int) *Attention {
	return &Attention{
		Wq:       NewLinear(hiddenSize, hiddenSize),
		Wk:       NewLinear(hiddenSize, hiddenSize),
		Wv:       NewLinear(hiddenSize, hiddenSize),
		Wo:       NewLinear(hiddenSize, hiddenSize),
		NumHeads: numHeads,
		HeadDim:  headDim,
	}
}

func (a *Attention) Forward(x *Tensor) *Tensor {
	q := a.Wq.Forward(x)
	k := a.Wk.Forward(x)
	v := a.Wv.Forward(x)

	attnWeights := Softmax(ScaledDotProduct(q, k))

	return a.Wo.Forward(Matmul(attnWeights, v))
}

func (a *Attention) Parameters() []*Tensor {
	return []*Tensor{a.Wq.Weight, a.Wq.Bias, a.Wk.Weight, a.Wk.Bias, a.Wv.Weight, a.Wv.Bias, a.Wo.Weight, a.Wo.Bias}
}

func (a *Attention) ZeroGrad() {
	a.Wq.ZeroGrad()
	a.Wk.ZeroGrad()
	a.Wv.ZeroGrad()
	a.Wo.ZeroGrad()
}

type MLP struct {
	GateProj *Linear
	UpProj   *Linear
	DownProj *Linear
}

func NewMLP(hiddenSize, intermediateSize int) *MLP {
	return &MLP{
		GateProj: NewLinear(hiddenSize, intermediateSize),
		UpProj:   NewLinear(hiddenSize, intermediateSize),
		DownProj: NewLinear(intermediateSize, hiddenSize),
	}
}

func (m *MLP) Forward(x *Tensor) *Tensor {
	gate := Gelu(m.GateProj.Forward(x))
	up := m.UpProj.Forward(x)
	return m.DownProj.Forward(Mul(gate, up))
}

func (m *MLP) Parameters() []*Tensor {
	return []*Tensor{m.GateProj.Weight, m.GateProj.Bias, m.UpProj.Weight, m.UpProj.Bias, m.DownProj.Weight, m.DownProj.Bias}
}

func (m *MLP) ZeroGrad() {
	m.GateProj.ZeroGrad()
	m.UpProj.ZeroGrad()
	m.DownProj.ZeroGrad()
}

type TransformerModel struct {
	EmbedTokens   *Embedding
	PositionEmbed *Embedding
	Layers        []*TransformerLayer
	Norm          *LayerNorm
	LMHead        *Linear
	Config        config.ModelConfig
}

func NewTransformerModel(cfg config.ModelConfig) *TransformerModel {
	layers := make([]*TransformerLayer, cfg.NumLayers)
	for i := range layers {
		layers[i] = NewTransformerLayer(cfg.HiddenSize, cfg.NumHeads, cfg.IntermediateSize)
	}

	return &TransformerModel{
		EmbedTokens:   NewEmbedding(cfg.VocabSize, cfg.HiddenSize),
		PositionEmbed: NewEmbedding(cfg.MaxPosition, cfg.HiddenSize),
		Layers:        layers,
		Norm:          NewLayerNorm(cfg.HiddenSize),
		LMHead:        NewLinear(cfg.HiddenSize, cfg.VocabSize),
		Config:        cfg,
	}
}

func (m *TransformerModel) Forward(x *Tensor) *Tensor {
	_ = x.Shape[0]
	seqLen := x.Shape[1]

	positions := make([]float32, seqLen)
	for i := range positions {
		positions[i] = float32(i % m.Config.MaxPosition)
	}
	posTensor := &Tensor{Data: positions, Shape: []int{1, seqLen}}

	hiddenStates := Add(m.EmbedTokens.Forward(x), m.EmbedTokens.Forward(posTensor))

	for _, layer := range m.Layers {
		hiddenStates = layer.Forward(hiddenStates)
	}

	hiddenStates = m.Norm.Forward(hiddenStates)
	return m.LMHead.Forward(hiddenStates)
}

func (m *TransformerModel) Parameters() []*Tensor {
	params := m.EmbedTokens.Parameters()
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	params = append(params, m.Norm.Parameters()...)
	params = append(params, m.LMHead.Parameters()...)
	return params
}

func (m *TransformerModel) ZeroGrad() {
	m.EmbedTokens.ZeroGrad()
	for _, layer := range m.Layers {
		layer.ZeroGrad()
	}
	m.Norm.ZeroGrad()
	m.LMHead.ZeroGrad()
}

func (m *TransformerModel) NumParameters() int {
	n := 0
	for _, p := range m.Parameters() {
		n += p.Numel()
	}
	return n
}

type ModelWrapper struct {
	BaseModel Module
	LoRAModel *LoRAModel
	cfg       config.FineTuneConfig
}

func NewModelWrapper(cfg config.FineTuneConfig) *ModelWrapper {
	baseModel := NewTransformerModel(cfg.Model)

	var loraModel *LoRAModel
	if cfg.Lora.Enabled {
		loraModel = NewLoRAModel(baseModel, cfg.Lora)
	}

	return &ModelWrapper{
		BaseModel: baseModel,
		LoRAModel: loraModel,
		cfg:       cfg,
	}
}

func (m *ModelWrapper) Forward(x *Tensor) *Tensor {
	return m.BaseModel.Forward(x)
}

func (m *ModelWrapper) Parameters() []*Tensor {
	if m.LoRAModel != nil {
		return m.LoRAModel.Parameters()
	}
	return m.BaseModel.Parameters()
}

func (m *ModelWrapper) ZeroGrad() {
	m.BaseModel.ZeroGrad()
	if m.LoRAModel != nil {
		for _, adapter := range m.LoRAModel.adapters {
			adapter.Adapter.ZeroGrad()
		}
	}
}

func (m *ModelWrapper) TrainMode() {
}

func (m *ModelWrapper) EvalMode() {
	if m.LoRAModel != nil {
		m.LoRAModel.DisableAllAdapters()
	}
}

func Add(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch")
	}
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + b.Data[i]
	}
	return out
}

func Mul(a, b *Tensor) *Tensor {
	if len(a.Data) != len(b.Data) {
		panic("tensor size mismatch")
	}
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * b.Data[i]
	}
	return out
}

func Softmax(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)

	rowLen := x.Shape[len(x.Shape)-1]
	numRows := x.Numel() / rowLen

	for r := 0; r < numRows; r++ {
		offset := r * rowLen
		maxVal := float32(-math.MaxFloat32)

		for i := 0; i < rowLen; i++ {
			if x.Data[offset+i] > maxVal {
				maxVal = x.Data[offset+i]
			}
		}

		sum := float32(0)
		for i := 0; i < rowLen; i++ {
			out.Data[offset+i] = float32(math.Exp(float64(x.Data[offset+i] - maxVal)))
			sum += out.Data[offset+i]
		}

		for i := 0; i < rowLen; i++ {
			out.Data[offset+i] /= sum
		}
	}

	return out
}

func ScaledDotProduct(q, k *Tensor) *Tensor {
	scale := 1.0 / math.Sqrt(float64(q.Shape[len(q.Shape)-1]))

	out := NewTensor(q.Shape...)
	kLen := k.Shape[len(k.Shape)-1]

	for i := range q.Data {
		out.Data[i] = q.Data[i] * k.Data[i%kLen] * float32(scale)
	}

	return out
}

func Matmul(a, b *Tensor) *Tensor {
	out := NewTensor(a.Shape[0], b.Shape[1])

	for i := 0; i < a.Shape[0]; i++ {
		for j := 0; j < b.Shape[1]; j++ {
			sum := float32(0)
			for k := 0; k < a.Shape[1]; k++ {
				sum += a.Data[i*a.Shape[1]+k] * b.Data[k*b.Shape[1]+j]
			}
			out.Data[i*b.Shape[1]+j] = sum
		}
	}

	return out
}

func Gelu(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)
	for i, v := range x.Data {
		out.Data[i] = float32(0.5 * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(float64(v)+0.044715*math.Pow(float64(v), 3)))))
	}
	return out
}

type ModelSaveable interface {
	Save(path string) error
	Load(path string) error
}

func SaveModel(model Module, path string) error {
	params := model.Parameters()
	fmt.Printf("Would save %d parameters to %s\n", len(params), path)
	return nil
}

func LoadModel(model Module, path string) error {
	fmt.Printf("Would load parameters from %s\n", path)
	return nil
}

type Checkpoint struct {
	Epoch     int
	Step      int
	Model     map[string][]float32
	Optimizer map[string][]float32
	Metrics   map[string]float64
}

func SaveCheckpoint(ckpt *Checkpoint, path string) error {
	fmt.Printf("Would save checkpoint to %s\n", path)
	return nil
}

func LoadCheckpoint(path string) (*Checkpoint, error) {
	fmt.Printf("Would load checkpoint from %s\n", path)
	return nil, nil
}
