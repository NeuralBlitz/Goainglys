package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"time"

	"github.com/user/transformer/tensor"
	"github.com/user/transformer/transformer"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	// Demo with multiple model sizes
	demoSmallModel()
	demoLargeModel()
}

func demoSmallModel() {
	fmt.Println("\n" + string(make([]byte, 70)))
	fmt.Println("  SMALL MODEL (Training Demo)")
	fmt.Println(string(make([]byte, 70)))

	config := transformer.Config{
		ModelDim:  256,
		NumHeads:  8,
		NumLayers: 2,
		FFNDim:    1024,
		VocabSize: 1000,
		MaxSeqLen: 32,
		Dropout:   0.1,
	}

	trainer := NewTrainer(config)
	trainer.Run(10, 2) // 10 epochs, batch size 2
}

func demoLargeModel() {
	fmt.Println("\n" + string(make([]byte, 70)))
	fmt.Println("  LARGE MODEL (Training Demo)")
	fmt.Println(string(make([]byte, 70)))

	config := transformer.Config{
		ModelDim:  512,
		NumHeads:  16,
		NumLayers: 4,
		FFNDim:    2048,
		VocabSize: 1000,
		MaxSeqLen: 32,
		Dropout:   0.1,
	}

	trainer := NewTrainer(config)
	trainer.Run(5, 2) // 5 epochs for speed
}

// ============================================================
//   TRAINER CLASS
// ============================================================

type Trainer struct {
	config    transformer.Config
	model     *transformer.Transformer
	optimizer *AdamOptimizer
	scheduler *WarmupScheduler
	metrics   TrainingMetrics
}

type TrainingMetrics struct {
	lossHistory    []float64
	valLossHistory []float64
	learningRates  []float64
	bestLoss       float64
	bestValLoss    float64
	earlyStopCount int
}

func NewTrainer(config transformer.Config) *Trainer {
	return &Trainer{
		config:    config,
		model:     transformer.NewTransformer(config),
		optimizer: NewAdamOptimizer(0.0001),
		scheduler: NewWarmupScheduler(config.ModelDim, 4000),
		metrics: TrainingMetrics{
			bestLoss:       math.MaxFloat64,
			bestValLoss:    math.MaxFloat64,
			earlyStopCount: 0,
		},
	}
}

// ============================================================
//   TRAINING LOOP
// ============================================================

func (t *Trainer) Run(numEpochs, batchSize int) {
	seqLen := 32

	fmt.Printf("Model: %d layers, %d heads, %d model dim\n",
		t.config.NumLayers, t.config.NumHeads, t.config.ModelDim)
	fmt.Printf("Training: %d epochs, batch size %d, seq len %d\n\n",
		numEpochs, batchSize, seqLen)

	startTime := time.Now()
	var epochTimes []time.Duration

	for epoch := 0; epoch < numEpochs; epoch++ {
		epochStart := time.Now()

		// Generate batch data
		src, tgt := generateBatch(batchSize, seqLen, t.config.VocabSize)

		// Create masks
		tgtMask := createCausalMask(batchSize, seqLen)

		// Get learning rate
		lr := t.scheduler.GetLR(float64(epoch + 1))
		t.optimizer.SetLR(lr)

		// Training step
		loss := t.trainStep(src, tgt, nil, tgtMask)

		// Track metrics
		t.metrics.lossHistory = append(t.metrics.lossHistory, loss)
		t.metrics.learningRates = append(t.metrics.learningRates, lr)

		// Update best loss
		if loss < t.metrics.bestLoss {
			t.metrics.bestLoss = loss
			t.metrics.earlyStopCount = 0
		} else {
			t.metrics.earlyStopCount++
		}

		// Validation evaluation every 2 epochs
		valLoss := t.metrics.bestValLoss
		if epoch%2 == 0 || epoch == numEpochs-1 {
			valLoss = t.evaluate(batchSize, seqLen)
			t.metrics.valLossHistory = append(t.metrics.valLossHistory, valLoss)
			if valLoss < t.metrics.bestValLoss {
				t.metrics.bestValLoss = valLoss
			}
		}

		epochTime := time.Since(epochStart)
		epochTimes = append(epochTimes, epochTime)

		// Print progress
		if epoch == 0 || (epoch+1)%2 == 0 || epoch == numEpochs-1 {
			fmt.Printf("Epoch %2d/%d | Loss: %.6f | Val Loss: %.6f | LR: %.8f | Time: %v",
				epoch+1, numEpochs, loss, valLoss, lr, epochTime.Round(time.Millisecond))
			if t.metrics.earlyStopCount > 0 {
				fmt.Printf(" | No improvement: %d", t.metrics.earlyStopCount)
			}
			fmt.Println()
		}

		// Early stopping
		if t.metrics.earlyStopCount >= 5 {
			fmt.Printf("\nEarly stopping triggered after epoch %d\n", epoch+1)
			break
		}
	}

	elapsed := time.Since(startTime)
	avgEpochTime := time.Duration(0)
	for _, et := range epochTimes {
		avgEpochTime += et
	}
	avgEpochTime = avgEpochTime / time.Duration(len(epochTimes))

	fmt.Printf("\nTraining completed in %v (avg %v per epoch)\n",
		elapsed.Round(time.Second), avgEpochTime.Round(time.Millisecond))
	fmt.Printf("Best train loss: %.6f\n", t.metrics.bestLoss)
	fmt.Printf("Best val loss: %.6f\n", t.metrics.bestValLoss)

	t.generateSample()
}

func (t *Trainer) evaluate(batchSize, seqLen int) float64 {
	// Generate validation batch
	src, tgt := generateBatch(batchSize, seqLen, t.config.VocabSize)
	tgtMask := createCausalMask(batchSize, seqLen)

	// Forward pass (no training)
	logits := t.model.Forward(src, tgt, nil, tgtMask, false)

	// Compute loss
	loss := CrossEntropyLoss(logits, tgt)

	return loss
}

func (t *Trainer) trainStep(src, tgt, srcMask, tgtMask *tensor.Tensor) float64 {
	t.optimizer.ZeroGrad()

	// Forward pass
	logits := t.model.Forward(src, tgt, srcMask, tgtMask, true)

	// Compute loss
	loss := CrossEntropyLoss(logits, tgt)

	// Compute gradients using backpropagation
	transformer.ComputeGradients(t.model, src, tgt, srcMask, tgtMask, loss)

	// Update parameters
	lr := t.scheduler.GetLR(float64(t.optimizer.step))
	t.UpdateParams(lr)

	return loss
}

func (t *Trainer) computeGradientsAndUpdate(logits, targets *tensor.Tensor, lr float64) {
	// For now, just update the model using a simplified approach
	// This function is kept for interface compatibility
	t.UpdateParams(lr)
}

// UpdateParams updates model parameters using gradients
func (t *Trainer) UpdateParams(lr float64) {
	// Update output projection
	for i := range t.model.OutputProj.Data.Data {
		grad := t.model.OutputProj.Grad.Data[i]
		t.model.OutputProj.Data.Data[i] -= lr * grad
		t.model.OutputProj.Grad.Data[i] = 0
	}

	// Update embedding
	for i := range t.model.Embedding.Weights.Data.Data {
		grad := t.model.Embedding.Weights.Grad.Data[i]
		t.model.Embedding.Weights.Data.Data[i] -= lr * grad
		t.model.Embedding.Weights.Grad.Data[i] = 0
	}

	// Update encoder layers
	for _, layer := range t.model.EncoderLayers {
		updateSubLayer(layer.SubLayer, lr)
	}

	// Update decoder layers
	for _, layer := range t.model.DecoderLayers {
		updateSubLayer(layer.SubLayer1, lr)
		updateSubLayer(layer.SubLayer2, lr)
		updateSubLayer(layer.SubLayer3, lr)
	}
}

func updateSubLayer(sl *transformer.SubLayer, lr float64) {
	// Update attention parameters
	updateParam(sl.Attention.Wq, lr)
	updateParam(sl.Attention.Wk, lr)
	updateParam(sl.Attention.Wv, lr)
	updateParam(sl.Attention.Wo, lr)

	// Update feedforward parameters
	updateParam(sl.Ffn.W1, lr)
	updateParam(sl.Ffn.W2, lr)
	updateParam(sl.Ffn.B1, lr)
	updateParam(sl.Ffn.B2, lr)

	// Update layer norm parameters
	updateParam(sl.Ln1, lr)
	updateParam(sl.Ln2, lr)
}

func updateParam(p *tensor.Param, lr float64) {
	for i := range p.Data.Data {
		p.Data.Data[i] -= lr * p.Grad.Data[i]
		p.Grad.Data[i] = 0
	}
}

func (t *Trainer) computeGradients(logits, targets *tensor.Tensor) *tensor.Tensor {
	// Simple gradient computation for cross-entropy loss
	// grad = softmax(logits) - one_hot(targets)
	batchSize := logits.Shape[0]
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	grad := tensor.New(batchSize, seqLen, vocabSize)

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			// Compute softmax
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

			// Set gradient
			for j := 0; j < vocabSize; j++ {
				prob := math.Exp(logits.Get(b, i, j)-maxVal) / sum
				target := int(targets.Get(b, i))
				if j == target {
					grad.Set(prob-1.0, b, i, j)
				} else {
					grad.Set(prob, b, i, j)
				}
			}
		}
	}

	return grad
}

// ============================================================
//   UTILITIES
// ============================================================

func generateBatch(batchSize, seqLen, vocabSize int) (*tensor.Tensor, *tensor.Tensor) {
	src := tensor.New(batchSize, seqLen)
	tgt := tensor.New(batchSize, seqLen)

	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			src.Set(float64((i*seqLen+j)%vocabSize), i, j)
			tgt.Set(float64((i*seqLen+j+1)%vocabSize), i, j)
		}
	}

	return src, tgt
}

func createCausalMask(batchSize, seqLen int) *tensor.Tensor {
	mask := tensor.New(batchSize, seqLen, seqLen)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if j <= i {
					mask.Set(1, b, i, j)
				} else {
					mask.Set(0, b, i, j)
				}
			}
		}
	}
	return mask
}

func CrossEntropyLoss(logits, targets *tensor.Tensor) float64 {
	batchSize := logits.Shape[0]
	seqLen := logits.Shape[1]
	vocabSize := logits.Shape[2]

	totalLoss := 0.0
	count := 0

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			target := int(targets.Get(b, i))
			if target < 0 || target >= vocabSize {
				continue
			}

			// Compute softmax with log-sum-exp trick
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

	if count == 0 {
		return 0
	}
	return totalLoss / float64(count)
}

// ============================================================
//   ADAM OPTIMIZER
// ============================================================

type AdamOptimizer struct {
	lr      float64
	beta1   float64
	beta2   float64
	epsilon float64
	step    int
}

func NewAdamOptimizer(lr float64) *AdamOptimizer {
	return &AdamOptimizer{
		lr:      lr,
		beta1:   0.9,
		beta2:   0.999,
		epsilon: 1e-8,
		step:    0,
	}
}

func (o *AdamOptimizer) ZeroGrad() {
	o.step++
}

func (o *AdamOptimizer) SetLR(lr float64) {
	o.lr = lr
}

// ============================================================
//   WARMUP SCHEDULER
// ============================================================

type WarmupScheduler struct {
	dModel      float64
	warmupSteps float64
}

func NewWarmupScheduler(dModel int, warmupSteps int) *WarmupScheduler {
	return &WarmupScheduler{
		dModel:      float64(dModel),
		warmupSteps: float64(warmupSteps),
	}
}

func (s *WarmupScheduler) GetLR(step float64) float64 {
	arg1 := math.Pow(s.dModel, -0.5)
	arg2 := math.Pow(step, -0.5)
	arg3 := math.Pow(s.warmupSteps, -1.5) * step

	return arg1 * math.Min(arg2, arg3)
}

// ============================================================
//   SAMPLE GENERATION
// ============================================================

func (t *Trainer) generateSample() {
	fmt.Println("\n" + string(make([]byte, 70)))
	fmt.Println("  SAMPLE PREDICTION (Autoregressive)")
	fmt.Println(string(make([]byte, 70)))

	batchSize := 2
	seqLen := 16

	// Generate input sequence
	src, tgt := generateBatch(batchSize, seqLen, t.config.VocabSize)

	// Print input sequences
	fmt.Println("\nInput sequences:")
	for b := 0; b < batchSize; b++ {
		fmt.Printf("  Batch %d: ", b)
		for i := 0; i < 8; i++ {
			fmt.Printf("%3d ", int(src.Get(b, i)))
		}
		fmt.Println("...")
	}

	// Generate target sequence autoregressively
	preds := tensor.New(batchSize, seqLen)

	// Initialize with start token (assuming 0 is start or just first token of input)
	// For this demo, we'll generate based on the input sequence provided
	// Actually, typical generation: start with BOS or input prefix.
	// Here we will just try to predict the next token given the input sequence up to that point.
	// We'll use the input sequence `src` as the prefix for generation.

	// Current sequence (starts with input)
	currentSeq := tensor.New(batchSize, 1)
	for b := 0; b < batchSize; b++ {
		currentSeq.Set(src.Get(b, 0), b, 0)
	}

	for i := 0; i < seqLen; i++ {
		// Create mask for current length
		tgtMask := createCausalMask(batchSize, i+1)

		logits := t.model.Forward(src, currentSeq, nil, tgtMask, false)

		// Get prediction for the last position
		lastPos := i

		// Update predictions
		for b := 0; b < batchSize; b++ {
			bestIdx := 0
			bestVal := math.Inf(-1)
			for j := 0; j < t.config.VocabSize; j++ {
				val := logits.Get(b, lastPos, j)
				if val > bestVal {
					bestVal = val
					bestIdx = j
				}
			}
			preds.Set(float64(bestIdx), b, i)
		}

		// Append to current sequence for next step
		newSeq := tensor.New(batchSize, i+2)
		for b := 0; b < batchSize; b++ {
			for k := 0; k <= i; k++ {
				newSeq.Set(currentSeq.Get(b, k), b, k)
			}
			newSeq.Set(preds.Get(b, i), b, i+1)
		}
		currentSeq = newSeq
	}

	// Print target sequences
	fmt.Println("\nTarget sequences:")
	for b := 0; b < batchSize; b++ {
		fmt.Printf("  Batch %d: ", b)
		for i := 0; i < 8; i++ {
			fmt.Printf("%3d ", int(tgt.Get(b, i)))
		}
		fmt.Println("...")
	}

	// Print predicted sequences
	fmt.Println("\nPredicted sequences:")
	for b := 0; b < batchSize; b++ {
		fmt.Printf("  Batch %d: ", b)
		for i := 0; i < 8; i++ {
			fmt.Printf("%3d ", int(preds.Get(b, i)))
		}
		fmt.Println("...")
	}

	// Calculate accuracy
	correct := 0
	total := 0
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			if int(preds.Get(b, i)) == int(tgt.Get(b, i)) {
				correct++
			}
			total++
		}
	}
	fmt.Printf("\nPrediction accuracy: %.2f%%\n", float64(correct)/float64(total)*100)

	// Visualize attention weights from the last step
	t.visualizeAttention()
}

func (t *Trainer) visualizeAttention() {
	fmt.Println("\n  ATTENTION VISUALIZATION (Last Decoder Layer, Self-Attention)")

	// Get the last decoder layer
	if len(t.model.DecoderLayers) == 0 {
		fmt.Println("No decoder layers found.")
		return
	}

	lastLayer := t.model.DecoderLayers[len(t.model.DecoderLayers)-1]
	// SubLayer1 is self-attention
	att := lastLayer.SubLayer1.Attention

	if att.LastAttentionWeights == nil {
		fmt.Println("No attention weights stored. Run inference first.")
		return
	}

	// weights shape: [batch, heads, seqLen, seqLen]
	weights := att.LastAttentionWeights
	_ = weights.Shape[0] // batchSize
	seqLen := weights.Shape[2]

	// Visualize first batch, all heads
	b := 0

	for h := 0; h < weights.Shape[1]; h++ {
		fmt.Printf("Batch %d, Head %d:\n", b, h)
		fmt.Print("   ")
		for i := 0; i < seqLen && i < 16; i++ {
			fmt.Printf("%3d", i)
		}
		fmt.Println()

		for i := 0; i < seqLen && i < 16; i++ {
			fmt.Printf("%2d ", i)
			for j := 0; j < seqLen && j < 16; j++ {
				val := weights.Get(b, h, i, j)
				// ASCII map: 0.0 -> ' ', 1.0 -> '#'
				char := ' '
				if val > 0.1 {
					char = '.'
				}
				if val > 0.3 {
					char = ':'
				}
				if val > 0.5 {
					char = '-'
				}
				if val > 0.7 {
					char = '+'
				}
				if val > 0.9 {
					char = '#'
				}
				fmt.Printf(" %c ", char)
			}
			fmt.Println()
		}
		fmt.Println()
	}
}

// ============================================================
//   TEXT GENERATION WITH TEMPERATURE
// ============================================================

func (t *Trainer) GenerateWithTemperature(seedTokens []int, maxLen int, temperature float64) []int {
	batchSize := 1
	seqLen := len(seedTokens)

	// Create source tensor
	src := tensor.New(batchSize, seqLen)
	for i := 0; i < seqLen; i++ {
		src.Set(float64(seedTokens[i]), 0, i)
	}

	result := make([]int, 0)
	result = append(result, seedTokens...)

	for len(result) < maxLen {
		currentSeqLen := len(result)
		tgt := tensor.New(1, currentSeqLen)
		for i := 0; i < currentSeqLen; i++ {
			tgt.Set(float64(result[i]), 0, i)
		}

		tgtMask := createCausalMask(1, currentSeqLen)

		logits := t.model.Forward(src, tgt, nil, tgtMask, false)

		// Get logits for the last position
		lastPos := currentSeqLen - 1
		vocabSize := t.config.VocabSize

		// Extract logits for last position
		lastLogits := make([]float64, vocabSize)
		for j := 0; j < vocabSize; j++ {
			lastLogits[j] = logits.Get(0, lastPos, j)
		}

		// Apply temperature
		for j := 0; j < vocabSize; j++ {
			lastLogits[j] /= temperature
		}

		// Softmax
		maxVal := math.Inf(-1)
		for j := 0; j < vocabSize; j++ {
			if lastLogits[j] > maxVal {
				maxVal = lastLogits[j]
			}
		}

		sum := 0.0
		for j := 0; j < vocabSize; j++ {
			sum += math.Exp(lastLogits[j] - maxVal)
		}
		logSumExp := maxVal + math.Log(sum)

		probs := make([]float64, vocabSize)
		for j := 0; j < vocabSize; j++ {
			probs[j] = math.Exp(lastLogits[j] - logSumExp)
		}

		// Sample from distribution
		nextToken := sampleFromProbs(probs)

		result = append(result, nextToken)

		// Extend src to include the new token for next iteration
		newSrc := tensor.New(1, currentSeqLen+1)
		for i := 0; i < currentSeqLen; i++ {
			newSrc.Set(float64(result[i]), 0, i)
		}
		newSrc.Set(float64(nextToken), 0, currentSeqLen)
		src = newSrc

		if nextToken == 0 { // Assume 0 is EOS
			break
		}
		if len(result) >= t.config.MaxSeqLen {
			break
		}
	}

	return result
}

func sampleFromProbs(probs []float64) int {
	// Simple sampling - could use more sophisticated methods
	r := rand.Float64()
	cumsum := 0.0
	for i, p := range probs {
		cumsum += p
		if r <= cumsum {
			return i
		}
	}
	return len(probs) - 1
}

// ============================================================
//   SAVE/LOAD MODEL
// ============================================================

type ModelCheckpoint struct {
	Config        transformer.Config    `json:"config"`
	BestLoss      float64               `json:"best_loss"`
	Epoch         int                   `json:"epoch"`
	Timestamp     string                `json:"timestamp"`
	Embeddings    []float64             `json:"embeddings"`
	OutputProj    []float64             `json:"output_proj"`
	EncoderLayers []EncoderLayerWeights `json:"encoder_layers"`
	DecoderLayers []DecoderLayerWeights `json:"decoder_layers"`
}

type EncoderLayerWeights struct {
	Attention [][][]float64 `json:"attention"` // Wq, Wk, Wv, Wo
	FFN       [][][]float64 `json:"ffn"`       // W1, W2
	LN        []float64     `json:"ln"`        // ln1, ln2
}

type DecoderLayerWeights struct {
	SubLayer1 [][][][]float64 `json:"sublayer1"`
	SubLayer2 [][][][]float64 `json:"sublayer2"`
	SubLayer3 [][][][]float64 `json:"sublayer3"`
}

func (t *Trainer) SaveModel(path string) error {
	checkpoint := ModelCheckpoint{
		Config:    t.config,
		BestLoss:  t.metrics.bestLoss,
		Epoch:     len(t.metrics.lossHistory),
		Timestamp: time.Now().Format(time.RFC3339),
	}

	// Save embeddings
	embeddings := make([]float64, t.model.Embedding.Weights.Data.Size())
	copy(embeddings, t.model.Embedding.Weights.Data.Data)
	checkpoint.Embeddings = embeddings

	// Save output projection
	outputProj := make([]float64, t.model.OutputProj.Data.Size())
	copy(outputProj, t.model.OutputProj.Data.Data)
	checkpoint.OutputProj = outputProj

	// Save encoder layers
	for _, layer := range t.model.EncoderLayers {
		weights := EncoderLayerWeights{}
		weights.Attention = saveAttentionWeights(layer.SubLayer.Attention)
		weights.FFN = saveFFNWeights(layer.SubLayer.Ffn)
		weights.LN = saveLNWeights(layer.SubLayer)
		checkpoint.EncoderLayers = append(checkpoint.EncoderLayers, weights)
	}

	// Save decoder layers
	for _, layer := range t.model.DecoderLayers {
		weights := DecoderLayerWeights{}
		weights.SubLayer1 = saveSubLayerWeights(layer.SubLayer1)
		weights.SubLayer2 = saveSubLayerWeights(layer.SubLayer2)
		weights.SubLayer3 = saveSubLayerWeights(layer.SubLayer3)
		checkpoint.DecoderLayers = append(checkpoint.DecoderLayers, weights)
	}

	jsonData, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		return err
	}

	return ioutil.WriteFile(path, jsonData, 0644)
}

func saveAttentionWeights(att *transformer.MultiHeadAttention) [][][]float64 {
	weights := [][][]float64{
		saveParam(att.Wq),
		saveParam(att.Wk),
		saveParam(att.Wv),
		saveParam(att.Wo),
	}
	return weights
}

func saveFFNWeights(ffn *transformer.FeedForward) [][][]float64 {
	return [][][]float64{
		saveParam(ffn.W1),
		saveParam(ffn.W2),
	}
}

func saveLNWeights(sl *transformer.SubLayer) []float64 {
	result := make([]float64, 0)
	result = append(result, saveParam1D(sl.Ln1)...)
	result = append(result, saveParam1D(sl.Ln2)...)
	return result
}

func saveParam(p *tensor.Param) [][]float64 {
	rows := p.Data.Shape[0]
	cols := p.Data.Shape[1]
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = p.Data.Get(i, j)
		}
	}
	return result
}

func saveParam1D(p *tensor.Param) []float64 {
	result := make([]float64, p.Data.Size())
	copy(result, p.Data.Data)
	return result
}

func saveSubLayerWeights(sl *transformer.SubLayer) [][][][]float64 {
	weights := [][][][]float64{
		saveAttentionWeights(sl.Attention),
		saveFFNWeights(sl.Ffn),
	}
	return weights
}

func (t *Trainer) LoadModel(path string) error {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return err
	}

	var checkpoint ModelCheckpoint
	if err := json.Unmarshal(data, &checkpoint); err != nil {
		return err
	}

	// Load embeddings
	if len(checkpoint.Embeddings) == t.model.Embedding.Weights.Data.Size() {
		copy(t.model.Embedding.Weights.Data.Data, checkpoint.Embeddings)
	}

	// Load output projection
	if len(checkpoint.OutputProj) == t.model.OutputProj.Data.Size() {
		copy(t.model.OutputProj.Data.Data, checkpoint.OutputProj)
	}

	// Load encoder layers
	for i, layer := range t.model.EncoderLayers {
		if i < len(checkpoint.EncoderLayers) {
			loadAttentionWeights(layer.SubLayer.Attention, checkpoint.EncoderLayers[i].Attention)
			loadFFNWeights(layer.SubLayer.Ffn, checkpoint.EncoderLayers[i].FFN)
		}
	}

	// Load decoder layers
	for i, layer := range t.model.DecoderLayers {
		if i < len(checkpoint.DecoderLayers) {
			loadSubLayerWeights(layer.SubLayer1, checkpoint.DecoderLayers[i].SubLayer1)
			loadSubLayerWeights(layer.SubLayer2, checkpoint.DecoderLayers[i].SubLayer2)
			loadSubLayerWeights(layer.SubLayer3, checkpoint.DecoderLayers[i].SubLayer3)
		}
	}

	fmt.Printf("Loaded model from epoch %d with loss: %.4f\n", checkpoint.Epoch, checkpoint.BestLoss)
	return nil
}

func loadAttentionWeights(att *transformer.MultiHeadAttention, weights [][][]float64) {
	if len(weights) >= 4 {
		loadParam(att.Wq, weights[0])
		loadParam(att.Wk, weights[1])
		loadParam(att.Wv, weights[2])
		loadParam(att.Wo, weights[3])
	}
}

func loadFFNWeights(ffn *transformer.FeedForward, weights [][][]float64) {
	if len(weights) >= 2 {
		loadParam(ffn.W1, weights[0])
		loadParam(ffn.W2, weights[1])
	}
}

func loadParam(p *tensor.Param, data [][]float64) {
	if len(data) != p.Data.Shape[0] || len(data[0]) != p.Data.Shape[1] {
		return
	}
	for i := 0; i < len(data); i++ {
		for j := 0; j < len(data[i]); j++ {
			p.Data.Set(data[i][j], i, j)
		}
	}
}

func loadSubLayerWeights(sl *transformer.SubLayer, weights [][][][]float64) {
	if len(weights) >= 2 {
		loadAttentionWeights(sl.Attention, weights[0])
		loadFFNWeights(sl.Ffn, weights[1])
	}
}
