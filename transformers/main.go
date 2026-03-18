package main

import (
	"fmt"

	"transformers/bert"
	"transformers/core"
	"transformers/gpt"
	"transformers/tokenizer"
	"transformers/training"
)

func main() {
	fmt.Println("=== Native Go Transformers (GPT/BERT) Demo ===")
	fmt.Println()

	fmt.Println("=== 1. Core Tensor Operations ===")
	t1 := core.NewTensor(2, 3, 4)
	fmt.Printf("Created tensor with shape: %v, numel: %d\n", t1.Shape, t1.Numel())

	t2 := core.TensorRandUniformSeeded(42, 0, 1, 3, 3)
	fmt.Printf("Random tensor (seeded): %v\n", t2.Shape)

	t3 := core.TensorZeros(2, 2)
	fmt.Printf("Zeros tensor: %v\n", t3.Shape)

	tEye := core.TensorEye(4)
	fmt.Printf("Identity tensor: %v\n", tEye.Shape)

	fmt.Println()
	fmt.Println("=== 2. GPT Model Configuration ===")
	gptConfig := gpt.GPT2Config()
	fmt.Printf("GPT-2 Config:\n")
	fmt.Printf("  Hidden Size: %d\n", gptConfig.HiddenSize)
	fmt.Printf("  Num Layers: %d\n", gptConfig.NumLayers)
	fmt.Printf("  Num Heads: %d\n", gptConfig.NumHeads)
	fmt.Printf("  Intermediate Size: %d\n", gptConfig.IntermediateSize)
	fmt.Printf("  Vocab Size: %d\n", gptConfig.VocabSize)
	fmt.Printf("  Max Position: %d\n", gptConfig.MaxPositionEmbed)

	gptModel := gpt.NewGPTModel(gptConfig)
	fmt.Printf("\nGPT-2 Model created with %d parameters\n", gptModel.NumParameters())

	fmt.Println()
	fmt.Println("=== 3. BERT Model Configuration ===")
	bertConfig := bert.DefaultBERTConfig()
	fmt.Printf("BERT Config:\n")
	fmt.Printf("  Hidden Size: %d\n", bertConfig.HiddenSize)
	fmt.Printf("  Num Layers: %d\n", bertConfig.NumLayers)
	fmt.Printf("  Num Heads: %d\n", bertConfig.NumHeads)
	fmt.Printf("  Intermediate Size: %d\n", bertConfig.IntermediateSize)
	fmt.Printf("  Vocab Size: %d\n", bertConfig.VocabSize)
	fmt.Printf("  Type Vocab Size: %d\n", bertConfig.TypeVocabSize)

	bertModel := bert.NewBERTModel(bertConfig)
	_ = bertModel
	fmt.Printf("\nBERT Model created\n")

	fmt.Println()
	fmt.Println("=== 4. Tokenizer Demo ===")
	tok := tokenizer.NewBasicTokenizer(1000)
	text := "Hello world, this is a test of the tokenizer"
	ids := tok.Encode(text)
	decoded := tok.Decode(ids)
	fmt.Printf("Original: \"%s\"\n", text)
	fmt.Printf("Encoded: %v\n", ids)
	fmt.Printf("Decoded: \"%s\"\n", decoded)
	fmt.Printf("Vocab size: %d\n", tok.VocabSize())

	fmt.Println()
	fmt.Println("=== 5. Optimizer Demo ===")
	params := make([]*training.AdamParam, 2)
	params[0] = &training.AdamParam{
		Data:     []float32{1.0, 2.0, 3.0, 4.0},
		Grad:     []float32{0.1, 0.1, 0.1, 0.1},
		ExpAvg:   []float32{0, 0, 0, 0},
		ExpAvgSq: []float32{0, 0, 0, 0},
		Step:     0,
	}
	params[1] = &training.AdamParam{
		Data:     []float32{0.5, 1.5, 2.5},
		Grad:     []float32{0.05, 0.05, 0.05},
		ExpAvg:   []float32{0, 0, 0},
		ExpAvgSq: []float32{0, 0, 0},
		Step:     0,
	}

	adamw := training.NewAdamW(params, 0.001, 0.9, 0.999, 1e-8, 0.01)
	fmt.Printf("AdamW optimizer created with %d parameter groups\n", len(params))
	fmt.Printf("Initial params[0]: %v\n", params[0].Data)

	for i := 0; i < 3; i++ {
		adamw.Step()
	}
	fmt.Printf("After 3 steps params[0]: %v\n", params[0].Data)

	adam := training.NewAdam(params, 0.001, 0.9, 0.999, 1e-8)
	for i := 0; i < 3; i++ {
		adam.Step()
	}
	fmt.Printf("After 3 Adam steps params[0]: %v\n", params[0].Data)

	sgd := training.NewSGD(params, 0.01, 0.9)
	for i := 0; i < 3; i++ {
		sgd.Step()
	}
	fmt.Printf("After 3 SGD steps params[0]: %v\n", params[0].Data)

	fmt.Println()
	fmt.Println("=== 6. Learning Rate Scheduler Demo ===")
	sched := training.NewScheduler(adamw, training.SchedulerCosine, 10, 100, 0.001)
	fmt.Println("Cosine LR Schedule (first 10 steps):")
	for i := 0; i < 10; i++ {
		sched.Step()
		fmt.Printf("  Step %2d: LR = %.6f\n", i+1, sched.GetCurrentLR())
	}

	fmt.Println()
	fmt.Println("=== 7. Metrics Demo ===")
	metrics := training.NewMetrics()
	metrics.AddTrainLoss(2.5)
	metrics.AddTrainLoss(2.3)
	metrics.AddTrainLoss(2.1)
	metrics.AddTrainLoss(2.0)
	metrics.AddEvalLoss(2.4)
	metrics.AddEvalLoss(2.2)
	metrics.AddLearningRate(0.001)
	fmt.Printf("Average train loss: %.4f\n", metrics.AverageTrainLoss())
	fmt.Printf("Average eval loss: %.4f\n", metrics.AverageEvalLoss())

	fmt.Println()
	fmt.Println("=== 8. Data Loader Demo ===")
	data := make([][]int, 100)
	labels := make([][]int, 100)
	for i := range data {
		data[i] = make([]int, 10)
		labels[i] = make([]int, 10)
		for j := range data[i] {
			data[i][j] = (i + j) % 50
			labels[i][j] = (i + j + 1) % 50
		}
	}

	loader := training.NewDataLoader(data, labels, 4, 10, true)
	loader.Init()
	batch, _, hasMore := loader.Next()
	fmt.Printf("DataLoader: batch size %d, has more: %v\n", len(batch), hasMore)
	loader.Reset()
	batch2, _, _ := loader.Next()
	fmt.Printf("After reset: new batch size %d\n", len(batch2))

	fmt.Println()
	fmt.Println("=== 9. Math Operations Demo ===")
	a := core.NewTensor(2, 3)
	b := core.NewTensor(2, 3)
	for i := range a.Data {
		a.Data[i] = float32(i + 1)
		b.Data[i] = 0.5
	}

	sum := core.Add(a, b)
	product := core.Mul(a, b)
	neg := core.Neg(a)
	fmt.Printf("Add result shape: %v\n", sum.Shape)
	fmt.Printf("Mul result shape: %v\n", product.Shape)
	fmt.Printf("Neg result shape: %v\n", neg.Shape)

	softmax := core.Softmax(a, 1)
	fmt.Printf("Softmax result shape: %v\n", softmax.Shape)

	gelu := gpt.Gelu(a)
	fmt.Printf("GELU result shape: %v\n", gelu.Shape)

	tanh := bert.Tanh(a)
	fmt.Printf("Tanh result shape: %v\n", tanh.Shape)

	fmt.Println()
	fmt.Println("=== 10. Embedding Demo ===")
	emb := gpt.NewEmbedding(100, 32)
	fmt.Printf("Embedding layer: vocab=%d, dim=%d\n", 100, 32)
	fmt.Printf("Embedding parameters: %d\n", emb.NumParameters())

	linear := gpt.NewLinear(64, 32)
	fmt.Printf("Linear layer: in=%d, out=%d\n", 64, 32)
	fmt.Printf("Linear parameters: %d\n", linear.NumParameters())

	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Println("Available Components:")
	fmt.Println("  - Core: Tensor, ops (Add, Mul, Softmax, GELU, Tanh, LayerNorm, etc.)")
	fmt.Println("  - GPT: GPTModel, TransformerBlock, MultiHeadAttention, MLP, LayerNorm")
	fmt.Println("  - BERT: BERTModel, BERTEncoder, BERTLayer, BERTAttention, PreTrainingHeads")
	fmt.Println("  - Tokenizer: BasicTokenizer, BPETokenizer")
	fmt.Println("  - Training: AdamW, Adam, SGD optimizers, LR Scheduler, Metrics, DataLoader")
	fmt.Println()
	fmt.Println("Model Architectures:")
	fmt.Println("  - GPT-2: 105M+ parameters (decoder-only transformer)")
	fmt.Println("  - BERT: Bidirectional encoder transformer")
	fmt.Println()
	fmt.Println("All components initialized successfully!")
}
