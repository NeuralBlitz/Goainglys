package main

import (
	"fmt"
	"time"

	"finetune/config"
	"finetune/models"
	"finetune/optimizers"
	"finetune/training"
)

func main() {
	fmt.Println("=== Go Fine-tuning Architecture Demo ===")
	fmt.Println()

	fmt.Println("=== Configuration Demo ===")
	cfg := config.DefaultConfig()
	cfg.Model.VocabSize = 256
	cfg.Model.HiddenSize = 64
	cfg.Model.NumLayers = 2
	cfg.Model.NumHeads = 4
	cfg.Model.IntermediateSize = 256
	cfg.Lora.Enabled = true
	cfg.Lora.Rank = 4
	cfg.Lora.Alpha = 8
	cfg.Lora.TargetModules = []string{"q_proj", "v_proj"}
	cfg.Training.LearningRate = 1e-3
	cfg.Training.LRScheduler = config.SchedulerCosine
	fmt.Printf("%s\n", cfg.String())

	fmt.Println()
	fmt.Println("=== Model Components Demo ===")

	linear := models.NewLinear(10, 5)
	fmt.Printf("Linear layer: %dx%d\n", 10, 5)
	input := models.NewTensor(2, 10)
	for i := range input.Data {
		input.Data[i] = float32(i % 10)
	}
	output := linear.Forward(input)
	fmt.Printf("Linear forward: %v -> %v\n", input.Shape, output.Shape)

	emb := models.NewEmbedding(100, 32)
	fmt.Printf("Embedding layer: vocab=%d, dim=%d\n", 100, 32)
	embInput := models.NewTensor(5)
	for i := range embInput.Data {
		embInput.Data[i] = float32(i * 10)
	}
	embOutput := emb.Forward(embInput)
	fmt.Printf("Embedding forward: %v -> %v\n", embInput.Shape, embOutput.Shape)

	ln := models.NewLayerNorm(64)
	fmt.Printf("LayerNorm: dim=%d\n", 64)
	lnInput := models.NewTensor(8, 64)
	for i := range lnInput.Data {
		lnInput.Data[i] = float32(i % 20)
	}
	lnOutput := ln.Forward(lnInput)
	fmt.Printf("LayerNorm forward: %v -> %v\n", lnInput.Shape, lnOutput.Shape)

	fmt.Println()
	fmt.Println("=== Transformer Model Demo ===")
	modelCfg := config.ModelConfig{
		VocabSize:        100,
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		MaxPosition:      32,
		IntermediateSize: 128,
	}
	model := models.NewTransformerModel(modelCfg)
	fmt.Printf("Transformer created with %d parameters\n", model.NumParameters())

	fmt.Println()
	fmt.Println("=== LoRA Model Demo ===")
	loraCfg := config.LoRAConfig{
		Enabled:       true,
		Rank:          4,
		Alpha:         8,
		Dropout:       0.1,
		TargetModules: []string{"q_proj", "v_proj"},
	}
	loraModel := models.NewLoRAModel(model, loraCfg)
	_ = loraModel
	fmt.Printf("LoRA model created\n")
	fmt.Printf("Trainable parameters: ~%d\n", 2*modelCfg.HiddenSize*loraCfg.Rank*len(loraCfg.TargetModules))

	fmt.Println()
	fmt.Println("=== Optimizer Demo ===")

	sgd := optimizers.NewSGD(0.01, 0.9)
	param := &optimizers.Tensor{Data: []float32{1.0, 2.0, 3.0, 4.0}, Shape: []int{4}}
	sgd.AddParams([]*optimizers.Tensor{param})
	fmt.Printf("SGD - Initial: %v\n", param.Data)
	for i := 0; i < 5; i++ {
		sgd.Step()
	}
	fmt.Printf("SGD - After 5 steps: %v\n", param.Data)

	adam := optimizers.NewAdam(1e-3, 0.9, 0.999, 1e-8)
	adamParam := &optimizers.Tensor{Data: []float32{0.5, 1.0, 1.5, 2.0}, Shape: []int{4}}
	adam.AddParams([]*optimizers.Tensor{adamParam})
	fmt.Printf("Adam - Initial: %v\n", adamParam.Data)
	for i := 0; i < 5; i++ {
		adam.Step()
	}
	fmt.Printf("Adam - After 5 steps: %v\n", adamParam.Data)

	adamw := optimizers.NewAdamW(1e-3, 0.9, 0.999, 1e-8, 0.01)
	adamwParam := &optimizers.Tensor{Data: []float32{1.0, 1.0, 1.0, 1.0}, Shape: []int{4}}
	adamw.AddParams([]*optimizers.Tensor{adamwParam})
	fmt.Printf("AdamW - Initial: %v\n", adamwParam.Data)
	for i := 0; i < 5; i++ {
		adamw.Step()
	}
	fmt.Printf("AdamW - After 5 steps: %v\n", adamwParam.Data)

	lion := optimizers.NewLion(1e-3, 0.9, 0.99)
	lionParam := &optimizers.Tensor{Data: []float32{0.0, 0.0, 0.0, 0.0}, Shape: []int{4}}
	lion.AddParams([]*optimizers.Tensor{lionParam})
	fmt.Printf("Lion - Initial: %v\n", lionParam.Data)
	for i := 0; i < 5; i++ {
		lion.Step()
	}
	fmt.Printf("Lion - After 5 steps: %v\n", lionParam.Data)

	rmsprop := optimizers.NewRMSprop(1e-3, 0.99, 1e-8, 0.9)
	rmsParam := &optimizers.Tensor{Data: []float32{5.0, 5.0, 5.0, 5.0}, Shape: []int{4}}
	rmsprop.AddParams([]*optimizers.Tensor{rmsParam})
	fmt.Printf("RMSprop - Initial: %v\n", rmsParam.Data)
	for i := 0; i < 5; i++ {
		rmsprop.Step()
	}
	fmt.Printf("RMSprop - After 5 steps: %v\n", rmsParam.Data)

	fmt.Println()
	fmt.Println("=== Scheduler Demo ===")

	opt := optimizers.NewAdamW(1e-3, 0.9, 0.999, 1e-8, 0.0)
	scheduler := training.NewScheduler(opt, cfg.Training, 100)
	fmt.Println("Cosine LR Schedule (first 10 steps):")
	for i := 0; i < 10; i++ {
		scheduler.Step()
		fmt.Printf("  Step %2d: LR = %.6f\n", i+1, scheduler.GetCurrentLR())
	}

	fmt.Println()
	fmt.Println("=== Metrics Demo ===")
	metrics := training.NewMetrics()
	metrics.AddTrainLoss(2.5)
	metrics.AddTrainLoss(2.3)
	metrics.AddTrainLoss(2.1)
	metrics.AddEvalLoss(2.4)
	metrics.AddEvalLoss(2.2)
	metrics.AddLearningRate(1e-3)
	metrics.AddStepTime(time.Duration(100) * time.Millisecond)
	metrics.AddStepTime(time.Duration(90) * time.Millisecond)
	fmt.Printf("Average train loss: %.4f\n", metrics.AverageTrainLoss())
	fmt.Printf("Average eval loss: %.4f\n", metrics.AverageEvalLoss())
	fmt.Printf("Average step time: %.4fs\n", metrics.AverageStepTime())

	fmt.Println()
	fmt.Println("=== Training State Demo ===")
	state := training.NewTrainingState(cfg)
	state.AdvanceBatch()
	state.AdvanceBatch()
	state.AdvanceEpoch()
	fmt.Printf("Training state - Epoch: %d, Step: %d, GlobalStep: %d, TokensSeen: %d\n",
		state.Epoch, state.Step, state.GlobalStep, state.TokensSeen)

	fmt.Println()
	fmt.Println("=== Model Wrapper Demo ===")
	fullCfg := *cfg
	fullCfg.Model = config.ModelConfig{
		VocabSize:        256,
		HiddenSize:       32,
		NumLayers:        2,
		NumHeads:         4,
		MaxPosition:      64,
		IntermediateSize: 128,
	}
	wrapper := models.NewModelWrapper(fullCfg)
	fmt.Printf("Model wrapper created with %d parameters\n",
		wrapper.BaseModel.(*models.TransformerModel).NumParameters())
	wrapper.TrainMode()
	wrapper.EvalMode()
	fmt.Printf("Model can switch between train/eval modes\n")

	fmt.Println()
	fmt.Println("=== Summary ===")
	fmt.Println("Components available:")
	fmt.Println("  - Models: Tensor, Linear, Embedding, LayerNorm, Attention, MLP, Transformer, LoRA")
	fmt.Println("  - Optimizers: SGD, Adam, AdamW, Lion, RMSprop")
	fmt.Println("  - Schedulers: Linear, Cosine, Constant with warmup")
	fmt.Println("  - Training: Trainer, Metrics, Scheduler, Evaluator")
	fmt.Println()
	fmt.Println("All components initialized successfully!")
}
