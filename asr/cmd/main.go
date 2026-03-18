package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/user/transformer/asr"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	fmt.Println("=== Go ASR Library Demo ===")
	fmt.Println("Automatic Speech Recognition in Pure Go")
	fmt.Println()

	// Create vocabulary (simplified English alphabet)
	vocabulary := []string{
		"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
		"k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
		"u", "v", "w", "x", "y", "z",
	}

	// Create ASR pipeline
	pipeline := asr.NewASRPipeline(vocabulary)
	fmt.Printf("Created ASR pipeline with vocabulary size: %d\n", len(vocabulary))
	fmt.Println()

	// Demo 1: Feature Extraction
	fmt.Println("=== Demo 1: Feature Extraction ===")
	sampleRate := 16000
	numFrames := 100
	numMFCC := pipeline.NumMFCC

	// Generate dummy audio samples
	dummySamples := make([]float64, numFrames*pipeline.HopSize)
	for i := range dummySamples {
		dummySamples[i] = rand.NormFloat64() * 0.1
	}

	dummyAudio := &asr.Audio{
		SampleRate: sampleRate,
		Samples:    dummySamples,
		Channels:   1,
		Duration:   float64(len(dummySamples)) / float64(sampleRate),
	}

	// Extract features
	features := pipeline.ExtractFeatures(dummyAudio)
	fmt.Printf("Extracted %d frames, %d MFCC coefficients per frame\n", len(features), numMFCC)
	fmt.Printf("First frame features (first 5 coefficients): ")
	for i := 0; i < 5 && i < numMFCC; i++ {
		fmt.Printf("%.3f ", features[0][i])
	}
	fmt.Println()
	fmt.Println()

	// Demo 2: Acoustic Model
	fmt.Println("=== Demo 2: Acoustic Model ===")
	model := pipeline.Model
	fmt.Printf("Model: LSTM hidden=%d, output=%d\n", model.HiddenSize, model.OutputSize)
	fmt.Printf("Input MFCC features: %d\n", model.InputSize)
	fmt.Println()

	// Forward pass
	start := time.Now()
	probs := model.Forward(features)
	elapsed := time.Since(start)
	fmt.Printf("Forward pass completed in %v\n", elapsed)
	fmt.Printf("Output probability shape: %d frames x %d classes\n", len(probs), len(probs[0]))
	fmt.Printf("First frame top probabilities: ")
	for i := 0; i < 3 && i < len(probs[0]); i++ {
		fmt.Printf("%.4f ", probs[0][i])
	}
	fmt.Println()
	fmt.Println()

	// Demo 3: Decoding
	fmt.Println("=== Demo 3: Decoding ===")
	start = time.Now()
	tokens := asr.GreedyDecode(probs, 0)
	elapsed = time.Since(start)
	fmt.Printf("Decoding completed in %v\n", elapsed)
	fmt.Printf("Decoded tokens count: %d\n", len(tokens))
	fmt.Println()

	// Demo 4: Training
	fmt.Println("=== Demo 4: Training (Simplified) ===")
	// Create dummy targets
	targets := make([]int, len(features))
	for i := range targets {
		targets[i] = rand.Intn(len(vocabulary)) + 1 // +1 for blank
	}

	// Single training step
	loss := pipeline.Train(features, targets)
	fmt.Printf("Training step completed\n")
	fmt.Printf("Loss: %.4f\n", loss)
	fmt.Println()

	// Demo 5: Performance Analysis
	fmt.Println("=== Demo 5: Performance Analysis ===")
	fmt.Println("Processing different audio lengths:")

	testLengths := []int{50, 100, 200, 500}
	for _, numFrames := range testLengths {
		testSamples := make([]float64, numFrames*pipeline.HopSize)
		for i := range testSamples {
			testSamples[i] = rand.NormFloat64() * 0.05
		}
		testAudio := &asr.Audio{
			SampleRate: sampleRate,
			Samples:    testSamples,
			Channels:   1,
		}

		start = time.Now()
		features := pipeline.ExtractFeatures(testAudio)
		_ = pipeline.Model.Forward(features)
		elapsed = time.Since(start)

		duration := float64(numFrames*pipeline.HopSize) / float64(sampleRate)
		fmt.Printf("  %d frames (%.2fs): %v (%.2fx realtime)\n",
			numFrames, duration, elapsed, duration/elapsed.Seconds())
	}

	fmt.Println()
	fmt.Println("=== ASR Library Demo Complete ===")
}
