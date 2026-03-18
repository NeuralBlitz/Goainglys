package asr

import (
	"math"
	"math/rand"
)

// RNNLayer implements a simple RNN layer
type RNNLayer struct {
	InputSize  int
	HiddenSize int
	Weights    [][]float64
	Hidden     []float64
}

// NewRNNLayer creates a new RNN layer
func NewRNNLayer(inputSize, hiddenSize int) *RNNLayer {
	layer := &RNNLayer{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Weights:    make([][]float64, hiddenSize),
		Hidden:     make([]float64, hiddenSize),
	}

	// Initialize weights with Xavier initialization
	scale := math.Sqrt(2.0 / float64(inputSize+hiddenSize))
	for i := 0; i < hiddenSize; i++ {
		layer.Weights[i] = make([]float64, inputSize+hiddenSize+1) // +1 for bias
		for j := range layer.Weights[i] {
			layer.Weights[i][j] = rand.NormFloat64() * scale
		}
	}

	return layer
}

// Forward performs forward pass
func (l *RNNLayer) Forward(input []float64) []float64 {
	// Concatenate input and previous hidden state
	concat := make([]float64, l.InputSize+l.HiddenSize)
	copy(concat, input)
	copy(concat[l.InputSize:], l.Hidden)

	// Compute new hidden state
	newHidden := make([]float64, l.HiddenSize)
	for i := 0; i < l.HiddenSize; i++ {
		sum := l.Weights[i][len(l.Weights[i])-1] // bias
		for j := 0; j < l.InputSize+l.HiddenSize; j++ {
			sum += l.Weights[i][j] * concat[j]
		}
		newHidden[i] = math.Tanh(sum)
	}

	l.Hidden = newHidden
	return newHidden
}

// Reset clears hidden state
func (l *RNNLayer) Reset() {
	for i := range l.Hidden {
		l.Hidden[i] = 0
	}
}

// LSTMLayer implements a simple LSTM layer
type LSTMLayer struct {
	InputSize      int
	HiddenSize     int
	Wf, Wi, Wo, Wg [][]float64 // Weight matrices
	Hidden         []float64
	Cell           []float64
}

// NewLSTMLayer creates a new LSTM layer
func NewLSTMLayer(inputSize, hiddenSize int) *LSTMLayer {
	layer := &LSTMLayer{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Hidden:     make([]float64, hiddenSize),
		Cell:       make([]float64, hiddenSize),
	}

	// Initialize weight matrices
	// Each matrix: [input + hidden + bias]
	dim := inputSize + hiddenSize + 1
	scale := math.Sqrt(2.0 / float64(inputSize+hiddenSize))

	for i := 0; i < hiddenSize; i++ {
		layer.Wf = append(layer.Wf, make([]float64, dim))
		layer.Wi = append(layer.Wi, make([]float64, dim))
		layer.Wo = append(layer.Wo, make([]float64, dim))
		layer.Wg = append(layer.Wg, make([]float64, dim))

		for j := 0; j < dim; j++ {
			layer.Wf[i][j] = rand.NormFloat64() * scale
			layer.Wi[i][j] = rand.NormFloat64() * scale
			layer.Wo[i][j] = rand.NormFloat64() * scale
			layer.Wg[i][j] = rand.NormFloat64() * scale
		}
	}

	return layer
}

// Forward performs LSTM forward pass
func (l *LSTMLayer) Forward(input []float64) []float64 {
	// Concatenate input and hidden state
	concat := make([]float64, l.InputSize+l.HiddenSize)
	copy(concat, input)
	copy(concat[l.InputSize:], l.Hidden)

	// Compute gate activations
	fForget := make([]float64, l.HiddenSize)
	fInput := make([]float64, l.HiddenSize)
	fOutput := make([]float64, l.HiddenSize)
	fGate := make([]float64, l.HiddenSize)

	for i := 0; i < l.HiddenSize; i++ {
		sumF := l.Wf[i][len(l.Wf[i])-1]
		sumI := l.Wi[i][len(l.Wi[i])-1]
		sumO := l.Wo[i][len(l.Wo[i])-1]
		sumG := l.Wg[i][len(l.Wg[i])-1]

		for j := 0; j < l.InputSize+l.HiddenSize; j++ {
			val := concat[j]
			sumF += l.Wf[i][j] * val
			sumI += l.Wi[i][j] * val
			sumO += l.Wo[i][j] * val
			sumG += l.Wg[i][j] * val
		}

		fForget[i] = 1 / (1 + math.Exp(-sumF)) // sigmoid
		fInput[i] = 1 / (1 + math.Exp(-sumI))
		fOutput[i] = 1 / (1 + math.Exp(-sumO))
		fGate[i] = math.Tanh(sumG)
	}

	// Update cell state
	newCell := make([]float64, l.HiddenSize)
	for i := 0; i < l.HiddenSize; i++ {
		newCell[i] = fForget[i]*l.Cell[i] + fInput[i]*fGate[i]
	}

	// Update hidden state
	newHidden := make([]float64, l.HiddenSize)
	for i := 0; i < l.HiddenSize; i++ {
		newHidden[i] = fOutput[i] * math.Tanh(newCell[i])
	}

	l.Cell = newCell
	l.Hidden = newHidden

	return newHidden
}

// Reset clears LSTM state
func (l *LSTMLayer) Reset() {
	for i := range l.Hidden {
		l.Hidden[i] = 0
		l.Cell[i] = 0
	}
}

// DenseLayer implements a fully connected layer
type DenseLayer struct {
	InputSize  int
	OutputSize int
	Weights    [][]float64
	Bias       []float64
}

// NewDenseLayer creates a new dense layer
func NewDenseLayer(inputSize, outputSize int) *DenseLayer {
	layer := &DenseLayer{
		InputSize:  inputSize,
		OutputSize: outputSize,
		Weights:    make([][]float64, outputSize),
		Bias:       make([]float64, outputSize),
	}

	scale := math.Sqrt(2.0 / float64(inputSize))
	for i := 0; i < outputSize; i++ {
		layer.Weights[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			layer.Weights[i][j] = rand.NormFloat64() * scale
		}
		layer.Bias[i] = 0
	}

	return layer
}

// Forward performs forward pass
func (l *DenseLayer) Forward(input []float64) []float64 {
	output := make([]float64, l.OutputSize)
	for i := 0; i < l.OutputSize; i++ {
		sum := l.Bias[i]
		for j := 0; j < l.InputSize; j++ {
			sum += l.Weights[i][j] * input[j]
		}
		output[i] = sum
	}
	return output
}

// Softmax applies softmax to vector
func Softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := 0.0
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = math.Exp(v - maxVal)
		sum += result[i]
	}

	for i := range result {
		result[i] /= sum
	}

	return result
}

// LogSoftmax applies log softmax
func LogSoftmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}

	sum := 0.0
	for _, v := range x {
		sum += math.Exp(v - maxVal)
	}

	logSum := maxVal + math.Log(sum)
	result := make([]float64, len(x))
	for i, v := range x {
		result[i] = v - logSum
	}

	return result
}

// AcousticModel represents the acoustic model
type AcousticModel struct {
	InputSize   int
	HiddenSize  int
	OutputSize  int
	RNN         *LSTMLayer
	OutputLayer *DenseLayer
}

// NewAcousticModel creates a new acoustic model
func NewAcousticModel(inputSize, hiddenSize, outputSize int) *AcousticModel {
	return &AcousticModel{
		InputSize:   inputSize,
		HiddenSize:  hiddenSize,
		OutputSize:  outputSize,
		RNN:         NewLSTMLayer(inputSize, hiddenSize),
		OutputLayer: NewDenseLayer(hiddenSize, outputSize),
	}
}

// Forward performs forward pass through the model
func (m *AcousticModel) Forward(features [][]float64) [][]float64 {
	numFrames := len(features)
	probabilities := make([][]float64, numFrames)

	m.RNN.Reset()
	for t := 0; t < numFrames; t++ {
		hidden := m.RNN.Forward(features[t])
		output := m.OutputLayer.Forward(hidden)
		probabilities[t] = Softmax(output)
	}

	return probabilities
}

// Predict performs prediction on features
func (m *AcousticModel) Predict(features [][]float64) []int {
	probs := m.Forward(features)
	predictions := make([]int, len(probs))

	for t, prob := range probs {
		maxIdx := 0
		maxVal := prob[0]
		for i, v := range prob {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		predictions[t] = maxIdx
	}

	return predictions
}

// CTCDecode performs CTC (Connectionist Temporal Classification) decoding
func CTCDecode(predictions []int, blankIdx int) []int {
	var result []int
	var prev int = -1

	for _, curr := range predictions {
		if curr != blankIdx && curr != prev {
			result = append(result, curr)
		}
		prev = curr
	}

	return result
}

// GreedyDecode performs greedy decoding with optional CTC
func GreedyDecode(probs [][]float64, blankIdx int) []int {
	predictions := make([]int, len(probs))
	for t, prob := range probs {
		maxIdx := 0
		maxVal := prob[0]
		for i, v := range prob {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		predictions[t] = maxIdx
	}
	return CTCDecode(predictions, blankIdx)
}

// BeamSearchDecode performs beam search decoding
func BeamSearchDecode(probs [][]float64, blankIdx int, beamWidth int, vocabulary []string) string {
	type Beam struct {
		Path  []int
		Score float64
	}

	beams := []Beam{{Path: []int{}, Score: 0.0}}

	for t := 0; t < len(probs); t++ {
		newBeams := []Beam{}
		for _, beam := range beams {
			for i, p := range probs[t] {
				if p < 1e-10 {
					continue
				}

				newPath := make([]int, len(beam.Path))
				copy(newPath, beam.Path)

				if i != blankIdx {
					if len(newPath) == 0 || newPath[len(newPath)-1] != i {
						newPath = append(newPath, i)
					}
				}

				newBeam := Beam{
					Path:  newPath,
					Score: beam.Score + math.Log(p),
				}
				newBeams = append(newBeams, newBeam)
			}
		}

		// Sort by score and keep top beams
		if len(newBeams) > beamWidth {
			newBeams = newBeams[:beamWidth]
		}
		beams = newBeams
	}

	// Convert best beam path to text
	if len(beams) == 0 {
		return ""
	}

	bestPath := beams[0].Path
	result := make([]rune, len(bestPath))
	for i, idx := range bestPath {
		if idx < len(vocabulary) {
			result[i] = rune(vocabulary[idx][0])
		}
	}

	return string(result)
}
