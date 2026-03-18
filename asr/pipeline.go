package asr

import (
	"math"
	"math/rand"
)

// ASRPipeline represents the complete ASR pipeline
type ASRPipeline struct {
	Model      *AcousticModel
	Vocabulary []string
	BlankIdx   int
	SampleRate int
	FrameSize  int
	HopSize    int
	NumMFCC    int
	NumFilters int
}

// NewASRPipeline creates a new ASR pipeline
func NewASRPipeline(vocabulary []string) *ASRPipeline {
	// Default configuration
	numMFCC := 13
	numFilters := 26
	hiddenSize := 128

	// Create vocabulary with blank symbol
	fullVocab := make([]string, len(vocabulary)+1)
	fullVocab[0] = "<blank>"
	copy(fullVocab[1:], vocabulary)

	pipeline := &ASRPipeline{
		Vocabulary: fullVocab,
		BlankIdx:   0,
		SampleRate: 16000,
		FrameSize:  400,
		HopSize:    160,
		NumMFCC:    numMFCC,
		NumFilters: numFilters,
		Model: NewAcousticModel(
			numMFCC,        // input size
			hiddenSize,     // hidden size
			len(fullVocab), // output size
		),
	}

	return pipeline
}

// ProcessAudio processes audio through the entire pipeline
func (p *ASRPipeline) ProcessAudio(audio *Audio) (string, error) {
	// Preprocess audio
	processed := p.preprocessAudio(audio)

	// Extract features
	features := p.ExtractFeatures(processed)

	// Run acoustic model
	probs := p.Model.Forward(features)

	// Decode
	tokens := GreedyDecode(probs, p.BlankIdx)

	// Convert to text
	text := p.tokensToText(tokens)

	return text, nil
}

// preprocessAudio preprocesses audio for ASR
func (p *ASRPipeline) preprocessAudio(audio *Audio) *Audio {
	// Resample to 16kHz if needed
	if audio.SampleRate != p.SampleRate {
		audio = audio.Resample(p.SampleRate)
	}

	// Normalize
	audio.Normalize()

	// Apply pre-emphasis
	audio.PreEmphasis(0.97)

	return audio
}

// extractFeatures extracts MFCC features from audio
func (p *ASRPipeline) ExtractFeatures(audio *Audio) [][]float64 {
	mfcc := ComputeMFCC(
		audio.Samples,
		p.SampleRate,
		p.FrameSize,
		p.HopSize,
		p.NumMFCC,
		p.NumFilters,
	)

	return mfcc.Coeffs
}

// tokensToText converts token indices to text
func (p *ASRPipeline) tokensToText(tokens []int) string {
	result := ""
	for _, token := range tokens {
		if token > 0 && token < len(p.Vocabulary) {
			result += p.Vocabulary[token]
		}
	}
	return result
}

// GenerateDummyFeatures generates random features for testing
func GenerateDummyFeatures(numFrames, inputSize int) [][]float64 {
	features := make([][]float64, numFrames)
	for i := 0; i < numFrames; i++ {
		features[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			features[i][j] = rand.NormFloat64()
		}
	}
	return features
}

// Train performs a single training step (simplified)
func (p *ASRPipeline) Train(features [][]float64, targets []int) float64 {
	// Forward pass
	probs := p.Model.Forward(features)

	// Compute CTC loss (simplified)
	loss := 0.0
	for t := 0; t < len(probs); t++ {
		if t < len(targets) {
			loss -= math.Log(probs[t][targets[t]])
		}
	}

	return loss
}

// CreateVocabulary creates a vocabulary from text
func CreateVocabulary(texts []string) []string {
	vocabMap := make(map[string]bool)
	vocab := []string{}

	// Add common characters
	commonChars := "abcdefghijklmnopqrstuvwxyz "
	for _, c := range commonChars {
		vocabMap[string(c)] = true
		vocab = append(vocab, string(c))
	}

	// Add any additional characters from texts
	for _, text := range texts {
		for _, r := range text {
			s := string(r)
			if s != " " && s != "\n" && s != "\t" {
				if _, exists := vocabMap[s]; !exists {
					vocabMap[s] = true
					vocab = append(vocab, s)
				}
			}
		}
	}

	return vocab
}
