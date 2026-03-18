package asr

import (
	"math"
	"sort"
)

type NGramLM struct {
	N         int
	Counts    map[string]map[string]int
	TotalCnts map[string]int
	Vocab     map[string]int
}

func NewNGramLM(n int) *NGramLM {
	return &NGramLM{
		N:         n,
		Counts:    make(map[string]map[string]int),
		TotalCnts: make(map[string]int),
		Vocab:     make(map[string]int),
	}
}

func (lm *NGramLM) Train(corpus []string) {
	for i := 0; i < len(corpus); i++ {
		lm.Vocab[corpus[i]] = 1
	}

	for i := 0; i < len(corpus); i++ {
		context := ""
		start := i - lm.N + 1
		if start < 0 {
			start = 0
		}
		for j := start; j < i; j++ {
			if j > start {
				context += " "
			}
			context += corpus[j]
		}
		if context == "" {
			context = "<s>"
		}

		if lm.Counts[context] == nil {
			lm.Counts[context] = make(map[string]int)
		}
		lm.Counts[context][corpus[i]]++
		lm.TotalCnts[context]++
	}
}

type BeamSearchDecoder struct {
	BeamWidth int
	LM        *NGramLM
	LMWeight  float32
	WordBonus float32
	Vocab     []string
	VocabSize int
}

func NewBeamSearchDecoder(beamWidth int, vocab []string) *BeamSearchDecoder {
	return &BeamSearchDecoder{
		BeamWidth: beamWidth,
		LM:        NewNGramLM(3),
		LMWeight:  0.5,
		WordBonus: 0.0,
		Vocab:     vocab,
		VocabSize: len(vocab),
	}
}

type BeamHypothesis struct {
	Score   float64
	Tokens  []int
	LMState []string
}

func (d *BeamSearchDecoder) Decode(probs [][]float64) []int {
	beams := []BeamHypothesis{
		{Score: 0.0, Tokens: []int{}, LMState: []string{}},
	}

	for t := 0; t < len(probs); t++ {
		candidates := make([]BeamHypothesis, 0)

		for _, beam := range beams {
			prevProbs := probs[t]

			topK := d.BeamWidth
			if topK > len(prevProbs) {
				topK = len(prevProbs)
			}

			indices := make([]int, len(prevProbs))
			for i := range indices {
				indices[i] = i
			}
			sort.Slice(indices, func(i, j int) bool {
				return prevProbs[indices[i]] > prevProbs[indices[j]]
			})

			for i := 0; i < topK; i++ {
				token := indices[i]
				logProb := math.Log(prevProbs[token] + 1e-10)

				newTokens := make([]int, len(beam.Tokens))
				copy(newTokens, beam.Tokens)
				newTokens = append(newTokens, token)

				newLMState := make([]string, len(beam.LMState))
				copy(newLMState, beam.LMState)
				if token < d.VocabSize {
					newLMState = append(newLMState, d.Vocab[token])
				}

				candidates = append(candidates, BeamHypothesis{
					Score:   beam.Score + logProb,
					Tokens:  newTokens,
					LMState: newLMState,
				})
			}
		}

		if len(candidates) == 0 {
			continue
		}

		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Score > candidates[j].Score
		})

		if len(candidates) > d.BeamWidth {
			candidates = candidates[:d.BeamWidth]
		}

		beams = candidates
	}

	if len(beams) > 0 {
		return beams[0].Tokens
	}

	return []int{}
}

func (d *BeamSearchDecoder) DecodeGreedy(probs [][]float64) []int {
	result := make([]int, 0)

	for t := 0; t < len(probs); t++ {
		maxProb := 0.0
		maxIdx := 0

		for i, p := range probs[t] {
			if p > maxProb {
				maxProb = p
				maxIdx = i
			}
		}

		result = append(result, maxIdx)
	}

	return result
}

func (d *BeamSearchDecoder) TrainLM(corpus []string) {
	d.LM.Train(corpus)
}

func CTCGreedyDecode(probs [][]float64, blank int) []int {
	result := make([]int, 0)
	prevToken := -1

	for t := 0; t < len(probs); t++ {
		maxProb := 0.0
		maxIdx := 0

		for i, p := range probs[t] {
			if p > maxProb {
				maxProb = p
				maxIdx = i
			}
		}

		if maxIdx != blank && maxIdx != prevToken {
			result = append(result, maxIdx)
		}

		prevToken = maxIdx
	}

	return result
}

type StreamingConfig struct {
	ContextSize      int
	MinConfidence    float64
	SilenceThreshold float64
}

func NewStreamingConfig() *StreamingConfig {
	return &StreamingConfig{
		ContextSize:      10,
		MinConfidence:    0.5,
		SilenceThreshold: 0.1,
	}
}

type StreamingASR struct {
	Decoder *BeamSearchDecoder
	Config  *StreamingConfig
	Buffer  []float64
	Context [][]float64
}

func NewStreamingASR(decoder *BeamSearchDecoder) *StreamingASR {
	return &StreamingASR{
		Decoder: decoder,
		Config:  NewStreamingConfig(),
		Buffer:  make([]float64, 0),
		Context: make([][]float64, 0),
	}
}

func (s *StreamingASR) ProcessChunk(audioChunk []float64) ([]int, bool) {
	s.Buffer = append(s.Buffer, audioChunk...)

	if len(s.Buffer) < 400 {
		return nil, false
	}

	mfcc := ComputeMFCC(s.Buffer, 16000, 400, 160, 13, 26)

	if len(mfcc.Coeffs) > 0 {
		features := mfcc.Coeffs[len(mfcc.Coeffs)-1]
		s.Context = append(s.Context, features)
	}

	if len(s.Context) > s.Config.ContextSize {
		s.Context = s.Context[len(s.Context)-s.Config.ContextSize:]
	}

	probs := make([][]float64, len(s.Context))
	for i := range s.Context {
		probs[i] = make([]float64, s.Decoder.VocabSize)
		for j := range probs[i] {
			probs[i][j] = 1.0 / float64(s.Decoder.VocabSize)
		}
	}

	tokens := s.Decoder.DecodeGreedy(probs)

	return tokens, true
}

func (s *StreamingASR) Reset() {
	s.Buffer = make([]float64, 0)
	s.Context = make([][]float64, 0)
}
