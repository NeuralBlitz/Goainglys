package main

import (
	"math"
	"sort"
	"strings"
)

type SemanticMetrics struct {
	EmbeddingSimilarity float64
	HallucinationScore  float64
	BiasScore           float64
	FairnessScore       float64
	ResponseTime        float64
	CostEstimation      float64
}

type EmbeddingProvider interface {
	GetEmbedding(text string) ([]float64, error)
	GetEmbeddings(texts []string) ([][]float64, error)
}

type SimpleEmbeddingProvider struct {
	Dimension int
}

func NewSimpleEmbeddingProvider(dim int) *SimpleEmbeddingProvider {
	return &SimpleEmbeddingProvider{Dimension: dim}
}

func (p *SimpleEmbeddingProvider) GetEmbedding(text string) ([]float64, error) {
	words := strings.Fields(strings.ToLower(text))
	embedding := make([]float64, p.Dimension)

	hash := 0
	for _, word := range words {
		for i, ch := range word {
			hash = hash*31 + int(ch) + i
		}
	}

	for i := 0; i < p.Dimension; i++ {
		embedding[i] = math.Sin(float64(hash+i)) * math.Cos(float64(hash-i))
	}

	norm := 0.0
	for _, v := range embedding {
		norm += v * v
	}
	norm = math.Sqrt(norm)
	if norm > 0 {
		for i := range embedding {
			embedding[i] /= norm
		}
	}

	return embedding, nil
}

func (p *SimpleEmbeddingProvider) GetEmbeddings(texts []string) ([][]float64, error) {
	result := make([][]float64, len(texts))
	for i, text := range texts {
		emb, err := p.GetEmbedding(text)
		if err != nil {
			return nil, err
		}
		result[i] = emb
	}
	return result, nil
}

func CosineSimilarityEmbedding(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}

	dot := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func SemanticSimilarity(reference, candidate string, provider EmbeddingProvider) (float64, error) {
	refEmb, err := provider.GetEmbedding(reference)
	if err != nil {
		return 0, err
	}

	candEmb, err := provider.GetEmbedding(candidate)
	if err != nil {
		return 0, err
	}

	return CosineSimilarityEmbedding(refEmb, candEmb), nil
}

func CalculateEmbeddingSimilarity(retrieved []string, generated string, provider EmbeddingProvider) float64 {
	genEmb, err := provider.GetEmbedding(generated)
	if err != nil {
		return 0
	}

	var totalSim float64
	for _, retText := range retrieved {
		retEmb, err := provider.GetEmbedding(retText)
		if err != nil {
			continue
		}
		totalSim += CosineSimilarityEmbedding(retEmb, genEmb)
	}

	if len(retrieved) == 0 {
		return 0
	}

	return totalSim / float64(len(retrieved))
}

func DetectHallucination(retrieved []string, generated string, provider EmbeddingProvider, threshold float64) (float64, bool) {
	similarity := CalculateEmbeddingSimilarity(retrieved, generated, provider)
	isHallucination := similarity < threshold
	return 1.0 - similarity, isHallucination
}

func CalculateBiasScore(generated string, sensitiveTerms []string) float64 {
	generatedLower := strings.ToLower(generated)

	hits := 0
	for _, term := range sensitiveTerms {
		if strings.Contains(generatedLower, strings.ToLower(term)) {
			hits++
		}
	}

	if len(sensitiveTerms) == 0 {
		return 0
	}

	return float64(hits) / float64(len(sensitiveTerms))
}

func CalculateFairnessScore(outputs []string, protectedAttributes []string) float64 {
	if len(outputs) == 0 || len(protectedAttributes) == 0 {
		return 1.0
	}

	var scores []float64
	for _, output := range outputs {
		scores = append(scores, 1.0-CalculateBiasScore(output, protectedAttributes))
	}

	var avg float64
	for _, s := range scores {
		avg += s
	}
	avg /= float64(len(scores))

	var variance float64
	for _, s := range scores {
		diff := s - avg
		variance += diff * diff
	}
	variance /= float64(len(scores))

	return 1.0 - math.Sqrt(variance)
}

type ResponseTimeMetrics struct {
	FirstTokenLatency float64
	TotalLatency      float64
	TokensPerSecond   float64
	TimeToFirstToken  float64
	TimePerToken      float64
}

func CalculateResponseTimeMetrics(totalTime float64, numTokens int) ResponseTimeMetrics {
	if numTokens == 0 {
		return ResponseTimeMetrics{
			TotalLatency:    totalTime,
			TokensPerSecond: 0,
			TimePerToken:    0,
		}
	}

	return ResponseTimeMetrics{
		TotalLatency:     totalTime,
		TokensPerSecond:  float64(numTokens) / totalTime,
		TimePerToken:     totalTime / float64(numTokens),
		TimeToFirstToken: totalTime * 0.1,
	}
}

type CostConfig struct {
	InputCostPer1K  float64
	OutputCostPer1K float64
}

func EstimateCost(inputTokens, outputTokens int, config CostConfig) float64 {
	inputCost := float64(inputTokens) / 1000.0 * config.InputCostPer1K
	outputCost := float64(outputTokens) / 1000.0 * config.OutputCostPer1K
	return inputCost + outputCost
}

func CalculateRAGQualityScore(retrievalRecall, retrievalPrecision, generationRelevance, faithfulness, semanticSim float64) float64 {
	weights := map[string]float64{
		"retrieval":    0.25,
		"generation":   0.25,
		"faithfulness": 0.25,
		"semantic":     0.25,
	}

	return weights["retrieval"]*retrievalRecall +
		weights["generation"]*generationRelevance +
		weights["faithfulness"]*faithfulness +
		weights["semantic"]*semanticSim
}

type RankingMetric struct {
	Query            string
	Retrieved        []string
	Relevance        []int
	K                int
	NDCG             float64
	AveragePrecision float64
}

func CalculateNDCG(retrieved []string, relevance []int, k int) float64 {
	if k <= 0 || len(retrieved) == 0 || len(relevance) == 0 {
		return 0
	}

	actualK := k
	if actualK > len(retrieved) {
		actualK = len(retrieved)
	}

	dcg := 0.0
	for i := 0; i < actualK; i++ {
		rel := 0
		if i < len(relevance) {
			rel = relevance[i]
		}
		dcg += float64(rel) / math.Log2(float64(i+2))
	}

	sortedRelevance := make([]int, len(relevance))
	copy(sortedRelevance, relevance)
	sort.Slice(sortedRelevance, func(i, j int) bool {
		return sortedRelevance[i] > sortedRelevance[j]
	})

	idcg := 0.0
	for i := 0; i < actualK; i++ {
		idcg += float64(sortedRelevance[i]) / math.Log2(float64(i+2))
	}

	if idcg == 0 {
		return 0
	}

	return dcg / idcg
}

func CalculateAveragePrecision(relevance []int) float64 {
	if len(relevance) == 0 {
		return 0
	}

	var precisionSum float64
	relevantCount := 0

	for i, rel := range relevance {
		if rel > 0 {
			relevantCount++
			precisionSum += float64(relevantCount) / float64(i+1)
		}
	}

	if relevantCount == 0 {
		return 0
	}

	return precisionSum / float64(relevantCount)
}
