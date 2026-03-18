package main

// SimilarityMetric defines how to calculate distance between vectors
type SimilarityMetric int

const (
	CosineSimilarity SimilarityMetric = iota
	EuclideanDistance
	DotProduct
)

// VectorItem represents a single vector with optional metadata
type VectorItem struct {
	ID       string
	Vector   []float64
	Metadata map[string]interface{}
	Score    float64
}
