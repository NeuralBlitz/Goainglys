package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// VectorDB represents the main vector database
type VectorDB struct {
	vectors   map[string][]float64
	metadata  map[string]map[string]interface{}
	metric    SimilarityMetric
	dimension int
	index     *HNSWIndex
	store     *PersistentStore // optional persistence
}

// NewVectorDB creates a new vector database
func NewVectorDB(dimension int, metric SimilarityMetric) *VectorDB {
	return &VectorDB{
		vectors:   make(map[string][]float64),
		metadata:  make(map[string]map[string]interface{}),
		metric:    metric,
		dimension: dimension,
		index:     NewHNSWIndex(dimension, metric),
	}
}

// NewPersistentVectorDB opens or creates a persistent vector database
func NewPersistentVectorDB(dimension int, metric SimilarityMetric, dataDir string) (*VectorDB, error) {
	store, err := NewPersistentStore(dataDir)
	if err != nil {
		return nil, err
	}

	db := &VectorDB{
		vectors:   make(map[string][]float64),
		metadata:  make(map[string]map[string]interface{}),
		metric:    metric,
		dimension: dimension,
		index:     NewHNSWIndex(dimension, metric),
		store:     store,
	}

	// Recover from snapshot + WAL
	if err := Recover(db, store); err != nil {
		return nil, fmt.Errorf("recover: %w", err)
	}

	// Open WAL for new writes
	if err := store.OpenWAL(); err != nil {
		return nil, err
	}

	return db, nil
}

// Close flushes and closes persistence resources
func (db *VectorDB) Close() error {
	if db.store != nil {
		return db.store.CloseWAL()
	}
	return nil
}

// Snapshot creates a point-in-time snapshot and compacts the WAL
func (db *VectorDB) Snapshot() (string, error) {
	if db.store == nil {
		return "", fmt.Errorf("persistence not enabled")
	}
	path, err := db.store.SaveSnapshot(db)
	if err != nil {
		return "", err
	}
	if err := db.store.CompactWAL(db); err != nil {
		return path, err
	}
	return path, nil
}

// Insert adds a vector to the database
func (db *VectorDB) Insert(id string, vector []float64, metadata map[string]interface{}) error {
	if len(vector) != db.dimension {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", db.dimension, len(vector))
	}

	db.vectors[id] = vector
	if metadata != nil {
		db.metadata[id] = metadata
	}

	db.index.Add(id, vector)

	if db.store != nil {
		if err := db.store.WALAppendInsert(id, vector, metadata); err != nil {
			return fmt.Errorf("WAL append: %w", err)
		}
	}
	return nil
}

// BulkInsert adds multiple vectors at once
func (db *VectorDB) BulkInsert(items []VectorItem) error {
	for _, item := range items {
		if len(item.Vector) != db.dimension {
			return fmt.Errorf("vector dimension mismatch for ID %s", item.ID)
		}

		db.vectors[item.ID] = item.Vector
		if item.Metadata != nil {
			db.metadata[item.ID] = item.Metadata
		}

		db.index.Add(item.ID, item.Vector)
	}
	return nil
}

// Search finds k nearest neighbors to query vector
func (db *VectorDB) Search(query []float64, k int) ([]VectorItem, error) {
	if len(query) != db.dimension {
		return nil, fmt.Errorf("query vector dimension mismatch")
	}

	results, err := db.index.Search(query, k, db.metric)
	if err != nil {
		return nil, err
	}

	// Add metadata to results
	for i := range results {
		if meta, ok := db.metadata[results[i].ID]; ok {
			results[i].Metadata = meta
		}
	}

	return results, nil
}

// SearchWithFilter finds k nearest neighbors with metadata filter
func (db *VectorDB) SearchWithFilter(query []float64, k int, filter func(map[string]interface{}) bool) ([]VectorItem, error) {
	if len(query) != db.dimension {
		return nil, fmt.Errorf("query vector dimension mismatch")
	}

	results, err := db.index.Search(query, k*2, db.metric)
	if err != nil {
		return nil, err
	}

	var filtered []VectorItem
	for _, item := range results {
		if meta, ok := db.metadata[item.ID]; ok && filter(meta) {
			item.Metadata = meta
			filtered = append(filtered, item)
			if len(filtered) == k {
				break
			}
		}
	}

	return filtered, nil
}

// Get retrieves a vector by ID
func (db *VectorDB) Get(id string) ([]float64, map[string]interface{}, error) {
	vector, ok := db.vectors[id]
	if !ok {
		return nil, nil, fmt.Errorf("vector not found: %s", id)
	}

	metadata := db.metadata[id]
	return vector, metadata, nil
}

// Update updates a vector's value and/or metadata
func (db *VectorDB) Update(id string, vector []float64, metadata map[string]interface{}) error {
	if len(vector) != db.dimension {
		return fmt.Errorf("vector dimension mismatch")
	}

	if _, ok := db.vectors[id]; !ok {
		return fmt.Errorf("vector not found: %s", id)
	}

	// For HNSW, we need to lock both DB and index
	// Simple approach: delete and re-insert
	delete(db.vectors, id)
	delete(db.metadata, id)
	db.index.Delete(id)

	db.vectors[id] = vector
	if metadata != nil {
		db.metadata[id] = metadata
	}
	db.index.Add(id, vector)

	return nil
}

// Delete removes a vector from the database
func (db *VectorDB) Delete(id string) error {
	if _, ok := db.vectors[id]; !ok {
		return fmt.Errorf("vector not found: %s", id)
	}

	delete(db.vectors, id)
	delete(db.metadata, id)
	db.index.Delete(id)

	return nil
}

// Count returns the number of vectors in the database
func (db *VectorDB) Count() int {
	return len(db.vectors)
}

// Clear removes all vectors from the database
func (db *VectorDB) Clear() {
	db.vectors = make(map[string][]float64)
	db.metadata = make(map[string]map[string]interface{})
	db.index = NewHNSWIndex(db.dimension, db.metric)
}

// CalculateSimilarity computes similarity between two vectors
func CalculateSimilarity(a, b []float64, metric SimilarityMetric) float64 {
	switch metric {
	case CosineSimilarity:
		return cosineSimilarity(a, b)
	case EuclideanDistance:
		return euclideanDistance(a, b)
	case DotProduct:
		return dotProduct(a, b)
	default:
		return cosineSimilarity(a, b)
	}
}

func cosineSimilarity(a, b []float64) float64 {
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

func euclideanDistance(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func dotProduct(a, b []float64) float64 {
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func randVec(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = rand.Float64()
	}
	return v
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Create a vector database with 128-dimensional vectors
	db := NewVectorDB(128, CosineSimilarity)

	fmt.Println("=== Native Vector Database in Go ===")
	fmt.Println("Created database with 128-dimensional vectors")
	fmt.Println("Similarity metric: Cosine Similarity")

	// Insert some sample vectors
	fmt.Println("\nInserting sample vectors...")
	start := time.Now()

	for i := 0; i < 10000; i++ {
		id := fmt.Sprintf("vec-%d", i)
		meta := map[string]interface{}{
			"category":  fmt.Sprintf("cat-%d", i%10),
			"timestamp": time.Now().Unix(),
		}
		db.Insert(id, randVec(128), meta)
	}

	elapsed := time.Since(start)
	fmt.Printf("Inserted 10,000 vectors in %v\n", elapsed)

	// Search for nearest neighbors
	queryVec := randVec(128)
	start = time.Now()

	results, err := db.Search(queryVec, 10)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	searchElapsed := time.Since(start)
	fmt.Printf("\nSearch completed in %v\n", searchElapsed)
	fmt.Printf("Found %d nearest neighbors:\n", len(results))

	for i, item := range results {
		fmt.Printf("  %d. ID: %s, Score: %.4f\n", i+1, item.ID, item.Score)
	}

	// Test CRUD operations
	fmt.Println("\n=== Testing CRUD Operations ===")

	// Update
	newVector := randVec(128)
	err = db.Update("vec-0", newVector, nil)
	if err != nil {
		fmt.Printf("Update error: %v\n", err)
	} else {
		fmt.Println("✓ Updated vec-0")
	}

	// Get
	vec, meta, err := db.Get("vec-1")
	if err != nil {
		fmt.Printf("Get error: %v\n", err)
	} else {
		fmt.Printf("✓ Retrieved vec-1 (dim=%d, meta=%v)\n", len(vec), len(meta))
	}

	// Delete
	err = db.Delete("vec-2")
	if err != nil {
		fmt.Printf("Delete error: %v\n", err)
	} else {
		fmt.Println("✓ Deleted vec-2")
	}

	fmt.Printf("\nFinal count: %d vectors\n", db.Count())

	// Search with filter
	fmt.Println("\n=== Testing Search with Filter ===")
	// First find a vector from cat-0
	filterResults, err := db.SearchWithFilter(queryVec, 5, func(meta map[string]interface{}) bool {
		cat, ok := meta["category"].(string)
		return ok && cat == "cat-0"
	})
	if err != nil {
		fmt.Printf("Filter search error: %v\n", err)
	} else {
		if len(filterResults) == 0 {
			// Try to get any vector from cat-0 to use as query
			vec0, _, err := db.Get("vec-0")
			if err == nil {
				fmt.Println("Using vec-0 (cat-0) as query for filtered search...")
				filterResults, _ = db.SearchWithFilter(vec0, 5, func(meta map[string]interface{}) bool {
					cat, ok := meta["category"].(string)
					return ok && cat == "cat-0"
				})
			}
		}
		fmt.Printf("Found %d results in category 'cat-0':\n", len(filterResults))
		for i, item := range filterResults {
			cat := "unknown"
			if c, ok := item.Metadata["category"].(string); ok {
				cat = c
			}
			fmt.Printf("  %d. ID: %s, Score: %.4f, Category: %s\n", i+1, item.ID, item.Score, cat)
		}
	}

	// Test multiple metric types
	fmt.Println("\n=== Testing Different Metrics ===")
	metrics := []SimilarityMetric{CosineSimilarity, EuclideanDistance, DotProduct}
	metricNames := []string{"Cosine Similarity", "Euclidean Distance", "Dot Product"}

	for idx, metric := range metrics {
		db2 := NewVectorDB(128, metric)
		// Insert a few vectors
		for i := 0; i < 100; i++ {
			db2.Insert(fmt.Sprintf("v-%d", i), randVec(128), nil)
		}
		qv := randVec(128)
		results, _ := db2.Search(qv, 3)
		fmt.Printf("%s: Found %d neighbors\n", metricNames[idx], len(results))
		if len(results) > 0 {
			fmt.Printf("  Best score: %.4f\n", results[0].Score)
		}
	}

	// Performance comparison: brute force vs HNSW
	fmt.Println("\n=== Performance Comparison ===")
	fmt.Printf("Vector count: %d\n", db.Count())

	// Test brute force search (sample)
	fmt.Println("Running brute force search on 100 random vectors...")
	start = time.Now()

	// Simulate brute force
	matches := 0
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("vec-%d", rand.Intn(db.Count()))
		vec, _, _ := db.Get(id)
		if vec != nil {
			// Calculate distance
			_ = CalculateSimilarity(queryVec, vec, db.metric)
			matches++
		}
	}
	bfTime := time.Since(start)
	fmt.Printf("Brute force: %d calculations in %v\n", matches, bfTime)

	fmt.Printf("\n=== Vector Database Complete ===\n")
}
