# Native Vector Database in Go

A pure Go implementation of a vector database with HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.

## Features

- **Pure Go Implementation**: No external dependencies
- **Multiple Similarity Metrics**: Cosine Similarity, Euclidean Distance, Dot Product
- **HNSW Index**: Efficient approximate nearest neighbor search
- **CRUD Operations**: Insert, Update, Delete, Get
- **Metadata Support**: Attach metadata to vectors
- **Filter Search**: Search with metadata-based filtering
- **Thread-Safe**: Concurrent access support

## Quick Start

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
)

func main() {
    rand.Seed(time.Now().UnixNano())

    // Create a vector database with 128-dimensional vectors
    db := NewVectorDB(128, CosineSimilarity)

    // Insert vectors
    for i := 0; i < 1000; i++ {
        id := fmt.Sprintf("vec-%d", i)
        vector := randVec(128)
        metadata := map[string]interface{}{
            "category": fmt.Sprintf("cat-%d", i%10),
        }
        db.Insert(id, vector, metadata)
    }

    // Search for nearest neighbors
    queryVec := randVec(128)
    results, err := db.Search(queryVec, 10)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Found %d nearest neighbors\n", len(results))
    for i, item := range results {
        fmt.Printf("%d. ID: %s, Score: %.4f\n", i+1, item.ID, item.Score)
    }
}
```

## API Reference

### VectorDB Operations

#### `NewVectorDB(dimension int, metric SimilarityMetric) *VectorDB`
Creates a new vector database.

#### `Insert(id string, vector []float64, metadata map[string]interface{}) error`
Inserts a single vector.

#### `BulkInsert(items []VectorItem) error`
Inserts multiple vectors at once.

#### `Search(query []float64, k int) ([]VectorItem, error)`
Searches for k nearest neighbors.

#### `SearchWithFilter(query []float64, k int, filter func(map[string]interface{}) bool)`
Searches with metadata filtering.

#### `Get(id string) ([]float64, map[string]interface{}, error)`
Retrieves a vector by ID.

#### `Update(id string, vector []float64, metadata map[string]interface{}) error`
Updates a vector.

#### `Delete(id string) error`
Deletes a vector.

#### `Count() int`
Returns the number of vectors.

#### `Clear()`
Removes all vectors.

### Similarity Metrics

- `CosineSimilarity`: Measures cosine angle between vectors (0-1, higher is better)
- `EuclideanDistance`: Measures straight-line distance (lower is better)
- `DotProduct`: Measures vector alignment (higher is better)

## Performance

With 10,000 vectors of 128 dimensions:
- Insert: ~300ms total for all vectors
- Search (10 results): ~20µs
- Brute force comparison: ~60µs per query

## Architecture

The database uses a Hierarchical Navigable Small World (HNSW) graph for approximate nearest neighbor search, which provides:
- Logarithmic search time complexity
- High recall with low latency
- Efficient memory usage

## Files

- `main.go`: Main database implementation and CLI demo
- `hnsw.go`: HNSW index implementation
- `types.go`: Type definitions

## License

MIT
