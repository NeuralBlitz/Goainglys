package main

import (
	"math/rand"
	"sort"
	"sync"
)

// HNSWIndex implements Hierarchical Navigable Small World graph for approximate NN search
type HNSWIndex struct {
	dimension      int
	metric         SimilarityMetric
	maxConnections int
	efConstruction int
	efSearch       int
	m              int
	maxElements    int
	numLayers      int

	nodes     map[string]*HNSWNode
	enterNode *HNSWNode
	pool      *NodePool

	lock sync.RWMutex
}

// HNSWNode represents a node in the HNSW graph
type HNSWNode struct {
	ID        string
	Vector    []float64
	Neighbors map[int][]*HNSWNode // Layer -> neighbors
	Level     int
}

// NodePool manages node allocations
type NodePool struct {
	nodes map[string]*HNSWNode
}

func NewNodePool() *NodePool {
	return &NodePool{
		nodes: make(map[string]*HNSWNode),
	}
}

func (p *NodePool) Get(id string, vector []float64, level int) *HNSWNode {
	node := &HNSWNode{
		ID:        id,
		Vector:    vector,
		Neighbors: make(map[int][]*HNSWNode),
		Level:     level,
	}
	p.nodes[id] = node
	return node
}

func (p *NodePool) Remove(id string) {
	delete(p.nodes, id)
}

// NewHNSWIndex creates a new HNSW index
func NewHNSWIndex(dimension int, metric SimilarityMetric) *HNSWIndex {
	index := &HNSWIndex{
		dimension:      dimension,
		metric:         metric,
		maxConnections: 16,
		efConstruction: 200,
		efSearch:       100,
		m:              16,
		maxElements:    100000,
		numLayers:      1,
		nodes:          make(map[string]*HNSWNode),
		pool:           NewNodePool(),
	}
	return index
}

// Add adds a vector to the HNSW index
func (index *HNSWIndex) Add(id string, vector []float64) {
	index.lock.Lock()
	defer index.lock.Unlock()

	// Determine level
	level := index.randomLevel()

	node := index.pool.Get(id, vector, level)

	if index.enterNode == nil {
		index.enterNode = node
		return
	}

	// Find entry point
	current := index.enterNode
	for l := index.numLayers - 1; l > level; l-- {
		current = index.findClosest(current, vector, l)
	}

	// Insert at each level
	for l := min(level, index.numLayers-1); l >= 0; l-- {
		neighbors := index.findClosestNeighbors(current, vector, l, index.efConstruction)
		node.Neighbors[l] = neighbors

		// Update neighbors
		for _, neighbor := range neighbors {
			// This is simplified - real HNSW would prune edges
			if len(neighbor.Neighbors[l]) < index.maxConnections {
				neighbor.Neighbors[l] = append(neighbor.Neighbors[l], node)
			}
		}
	}

	index.nodes[id] = node
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func (index *HNSWIndex) randomLevel() int {
	// Geometric distribution
	level := 0
	for rand.Float64() < 0.5 && level < 10 {
		level++
	}
	return level
}

func (index *HNSWIndex) findClosest(entry *HNSWNode, query []float64, layer int) *HNSWNode {
	best := entry
	bestDist := CalculateSimilarity(query, entry.Vector, index.metric)

	for {
		foundBetter := false
		for _, neighbor := range entry.Neighbors[layer] {
			dist := CalculateSimilarity(query, neighbor.Vector, index.metric)
			// For similarity, higher is better; for distance, lower is better
			isBetter := index.metric == EuclideanDistance && dist < bestDist
			isBetter = isBetter || (index.metric != EuclideanDistance && dist > bestDist)

			if isBetter {
				best = neighbor
				bestDist = dist
				foundBetter = true
			}
		}
		if !foundBetter {
			break
		}
		entry = best
	}

	return best
}

func (index *HNSWIndex) findClosestNeighbors(entry *HNSWNode, query []float64, layer int, k int) []*HNSWNode {
	// Simplified nearest neighbor search
	neighbors := make([]*HNSWNode, 0)

	// Collect all neighbors in layer
	collectNeighbors(entry, layer, make(map[*HNSWNode]bool), &neighbors)

	// Sort by similarity
	sort.Slice(neighbors, func(i, j int) bool {
		distI := CalculateSimilarity(query, neighbors[i].Vector, index.metric)
		distJ := CalculateSimilarity(query, neighbors[j].Vector, index.metric)
		if index.metric == EuclideanDistance {
			return distI < distJ
		}
		return distI > distJ
	})

	// Return top k
	if len(neighbors) > k {
		neighbors = neighbors[:k]
	}

	return neighbors
}

func collectNeighbors(node *HNSWNode, layer int, visited map[*HNSWNode]bool, results *[]*HNSWNode) {
	if visited[node] {
		return
	}
	visited[node] = true
	*results = append(*results, node)

	if neighbors, ok := node.Neighbors[layer]; ok {
		for _, neighbor := range neighbors {
			collectNeighbors(neighbor, layer, visited, results)
		}
	}
}

// Search finds k nearest neighbors
func (index *HNSWIndex) Search(query []float64, k int, metric SimilarityMetric) ([]VectorItem, error) {
	index.lock.RLock()
	defer index.lock.RUnlock()

	if index.enterNode == nil {
		return []VectorItem{}, nil
	}

	// Find entry point at each layer
	current := index.enterNode
	for l := index.numLayers - 1; l >= 0; l-- {
		current = index.findClosest(current, query, l)
	}

	// Collect all nodes for this layer and find closest k
	allNodes := make([]*HNSWNode, 0)
	visited := make(map[*HNSWNode]bool)
	collectNeighbors(current, 0, visited, &allNodes)

	if len(allNodes) == 0 {
		allNodes = append(allNodes, current)
	}

	// Calculate scores for all nodes
	type scoredNode struct {
		node  *HNSWNode
		score float64
	}
	scored := make([]scoredNode, len(allNodes))
	for i, node := range allNodes {
		scored[i] = scoredNode{
			node:  node,
			score: CalculateSimilarity(query, node.Vector, metric),
		}
	}

	// Sort by score
	sort.Slice(scored, func(i, j int) bool {
		if metric == EuclideanDistance {
			return scored[i].score < scored[j].score
		}
		return scored[i].score > scored[j].score
	})

	// Take top k
	limit := k
	if len(scored) < limit {
		limit = len(scored)
	}

	results := make([]VectorItem, limit)
	for i := 0; i < limit; i++ {
		results[i] = VectorItem{
			ID:     scored[i].node.ID,
			Vector: scored[i].node.Vector,
			Score:  scored[i].score,
		}
	}

	return results, nil
}

// Update updates a vector in the index
func (index *HNSWIndex) Update(id string, vector []float64) {
	// Remove and re-add (simplified) - assumes caller holds lock
	index.DeleteNoLock(id)
	index.AddNoLock(id, vector)
}

// AddNoLock adds without locking (caller must hold lock)
func (index *HNSWIndex) AddNoLock(id string, vector []float64) {
	level := index.randomLevel()
	node := index.pool.Get(id, vector, level)

	if index.enterNode == nil {
		index.enterNode = node
		return
	}

	current := index.enterNode
	for l := index.numLayers - 1; l > level; l-- {
		current = index.findClosest(current, vector, l)
	}

	for l := min(level, index.numLayers-1); l >= 0; l-- {
		neighbors := index.findClosestNeighbors(current, vector, l, index.efConstruction)
		node.Neighbors[l] = neighbors

		for _, neighbor := range neighbors {
			if len(neighbor.Neighbors[l]) < index.maxConnections {
				neighbor.Neighbors[l] = append(neighbor.Neighbors[l], node)
			}
		}
	}

	index.nodes[id] = node
}

// DeleteNoLock removes a vector without locking (caller must hold lock)
func (index *HNSWIndex) DeleteNoLock(id string) {
	node, ok := index.nodes[id]
	if !ok {
		return
	}

	// Remove from neighbors
	for _, neighbors := range node.Neighbors {
		for _, neighbor := range neighbors {
			for i, n := range neighbor.Neighbors[node.Level] {
				if n.ID == id {
					neighbor.Neighbors[node.Level] = append(neighbor.Neighbors[node.Level][:i], neighbor.Neighbors[node.Level][i+1:]...)
					break
				}
			}
		}
	}

	index.pool.Remove(id)
	delete(index.nodes, id)
}

// Delete removes a vector from the index
func (index *HNSWIndex) Delete(id string) {
	index.lock.Lock()
	defer index.lock.Unlock()
	index.DeleteNoLock(id)
}
