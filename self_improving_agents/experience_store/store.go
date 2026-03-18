package experience_store

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"self_improving_agents/core"
)

// Store holds successful improvements that can be replayed
type Store struct {
	experiences []core.Experience
	maxSize     int
}

// NewStore creates a new experience store
func NewStore(maxSize int) *Store {
	return &Store{
		experiences: make([]core.Experience, 0, maxSize),
		maxSize:     maxSize,
	}
}

// Store adds an experience to the store, removing the oldest if at capacity
func (s *Store) Store(exp core.Experience) {
	s.experiences = append(s.experiences, exp)
	if len(s.experiences) > s.maxSize {
		s.experiences = s.experiences[1:]
	}
}

// GetAll returns all experiences in the store
func (s *Store) GetAll() []core.Experience {
	exps := make([]core.Experience, len(s.experiences))
	copy(exps, s.experiences)
	return exps
}

// Save persists the store to a file
func (s *Store) Save(path string) error {
	// Create directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	data, err := json.MarshalIndent(s.experiences, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// Load restores the store from a file
func (s *Store) Load(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	var experiences []core.Experience
	if err := json.Unmarshal(data, &experiences); err != nil {
		return err
	}
	s.experiences = experiences
	return nil
}
