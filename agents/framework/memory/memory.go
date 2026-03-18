package memory

import (
	"strings"
	"sync"
	"time"

	"agents/framework/core"
)

type ConversationMemory struct {
	messages []core.Message
	maxSize  int
	mu       sync.RWMutex
}

func NewConversationMemory(maxSize int) *ConversationMemory {
	return &ConversationMemory{
		messages: make([]core.Message, 0),
		maxSize:  maxSize,
	}
}

func (m *ConversationMemory) Add(msg core.Message) {
	m.mu.Lock()
	defer m.mu.Unlock()

	m.messages = append(m.messages, msg)
	if len(m.messages) > m.maxSize && m.maxSize > 0 {
		m.messages = m.messages[len(m.messages)-m.maxSize:]
	}
}

func (m *ConversationMemory) GetMessages(n int) []core.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if n <= 0 || n > len(m.messages) {
		result := make([]core.Message, len(m.messages))
		copy(result, m.messages)
		return result
	}

	result := make([]core.Message, n)
	copy(result, m.messages[len(m.messages)-n:])
	return result
}

func (m *ConversationMemory) Search(query string, limit int) []core.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var results []core.Message
	query = strings.ToLower(query)

	for i := len(m.messages) - 1; i >= 0 && len(results) < limit; i-- {
		if strings.Contains(strings.ToLower(m.messages[i].Content), query) {
			results = append(results, m.messages[i])
		}
	}

	return results
}

func (m *ConversationMemory) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = make([]core.Message, 0)
}

func (m *ConversationMemory) Summary() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.messages) == 0 {
		return "No messages in memory."
	}

	var builder strings.Builder
	builder.WriteString("Recent conversation:\n")

	start := 0
	if len(m.messages) > 5 {
		start = len(m.messages) - 5
	}
	for _, msg := range m.messages[start:] {
		builder.WriteString(msg.Role.String() + ": " + msg.Content + "\n")
	}

	return builder.String()
}

func (m *ConversationMemory) Count() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.messages)
}

type VectorMemory struct {
	entries []MemoryEntry
	index   map[string]int
	maxSize int
	mu      sync.RWMutex
}

type MemoryEntry struct {
	ID        string
	Content   string
	Metadata  map[string]any
	Timestamp time.Time
	Embedding []float32
}

func NewVectorMemory(maxSize int) *VectorMemory {
	return &VectorMemory{
		entries: make([]MemoryEntry, 0),
		index:   make(map[string]int),
		maxSize: maxSize,
	}
}

func (m *VectorMemory) Add(entry MemoryEntry) {
	m.mu.Lock()
	defer m.mu.Unlock()

	entry.ID = core.GenerateID()
	entry.Timestamp = time.Now()

	if len(m.entries) >= m.maxSize && m.maxSize > 0 {
		oldest := m.entries[0]
		delete(m.index, oldest.ID)
		m.entries = m.entries[1:]
	}

	m.entries = append(m.entries, entry)
	m.index[entry.ID] = len(m.entries) - 1
}

func (m *VectorMemory) Search(query string, limit int) []core.Message {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var results []core.Message
	query = strings.ToLower(query)

	for i := len(m.entries) - 1; i >= 0 && len(results) < limit; i-- {
		if strings.Contains(strings.ToLower(m.entries[i].Content), query) {
			results = append(results, core.Message{
				ID:        m.entries[i].ID,
				Content:   m.entries[i].Content,
				Timestamp: m.entries[i].Timestamp,
			})
		}
	}

	return results
}

func (m *VectorMemory) GetRecent(n int) []MemoryEntry {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if n <= 0 || n > len(m.entries) {
		result := make([]MemoryEntry, len(m.entries))
		copy(result, m.entries)
		return result
	}

	result := make([]MemoryEntry, n)
	copy(result, m.entries[len(m.entries)-n:])
	return result
}

func (m *VectorMemory) Clear() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = make([]MemoryEntry, 0)
	m.index = make(map[string]int)
}

func (m *VectorMemory) Summary() string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	return "Vector memory with " + string(rune(len(m.entries))) + " entries"
}

type SlidingWindowMemory struct {
	*ConversationMemory
	windowSize int
}

func NewSlidingWindowMemory(windowSize int) *SlidingWindowMemory {
	return &SlidingWindowMemory{
		ConversationMemory: NewConversationMemory(0),
		windowSize:         windowSize,
	}
}

func (m *SlidingWindowMemory) GetMessages(n int) []core.Message {
	if n <= 0 {
		n = m.windowSize
	}
	return m.ConversationMemory.GetMessages(n)
}

type BufferedMemory struct {
	*ConversationMemory
	buffer    []core.Message
	flushSize int
}

func NewBufferedMemory(flushSize int) *BufferedMemory {
	return &BufferedMemory{
		ConversationMemory: NewConversationMemory(0),
		buffer:             make([]core.Message, 0),
		flushSize:          flushSize,
	}
}

func (m *BufferedMemory) Add(msg core.Message) {
	m.buffer = append(m.buffer, msg)
	if len(m.buffer) >= m.flushSize {
		m.Flush()
	}
}

func (m *BufferedMemory) Flush() {
	for _, msg := range m.buffer {
		m.ConversationMemory.Add(msg)
	}
	m.buffer = make([]core.Message, 0)
}

func (m *BufferedMemory) GetBufferedMessages() []core.Message {
	result := make([]core.Message, len(m.buffer))
	copy(result, m.buffer)
	return result
}
