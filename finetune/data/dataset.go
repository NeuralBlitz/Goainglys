package data

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math/rand"
	"os"
	"strings"
	"sync"

	"finetune/config"
)

type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
	VocabSize() int
	PadToken() int
	unkToken() int
}

type Dataset struct {
	Tokens [][]int
	Labels [][]int
	Names  []string
}

func NewSimpleDataset(vocabSize, seqLength, numSamples int) *Dataset {
	ds := &Dataset{
		Tokens: make([][]int, numSamples),
		Labels: make([][]int, numSamples),
		Names:  make([]string, numSamples),
	}
	for i := 0; i < numSamples; i++ {
		ds.Tokens[i] = make([]int, seqLength)
		ds.Labels[i] = make([]int, seqLength)
		for j := 0; j < seqLength; j++ {
			ds.Tokens[i][j] = rand.Intn(vocabSize)
			ds.Labels[i][j] = rand.Intn(vocabSize)
		}
		ds.Names[i] = fmt.Sprintf("sample-%d", i)
	}
	return ds
}

type DataLoader struct {
	dataset   *Dataset
	batchSize int
	seqLength int
	shuffle   bool
	dropLast  bool
	index     int
	indices   []int
	mu        sync.Mutex
}

func NewDataLoader(dataset *Dataset, cfg config.DataConfig) *DataLoader {
	return &DataLoader{
		dataset:   dataset,
		batchSize: cfg.MaxLength,
		seqLength: cfg.MaxLength,
		shuffle:   cfg.Shuffle,
		dropLast:  false,
		index:     0,
		indices:   make([]int, len(dataset.Tokens)),
	}
}

func (d *DataLoader) Init() {
	for i := range d.indices {
		d.indices[i] = i
	}
	if d.shuffle {
		d.Shuffle()
	}
}

func (d *DataLoader) Shuffle() {
	for i := len(d.indices) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		d.indices[i], d.indices[j] = d.indices[j], d.indices[i]
	}
}

func (d *DataLoader) Next() ([]int, []int, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.index >= len(d.indices) {
		return nil, nil, false
	}

	idx := d.indices[d.index]
	d.index++

	tokens := d.dataset.Tokens[idx]
	labels := d.dataset.Labels[idx]

	if len(tokens) > d.seqLength {
		tokens = tokens[:d.seqLength]
		labels = labels[:d.seqLength]
	}

	return tokens, labels, true
}

func (d *DataLoader) Reset() {
	d.index = 0
	if d.shuffle {
		d.Shuffle()
	}
}

func (d *DataLoader) Size() int {
	return len(d.dataset.Tokens)
}

func (d *DataLoader) Batch() ([][]int, [][]int, bool) {
	d.mu.Lock()
	defer d.mu.Unlock()

	var batchTokens [][]int
	var batchLabels [][]int

	for len(batchTokens) < d.batchSize && d.index < len(d.indices) {
		idx := d.indices[d.index]
		d.index++

		tokens := d.dataset.Tokens[idx]
		labels := d.dataset.Labels[idx]

		if len(tokens) > d.seqLength {
			tokens = tokens[:d.seqLength]
			labels = labels[:d.seqLength]
		}

		batchTokens = append(batchTokens, tokens)
		batchLabels = append(batchLabels, labels)
	}

	if len(batchTokens) == 0 {
		return nil, nil, false
	}

	return batchTokens, batchLabels, true
}

type SFTDataset struct {
	conversations []Conversation
	tokenizer     Tokenizer
	maxLength     int
}

type Conversation struct {
	Role    string
	Content string
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatMLSample struct {
	Messages []Message `json:"messages"`
}

func LoadChatMLDataset(path string, tokenizer Tokenizer, maxLength int) (*Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var tokens [][]int
	var labels [][]int
	var names []string

	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		line := scanner.Bytes()
		var sample ChatMLSample
		if err := json.Unmarshal(line, &sample); err != nil {
			continue
		}

		seq := BuildChatMLSequence(sample.Messages, tokenizer)
		if len(seq) < 2 {
			continue
		}

		inputTokens := seq[:len(seq)-1]
		labelTokens := seq[1:]

		if len(inputTokens) > maxLength {
			inputTokens = inputTokens[:maxLength]
			labelTokens = labelTokens[:maxLength]
		}

		tokens = append(tokens, inputTokens)
		labels = append(labels, labelTokens)
		names = append(names, fmt.Sprintf("sample_%d", lineNum))
		lineNum++
	}

	return &Dataset{
		Tokens: tokens,
		Labels: labels,
		Names:  names,
	}, nil
}

func BuildChatMLSequence(messages []Message, tokenizer Tokenizer) []int {
	var tokens []int

	systemPrompt := "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
	tokens = append(tokens, tokenizer.Encode(systemPrompt)...)

	for _, msg := range messages {
		role := msg.Role
		if role == "assistant" {
			role = "assistant"
		}

		content := fmt.Sprintf("<|im_start|>%s\n%s<|im_end|>", role, msg.Content)
		tokens = append(tokens, tokenizer.Encode(content)...)
	}

	eosToken := []int{tokenizer.PadToken()}
	tokens = append(tokens, eosToken...)

	return tokens
}

type Preprocessor struct {
	tokenizer  Tokenizer
	maxLength  int
	numWorkers int
}

func NewPreprocessor(tokenizer Tokenizer, maxLength int, numWorkers int) *Preprocessor {
	return &Preprocessor{
		tokenizer:  tokenizer,
		maxLength:  maxLength,
		numWorkers: numWorkers,
	}
}

func (p *Preprocessor) PreprocessText(text string) ([]int, []int) {
	tokens := p.tokenizer.Encode(text)
	labels := make([]int, len(tokens))
	copy(labels, tokens)

	if len(tokens) > p.maxLength {
		tokens = tokens[:p.maxLength]
		labels = labels[:p.maxLength]
	}

	return tokens, labels
}

func (p *Preprocessor) PreprocessBatch(texts []string) ([][]int, [][]int) {
	inputs := make([][]int, len(texts))
	labels := make([][]int, len(texts))

	for i, text := range texts {
		inputs[i], labels[i] = p.PreprocessText(text)
	}

	return inputs, labels
}

type DatasetBuilder struct {
	tokenizer Tokenizer
	maxLength int
	format    string
}

func NewDatasetBuilder(tokenizer Tokenizer, maxLength int, format string) *DatasetBuilder {
	return &DatasetBuilder{
		tokenizer: tokenizer,
		maxLength: maxLength,
		format:    format,
	}
}

func (b *DatasetBuilder) Load(path string) (*Dataset, error) {
	switch b.format {
	case "chatml", "jsonl":
		return LoadChatMLDataset(path, b.tokenizer, b.maxLength)
	case "text":
		return LoadTextDataset(path, b.tokenizer, b.maxLength)
	default:
		return LoadTextDataset(path, b.tokenizer, b.maxLength)
	}
}

func LoadTextDataset(path string, tokenizer Tokenizer, maxLength int) (*Dataset, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	lines := strings.Split(string(content), "\n")
	var tokens [][]int
	var labels [][]int

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		seq := tokenizer.Encode(line)
		if len(seq) < 2 {
			continue
		}

		inputTokens := seq[:len(seq)-1]
		labelTokens := seq[1:]

		if len(inputTokens) > maxLength {
			inputTokens = inputTokens[:maxLength]
			labelTokens = labelTokens[:maxLength]
		}

		tokens = append(tokens, inputTokens)
		labels = append(labels, labelTokens)
	}

	return &Dataset{
		Tokens: tokens,
		Labels: labels,
		Names:  make([]string, len(tokens)),
	}, nil
}

type Collator struct {
	tokenizer  Tokenizer
	maxLength  int
	padTokenID int
}

func NewCollator(tokenizer Tokenizer, maxLength int) *Collator {
	return &Collator{
		tokenizer:  tokenizer,
		maxLength:  maxLength,
		padTokenID: tokenizer.PadToken(),
	}
}

func (c *Collator) Collate(batch [][]int) ([][]int, []int) {
	batchSize := len(batch)
	seqLength := c.maxLength

	inputs := make([][]int, batchSize)
	attentionMasks := make([][]int, batchSize)

	for i, seq := range batch {
		padded := make([]int, seqLength)
		mask := make([]int, seqLength)

		copy(padded, seq)
		for j := 0; j < len(seq) && j < seqLength; j++ {
			mask[j] = 1
		}

		inputs[i] = padded
		attentionMasks[i] = mask
	}

	return inputs, attentionMasks[0]
}

type Split int

const (
	SplitTrain Split = iota
	SplitEval
	SplitTest
)

func SplitDataset(dataset *Dataset, trainRatio float64) (*Dataset, *Dataset) {
	n := len(dataset.Tokens)
	splitIdx := int(float64(n) * trainRatio)

	return &Dataset{
			Tokens: dataset.Tokens[:splitIdx],
			Labels: dataset.Labels[:splitIdx],
			Names:  dataset.Names[:splitIdx],
		}, &Dataset{
			Tokens: dataset.Tokens[splitIdx:],
			Labels: dataset.Labels[splitIdx:],
			Names:  dataset.Names[splitIdx:],
		}
}

func (d *Dataset) Size() int {
	return len(d.Tokens)
}

func (d *Dataset) Get(index int) ([]int, []int) {
	return d.Tokens[index], d.Labels[index]
}
