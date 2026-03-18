package tokenizer

import (
	"fmt"
	"strings"
	"unicode"
)

type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
	VocabSize() int
	TokenToId(token string) int
	IdToToken(id int) string
}

type BasicTokenizer struct {
	Vocab        map[string]int
	IdToVocab    []string
	UnknownToken string
	PadToken     string
	ClsToken     string
	SepToken     string
	MaskToken    string
	UNK          int
	PAD          int
	CLS          int
	SEP          int
	MASK         int
}

func NewBasicTokenizer(vocabSize int) *BasicTokenizer {
	t := &BasicTokenizer{
		Vocab:        make(map[string]int),
		IdToVocab:    make([]string, vocabSize),
		UnknownToken: "[UNK]",
		PadToken:     "[PAD]",
		ClsToken:     "[CLS]",
		SepToken:     "[SEP]",
		MaskToken:    "[MASK]",
	}

	t.Vocab["[PAD]"] = 0
	t.Vocab["[UNK]"] = 1
	t.Vocab["[CLS]"] = 2
	t.Vocab["[SEP]"] = 3
	t.Vocab["[MASK]"] = 4

	t.PAD = 0
	t.UNK = 1
	t.CLS = 2
	t.SEP = 3
	t.MASK = 4

	t.IdToVocab[0] = "[PAD]"
	t.IdToVocab[1] = "[UNK]"
	t.IdToVocab[2] = "[CLS]"
	t.IdToVocab[3] = "[SEP]"
	t.IdToVocab[4] = "[MASK]"

	nextId := 5
	commonWords := []string{
		"the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
		"it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
		"this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
		"or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
		"so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
	}

	for _, word := range commonWords {
		if nextId < vocabSize {
			t.Vocab[word] = nextId
			t.IdToVocab[nextId] = word
			nextId++
		}
	}

	for i := 0; i < 26 && nextId < vocabSize; i++ {
		for j := 0; j < 26 && nextId < vocabSize; j++ {
			token := string(rune('a'+i)) + string(rune('a'+j))
			t.Vocab[token] = nextId
			t.IdToVocab[nextId] = token
			nextId++
		}
	}

	for i := 0; i < 256 && nextId < vocabSize; i++ {
		token := fmt.Sprintf("##%d", i)
		t.Vocab[token] = nextId
		t.IdToVocab[nextId] = token
		nextId++
	}

	return t
}

func (t *BasicTokenizer) Encode(text string) []int {
	tokens := t.tokenize(text)
	ids := make([]int, len(tokens))
	for i, token := range tokens {
		ids[i] = t.TokenToId(token)
	}
	return ids
}

func (t *BasicTokenizer) tokenize(text string) []string {
	text = strings.ToLower(text)
	text = strings.TrimSpace(text)

	var tokens []string
	var currentToken strings.Builder

	for _, r := range text {
		if unicode.IsSpace(r) {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentToken.Reset()
			}
			continue
		}

		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			currentToken.WriteRune(r)
		} else {
			if currentToken.Len() > 0 {
				tokens = append(tokens, currentToken.String())
				currentTokenizer := new(strings.Builder)
				currentTokenizer.WriteRune(r)
				tokens = append(tokens, currentTokenizer.String())
				currentToken.Reset()
			} else {
				currentToken.WriteRune(r)
			}
		}
	}

	if currentToken.Len() > 0 {
		tokens = append(tokens, currentToken.String())
	}

	return tokens
}

func (t *BasicTokenizer) Decode(tokens []int) string {
	var sb strings.Builder
	for i, id := range tokens {
		if id == t.PAD || id == t.UNK || id == t.CLS || id == t.SEP {
			continue
		}
		token := t.IdToToken(id)
		if strings.HasPrefix(token, "##") {
			sb.WriteString(token[2:])
		} else {
			if i > 0 && sb.Len() > 0 {
				sb.WriteString(" ")
			}
			sb.WriteString(token)
		}
	}
	return sb.String()
}

func (t *BasicTokenizer) VocabSize() int {
	return len(t.Vocab)
}

func (t *BasicTokenizer) TokenToId(token string) int {
	if id, ok := t.Vocab[token]; ok {
		return id
	}
	return t.UNK
}

func (t *BasicTokenizer) IdToToken(id int) string {
	if id >= 0 && id < len(t.IdToVocab) {
		return t.IdToVocab[id]
	}
	return t.UnknownToken
}

func (t *BasicTokenizer) AddTokens(tokens []string) int {
	added := 0
	for _, token := range tokens {
		if _, ok := t.Vocab[token]; !ok {
			t.Vocab[token] = len(t.IdToVocab)
			t.IdToVocab = append(t.IdToVocab, token)
			added++
		}
	}
	return added
}

func (t *BasicTokenizer) BuildInputWithSpecialTokens(inputIds []int) []int {
	return append([]int{t.CLS}, append(inputIds, t.SEP)...)
}

type WordPieceTokenizer struct {
	Tokenizer     *BasicTokenizer
	MaxInputChars int
}

func NewWordPieceTokenizer(tokenizer *BasicTokenizer) *WordPieceTokenizer {
	return &WordPieceTokenizer{
		Tokenizer:     tokenizer,
		MaxInputChars: 200,
	}
}

func (t *WordPieceTokenizer) Encode(text string) []int {
	return t.Tokenizer.Encode(text)
}

func (t *WordPieceTokenizer) Decode(tokens []int) string {
	return t.Tokenizer.Decode(tokens)
}

func (t *WordPieceTokenizer) VocabSize() int {
	return t.Tokenizer.VocabSize()
}

type BPETokenizer struct {
	Vocab     map[string]int
	IdToVocab []string
	Merges    []string
	Tokenizer *BasicTokenizer
}

func NewBPETokenizer(vocabSize int) *BPETokenizer {
	bpe := &BPETokenizer{
		Vocab:     make(map[string]int),
		IdToVocab: make([]string, vocabSize),
		Merges:    make([]string, 0),
		Tokenizer: NewBasicTokenizer(vocabSize),
	}

	for i := 0; i < 256; i++ {
		token := string(rune(i))
		bpe.Vocab[token] = i
		bpe.IdToVocab[i] = token
	}

	return bpe
}

func (t *BPETokenizer) Encode(text string) []int {
	return t.Tokenizer.Encode(text)
}

func (t *BPETokenizer) Decode(tokens []int) string {
	return t.Tokenizer.Decode(tokens)
}

func (t *BPETokenizer) VocabSize() int {
	return len(t.Vocab)
}

func (t *BPETokenizer) TokenToId(token string) int {
	if id, ok := t.Vocab[token]; ok {
		return id
	}
	return 0
}

func (t *BPETokenizer) IdToToken(id int) string {
	if id >= 0 && id < len(t.IdToVocab) {
		return t.IdToVocab[id]
	}
	return ""
}

func CreateSentencePair(text1, text2 string, tokenizer Tokenizer) ([]int, []int, []int) {
	ids1 := tokenizer.Encode(text1)
	ids2 := tokenizer.Encode(text2)

	segmentIds1 := make([]int, len(ids1))
	segmentIds2 := make([]int, len(ids2))

	return ids1, ids2, append(segmentIds1, segmentIds2...)
}

func PadSequence(ids []int, maxLength int, padToken int) []int {
	if len(ids) >= maxLength {
		return ids[:maxLength]
	}
	padding := make([]int, maxLength-len(ids))
	for i := range padding {
		padding[i] = padToken
	}
	return append(ids, padding...)
}

func CreateAttentionMask(ids []int, padToken int) []int {
	mask := make([]int, len(ids))
	for i, id := range ids {
		if id != padToken {
			mask[i] = 1
		}
	}
	return mask
}
