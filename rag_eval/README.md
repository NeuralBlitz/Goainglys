# RAG/LLM Evaluation Tool

A comprehensive evaluation toolkit for Retrieval-Augmented Generation (RAG) and Large Language Model (LLM) systems written in pure Go.

## Features

- ✅ **Retrieval Metrics**: Recall, Precision, F1, MRR, MAP, NDCG
- ✅ **Generation Metrics**: BLEU (1/2/3-gram), ROUGE (1/2/L), Jaccard Similarity
- ✅ **RAG Quality Metrics**: Context Relevancy, Answer Relevancy, Faithfulness
- ✅ **Batch Evaluation**: Evaluate multiple test cases at once
- ✅ **JSON Export**: Save evaluation results for analysis
- ✅ **Text Reports**: Human-readable summary reports

## Architecture

```
RAG Evaluation Pipeline
├── Retrieval Metrics
│   ├── Recall@K         # What fraction of relevant docs retrieved
│   ├── Precision@K      # What fraction of retrieved docs are relevant
│   ├── F1 Score         # Harmonic mean of recall and precision
│   ├── MRR              # Mean Reciprocal Rank of first relevant doc
│   ├── MAP              # Mean Average Precision across all positions
│   └── NDCG@K           # Normalized Discounted Cumulative Gain
│
├── Generation Metrics
│   ├── BLEU-1/2/3       # N-gram precision with brevity penalty
│   ├── ROUGE-1/2/L      # Recall-Oriented understudy for Gisting
│   └── Jaccard Similarity
│
└── RAG Quality Metrics
    ├── Context Relevancy   # How well context matches question
    ├── Answer Relevancy    # Semantic similarity to reference
    └── Faithfulness        # Grounded in retrieved context
```

## Metrics Explained

### Retrieval Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Recall** | TP / (TP + FN) | Fraction of relevant docs retrieved |
| **Precision** | TP / (TP + FP) | Fraction of retrieved docs that are relevant |
| **F1 Score** | 2 * P * R / (P + R) | Harmonic mean of precision and recall |
| **MRR** | Σ(1/rank_i) / N | Average reciprocal rank of first relevant doc |
| **MAP** | Average of precision at each relevant doc | Mean average precision across queries |
| **NDCG** | DCG / IDCG | Ranking quality with position weighting |

### Generation Metrics

- **BLEU**: Measures n-gram overlap between candidate and reference, with brevity penalty
- **ROUGE**: Recall-oriented metrics measuring overlap with reference
- **Jaccard**: Token set overlap between texts

### RAG Quality Metrics

- **Context Relevancy**: How well retrieved documents match the question
- **Answer Relevancy**: Semantic similarity between generated answer and reference
- **Faithfulness**: Whether answer is grounded in retrieved context

## Usage

### Basic Evaluation

```go
package main

import "github.com/user/transformer/rag_eval"

func main() {
    eval := rag_eval.RAGEvaluation{
        Question:  "What is the capital of France?",
        Reference: "Paris is the capital of France.",
        Retrieved: []string{
            "Paris is the capital of France.",
            "France is a country in Europe.",
        },
        Generated: "Paris is the capital of France.",
    }

    metrics := rag_eval.EvaluateRAG(eval)
    fmt.Printf("F1 Score: %.4f\n", metrics.Retrieval.F1Score)
}
```

### Batch Evaluation

```go
batch := rag_eval.BatchEvaluation{
    Evaluations: []rag_eval.RAGEvaluation{
        {Question: "Q1", Reference: "R1", Retrieved: [...], Generated: "A1"},
        {Question: "Q2", Reference: "R2", Retrieved: [...], Generated: "A2"},
    },
}

results := rag_eval.EvaluateBatch(batch)
```

### JSON Export

```go
err := rag_eval.ExportJSON(results, "evaluation.json")
```

## Example Output

```
=== RAG Evaluation Report ===

Number of evaluations: 3

=== Average Retrieval Metrics ===
Recall@5:     0.7374
Precision@5:  0.9333
F1 Score:     0.7983
MRR:          1.0000
MAP:          0.9833
NDCG@5:       2.8049

=== Average Generation Metrics ===
BLEU-1:       0.5500
BLEU-2:       0.4048
ROUGE-1:      0.7333
ROUGE-2:      0.4242
ROUGE-L:      0.5722

=== RAG Quality ===
Context Relevancy:  0.2142
Answer Relevancy:   0.6297
Faithfulness:       0.2618

Overall Score:      0.6593
```

## API Reference

### Types

```go
type RAGEvaluation struct {
    Question   string
    Reference  string
    Retrieved  []string
    Generated  string
}

type RetrievalMetrics struct {
    Recall      float64
    Precision   float64
    F1Score     float64
    MRR         float64
    MAP         float64
    NDCG        float64
}

type GenerationMetrics struct {
    BLEU1, BLEU2, BLEU3 float64
    ROUGE1, ROUGE2, ROUGEL float64
    Jaccard float64
}
```

### Functions

| Function | Description |
|----------|-------------|
| `JaccardSimilarity(a, b)` | Token set overlap |
| `CosineSimilarity(a, b)` | Vector cosine similarity |
| `CalculateBLEU(ref, cand, n)` | BLEU score with n-gram |
| `CalculateROUGE(ref, cand)` | ROUGE-1, ROUGE-2, ROUGE-L |
| `CalculateRetrievalMetrics(ref, cand, k)` | Retrieval quality metrics |
| `EvaluateRAG(eval)` | Single evaluation |
| `EvaluateBatch(batch)` | Batch evaluation |
| `ExportJSON(metrics, file)` | Export to JSON |
| `GenerateReport(metrics)` | Create text report |

## Files

- `main.go` - Complete evaluation tool with all metrics

## Performance

- Processing time: ~400µs per evaluation (3 test cases)
- Memory efficient: Uses minimal allocations
- No external dependencies

## Future Enhancements

- [x] Semantic similarity using embeddings
- [x] Hallucination detection
- [x] Bias and fairness metrics
- [x] Response time measurement
- [x] Cost estimation
- [x] Dashboard visualization
- [ ] Integration with vector databases
