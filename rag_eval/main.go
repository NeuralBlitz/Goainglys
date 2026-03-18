package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"time"
	"unicode"
)

// ============================================
// Similarity Metrics
// ============================================

// JaccardSimilarity computes Jaccard similarity between two strings
func JaccardSimilarity(a, b string) float64 {
	tokensA := tokenize(a)
	tokensB := tokenize(b)

	setA := make(map[string]bool)
	setB := make(map[string]bool)

	for _, t := range tokensA {
		setA[t] = true
	}
	for _, t := range tokensB {
		setB[t] = true
	}

	union := len(setA) + len(setB)
	intersection := 0
	for t := range setA {
		if setB[t] {
			intersection++
		}
	}
	union -= intersection

	if union == 0 {
		return 1.0
	}
	return float64(intersection) / float64(union)
}

// CosineSimilarity computes cosine similarity between two vectors
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
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
		return 0.0
	}

	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

// ============================================
// Text Processing
// ============================================

func tokenize(text string) []string {
	text = strings.ToLower(text)
	var tokens []string
	var current strings.Builder

	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			current.WriteRune(r)
		} else {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

func ngrams(text string, n int) []string {
	tokens := tokenize(text)
	if len(tokens) < n {
		return nil
	}
	var grams []string
	for i := 0; i <= len(tokens)-n; i++ {
		grams = append(grams, strings.Join(tokens[i:i+n], " "))
	}
	return grams
}

// ============================================
// Retrieval Metrics
// ============================================

type RetrievalMetrics struct {
	Recall    float64 `json:"recall"`
	Precision float64 `json:"precision"`
	F1Score   float64 `json:"f1_score"`
	MRR       float64 `json:"mrr"`
	MAP       float64 `json:"map"`
	NDCG      float64 `json:"ndcg"`
}

func CalculateRetrievalMetrics(relevantDocs, retrievedDocs []string, k int) RetrievalMetrics {
	relevantTokens := make(map[string]bool)
	for _, doc := range relevantDocs {
		for _, t := range tokenize(doc) {
			relevantTokens[t] = true
		}
	}

	uniqueRelevantFound := make(map[string]bool)
	truePositives := 0

	for i, doc := range retrievedDocs {
		if i >= k {
			break
		}
		tokens := tokenize(doc)
		found := false
		for _, t := range tokens {
			if relevantTokens[t] {
				uniqueRelevantFound[t] = true
				if !found {
					truePositives++
					found = true
				}
			}
		}
	}

	recall := 0.0
	if len(relevantTokens) > 0 {
		recall = float64(len(uniqueRelevantFound)) / float64(len(relevantTokens))
	}

	precision := 0.0
	if len(retrievedDocs) > 0 && k > 0 {
		precision = float64(truePositives) / float64(min(k, len(retrievedDocs)))
	}

	f1 := 0.0
	if recall+precision > 0 {
		f1 = 2 * recall * precision / (recall + precision)
	}

	// MRR
	mrr := 0.0
	for i, doc := range retrievedDocs {
		if i >= k {
			break
		}
		tokens := tokenize(doc)
		for _, t := range tokens {
			if relevantTokens[t] {
				mrr = 1.0 / float64(i+1)
				break
			}
		}
		if mrr > 0 {
			break
		}
	}

	// MAP
	var precisionAtK []float64
	relevantCount := 0
	for i, doc := range retrievedDocs {
		if i >= k {
			break
		}
		tokens := tokenize(doc)
		found := false
		for _, t := range tokens {
			if relevantTokens[t] {
				found = true
				break
			}
		}
		if found {
			relevantCount++
			precisionAtK = append(precisionAtK, float64(relevantCount)/float64(i+1))
		}
	}
	mapScore := 0.0
	if len(precisionAtK) > 0 {
		for _, p := range precisionAtK {
			mapScore += p
		}
		mapScore /= float64(len(precisionAtK))
	}

	// NDCG
	ndcg := 0.0
	if len(retrievedDocs) > 0 {
		relevanceScores := make([]float64, k)
		for i := 0; i < k && i < len(retrievedDocs); i++ {
			tokens := tokenize(retrievedDocs[i])
			for _, t := range tokens {
				if relevantTokens[t] {
					relevanceScores[i] = 1.0
					break
				}
			}
		}
		ndcg = calculateDCG(relevanceScores)
		idealScores := make([]float64, k)
		for i := 0; i < min(k, len(relevantDocs)); i++ {
			idealScores[i] = 1.0
		}
		idcg := calculateDCG(idealScores)
		if idcg > 0 {
			ndcg /= idcg
		}
	}

	return RetrievalMetrics{
		Recall:    recall,
		Precision: precision,
		F1Score:   f1,
		MRR:       mrr,
		MAP:       mapScore,
		NDCG:      ndcg,
	}
}

func calculateDCG(scores []float64) float64 {
	dcg := 0.0
	for i, score := range scores {
		dcg += score / math.Log2(float64(i+2))
	}
	return dcg
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// ============================================
// Generation Quality Metrics
// ============================================

type GenerationMetrics struct {
	BLEU1   float64 `json:"bleu1"`
	BLEU2   float64 `json:"bleu2"`
	BLEU3   float64 `json:"bleu3"`
	ROUGE1  float64 `json:"rouge1"`
	ROUGE2  float64 `json:"rouge2"`
	ROUGEL  float64 `json:"rougeL"`
	Jaccard float64 `json:"jaccard"`
}

func CalculateBLEU(reference, candidate string, n int) float64 {
	refNGrams := ngrams(reference, n)
	candNGrams := ngrams(candidate, n)

	if len(candNGrams) == 0 {
		return 0.0
	}

	candCount := make(map[string]int)
	for _, gram := range candNGrams {
		candCount[gram]++
	}

	refCount := make(map[string]int)
	for _, gram := range refNGrams {
		refCount[gram]++
	}

	matchCount := 0
	for gram, count := range candCount {
		if refCount[gram] > 0 {
			matchCount += min(count, refCount[gram])
		}
	}

	precision := float64(matchCount) / float64(len(candNGrams))

	// Brevity penalty
	refLen := len(refNGrams)
	candLen := len(candNGrams)
	brevityPenalty := 1.0
	if candLen < refLen {
		brevityPenalty = math.Exp(1.0 - float64(refLen)/float64(candLen))
	}

	return brevityPenalty * precision
}

func CalculateROUGE(reference, candidate string) (rouge1, rouge2, rougeL float64) {
	refTokens := tokenize(reference)
	candTokens := tokenize(candidate)

	// ROUGE-1
	refSet := make(map[string]bool)
	for _, t := range refTokens {
		refSet[t] = true
	}
	match1 := 0
	for _, t := range candTokens {
		if refSet[t] {
			match1++
		}
	}
	rouge1 = float64(match1) / float64(len(candTokens))
	if len(refTokens) > 0 && float64(match1)/float64(len(refTokens)) > rouge1 {
		rouge1 = float64(match1) / float64(len(refTokens))
	}

	// ROUGE-2
	refBigrams := ngrams(reference, 2)
	candBigrams := ngrams(candidate, 2)
	if len(candBigrams) > 0 {
		refBigramSet := make(map[string]bool)
		for _, b := range refBigrams {
			refBigramSet[b] = true
		}
		match2 := 0
		for _, b := range candBigrams {
			if refBigramSet[b] {
				match2++
			}
		}
		rouge2 = float64(match2) / float64(len(candBigrams))
	}

	// ROUGE-L (simplified)
	lcsLen := lcs(refTokens, candTokens)
	rougeL = 0.0
	if len(candTokens) > 0 {
		rougeL = float64(lcsLen) / float64(len(candTokens))
	}

	return
}

func lcs(a, b []string) int {
	m, n := len(a), len(b)
	dp := make([][]int, m+1)
	for i := range dp {
		dp[i] = make([]int, n+1)
	}

	for i := 1; i <= m; i++ {
		for j := 1; j <= n; j++ {
			if a[i-1] == b[j-1] {
				dp[i][j] = dp[i-1][j-1] + 1
			} else {
				dp[i][j] = max(dp[i-1][j], dp[i][j-1])
			}
		}
	}
	return dp[m][n]
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// ============================================
// RAG Evaluation Pipeline
// ============================================

type RAGEvaluation struct {
	Question  string   `json:"question"`
	Reference string   `json:"reference"`
	Retrieved []string `json:"retrieved"`
	Generated string   `json:"generated"`
}

type RAGMetrics struct {
	Question         string            `json:"question"`
	Retrieval        RetrievalMetrics  `json:"retrieval"`
	Generation       GenerationMetrics `json:"generation"`
	ContextRelevancy float64           `json:"context_relevancy"`
	AnswerRelevancy  float64           `json:"answer_relevancy"`
	Faithfulness     float64           `json:"faithfulness"`
}

func EvaluateRAG(eval RAGEvaluation) RAGMetrics {
	metrics := RAGMetrics{
		Question: eval.Question,
	}

	// Retrieval metrics
	relevantDocs := []string{eval.Reference}
	metrics.Retrieval = CalculateRetrievalMetrics(relevantDocs, eval.Retrieved, 5)

	// Generation metrics
	metrics.Generation.BLEU1 = CalculateBLEU(eval.Reference, eval.Generated, 1)
	metrics.Generation.BLEU2 = CalculateBLEU(eval.Reference, eval.Generated, 2)
	metrics.Generation.BLEU3 = CalculateBLEU(eval.Reference, eval.Generated, 3)
	metrics.Generation.ROUGE1, metrics.Generation.ROUGE2, metrics.Generation.ROUGEL = CalculateROUGE(eval.Reference, eval.Generated)
	metrics.Generation.Jaccard = JaccardSimilarity(eval.Reference, eval.Generated)

	// Context relevance
	if len(eval.Retrieved) > 0 {
		contextScore := 0.0
		for _, doc := range eval.Retrieved {
			contextScore += JaccardSimilarity(eval.Question, doc)
		}
		metrics.ContextRelevancy = contextScore / float64(len(eval.Retrieved))
	}

	// Answer relevance
	tokensRef := tokenize(eval.Reference)
	tokensAns := tokenize(eval.Generated)

	vocab := make(map[string]int)
	idx := 0

	for _, t := range tokensRef {
		if _, exists := vocab[t]; !exists {
			vocab[t] = idx
			idx++
		}
	}
	for _, t := range tokensAns {
		if _, exists := vocab[t]; !exists {
			vocab[t] = idx
			idx++
		}
	}

	vecRef := make([]float64, len(vocab))
	vecAns := make([]float64, len(vocab))

	for _, t := range tokensRef {
		vecRef[vocab[t]] += 1.0
	}
	for _, t := range tokensAns {
		vecAns[vocab[t]] += 1.0
	}

	metrics.AnswerRelevancy = CosineSimilarity(vecRef, vecAns)

	// Faithfulness
	faithScore := 0.0
	for _, doc := range eval.Retrieved {
		faithScore += JaccardSimilarity(eval.Generated, doc)
	}
	metrics.Faithfulness = faithScore / float64(len(eval.Retrieved))

	return metrics
}

// ============================================
// Batch Evaluation
// ============================================

type BatchEvaluation struct {
	Evaluations []RAGEvaluation `json:"evaluations"`
}

type BatchMetrics struct {
	AvgRetrieval    RetrievalMetrics  `json:"avg_retrieval"`
	AvgGeneration   GenerationMetrics `json:"avg_generation"`
	AvgContextRel   float64           `json:"avg_context_relevancy"`
	AvgAnswerRel    float64           `json:"avg_answer_relevancy"`
	AvgFaithfulness float64           `json:"avg_faithfulness"`
	AvgOverall      float64           `json:"avg_overall"`
	Metrics         []RAGMetrics      `json:"metrics"`
}

func EvaluateBatch(batch BatchEvaluation) BatchMetrics {
	var metrics []RAGMetrics
	var totalRet RetrievalMetrics
	var totalGen GenerationMetrics
	totalContext, totalAnswer, totalFaith := 0.0, 0.0, 0.0

	for _, eval := range batch.Evaluations {
		m := EvaluateRAG(eval)
		metrics = append(metrics, m)

		totalRet.Recall += m.Retrieval.Recall
		totalRet.Precision += m.Retrieval.Precision
		totalRet.F1Score += m.Retrieval.F1Score
		totalRet.MRR += m.Retrieval.MRR
		totalRet.MAP += m.Retrieval.MAP
		totalRet.NDCG += m.Retrieval.NDCG

		totalGen.BLEU1 += m.Generation.BLEU1
		totalGen.BLEU2 += m.Generation.BLEU2
		totalGen.BLEU3 += m.Generation.BLEU3
		totalGen.ROUGE1 += m.Generation.ROUGE1
		totalGen.ROUGE2 += m.Generation.ROUGE2
		totalGen.ROUGEL += m.Generation.ROUGEL
		totalGen.Jaccard += m.Generation.Jaccard

		totalContext += m.ContextRelevancy
		totalAnswer += m.AnswerRelevancy
		totalFaith += m.Faithfulness
	}

	n := float64(len(batch.Evaluations))
	return BatchMetrics{
		AvgRetrieval: RetrievalMetrics{
			Recall:    totalRet.Recall / n,
			Precision: totalRet.Precision / n,
			F1Score:   totalRet.F1Score / n,
			MRR:       totalRet.MRR / n,
			MAP:       totalRet.MAP / n,
			NDCG:      totalRet.NDCG / n,
		},
		AvgGeneration: GenerationMetrics{
			BLEU1:   totalGen.BLEU1 / n,
			BLEU2:   totalGen.BLEU2 / n,
			BLEU3:   totalGen.BLEU3 / n,
			ROUGE1:  totalGen.ROUGE1 / n,
			ROUGE2:  totalGen.ROUGE2 / n,
			ROUGEL:  totalGen.ROUGEL / n,
			Jaccard: totalGen.Jaccard / n,
		},
		AvgContextRel:   totalContext / n,
		AvgAnswerRel:    totalAnswer / n,
		AvgFaithfulness: totalFaith / n,
		AvgOverall:      (totalRet.F1Score + totalGen.BLEU1 + totalAnswer) / (3 * n),
		Metrics:         metrics,
	}
}

// ============================================
// Export Functions
// ============================================

func ExportJSON(metrics BatchMetrics, filename string) error {
	data, err := json.MarshalIndent(metrics, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filename, data, 0644)
}

func GenerateReport(metrics BatchMetrics) string {
	var report strings.Builder

	report.WriteString("=== RAG Evaluation Report ===\n\n")
	report.WriteString(fmt.Sprintf("Number of evaluations: %d\n", len(metrics.Metrics)))
	report.WriteString("\n")

	report.WriteString("=== Average Retrieval Metrics ===\n")
	report.WriteString(fmt.Sprintf("Recall@5:     %.4f\n", metrics.AvgRetrieval.Recall))
	report.WriteString(fmt.Sprintf("Precision@5:  %.4f\n", metrics.AvgRetrieval.Precision))
	report.WriteString(fmt.Sprintf("F1 Score:     %.4f\n", metrics.AvgRetrieval.F1Score))
	report.WriteString(fmt.Sprintf("MRR:          %.4f\n", metrics.AvgRetrieval.MRR))
	report.WriteString(fmt.Sprintf("MAP:          %.4f\n", metrics.AvgRetrieval.MAP))
	report.WriteString(fmt.Sprintf("NDCG@5:       %.4f\n", metrics.AvgRetrieval.NDCG))
	report.WriteString("\n")

	report.WriteString("=== Average Generation Metrics ===\n")
	report.WriteString(fmt.Sprintf("BLEU-1:       %.4f\n", metrics.AvgGeneration.BLEU1))
	report.WriteString(fmt.Sprintf("BLEU-2:       %.4f\n", metrics.AvgGeneration.BLEU2))
	report.WriteString(fmt.Sprintf("BLEU-3:       %.4f\n", metrics.AvgGeneration.BLEU3))
	report.WriteString(fmt.Sprintf("ROUGE-1:      %.4f\n", metrics.AvgGeneration.ROUGE1))
	report.WriteString(fmt.Sprintf("ROUGE-2:      %.4f\n", metrics.AvgGeneration.ROUGE2))
	report.WriteString(fmt.Sprintf("ROUGE-L:      %.4f\n", metrics.AvgGeneration.ROUGEL))
	report.WriteString(fmt.Sprintf("Jaccard:      %.4f\n", metrics.AvgGeneration.Jaccard))
	report.WriteString("\n")

	report.WriteString("=== RAG Quality ===\n")
	report.WriteString(fmt.Sprintf("Context Relevancy:  %.4f\n", metrics.AvgContextRel))
	report.WriteString(fmt.Sprintf("Answer Relevancy:   %.4f\n", metrics.AvgAnswerRel))
	report.WriteString(fmt.Sprintf("Faithfulness:       %.4f\n", metrics.AvgFaithfulness))
	report.WriteString("\n")

	report.WriteString(fmt.Sprintf("Overall Score:      %.4f\n", metrics.AvgOverall))
	report.WriteString("\n")

	report.WriteString("=== Individual Test Cases ===\n")
	for i, m := range metrics.Metrics {
		report.WriteString(fmt.Sprintf("\nTest %d: %s\n", i+1, m.Question))
		report.WriteString(fmt.Sprintf("  Retrieval F1: %.4f | BLEU-1: %.4f | Answer Relevance: %.4f\n",
			m.Retrieval.F1Score, m.Generation.BLEU1, m.AnswerRelevancy))
	}

	return report.String()
}

// ============================================
// Main Demo
// ============================================

func main() {
	fmt.Println("=== RAG/LLM Evaluation Tool ===")
	fmt.Println("Comprehensive evaluation metrics for Retrieval-Augmented Generation")
	fmt.Println()

	// Create test dataset
	evaluations := []RAGEvaluation{
		{
			Question:  "What is the capital of France?",
			Reference: "Paris is the capital and largest city of France.",
			Retrieved: []string{
				"Paris is the capital of France.",
				"France is a country in Western Europe.",
				"London is the capital of the United Kingdom.",
				"Paris is known for the Eiffel Tower.",
				"The capital of France is Paris.",
			},
			Generated: "Paris is the capital of France, located in Western Europe.",
		},
		{
			Question:  "What is photosynthesis?",
			Reference: "Photosynthesis is the process by which plants convert sunlight into energy.",
			Retrieved: []string{
				"Plants use sunlight to make food.",
				"Photosynthesis converts CO2 and water into glucose.",
				"Chlorophyll is the green pigment in plants.",
				"Plants release oxygen during photosynthesis.",
				"Sunlight is essential for plant growth.",
			},
			Generated: "Photosynthesis is how plants convert sunlight into energy using chlorophyll.",
		},
		{
			Question:  "Who wrote Romeo and Juliet?",
			Reference: "William Shakespeare wrote Romeo and Juliet.",
			Retrieved: []string{
				"Shakespeare was an English playwright.",
				"Romeo and Juliet is a tragedy.",
				"Shakespeare wrote many famous plays.",
				"The play is set in Verona, Italy.",
				"William Shakespeare lived in the 16th century.",
			},
			Generated: "William Shakespeare is the author of Romeo and Juliet, a famous tragedy.",
		},
	}

	batch := BatchEvaluation{Evaluations: evaluations}

	start := time.Now()
	results := EvaluateBatch(batch)
	elapsed := time.Since(start)

	// Generate and display report
	report := GenerateReport(results)
	fmt.Println(report)

	// Export to JSON
	jsonFile := "rag_evaluation.json"
	if err := ExportJSON(results, jsonFile); err != nil {
		fmt.Printf("Error exporting JSON: %v\n", err)
	} else {
		fmt.Printf("Results exported to %s\n", jsonFile)
	}

	fmt.Printf("\nProcessing time: %v\n", elapsed)
	fmt.Println("\n=== Evaluation Complete ===")
}
