package core

import (
	"math"
)

func Matmul(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("matmul requires 2D tensors")
	}
	if a.Shape[1] != b.Shape[0] {
		panic("matrix dimension mismatch")
	}
	out := NewTensor(a.Shape[0], b.Shape[1])
	for i := 0; i < a.Shape[0]; i++ {
		for j := 0; j < b.Shape[1]; j++ {
			sum := float32(0)
			for k := 0; k < a.Shape[1]; k++ {
				sum += a.Data[i*a.Shape[1]+k] * b.Data[k*b.Shape[1]+j]
			}
			out.Data[i*b.Shape[1]+j] = sum
		}
	}
	return out
}

func MatmulTransposeB(a, b *Tensor) *Tensor {
	if len(a.Shape) != 2 || len(b.Shape) != 2 {
		panic("matmul requires 2D tensors")
	}
	if a.Shape[1] != b.Shape[1] {
		panic("matrix dimension mismatch")
	}
	out := NewTensor(a.Shape[0], b.Shape[0])
	for i := 0; i < a.Shape[0]; i++ {
		for j := 0; j < b.Shape[0]; j++ {
			sum := float32(0)
			for k := 0; k < a.Shape[1]; k++ {
				sum += a.Data[i*a.Shape[1]+k] * b.Data[j*b.Shape[1]+k]
			}
			out.Data[i*b.Shape[0]+j] = sum
		}
	}
	return out
}

func Transpose2D(t *Tensor) *Tensor {
	if len(t.Shape) != 2 {
		panic("transpose2d requires 2D tensor")
	}
	out := NewTensor(t.Shape[1], t.Shape[0])
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			out.Data[j*t.Shape[0]+i] = t.Data[i*t.Shape[1]+j]
		}
	}
	return out
}

func ScaledDotProductAttention(query, key, value *Tensor, mask *Tensor, scale float32) *Tensor {
	seqLen := query.Shape[0]
	headDim := query.Shape[1]

	scores := NewTensor(seqLen, seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			sum := float32(0)
			for k := 0; k < headDim; k++ {
				sum += query.Data[i*headDim+k] * key.Data[j*headDim+k]
			}
			scores.Data[i*seqLen+j] = sum * scale
		}
	}

	if mask != nil {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				if mask.Data[i*seqLen+j] == 0 {
					scores.Data[i*seqLen+j] = float32(-1e9)
				}
			}
		}
	}

	attnWeights := Softmax(scores, 1)

	out := NewTensor(seqLen, headDim)
	for i := 0; i < seqLen; i++ {
		for k := 0; k < headDim; k++ {
			sum := float32(0)
			for j := 0; j < seqLen; j++ {
				sum += attnWeights.Data[i*seqLen+j] * value.Data[j*headDim+k]
			}
			out.Data[i*headDim+k] = sum
		}
	}
	return out
}

func CreateCausalMask(seqLen int) *Tensor {
	mask := NewTensor(seqLen, seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i {
				mask.Data[i*seqLen+j] = 1
			} else {
				mask.Data[i*seqLen+j] = 0
			}
		}
	}
	return mask
}

func CreatePaddingMask(lengths []int, maxLen int) *Tensor {
	batchSize := len(lengths)
	mask := NewTensor(batchSize, maxLen)
	for i, length := range lengths {
		for j := 0; j < maxLen; j++ {
			if j < length {
				mask.Data[i*maxLen+j] = 1
			} else {
				mask.Data[i*maxLen+j] = 0
			}
		}
	}
	return mask
}

func CreateAttentionMask(input_ids *Tensor, pad_token_id int) *Tensor {
	batchSize := input_ids.Shape[0]
	seqLen := input_ids.Shape[1]
	mask := NewTensor(batchSize, seqLen)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			if int(input_ids.Data[i*seqLen+j]) == pad_token_id {
				mask.Data[i*seqLen+j] = 0
			} else {
				mask.Data[i*seqLen+j] = 1
			}
		}
	}
	return mask
}

func Create三角矩阵(size int) *Tensor {
	matrix := NewTensor(size, size)
	for i := 0; i < size; i++ {
		for j := 0; j <= i; j++ {
			matrix.Data[i*size+j] = 1
		}
	}
	return matrix
}

type TriangularMask struct {
	Data  []float32
	Shape []int
}

func NewTriangularMask(seqLen int, device string) *TriangularMask {
	data := make([]float32, seqLen*seqLen)
	for i := 0; i < seqLen; i++ {
		for j := 0; j < seqLen; j++ {
			if j <= i {
				data[i*seqLen+j] = 0
			} else {
				data[i*seqLen+j] = float32(-1e9)
			}
		}
	}
	return &TriangularMask{
		Data:  data,
		Shape: []int{seqLen, seqLen},
	}
}

type AttentionMask struct {
	Data  []float32
	Shape []int
}

func NewAttentionMaskFromPadding(paddingMask *Tensor) *AttentionMask {
	batchSize := paddingMask.Shape[0]
	seqLen := paddingMask.Shape[1]
	data := make([]float32, batchSize*seqLen*seqLen)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				idx := (b*seqLen+i)*seqLen + j
				padVal := paddingMask.Data[b*seqLen+i]
				keyPadVal := paddingMask.Data[b*seqLen+j]
				if padVal == 0 || keyPadVal == 0 {
					data[idx] = float32(-1e9)
				} else {
					data[idx] = 0
				}
			}
		}
	}
	return &AttentionMask{
		Data:  data,
		Shape: []int{batchSize, seqLen, seqLen},
	}
}

func MaskedFill(t *Tensor, mask *Tensor, value float32) *Tensor {
	out := t.Copy()
	for i := range mask.Data {
		if mask.Data[i] == 0 {
			out.Data[i] = value
		}
	}
	return out
}

func CrossEntropyLoss(logits, targets *Tensor, ignoreIndex int) float32 {
	if len(logits.Shape) != 2 || len(targets.Shape) != 1 {
		panic("cross entropy requires 2D logits and 1D targets")
	}
	batchSize := logits.Shape[0]
	seqLen := logits.Shape[1]
	loss := float32(0)
	count := 0
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLen; j++ {
			target := int(targets.Data[i*seqLen+j])
			if target == ignoreIndex {
				continue
			}
			maxLogit := float32(-math.MaxFloat32)
			for k := 0; k < logits.Shape[1]; k++ {
				if logits.Data[i*seqLen+k] > maxLogit {
					maxLogit = logits.Data[i*seqLen+k]
				}
			}
			sum := float32(0)
			for k := 0; k < logits.Shape[1]; k++ {
				sum += float32(math.Exp(float64(logits.Data[i*seqLen+k] - maxLogit)))
			}
			logProb := float64(logits.Data[i*seqLen+target]-maxLogit) - math.Log(float64(sum))
			loss -= float32(logProb)
			count++
		}
	}
	if count > 0 {
		loss /= float32(count)
	}
	return loss
}
