package core

import (
	"math"
)

func Add(a, b *Tensor) *Tensor {
	if len(a.Shape) == 1 && len(b.Shape) == 1 && a.Shape[0] == b.Shape[0] {
		out := NewTensor(a.Shape[0])
		for i := range a.Data {
			out.Data[i] = a.Data[i] + b.Data[i]
		}
		return out
	}
	if len(a.Shape) == 2 && len(b.Shape) == 2 {
		out := NewTensor(a.Shape[0], a.Shape[1])
		for i := range a.Data {
			out.Data[i] = a.Data[i] + b.Data[i]
		}
		return out
	}
	if len(a.Shape) == 2 && len(b.Shape) == 1 {
		out := NewTensor(a.Shape[0], a.Shape[1])
		for i := 0; i < a.Shape[0]; i++ {
			for j := 0; j < a.Shape[1]; j++ {
				out.Data[i*a.Shape[1]+j] = a.Data[i*a.Shape[1]+j] + b.Data[j]
			}
		}
		return out
	}
	if len(a.Shape) == 3 && len(b.Shape) == 3 {
		out := NewTensor(a.Shape[0], a.Shape[1], a.Shape[2])
		for i := range a.Data {
			out.Data[i] = a.Data[i] + b.Data[i]
		}
		return out
	}
	if len(a.Shape) == 3 && (len(b.Shape) == 2 || len(b.Shape) == 1) {
		out := NewTensor(a.Shape[0], a.Shape[1], a.Shape[2])
		for b2 := 0; b2 < a.Shape[0]; b2++ {
			for i := 0; i < a.Shape[1]; i++ {
				for j := 0; j < a.Shape[2]; j++ {
					var bVal float32
					if len(b.Shape) == 2 {
						bVal = b.Data[i*b.Shape[1]+j]
					} else {
						bVal = b.Data[j]
					}
					out.Data[(b2*a.Shape[1]+i)*a.Shape[2]+j] = a.Data[(b2*a.Shape[1]+i)*a.Shape[2]+j] + bVal
				}
			}
		}
		return out
	}
	panic("unsupported add shapes")
}

func Sub(a, b *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] - b.Data[i]
	}
	return out
}

func Mul(a, b *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * b.Data[i]
	}
	return out
}

func Div(a, b *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] / b.Data[i]
	}
	return out
}

func Neg(a *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = -a.Data[i]
	}
	return out
}

func Abs(a *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = float32(math.Abs(float64(a.Data[i])))
	}
	return out
}

func Square(a *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * a.Data[i]
	}
	return out
}

func Sqrt(a *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = float32(math.Sqrt(float64(a.Data[i])))
	}
	return out
}

func Pow(a *Tensor, exp float32) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = float32(math.Pow(float64(a.Data[i]), float64(exp)))
	}
	return out
}

func Exp(a *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = float32(math.Exp(float64(a.Data[i])))
	}
	return out
}

func Log(a *Tensor) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = float32(math.Log(float64(a.Data[i])))
	}
	return out
}

func Sum(a *Tensor) float32 {
	var sum float32
	for _, v := range a.Data {
		sum += v
	}
	return sum
}

func Mean(a *Tensor) float32 {
	if a.Numel() == 0 {
		return 0
	}
	return Sum(a) / float32(a.Numel())
}

func Max(a *Tensor) float32 {
	if len(a.Data) == 0 {
		return 0
	}
	max := a.Data[0]
	for _, v := range a.Data[1:] {
		if v > max {
			max = v
		}
	}
	return max
}

func Min(a *Tensor) float32 {
	if len(a.Data) == 0 {
		return 0
	}
	min := a.Data[0]
	for _, v := range a.Data[1:] {
		if v < min {
			min = v
		}
	}
	return min
}

func SumDim(a *Tensor, dim int) *Tensor {
	if dim < 0 || dim >= len(a.Shape) {
		panic("dim out of range")
	}
	shape := make([]int, len(a.Shape))
	copy(shape, a.Shape)
	shape[dim] = 1
	out := NewTensor(shape...)
	stride := 1
	for i := dim + 1; i < len(a.Shape); i++ {
		stride *= a.Shape[i]
	}
	for i := 0; i < a.Shape[dim]; i++ {
		offset := i * stride
		for j := 0; j < stride; j++ {
			out.Data[j] += a.Data[offset+j]
		}
	}
	return out
}

func MeanDim(a *Tensor, dim int) *Tensor {
	sum := SumDim(a, dim)
	scale := float32(1.0 / float32(a.Shape[dim]))
	return MulScalar(sum, scale)
}

func MaxDim(a *Tensor, dim int) *Tensor {
	if dim < 0 || dim >= len(a.Shape) {
		panic("dim out of range")
	}
	shape := make([]int, len(a.Shape))
	copy(shape, a.Shape)
	shape[dim] = 1
	out := NewTensor(shape...)
	stride := 1
	for i := dim + 1; i < len(a.Shape); i++ {
		stride *= a.Shape[i]
	}
	for i := 0; i < a.Shape[dim]; i++ {
		offset := i * stride
		for j := 0; j < stride; j++ {
			if i == 0 || a.Data[offset+j] > out.Data[j] {
				out.Data[j] = a.Data[offset+j]
			}
		}
	}
	return out
}

func Argmax(a *Tensor, dim int) *Tensor {
	if dim < 0 || dim >= len(a.Shape) {
		panic("dim out of range")
	}
	shape := make([]int, len(a.Shape))
	copy(shape, a.Shape)
	shape[dim] = 1
	out := NewTensor(shape...)
	stride := 1
	for i := dim + 1; i < len(a.Shape); i++ {
		stride *= a.Shape[i]
	}
	outerDim := 1
	for i := 0; i < dim; i++ {
		outerDim *= a.Shape[i]
	}
	for o := 0; o < outerDim; o++ {
		offset := o * a.Shape[dim] * stride
		maxIdx := 0
		maxVal := a.Data[offset]
		for i := 1; i < a.Shape[dim]; i++ {
			if a.Data[offset+i*stride] > maxVal {
				maxVal = a.Data[offset+i*stride]
				maxIdx = i
			}
		}
		out.Data[o*stride] = float32(maxIdx)
	}
	return out
}

func AddScalar(a *Tensor, scalar float32) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] + scalar
	}
	return out
}

func MulScalar(a *Tensor, scalar float32) *Tensor {
	out := NewTensor(a.Shape...)
	for i := range a.Data {
		out.Data[i] = a.Data[i] * scalar
	}
	return out
}

func Softmax(a *Tensor, dim int) *Tensor {
	if dim < 0 || dim >= len(a.Shape) {
		panic("dim out of range")
	}
	out := NewTensor(a.Shape...)
	stride := 1
	for i := dim + 1; i < len(a.Shape); i++ {
		stride *= a.Shape[i]
	}
	outerDim := 1
	for i := 0; i < dim; i++ {
		outerDim *= a.Shape[i]
	}
	for o := 0; o < outerDim; o++ {
		offset := o * a.Shape[dim] * stride
		maxVal := float32(-math.MaxFloat32)
		for i := 0; i < a.Shape[dim]; i++ {
			if a.Data[offset+i*stride] > maxVal {
				maxVal = a.Data[offset+i*stride]
			}
		}
		sum := float32(0)
		for i := 0; i < a.Shape[dim]; i++ {
			exp := float32(math.Exp(float64(a.Data[offset+i*stride] - maxVal)))
			out.Data[offset+i*stride] = exp
			sum += exp
		}
		for i := 0; i < a.Shape[dim]; i++ {
			out.Data[offset+i*stride] /= sum
		}
	}
	return out
}

func LogSoftmax(a *Tensor, dim int) *Tensor {
	sm := Softmax(a, dim)
	return Log(sm)
}

func Gelu(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)
	sqrt2pi := float32(math.Sqrt(2 / math.Pi))
	c1 := float32(0.044715)
	for i, v := range x.Data {
		v3 := v * v * v
		t := float32(math.Tanh(float64(sqrt2pi * (v + c1*v3))))
		out.Data[i] = 0.5 * v * (1 + t)
	}
	return out
}

func Tanh(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)
	for i, v := range x.Data {
		out.Data[i] = float32(math.Tanh(float64(v)))
	}
	return out
}

func Sigmoid(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)
	for i, v := range x.Data {
		out.Data[i] = float32(1.0 / (1.0 + math.Exp(float64(-v))))
	}
	return out
}

func Relu(x *Tensor) *Tensor {
	out := NewTensor(x.Shape...)
	for i, v := range x.Data {
		if v > 0 {
			out.Data[i] = v
		}
	}
	return out
}

func LeakyRelu(x *Tensor, alpha float32) *Tensor {
	out := NewTensor(x.Shape...)
	for i, v := range x.Data {
		if v > 0 {
			out.Data[i] = v
		} else {
			out.Data[i] = alpha * v
		}
	}
	return out
}

func Dropout(x *Tensor, p float32, train bool) *Tensor {
	if !train || p == 0 {
		return x.Copy()
	}
	out := NewTensor(x.Shape...)
	mask := TensorRandUniform(0, 1, x.Shape...)
	for i, v := range mask.Data {
		if v > p {
			out.Data[i] = x.Data[i] / (1 - p)
		}
	}
	return out
}

func LayerNorm(x *Tensor, normalizedShape int, weight, bias *Tensor, eps float32) *Tensor {
	if len(x.Shape) == 1 {
		mean := Mean(x)
		var variance float32
		for _, v := range x.Data {
			diff := v - mean
			variance += diff * diff
		}
		variance /= float32(len(x.Data))
		out := NewTensor(x.Shape...)
		for i, v := range x.Data {
			norm := (v - mean) / float32(math.Sqrt(float64(variance)+float64(eps)))
			out.Data[i] = norm*weight.Data[i] + bias.Data[i]
		}
		return out
	}
	seqLen := x.Shape[0]
	out := NewTensor(x.Shape...)
	for i := 0; i < seqLen; i++ {
		var sum float32
		for j := 0; j < normalizedShape; j++ {
			sum += x.Data[i*normalizedShape+j]
		}
		mean := sum / float32(normalizedShape)
		var variance float32
		for j := 0; j < normalizedShape; j++ {
			diff := x.Data[i*normalizedShape+j] - mean
			variance += diff * diff
		}
		variance /= float32(normalizedShape)
		std := float32(math.Sqrt(float64(variance) + float64(eps)))
		for j := 0; j < normalizedShape; j++ {
			norm := (x.Data[i*normalizedShape+j] - mean) / std
			out.Data[i*normalizedShape+j] = norm*weight.Data[j] + bias.Data[j]
		}
	}
	return out
}

func BatchNorm(x *Tensor, numFeatures int, runningMean, runningVar *Tensor, weight, bias *Tensor, training bool, momentum, eps float32) *Tensor {
	batchSize := x.Shape[0]
	out := NewTensor(x.Shape...)
	if training {
		for i := 0; i < batchSize; i++ {
			var sum float32
			for j := 0; j < numFeatures; j++ {
				sum += x.Data[i*numFeatures+j]
			}
			mean := sum / float32(numFeatures)
			var variance float32
			for j := 0; j < numFeatures; j++ {
				diff := x.Data[i*numFeatures+j] - mean
				variance += diff * diff
			}
			variance /= float32(numFeatures)
			std := float32(math.Sqrt(float64(variance) + float64(eps)))
			for j := 0; j < numFeatures; j++ {
				norm := (x.Data[i*numFeatures+j] - mean) / std
				out.Data[i*numFeatures+j] = norm*weight.Data[j] + bias.Data[j]
			}
		}
	} else {
		for i := 0; i < batchSize; i++ {
			for j := 0; j < numFeatures; j++ {
				norm := (x.Data[i*numFeatures+j] - runningMean.Data[j]) / float32(math.Sqrt(float64(runningVar.Data[j])+float64(eps)))
				out.Data[i*numFeatures+j] = norm*weight.Data[j] + bias.Data[j]
			}
		}
	}
	return out
}

func Embedding(input *Tensor, weight *Tensor) *Tensor {
	vocabSize := weight.Shape[0]
	embDim := weight.Shape[1]
	out := NewTensor(len(input.Data), embDim)
	for i, idx := range input.Data {
		row := int(idx)
		if row >= 0 && row < vocabSize {
			for j := 0; j < embDim; j++ {
				out.Data[i*embDim+j] = weight.Data[row*embDim+j]
			}
		}
	}
	return out
}

func Gather(weight *Tensor, indices *Tensor) *Tensor {
	vocabSize := weight.Shape[0]
	embDim := weight.Shape[1]
	out := NewTensor(len(indices.Data), embDim)
	for i, idx := range indices.Data {
		row := int(idx)
		if row >= 0 && row < vocabSize {
			for j := 0; j < embDim; j++ {
				out.Data[i*embDim+j] = weight.Data[row*embDim+j]
			}
		}
	}
	return out
}

func Concat(dim int, tensors ...*Tensor) *Tensor {
	if len(tensors) == 0 {
		panic("no tensors to concat")
	}
	if len(tensors) == 1 {
		return tensors[0].Copy()
	}
	totalSize := 0
	for _, t := range tensors {
		totalSize += t.Numel()
	}
	shape := make([]int, len(tensors[0].Shape))
	copy(shape, tensors[0].Shape)
	shape[dim] = 0
	for _, t := range tensors {
		shape[dim] += t.Shape[dim]
	}
	out := NewTensor(shape...)
	offset := 0
	stride := 1
	for i := dim + 1; i < len(shape); i++ {
		stride *= shape[i]
	}
	for _, t := range tensors {
		copy(out.Data[offset:offset+t.Numel()], t.Data)
		offset += t.Numel()
	}
	return out
}

func Stack(dim int, tensors ...*Tensor) *Tensor {
	if len(tensors) == 0 {
		panic("no tensors to stack")
	}
	shape := make([]int, len(tensors[0].Shape)+1)
	for i := 0; i < dim; i++ {
		shape[i] = tensors[0].Shape[i]
	}
	shape[dim] = len(tensors)
	for i := dim + 1; i < len(shape); i++ {
		shape[i] = tensors[0].Shape[i-1]
	}
	out := NewTensor(shape...)
	elementSize := 1
	for i := dim + 1; i < len(shape); i++ {
		elementSize *= shape[i]
	}
	for i, t := range tensors {
		copy(out.Data[i*elementSize:(i+1)*elementSize], t.Data)
	}
	return out
}

func TransposeMatrix(t *Tensor) *Tensor {
	if len(t.Shape) != 2 {
		return t.Copy()
	}
	rows, cols := t.Shape[0], t.Shape[1]
	out := NewTensor(cols, rows)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			out.Data[j*rows+i] = t.Data[i*cols+j]
		}
	}
	return out
}
