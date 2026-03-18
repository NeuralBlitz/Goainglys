package core

import (
	"math"
)

type GradCache struct {
	Input     *Tensor
	Output    *Tensor
	InputGrad *Tensor
}

type Operation interface {
	Forward() *Tensor
	Backward(grad *Tensor)
	GetCache() *GradCache
}

type AddOp struct {
	A, B   *Tensor
	Output *Tensor
}

func (op *AddOp) Forward() *Tensor {
	op.Output = Add(op.A, op.B)
	return op.Output
}

func (op *AddOp) Backward(grad *Tensor) {
	if op.A.RequiresGrad {
		if op.A.Grad == nil {
			op.A.Grad = NewTensor(op.A.Shape...)
		}
		for i := range op.A.Grad.Data {
			op.A.Grad.Data[i] += grad.Data[i]
		}
	}

	if op.B.RequiresGrad {
		if op.B.Grad == nil {
			op.B.Grad = NewTensor(op.B.Shape...)
		}
		for i := range op.B.Grad.Data {
			op.B.Grad.Data[i] += grad.Data[i]
		}
	}
}

func (op *AddOp) GetCache() *GradCache {
	return &GradCache{Input: op.A, Output: op.Output}
}

type MulOp struct {
	A, B   *Tensor
	Output *Tensor
}

func (op *MulOp) Forward() *Tensor {
	op.Output = Mul(op.A, op.B)
	return op.Output
}

func (op *MulOp) Backward(grad *Tensor) {
	a := op.A
	b := op.B

	if len(a.Shape) == len(b.Shape) {
		if a.RequiresGrad {
			if a.Grad == nil {
				a.Grad = NewTensor(a.Shape...)
			}
			for i := range a.Grad.Data {
				a.Grad.Data[i] += grad.Data[i] * b.Data[i]
			}
		}
		if b.RequiresGrad {
			if b.Grad == nil {
				b.Grad = NewTensor(b.Shape...)
			}
			for i := range b.Grad.Data {
				b.Grad.Data[i] += grad.Data[i] * a.Data[i]
			}
		}
	}
}

func (op *MulOp) GetCache() *GradCache {
	return &GradCache{Input: op.A, Output: op.Output}
}

type MatmulOp struct {
	A, B   *Tensor
	Output *Tensor
}

func (op *MatmulOp) Forward() *Tensor {
	op.Output = Matmul(op.A, op.B)
	return op.Output
}

func (op *MatmulOp) Backward(grad *Tensor) {
	a := op.A
	b := op.B

	if a.RequiresGrad {
		if a.Grad == nil {
			a.Grad = NewTensor(a.Shape...)
		}
		gradB := TransposeMatrix(b)
		aGrad := Matmul(grad, gradB)
		for i := range a.Grad.Data {
			a.Grad.Data[i] += aGrad.Data[i]
		}
	}

	if b.RequiresGrad {
		if b.Grad == nil {
			b.Grad = NewTensor(b.Shape...)
		}
		gradA := TransposeMatrix(a)
		bGrad := Matmul(gradA, grad)
		for i := range b.Grad.Data {
			b.Grad.Data[i] += bGrad.Data[i]
		}
	}
}

func (op *MatmulOp) GetCache() *GradCache {
	return &GradCache{Input: op.A, Output: op.Output}
}

type SoftmaxOp struct {
	Input  *Tensor
	Output *Tensor
	Dim    int
}

func (op *SoftmaxOp) Forward() *Tensor {
	op.Output = Softmax(op.Input, op.Dim)
	return op.Output
}

func (op *SoftmaxOp) Backward(grad *Tensor) {
	if !op.Input.RequiresGrad {
		return
	}

	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape...)
	}

	for i := 0; i < op.Input.Numel(); i++ {
		var sum float32
		for j := 0; j < op.Output.Numel(); j++ {
			gradVal := grad.Data[j]
			probI := op.Output.Data[i]
			probJ := op.Output.Data[j]
			if i == j {
				sum += gradVal * probI * (1 - probI)
			} else {
				sum += gradVal * (-probI * probJ)
			}
		}
		op.Input.Grad.Data[i] += sum
	}
}

func (op *SoftmaxOp) GetCache() *GradCache {
	return &GradCache{Input: op.Input, Output: op.Output}
}

type ReluOp struct {
	Input  *Tensor
	Output *Tensor
	Mask   *Tensor
}

func (op *ReluOp) Forward() *Tensor {
	op.Output = NewTensor(op.Input.Shape...)
	op.Mask = NewTensor(op.Input.Shape...)
	for i := range op.Input.Data {
		if op.Input.Data[i] > 0 {
			op.Output.Data[i] = op.Input.Data[i]
			op.Mask.Data[i] = 1
		} else {
			op.Output.Data[i] = 0
			op.Mask.Data[i] = 0
		}
	}
	return op.Output
}

func (op *ReluOp) Backward(grad *Tensor) {
	if !op.Input.RequiresGrad {
		return
	}

	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape...)
	}

	for i := range op.Input.Grad.Data {
		op.Input.Grad.Data[i] += grad.Data[i] * op.Mask.Data[i]
	}
}

func (op *ReluOp) GetCache() *GradCache {
	return &GradCache{Input: op.Input, Output: op.Output}
}

type TanhOp struct {
	Input  *Tensor
	Output *Tensor
}

func (op *TanhOp) Forward() *Tensor {
	op.Output = Tanh(op.Input)
	return op.Output
}

func (op *TanhOp) Backward(grad *Tensor) {
	if !op.Input.RequiresGrad {
		return
	}

	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape...)
	}

	for i := range op.Input.Grad.Data {
		derivative := 1 - op.Output.Data[i]*op.Output.Data[i]
		op.Input.Grad.Data[i] += grad.Data[i] * derivative
	}
}

func (op *TanhOp) GetCache() *GradCache {
	return &GradCache{Input: op.Input, Output: op.Output}
}

type SigmoidOp struct {
	Input  *Tensor
	Output *Tensor
}

func (op *SigmoidOp) Forward() *Tensor {
	op.Output = NewTensor(op.Input.Shape...)
	for i := range op.Input.Data {
		op.Output.Data[i] = 1 / (1 + float32(math.Exp(float64(-op.Input.Data[i]))))
	}
	return op.Output
}

func (op *SigmoidOp) Backward(grad *Tensor) {
	if !op.Input.RequiresGrad {
		return
	}

	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape...)
	}

	for i := range op.Input.Grad.Data {
		derivative := op.Output.Data[i] * (1 - op.Output.Data[i])
		op.Input.Grad.Data[i] += grad.Data[i] * derivative
	}
}

func (op *SigmoidOp) GetCache() *GradCache {
	return &GradCache{Input: op.Input, Output: op.Output}
}

type LinearOp struct {
	Input  *Tensor
	Weight *Tensor
	Bias   *Tensor
	Output *Tensor
}

func (op *LinearOp) Forward() *Tensor {
	op.Output = Matmul(op.Input, op.Weight)
	if op.Bias != nil {
		op.Output = Add(op.Output, op.Bias)
	}
	return op.Output
}

func (op *LinearOp) Backward(grad *Tensor) {
	input := op.Input
	weight := op.Weight

	if input.RequiresGrad {
		if input.Grad == nil {
			input.Grad = NewTensor(input.Shape...)
		}
		weightT := TransposeMatrix(weight)
		inputGrad := Matmul(grad, weightT)
		for i := range input.Grad.Data {
			input.Grad.Data[i] += inputGrad.Data[i]
		}
	}

	if weight.RequiresGrad {
		if weight.Grad == nil {
			weight.Grad = NewTensor(weight.Shape...)
		}
		inputT := TransposeMatrix(input)
		weightGrad := Matmul(inputT, grad)
		for i := range weight.Grad.Data {
			weight.Grad.Data[i] += weightGrad.Data[i]
		}
	}

	if op.Bias != nil && op.Bias.RequiresGrad {
		if op.Bias.Grad == nil {
			op.Bias.Grad = NewTensor(op.Bias.Shape...)
		}
		for i := range op.Bias.Grad.Data {
			op.Bias.Grad.Data[i] += grad.Data[i]
		}
	}
}

func (op *LinearOp) GetCache() *GradCache {
	return &GradCache{Input: op.Input, Output: op.Output}
}

func Backward(loss *Tensor) {
	if loss.Grad == nil {
		loss.Grad = NewTensor(loss.Shape...)
		for i := range loss.Grad.Data {
			loss.Grad.Data[i] = 1
		}
	}
}

func ZeroGrad(model map[string]*Tensor) {
	for _, t := range model {
		if t.Grad != nil {
			for i := range t.Grad.Data {
				t.Grad.Data[i] = 0
			}
		}
	}
}
