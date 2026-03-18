package core

import (
	"fmt"
	"math"
	"math/rand"
)

type Tensor struct {
	Data         []float32
	Shape        []int
	Grad         *Tensor
	RequiresGrad bool
	Device       string
}

func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return &Tensor{
		Data:         make([]float32, size),
		Shape:        shape,
		RequiresGrad: true,
		Device:       "CPU",
	}
}

func NewTensorWithData(data []float32, shape []int) *Tensor {
	return &Tensor{
		Data:         data,
		Shape:        shape,
		RequiresGrad: true,
		Device:       "CPU",
	}
}

func (t *Tensor) Numel() int {
	n := 1
	for _, s := range t.Shape {
		n *= s
	}
	return n
}

func (t *Tensor) Dim() int {
	return len(t.Shape)
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	if size != t.Numel() {
		panic(fmt.Sprintf("reshape size mismatch: got %d, want %d", size, t.Numel()))
	}
	return &Tensor{
		Data:         t.Data,
		Shape:        shape,
		Grad:         t.Grad,
		RequiresGrad: t.RequiresGrad,
	}
}

func (t *Tensor) Copy() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	return &Tensor{
		Data:         data,
		Shape:        t.Shape,
		Grad:         nil,
		RequiresGrad: t.RequiresGrad,
	}
}

func (t *Tensor) SetData(data []float32) {
	if len(data) != len(t.Data) {
		panic("data size mismatch")
	}
	copy(t.Data, data)
}

func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad.Data {
			t.Grad.Data[i] = 0
		}
	}
}

func (t *Tensor) AddGrad(grad *Tensor) {
	if t.Grad == nil {
		t.Grad = grad.Copy()
		return
	}
	for i := range t.Grad.Data {
		t.Grad.Data[i] += grad.Data[i]
	}
}

func TensorZeros(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = 0
	}
	return t
}

func TensorOnes(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = 1
	}
	return t
}

func TensorFull(value float32, shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = value
	}
	return t
}

func TensorRandn(shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = float32(randn())
	}
	return t
}

func TensorRandUniform(low, high float32, shape ...int) *Tensor {
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = low + float32(rand.Float64())*(high-low)
	}
	return t
}

func randn() float64 {
	u1 := rand.Float64()
	u2 := rand.Float64()
	return math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2)
}

func TensorRandnSeeded(seed int64, shape ...int) *Tensor {
	r := rand.New(rand.NewSource(seed))
	t := NewTensor(shape...)
	for i := range t.Data {
		u1 := r.Float64()
		u2 := r.Float64()
		t.Data[i] = float32(math.Sqrt(-2*math.Log(u1)) * math.Cos(2*math.Pi*u2))
	}
	return t
}

func TensorRandUniformSeeded(seed int64, low, high float32, shape ...int) *Tensor {
	r := rand.New(rand.NewSource(seed))
	t := NewTensor(shape...)
	for i := range t.Data {
		t.Data[i] = low + float32(r.Float64())*(high-low)
	}
	return t
}

func TensorEye(n int) *Tensor {
	t := NewTensor(n, n)
	for i := 0; i < n; i++ {
		t.Data[i*t.Shape[1]+i] = 1
	}
	return t
}

func TensorArange(start, end, step float32) *Tensor {
	n := int((float64(end) - float64(start)) / float64(step))
	data := make([]float32, n)
	for i := 0; i < n; i++ {
		data[i] = start + float32(i)*step
	}
	return NewTensorWithData(data, []int{n})
}

func (t *Tensor) View(shape ...int) *Tensor {
	return t.Reshape(shape...)
}

func (t *Tensor) Squeeze(dim int) *Tensor {
	if dim < 0 || dim >= len(t.Shape) {
		panic("dim out of range")
	}
	if t.Shape[dim] != 1 {
		return t.Copy()
	}
	shape := make([]int, 0, len(t.Shape)-1)
	for i := range t.Shape {
		if i != dim {
			shape = append(shape, t.Shape[i])
		}
	}
	return t.Reshape(shape...)
}

func (t *Tensor) Unsqueeze(dim int) *Tensor {
	if dim < 0 || dim > len(t.Shape) {
		panic("dim out of range")
	}
	shape := make([]int, len(t.Shape)+1)
	for i := 0; i < dim; i++ {
		shape[i] = t.Shape[i]
	}
	shape[dim] = 1
	for i := dim; i < len(t.Shape); i++ {
		shape[i+1] = t.Shape[i]
	}
	return t.Reshape(shape...)
}

func (t *Tensor) Expand(shape ...int) *Tensor {
	if len(shape) != len(t.Shape) {
		panic("shape length mismatch")
	}
	for i := range t.Shape {
		if t.Shape[i] != 1 && t.Shape[i] != shape[i] {
			panic("cannot expand dimension")
		}
	}
	return &Tensor{
		Data:         t.Data,
		Shape:        shape,
		Grad:         t.Grad,
		RequiresGrad: t.RequiresGrad,
	}
}

func (t *Tensor) ToDevice(device string) *Tensor {
	return t
}

func (t *Tensor) CPU() *Tensor {
	return t
}

func (t *Tensor) GPU() *Tensor {
	return t
}
