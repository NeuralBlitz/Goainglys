package tensor

import (
	"math"
	"math/rand"
)

type Tensor struct {
	Data    []float64
	Shape   []int
	Grad    []float64
	ReqGrad bool
}

func New(shape ...int) *Tensor {
	size := 1
	for _, d := range shape {
		size *= d
	}
	return &Tensor{
		Data:  make([]float64, size),
		Shape: shape,
		Grad:  make([]float64, size),
	}
}

func (t *Tensor) Dim(i int) int {
	if i < 0 || i >= len(t.Shape) {
		return 0
	}
	return t.Shape[i]
}

func (t *Tensor) Size() int {
	size := 1
	for _, d := range t.Shape {
		size *= d
	}
	return size
}

func (t *Tensor) Index(indices ...int) int {
	if len(indices) != len(t.Shape) {
		panic("indices dimension mismatch")
	}
	idx := 0
	multiplier := 1
	for i := len(indices) - 1; i >= 0; i-- {
		idx += indices[i] * multiplier
		multiplier *= t.Shape[i]
	}
	return idx
}

func (t *Tensor) Get(indices ...int) float64 {
	return t.Data[t.Index(indices...)]
}

func (t *Tensor) Set(v float64, indices ...int) {
	t.Data[t.Index(indices...)] = v
}

func (t *Tensor) Reshape(shape ...int) *Tensor {
	newSize := 1
	for _, d := range shape {
		newSize *= d
	}
	if newSize != t.Size() {
		panic("cannot reshape: size mismatch")
	}
	return &Tensor{
		Data:  t.Data,
		Shape: shape,
		Grad:  t.Grad,
	}
}

func (t *Tensor) Transpose() *Tensor {
	if len(t.Shape) != 2 {
		panic("transpose only supports 2D tensors")
	}
	result := New(t.Shape[1], t.Shape[0])
	for i := 0; i < t.Shape[0]; i++ {
		for j := 0; j < t.Shape[1]; j++ {
			result.Set(t.Get(i, j), j, i)
		}
	}
	return result
}

func (t *Tensor) Clone() *Tensor {
	data := make([]float64, t.Size())
	grad := make([]float64, t.Size())
	copy(data, t.Data)
	copy(grad, t.Grad)
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	return &Tensor{
		Data:    data,
		Shape:   shape,
		Grad:    grad,
		ReqGrad: t.ReqGrad,
	}
}

func (t *Tensor) ZeroGrad() {
	for i := range t.Grad {
		t.Grad[i] = 0
	}
}

type Param struct {
	Data *Tensor
	Grad *Tensor
}

func NewParam(shape ...int) *Param {
	data := New(shape...)
	data.ReqGrad = true
	for i := range data.Data {
		scale := math.Sqrt(2.0 / float64(shape[0]))
		data.Data[i] = rand.NormFloat64() * scale
	}
	return &Param{
		Data: data,
		Grad: New(shape...),
	}
}

func (p *Param) ZeroGrad() {
	p.Grad.ZeroGrad()
}

func (p *Param) SetGrad(d *Tensor) {
	copy(p.Grad.Data, d.Data)
}

func (p *Param) AddGrad(d *Tensor) {
	for i := range p.Grad.Data {
		p.Grad.Data[i] += d.Data[i]
	}
}
