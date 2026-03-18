package core

import (
	"errors"
	"math"
)

// QuantizationConfig holds configuration for quantization
type QuantizationConfig struct {
	Bits        int
	Symmetric   bool
	PerChannel  bool
	QuantMin    float32
	QuantMax    float32
	Scale       []float32
	ZeroPoint   []int32
	IsQuantized bool
}

// NewQuantizationConfig creates a new quantization config
func NewQuantizationConfig(bits int, symmetric bool, perChannel bool) *QuantizationConfig {
	var qmin, qmax float32
	if symmetric {
		shift := bits - 1
		qmin = -float32(int(1)<<shift) + 1
		qmax = float32(int(1)<<shift) - 1
	} else {
		qmin = 0
		qmax = float32(int(1)<<bits) - 1
	}

	return &QuantizationConfig{
		Bits:        bits,
		Symmetric:   symmetric,
		PerChannel:  perChannel,
		QuantMin:    qmin,
		QuantMax:    qmax,
		IsQuantized: false,
	}
}

// ErrEmptyTensor is returned when trying to quantize an empty tensor
var ErrEmptyTensor = errors.New("cannot quantize empty tensor")

// QuantizeTensor quantizes a tensor to lower precision
func QuantizeTensor(t *Tensor, config *QuantizationConfig) (*Tensor, error) {
	if t.Data == nil || len(t.Data) == 0 {
		return nil, ErrEmptyTensor
	}

	// Calculate scale and zero point
	var scale []float32
	var zeroPoint []int32

	if config.PerChannel && len(t.Shape) > 1 {
		// Per-channel quantization (for weight matrices)
		channels := t.Shape[0]
		scale = make([]float32, channels)
		zeroPoint = make([]int32, channels)

		for c := 0; c < channels; c++ {
			// Find min/max for this channel
			var minVal, maxVal float32
			if channels == t.Shape[0] {
				// First dimension is channels
				startIdx := c * t.Numel() / channels
				endIdx := (c + 1) * t.Numel() / channels
				if endIdx > len(t.Data) {
					endIdx = len(t.Data)
				}
				minVal, maxVal = findMinMaxQ(t.Data[startIdx:endIdx])
			} else {
				// Fallback to tensor-wide
				minVal, maxVal = findMinMaxQ(t.Data)
			}

			s, zp := calcScaleAndZP(minVal, maxVal, config.QuantMin, config.QuantMax, config.Symmetric)
			scale[c] = s
			zeroPoint[c] = zp
		}
	} else {
		// Tensor-wide quantization
		minVal, maxVal := findMinMaxQ(t.Data)
		s, zp := calcScaleAndZP(minVal, maxVal, config.QuantMin, config.QuantMax, config.Symmetric)
		scale = []float32{s}
		zeroPoint = []int32{zp}
	}

	config.Scale = scale
	config.ZeroPoint = zeroPoint
	config.IsQuantized = true

	// Create quantized tensor
	quantizedData := make([]float32, len(t.Data))
	if config.PerChannel && len(t.Shape) > 1 {
		channels := t.Shape[0]
		elementsPerChannel := len(t.Data) / channels

		for c := 0; c < channels; c++ {
			startIdx := c * elementsPerChannel
			endIdx := startIdx + elementsPerChannel
			s := scale[c]
			zp := zeroPoint[c]

			for i := startIdx; i < endIdx; i++ {
				q := float32(zp) + t.Data[i]/s
				// Clamp to quantized range
				if q < config.QuantMin {
					q = config.QuantMin
				} else if q > config.QuantMax {
					q = config.QuantMax
				}
				quantizedData[i] = q
			}
		}
	} else {
		s := scale[0]
		zp := zeroPoint[0]
		for i := 0; i < len(t.Data); i++ {
			q := float32(zp) + t.Data[i]/s
			// Clamp to quantized range
			if q < config.QuantMin {
				q = config.QuantMin
			} else if q > config.QuantMax {
				q = config.QuantMax
			}
			quantizedData[i] = q
		}
	}

	return &Tensor{
		Data:         quantizedData,
		Shape:        t.Shape,
		Grad:         nil,
		RequiresGrad: false,
		Device:       t.Device,
	}, nil
}

// DequantizeTensor converts quantized tensor back to floating point
func DequantizeTensor(t *Tensor, config *QuantizationConfig) (*Tensor, error) {
	if !config.IsQuantized {
		return t.Copy(), nil
	}

	dequantizedData := make([]float32, len(t.Data))
	if config.PerChannel && len(t.Shape) > 1 {
		channels := t.Shape[0]
		elementsPerChannel := len(t.Data) / channels

		for c := 0; c < channels; c++ {
			startIdx := c * elementsPerChannel
			endIdx := startIdx + elementsPerChannel
			s := config.Scale[c]
			zp := config.ZeroPoint[c]

			for i := startIdx; i < endIdx; i++ {
				dequantizedData[i] = (t.Data[i] - float32(zp)) * s
			}
		}
	} else {
		s := config.Scale[0]
		zp := config.ZeroPoint[0]
		for i := 0; i < len(t.Data); i++ {
			dequantizedData[i] = (t.Data[i] - float32(zp)) * s
		}
	}

	return &Tensor{
		Data:         dequantizedData,
		Shape:        t.Shape,
		Grad:         nil,
		RequiresGrad: false,
		Device:       t.Device,
	}, nil
}

// Helper functions
func findMinMaxQ(data []float32) (float32, float32) {
	if len(data) == 0 {
		return 0, 0
	}
	minVal, maxVal := data[0], data[0]
	for _, v := range data[1:] {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}
	return minVal, maxVal
}

func calcScaleAndZP(minVal, maxVal, qmin, qmax float32, symmetric bool) (float32, int32) {
	if symmetric {
		// Symmetric quantization
		maxAbs := float32(math.Max(float64(math.Abs(float64(minVal))), float64(math.Abs(float64(maxVal)))))
		if maxAbs == 0 {
			return 1.0, 0
		}
		scale := maxAbs / ((qmax - qmin) / 2)
		return scale, 0
	} else {
		// Asymmetric quantization
		if maxVal == minVal {
			return 1.0, int32(qmin)
		}
		scale := (maxVal - minVal) / (qmax - qmin)
		zeroPoint := qmin - minVal/scale
		return scale, int32(zeroPoint)
	}
}

// Int8Quantize creates an INT8 quantized version of a tensor
func Int8Quantize(t *Tensor, perChannel bool) (*Tensor, *QuantizationConfig) {
	config := NewQuantizationConfig(8, true, perChannel)
	quantized, _ := QuantizeTensor(t, config)
	return quantized, config
}

// Int4Quantize creates an INT4 quantized version of a tensor
func Int4Quantize(t *Tensor, perChannel bool) (*Tensor, *QuantizationConfig) {
	config := NewQuantizationConfig(4, true, perChannel)
	quantized, _ := QuantizeTensor(t, config)
	return quantized, config
}
