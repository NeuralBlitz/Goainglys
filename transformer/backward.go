package transformer

import (
	"math"

	"github.com/user/transformer/tensor"
)

func ComputeGradients(model *Transformer, src, tgt, srcMask, tgtMask *tensor.Tensor, loss float64) {
	batchSize := src.Shape[0]
	seqLen := src.Shape[1]
	dModel := model.Config.ModelDim
	vocabSize := model.Config.VocabSize

	logits := model.Forward(src, tgt, srcMask, tgtMask, true)

	gradLogits := computeLossGradient(logits, tgt, vocabSize)

	gradW := make([]float64, dModel*vocabSize)
	for i := 0; i < dModel; i++ {
		for j := 0; j < vocabSize; j++ {
			sum := 0.0
			for b := 0; b < batchSize; b++ {
				for k := 0; k < seqLen; k++ {
					sum += logits.Get(b, k, i) * gradLogits.Get(b, k, j)
				}
			}
			gradW[i*vocabSize+j] = sum
		}
	}

	model.OutputProj.Grad = tensor.New(dModel, vocabSize)
	copy(model.OutputProj.Grad.Data, gradW)

	gradInput := tensor.New(batchSize, seqLen, dModel)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for k := 0; k < dModel; k++ {
				sum := 0.0
				for j := 0; j < vocabSize; j++ {
					sum += model.OutputProj.Data.Get(k, j) * gradLogits.Get(b, i, j)
				}
				gradInput.Set(sum, b, i, k)
			}
		}
	}

	for _, layer := range model.EncoderLayers {
		layer.SubLayer.Attention.Backward(gradInput, srcMask)
		layer.SubLayer.Ffn.Backward(gradInput)
	}

	for _, layer := range model.DecoderLayers {
		layer.SubLayer1.Attention.Backward(gradInput, tgtMask)
		layer.SubLayer1.Ffn.Backward(gradInput)
		layer.SubLayer2.Attention.BackwardCross(gradInput, gradInput, srcMask)
		layer.SubLayer2.Ffn.Backward(gradInput)
		layer.SubLayer3.Attention.Backward(gradInput, tgtMask)
		layer.SubLayer3.Ffn.Backward(gradInput)
	}

	model.Embedding.Weights.Grad = tensor.New(model.Embedding.Weights.Data.Shape[0], model.Embedding.Weights.Data.Shape[1])
}

func computeLossGradient(logits, targets *tensor.Tensor, vocabSize int) *tensor.Tensor {
	batchSize := logits.Shape[0]
	seqLen := logits.Shape[1]
	grad := tensor.New(batchSize, seqLen, vocabSize)

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			maxVal := math.Inf(-1)
			for j := 0; j < vocabSize; j++ {
				if logits.Get(b, i, j) > maxVal {
					maxVal = logits.Get(b, i, j)
				}
			}

			sum := 0.0
			for j := 0; j < vocabSize; j++ {
				sum += math.Exp(logits.Get(b, i, j) - maxVal)
			}

			target := int(targets.Get(b, i))
			for j := 0; j < vocabSize; j++ {
				prob := math.Exp(logits.Get(b, i, j)-maxVal) / sum
				if j == target {
					grad.Set(prob-1.0, b, i, j)
				} else {
					grad.Set(prob, b, i, j)
				}
			}
		}
	}

	return grad
}

func computeOutputProjGrad(decOutput *tensor.Tensor, outputProj *tensor.Param, gradLogits *tensor.Tensor) *tensor.Tensor {
	batchSize := decOutput.Shape[0]
	seqLen := decOutput.Shape[1]
	dModel := decOutput.Shape[2]
	vocabSize := outputProj.Data.Shape[1]

	gradW := make([]float64, outputProj.Data.Shape[0]*outputProj.Data.Shape[1])
	for i := 0; i < outputProj.Data.Shape[0]; i++ {
		for j := 0; j < outputProj.Data.Shape[1]; j++ {
			sum := 0.0
			for b := 0; b < batchSize; b++ {
				for k := 0; k < seqLen; k++ {
					sum += decOutput.Get(b, k, i) * gradLogits.Get(b, k, j)
				}
			}
			gradW[i*outputProj.Data.Shape[1]+j] = sum
		}
	}

	outputProj.Grad = tensor.New(outputProj.Data.Shape[0], outputProj.Data.Shape[1])
	copy(outputProj.Grad.Data, gradW)

	gradDecOutput := tensor.New(batchSize, seqLen, dModel)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for k := 0; k < dModel; k++ {
				sum := 0.0
				for j := 0; j < vocabSize; j++ {
					sum += outputProj.Data.Get(k, j) * gradLogits.Get(b, i, j)
				}
				gradDecOutput.Set(sum, b, i, k)
			}
		}
	}

	return gradDecOutput
}

func (el *EncoderLayer) Backward(gradOutput, mask *tensor.Tensor) *tensor.Tensor {
	gradX := el.SubLayer.Backward(gradOutput, mask)
	return gradX
}

func (dl *DecoderLayer) Backward(gradOutput, encOutput, mask, srcMask *tensor.Tensor) *tensor.Tensor {
	gradX := dl.SubLayer3.Backward(gradOutput, mask)
	gradEnc := dl.SubLayer2.BackwardCross(gradX, encOutput, srcMask)
	gradX = dl.SubLayer1.Backward(gradEnc, mask)
	return gradX
}

func (sl *SubLayer) Backward(gradOutput, mask *tensor.Tensor) *tensor.Tensor {
	gradLN2 := layerNormGrad(sl.Ln2.Data, gradOutput)
	gradFFN := sl.Ffn.Backward(gradLN2)

	gradAdd1 := tensor.Add(gradFFN, gradOutput)

	gradLN1 := layerNormGrad(sl.Ln1.Data, gradAdd1)
	gradAttn := sl.Attention.Backward(gradLN1, mask)

	gradX := tensor.Add(gradAttn, gradAdd1)

	return gradX
}

func (sl *SubLayer) BackwardCross(gradOutput, encOutput, mask *tensor.Tensor) *tensor.Tensor {
	gradLN2 := layerNormGrad(sl.Ln2.Data, gradOutput)
	gradFFN := sl.Ffn.Backward(gradLN2)

	gradAdd1 := tensor.Add(gradFFN, gradOutput)

	gradLN1 := layerNormGrad(sl.Ln1.Data, gradAdd1)
	gradAttn, _ := sl.Attention.BackwardCross(gradLN1, encOutput, mask)

	gradX := tensor.Add(gradAttn, gradAdd1)

	return gradX
}

func layerNormGrad(gamma *tensor.Tensor, gradOutput *tensor.Tensor) *tensor.Tensor {
	if len(gradOutput.Shape) == 3 {
		batchSize := gradOutput.Shape[0]
		seqLen := gradOutput.Shape[1]
		dModel := gradOutput.Shape[2]

		gradInput := tensor.New(batchSize, seqLen, dModel)

		for b := 0; b < batchSize; b++ {
			for i := 0; i < seqLen; i++ {
				for j := 0; j < dModel; j++ {
					gradInput.Set(gradOutput.Get(b, i, j)*gamma.Get(j), b, i, j)
				}
			}
		}

		return gradInput
	}
	return gradOutput
}

func (ffn *FeedForward) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	batchSize := gradOutput.Shape[0]
	seqLen := gradOutput.Shape[1]
	dModel := ffn.W2.Data.Shape[1]

	gradW2 := make([]float64, ffn.W2.Data.Shape[0]*ffn.W2.Data.Shape[1])
	gradB2 := make([]float64, ffn.B2.Data.Shape[0])

	gradHidden := tensor.New(batchSize, seqLen, ffn.W2.Data.Shape[0])
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < ffn.W2.Data.Shape[0]; j++ {
				sum := 0.0
				for k := 0; k < dModel; k++ {
					sum += gradOutput.Get(b, i, k) * ffn.W2.Data.Get(j, k)
				}
				gradHidden.Set(sum, b, i, j)
			}
		}
	}

	gradReLU := tensor.New(gradHidden.Shape...)
	for i := 0; i < gradHidden.Size(); i++ {
		if gradHidden.Data[i] > 0 {
			gradReLU.Data[i] = gradHidden.Data[i]
		}
	}

	gradW1 := make([]float64, ffn.W1.Data.Shape[0]*ffn.W1.Data.Shape[1])
	gradB1 := make([]float64, ffn.B1.Data.Shape[0])

	gradInput := tensor.New(batchSize, seqLen, ffn.W1.Data.Shape[1])
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < ffn.W1.Data.Shape[1]; j++ {
				sum := 0.0
				for k := 0; k < ffn.W1.Data.Shape[0]; k++ {
					sum += gradReLU.Get(b, i, k) * ffn.W1.Data.Get(k, j)
				}
				gradInput.Set(sum, b, i, j)
			}
		}
	}

	ffn.W1.Grad = tensor.New(ffn.W1.Data.Shape[0], ffn.W1.Data.Shape[1])
	copy(ffn.W1.Grad.Data, gradW1)
	ffn.W2.Grad = tensor.New(ffn.W2.Data.Shape[0], ffn.W2.Data.Shape[1])
	copy(ffn.W2.Grad.Data, gradW2)
	ffn.B1.Grad = tensor.New(ffn.B1.Data.Shape[0])
	copy(ffn.B1.Grad.Data, gradB1)
	ffn.B2.Grad = tensor.New(ffn.B2.Data.Shape[0])
	copy(ffn.B2.Grad.Data, gradB2)

	return gradInput
}

func (mha *MultiHeadAttention) Backward(gradOutput, mask *tensor.Tensor) *tensor.Tensor {
	batchSize := gradOutput.Shape[0]
	seqLen := gradOutput.Shape[1]
	dModel := gradOutput.Shape[2]

	flatGrad := gradOutput.Reshape(batchSize*seqLen, dModel)

	gradOutProj := tensor.New(batchSize*seqLen, mha.config.ModelDim)
	for i := 0; i < batchSize*seqLen; i++ {
		for j := 0; j < mha.config.ModelDim; j++ {
			sum := 0.0
			for k := 0; k < mha.config.ModelDim; k++ {
				sum += flatGrad.Get(i, k) * mha.Wo.Data.Get(k, j)
			}
			gradOutProj.Set(sum, i, j)
		}
	}

	mha.Wo.Grad = tensor.New(mha.Wo.Data.Shape[0], mha.Wo.Data.Shape[1])
	for i := 0; i < mha.Wo.Data.Shape[0]; i++ {
		for j := 0; j < mha.Wo.Data.Shape[1]; j++ {
			sum := 0.0
			for b := 0; b < batchSize; b++ {
				for k := 0; k < seqLen; k++ {
					sum += gradOutput.Get(b, k, i) * mha.Wo.Data.Get(i, j)
				}
			}
			mha.Wo.Grad.Set(sum, i, j)
		}
	}

	gradInput := tensor.New(batchSize, seqLen, dModel)

	mha.Wq.Grad = tensor.New(mha.Wq.Data.Shape[0], mha.Wq.Data.Shape[1])
	mha.Wk.Grad = tensor.New(mha.Wk.Data.Shape[0], mha.Wk.Data.Shape[1])
	mha.Wv.Grad = tensor.New(mha.Wv.Data.Shape[0], mha.Wv.Data.Shape[1])

	return gradInput
}

func (mha *MultiHeadAttention) BackwardCross(gradOutput, encOutput, mask *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor) {
	batchSize := gradOutput.Shape[0]
	seqLen := gradOutput.Shape[1]
	dModel := gradOutput.Shape[2]

	gradInput := tensor.New(batchSize, seqLen, dModel)
	gradEncOutput := tensor.New(encOutput.Shape[0], encOutput.Shape[1], encOutput.Shape[2])

	mha.Wq.Grad = tensor.New(mha.Wq.Data.Shape[0], mha.Wq.Data.Shape[1])
	mha.Wk.Grad = tensor.New(mha.Wk.Data.Shape[0], mha.Wk.Data.Shape[1])
	mha.Wv.Grad = tensor.New(mha.Wv.Data.Shape[0], mha.Wv.Data.Shape[1])
	mha.Wo.Grad = tensor.New(mha.Wo.Data.Shape[0], mha.Wo.Data.Shape[1])

	return gradInput, gradEncOutput
}

func (e *Embedding) Backward(gradOutput *tensor.Tensor) *tensor.Tensor {
	vocabSize := e.Weights.Data.Shape[0]
	dModel := e.Weights.Data.Shape[1]

	gradWeights := make([]float64, vocabSize*dModel)

	batchSize := gradOutput.Shape[0]
	seqLen := gradOutput.Shape[1]

	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLen; i++ {
			for j := 0; j < dModel; j++ {
				gradWeights[i*dModel+j] += gradOutput.Get(b, i, j)
			}
		}
	}

	e.Weights.Grad = tensor.New(vocabSize, dModel)
	copy(e.Weights.Grad.Data, gradWeights)

	return tensor.New(batchSize, seqLen, dModel)
}
