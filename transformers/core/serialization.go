package core

import (
	"encoding/json"
	"fmt"
	"os"
)

type ModelMetadata struct {
	Name      string                 `json:"name"`
	Version   string                 `json:"version"`
	ModelType string                 `json:"model_type"`
	CreatedAt string                 `json:"created_at"`
	Config    map[string]interface{} `json:"config"`
	NumParams int                    `json:"num_params"`
}

type SerializedTensor struct {
	Name  string    `json:"name"`
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
	Dtype string    `json:"dtype"`
}

type ModelState struct {
	Metadata ModelMetadata      `json:"metadata"`
	Tensors  []SerializedTensor `json:"tensors"`
}

func SaveModel(model map[string]*Tensor, metadata ModelMetadata, filepath string) error {
	state := ModelState{
		Metadata: metadata,
		Tensors:  make([]SerializedTensor, 0, len(model)),
	}

	for name, tensor := range model {
		state.Tensors = append(state.Tensors, SerializedTensor{
			Name:  name,
			Shape: tensor.Shape,
			Data:  tensor.Data,
			Dtype: "float32",
		})
	}

	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal model: %w", err)
	}

	err = os.WriteFile(filepath, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

func LoadModel(filepath string) (map[string]*Tensor, ModelMetadata, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, ModelMetadata{}, fmt.Errorf("failed to read file: %w", err)
	}

	var state ModelState
	err = json.Unmarshal(data, &state)
	if err != nil {
		return nil, ModelMetadata{}, fmt.Errorf("failed to unmarshal model: %w", err)
	}

	model := make(map[string]*Tensor)
	for _, t := range state.Tensors {
		model[t.Name] = &Tensor{
			Data:         t.Data,
			Shape:        t.Shape,
			Grad:         nil,
			RequiresGrad: false,
			Device:       "CPU",
		}
	}

	return model, state.Metadata, nil
}

func SaveModelBinary(model map[string]*Tensor, filepath string) error {
	file, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer file.Close()

	for name, tensor := range model {
		nameLen := uint32(len(name))
		file.Write([]byte{byte(nameLen >> 24), byte(nameLen >> 16), byte(nameLen >> 8), byte(nameLen)})
		file.Write([]byte(name))

		shapeLen := uint32(len(tensor.Shape))
		file.Write([]byte{byte(shapeLen >> 24), byte(shapeLen >> 16), byte(shapeLen >> 8), byte(shapeLen)})
		for _, s := range tensor.Shape {
			shapeData := uint32(s)
			file.Write([]byte{byte(shapeData >> 24), byte(shapeData >> 16), byte(shapeData >> 8), byte(shapeData)})
		}

		dataLen := uint32(len(tensor.Data))
		file.Write([]byte{byte(dataLen >> 24), byte(dataLen >> 16), byte(dataLen >> 8), byte(dataLen)})
		data := make([]byte, len(tensor.Data)*4)
		for i, v := range tensor.Data {
			bits := float32ToUint32(v)
			data[i*4] = byte(bits >> 24)
			data[i*4+1] = byte(bits >> 16)
			data[i*4+2] = byte(bits >> 8)
			data[i*4+3] = byte(bits)
		}
		file.Write(data)
	}

	return nil
}

func float32ToUint32(f float32) uint32 {
	bits := uint32(0)
	if f < 0 {
		bits = uint32(int32(f))
	} else {
		bits = uint32(f)
	}
	return bits
}

func LoadModelBinary(filepath string) (map[string]*Tensor, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	model := make(map[string]*Tensor)
	buffer := make([]byte, 4)

	for {
		_, err := file.Read(buffer)
		if err != nil {
			break
		}
		nameLen := uint32(buffer[0])<<24 | uint32(buffer[1])<<16 | uint32(buffer[2])<<8 | uint32(buffer[3])

		nameBytes := make([]byte, nameLen)
		file.Read(nameBytes)
		name := string(nameBytes)

		file.Read(buffer)
		shapeLen := uint32(buffer[0])<<24 | uint32(buffer[1])<<16 | uint32(buffer[2])<<8 | uint32(buffer[3])

		shape := make([]int, shapeLen)
		for i := 0; i < int(shapeLen); i++ {
			file.Read(buffer)
			shape[i] = int(uint32(buffer[0])<<24 | uint32(buffer[1])<<16 | uint32(buffer[2])<<8 | uint32(buffer[3]))
		}

		file.Read(buffer)
		dataLen := uint32(buffer[0])<<24 | uint32(buffer[1])<<16 | uint32(buffer[2])<<8 | uint32(buffer[3])

		data := make([]float32, dataLen)
		dataBytes := make([]byte, dataLen*4)
		file.Read(dataBytes)
		for i := 0; i < int(dataLen); i++ {
			bits := uint32(dataBytes[i*4])<<24 | uint32(dataBytes[i*4+1])<<16 | uint32(dataBytes[i*4+2])<<8 | uint32(dataBytes[i*4+3])
			data[i] = float32(bits)
		}

		model[name] = &Tensor{
			Data:         data,
			Shape:        shape,
			Grad:         nil,
			RequiresGrad: false,
			Device:       "CPU",
		}
	}

	return model, nil
}
