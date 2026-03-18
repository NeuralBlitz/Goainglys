package asr

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

// Audio represents loaded audio data
type Audio struct {
	SampleRate int
	Channels   int
	Samples    []float64
	Duration   float64
}

// LoadWAV loads a WAV file from disk
func LoadWAV(filename string) (*Audio, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read RIFF header
	var riff [4]byte
	if _, err := f.Read(riff[:]); err != nil {
		return nil, err
	}
	if string(riff[:]) != "RIFF" {
		return nil, fmt.Errorf("not a WAV file")
	}

	// Read file size (skip)
	var fileSize int32
	binary.Read(f, binary.LittleEndian, &fileSize)

	// Read WAVE header
	var wave [4]byte
	if _, err := f.Read(wave[:]); err != nil {
		return nil, err
	}
	if string(wave[:]) != "WAVE" {
		return nil, fmt.Errorf("not a WAV file")
	}

	// Read fmt chunk
	var fmtChunk [4]byte
	if _, err := f.Read(fmtChunk[:]); err != nil {
		return nil, err
	}
	if string(fmtChunk[:]) != "fmt " {
		return nil, fmt.Errorf("unexpected format chunk")
	}

	// Read fmt chunk size
	var fmtSize int32
	binary.Read(f, binary.LittleEndian, &fmtSize)

	// Read audio format
	var audioFormat int16
	binary.Read(f, binary.LittleEndian, &audioFormat)
	if audioFormat != 1 {
		return nil, fmt.Errorf("only PCM format supported")
	}

	// Read number of channels
	var numChannels int16
	binary.Read(f, binary.LittleEndian, &numChannels)

	// Read sample rate
	var sampleRate int32
	binary.Read(f, binary.LittleEndian, &sampleRate)

	// Read byte rate (skip)
	var byteRate int32
	binary.Read(f, binary.LittleEndian, &byteRate)

	// Read block align (skip)
	var blockAlign int16
	binary.Read(f, binary.LittleEndian, &blockAlign)

	// Read bits per sample
	var bitsPerSample int16
	binary.Read(f, binary.LittleEndian, &bitsPerSample)

	// Skip to data chunk
	dataChunk := make([]byte, 4)
	for {
		if _, err := f.Read(dataChunk); err != nil {
			return nil, err
		}
		if string(dataChunk) == "data" {
			break
		}
		f.Seek(-3, io.SeekCurrent) // Go back 3 bytes
	}

	// Read data size
	var dataSize int32
	binary.Read(f, binary.LittleEndian, &dataSize)

	// Read samples
	numSamples := dataSize / int32(numChannels*bitsPerSample/8)
	samples := make([]float64, numSamples)

	bytesPerSample := bitsPerSample / 8
	for i := int32(0); i < numSamples; i++ {
		var sample int16
		if bytesPerSample == 2 {
			binary.Read(f, binary.LittleEndian, &sample)
			samples[i] = float64(sample) / 32768.0
		} else if bytesPerSample == 1 {
			var sample8 byte
			binary.Read(f, binary.LittleEndian, &sample8)
			samples[i] = (float64(sample8) - 128) / 128.0
		}
	}

	audio := &Audio{
		SampleRate: int(sampleRate),
		Channels:   int(numChannels),
		Samples:    samples,
		Duration:   float64(numSamples) / float64(sampleRate),
	}

	return audio, nil
}

// SaveWAV saves audio to a WAV file
func (a *Audio) SaveWAV(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	// Write RIFF header
	f.Write([]byte("RIFF"))
	fileSize := 36 + len(a.Samples)*2
	binary.Write(f, binary.LittleEndian, int32(fileSize))
	f.Write([]byte("WAVE"))

	// Write fmt chunk
	f.Write([]byte("fmt "))
	binary.Write(f, binary.LittleEndian, int32(16)) // fmt chunk size
	binary.Write(f, binary.LittleEndian, int16(1))  // PCM format
	binary.Write(f, binary.LittleEndian, int16(a.Channels))
	binary.Write(f, binary.LittleEndian, int32(a.SampleRate))
	byteRate := a.SampleRate * a.Channels * 2
	binary.Write(f, binary.LittleEndian, int32(byteRate))
	blockAlign := int16(a.Channels * 2)
	binary.Write(f, binary.LittleEndian, blockAlign)
	binary.Write(f, binary.LittleEndian, int16(16)) // 16-bit

	// Write data chunk
	f.Write([]byte("data"))
	binary.Write(f, binary.LittleEndian, int32(len(a.Samples)*2))

	// Write samples
	for _, sample := range a.Samples {
		sampleInt := int16(math.Max(-1, math.Min(1, sample)) * 32767)
		binary.Write(f, binary.LittleEndian, sampleInt)
	}

	return nil
}

// Resample resamples the audio to a new sample rate
func (a *Audio) Resample(targetRate int) *Audio {
	if a.SampleRate == targetRate {
		return a
	}

	ratio := float64(targetRate) / float64(a.SampleRate)
	newLen := int(float64(len(a.Samples)) * ratio)
	newSamples := make([]float64, newLen)

	for i := 0; i < newLen; i++ {
		pos := float64(i) / ratio
		idx := int(pos)
		frac := pos - float64(idx)

		if idx >= len(a.Samples)-1 {
			newSamples[i] = a.Samples[len(a.Samples)-1]
		} else {
			newSamples[i] = a.Samples[idx]*(1-frac) + a.Samples[idx+1]*frac
		}
	}

	return &Audio{
		SampleRate: targetRate,
		Channels:   a.Channels,
		Samples:    newSamples,
		Duration:   a.Duration,
	}
}

// Normalize normalizes audio to [-1, 1] range
func (a *Audio) Normalize() {
	maxVal := 0.0
	for _, s := range a.Samples {
		if math.Abs(s) > maxVal {
			maxVal = math.Abs(s)
		}
	}

	if maxVal > 0 {
		for i := range a.Samples {
			a.Samples[i] /= maxVal
		}
	}
}

// PreEmphasis applies pre-emphasis filter to boost high frequencies
func (a *Audio) PreEmphasis(coef float64) {
	for i := len(a.Samples) - 1; i > 0; i-- {
		a.Samples[i] = a.Samples[i] - coef*a.Samples[i-1]
	}
	a.Samples[0] *= (1 - coef)
}
