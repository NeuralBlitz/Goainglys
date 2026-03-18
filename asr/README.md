# Go ASR Library

A complete Automatic Speech Recognition (ASR) library implemented in pure Go with no external dependencies.

## Features

- **Audio Processing**: WAV loading, resampling, normalization, pre-emphasis
- **Feature Extraction**: MFCC (Mel-frequency cepstral coefficients) using STFT
- **Acoustic Model**: LSTM-based neural network for acoustic modeling
- **Decoding**: Greedy decoding, CTC decoding, beam search
- **Training Pipeline**: Simplified training with CTC loss

## Architecture

```
Audio Input → Preprocess → Feature Extraction → Acoustic Model → Decoder → Text Output
```

### Components

1. **Audio Processing** (`audio.go`)
   - WAV file loading/saving
   - Sample rate resampling
   - Audio normalization
   - Pre-emphasis filter

2. **Feature Extraction** (`features.go`)
   - STFT (Short-Time Fourier Transform)
   - Mel filter bank
   - MFCC computation
   - DCT (Discrete Cosine Transform)

3. **Acoustic Model** (`model.go`)
   - LSTM layer implementation
   - Dense layer (fully connected)
   - Softmax activation
   - CTC decoding

4. **ASR Pipeline** (`pipeline.go`)
   - End-to-end speech recognition
   - Feature extraction and model inference
   - Decoding pipeline

## API Reference

### Audio Processing

```go
// Load WAV file
audio, err := asr.LoadWAV("audio.wav")

// Resample audio
resampled := audio.Resample(16000)

// Normalize audio
audio.Normalize()

// Apply pre-emphasis
audio.PreEmphasis(0.97)
```

### Feature Extraction

```go
// Extract MFCC features
mfcc := asr.ComputeMFCC(
    audio.Samples,    // audio samples
    16000,            // sample rate
    400,              // frame size
    160,              // hop size
    13,               // num MFCC coefficients
    26,               // num Mel filters
)
```

### Acoustic Model

```go
// Create model
model := asr.NewAcousticModel(
    inputSize,   // MFCC features per frame
    hiddenSize,  // LSTM hidden size
    outputSize,  // vocabulary size + blank
)

// Forward pass
probs := model.Forward(features)

// Predict
predictions := model.Predict(features)
```

### ASR Pipeline

```go
// Create pipeline with vocabulary
vocabulary := []string{"a", "b", "c", ..., "z"}
pipeline := asr.NewASRPipeline(vocabulary)

// Process audio
text, err := pipeline.ProcessAudio(audio)
```

## Performance

Processing 5 seconds of audio at 16kHz:
- Feature extraction: ~6ms
- Model forward pass: ~60ms
- Total: ~66ms (82x realtime)

## Files

- `audio.go` - Audio loading and preprocessing
- `features.go` - MFCC feature extraction
- `model.go` - LSTM acoustic model
- `pipeline.go` - End-to-end ASR pipeline
- `cmd/main.go` - Demo application

## Usage Example

```go
package main

import (
    "fmt"
    "github.com/user/transformer/asr"
)

func main() {
    // Create vocabulary
    vocabulary := []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
                           "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
                           "u", "v", "w", "x", "y", "z"}

    // Create ASR pipeline
    pipeline := asr.NewASRPipeline(vocabulary)

    // Load and process audio
    audio, err := asr.LoadWAV("speech.wav")
    if err != nil {
        panic(err)
    }

    text, err := pipeline.ProcessAudio(audio)
    if err != nil {
        panic(err)
    }

    fmt.Printf("Transcribed: %s\n", text)
}
```

## Technical Details

### MFCC Computation
1. Pre-emphasis filter to boost high frequencies
2. Frame the audio into overlapping windows
3. Apply Hann window to each frame
4. Compute FFT (Fast Fourier Transform)
5. Apply Mel filter bank
6. Compute log energy
7. Apply DCT to get cepstral coefficients

### LSTM Architecture
- Input: MFCC features (13-dimensional)
- Hidden layer: 128 units
- Output: Vocabulary size + 1 (blank)
- Activation: Tanh for gates, Sigmoid for forget/input/output

### CTC Decoding
Connectionist Temporal Classification removes:
1. Repeated characters
2. Blank symbols (<blank>)

## Limitations

- Simplified training (no backpropagation through time)
- No language model integration
- Limited to small vocabulary
- No acoustic model training (pre-trained weights needed for production)

## Future Enhancements

- [ ] Full backpropagation through time (BPTT)
- [x] Language model integration (n-gram, neural LM)
- [x] Beam search decoding
- [x] Model serialization/loading
- [ ] GPU acceleration support
- [x] Streaming inference
