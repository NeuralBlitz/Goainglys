package asr

import (
	"math"
)

// Spectrogram represents a spectrogram of audio
type Spectrogram struct {
	FreqBins []float64
	TimeBins []float64
	Power    [][]float64
}

// MFCC represents Mel-frequency cepstral coefficients
type MFCC struct {
	Coeffs    [][]float64
	NumFrames int
	NumCoeffs int
}

// STFT computes Short-Time Fourier Transform
func STFT(samples []float64, sampleRate, frameSize, hopSize int) *Spectrogram {
	numFrames := (len(samples)-frameSize)/hopSize + 1
	power := make([][]float64, numFrames)

	for i := 0; i < numFrames; i++ {
		start := i * hopSize
		frame := samples[start : start+frameSize]
		// Apply window (Hann window)
		windowed := applyWindow(frame)
		// Compute FFT
		fft := computeFFT(windowed)
		// Compute power spectrum
		power[i] = make([]float64, frameSize/2)
		for j := 0; j < frameSize/2; j++ {
			real := fft[j*2]
			imag := fft[j*2+1]
			power[i][j] = real*real + imag*imag
		}
	}

	// Create frequency bins
	freqBins := make([]float64, frameSize/2)
	for i := 0; i < frameSize/2; i++ {
		freqBins[i] = float64(i*sampleRate) / float64(frameSize)
	}

	// Create time bins
	timeBins := make([]float64, numFrames)
	for i := 0; i < numFrames; i++ {
		timeBins[i] = float64(i*hopSize) / float64(sampleRate)
	}

	return &Spectrogram{
		FreqBins: freqBins,
		TimeBins: timeBins,
		Power:    power,
	}
}

// applyWindow applies Hann window to frame
func applyWindow(frame []float64) []float64 {
	windowed := make([]float64, len(frame))
	for i, val := range frame {
		w := 0.5 * (1 - math.Cos(2*math.Pi*float64(i)/float64(len(frame)-1)))
		windowed[i] = val * w
	}
	return windowed
}

// computeFFT computes FFT using Cooley-Tukey algorithm
func computeFFT(samples []float64) []float64 {
	n := len(samples)
	if n == 1 {
		return []float64{samples[0], 0}
	}

	// Convert to complex format
	complex := make([]float64, n*2)
	for i := 0; i < n; i++ {
		complex[i*2] = samples[i]
		complex[i*2+1] = 0
	}

	// Recursive FFT
	fftRecursive(complex, n)

	return complex
}

// fftRecursive performs recursive FFT
func fftRecursive(x []float64, n int) {
	if n <= 1 {
		return
	}

	// Split into even and odd
	even := make([]float64, n)
	odd := make([]float64, n)
	for i := 0; i < n/2; i++ {
		even[i*2] = x[i*2]
		even[i*2+1] = x[i*2+1]
		odd[i*2] = x[(i+n/2)*2]
		odd[i*2+1] = x[(i+n/2)*2+1]
	}

	fftRecursive(even, n/2)
	fftRecursive(odd, n/2)

	for k := 0; k < n/2; k++ {
		angle := -2 * math.Pi * float64(k) / float64(n)
		wReal := math.Cos(angle)
		wImag := math.Sin(angle)

		// Multiply odd by twiddle factor
		oddReal := odd[k*2]*wReal - odd[k*2+1]*wImag
		oddImag := odd[k*2]*wImag + odd[k*2+1]*wReal

		// Combine
		x[k*2] = even[k*2] + oddReal
		x[k*2+1] = even[k*2+1] + oddImag
		x[(k+n/2)*2] = even[k*2] - oddReal
		x[(k+n/2)*2+1] = even[k*2+1] - oddImag
	}
}

// MelFilterBank creates Mel filter bank
func MelFilterBank(numFilters, fftSize, sampleRate int) [][]float64 {
	filters := make([][]float64, numFilters)

	minFreq := 20.0
	maxFreq := float64(sampleRate / 2)
	minMel := freqToMel(minFreq)
	maxMel := freqToMel(maxFreq)

	// Create linearly spaced Mel points
	melPoints := make([]float64, numFilters+2)
	for i := 0; i < numFilters+2; i++ {
		melPoints[i] = minMel + (maxMel-minMel)*float64(i)/float64(numFilters+1)
	}

	// Convert back to frequency
	freqPoints := make([]float64, numFilters+2)
	for i := 0; i < numFilters+2; i++ {
		freqPoints[i] = melToFreq(melPoints[i])
	}

	// Create filters
	for i := 0; i < numFilters; i++ {
		filters[i] = make([]float64, fftSize/2)
		for j := 0; j < fftSize/2; j++ {
			freq := float64(j*sampleRate) / float64(fftSize)

			if freq < freqPoints[i] {
				filters[i][j] = 0
			} else if freq < freqPoints[i+1] {
				filters[i][j] = (freq - freqPoints[i]) / (freqPoints[i+1] - freqPoints[i])
			} else if freq < freqPoints[i+2] {
				filters[i][j] = (freqPoints[i+2] - freq) / (freqPoints[i+2] - freqPoints[i+1])
			} else {
				filters[i][j] = 0
			}
		}
	}

	return filters
}

func freqToMel(freq float64) float64 {
	return 2595 * math.Log10(1+freq/700)
}

func melToFreq(mel float64) float64 {
	return 700 * (math.Pow(10, mel/2595) - 1)
}

// ComputeMFCC computes MFCC features from audio
func ComputeMFCC(samples []float64, sampleRate, frameSize, hopSize, numCoeffs, numFilters int) *MFCC {
	// Compute spectrogram
	spec := STFT(samples, sampleRate, frameSize, hopSize)

	// Create Mel filter bank
	filters := MelFilterBank(numFilters, frameSize, sampleRate)

	// Apply filter bank
	numFrames := len(spec.Power)
	melSpec := make([][]float64, numFrames)
	for i := 0; i < numFrames; i++ {
		melSpec[i] = make([]float64, numFilters)
		for j := 0; j < numFilters; j++ {
			sum := 0.0
			for k := 0; k < frameSize/2; k++ {
				sum += spec.Power[i][k] * filters[j][k]
			}
			// Apply log scale
			melSpec[i][j] = math.Log10(sum + 1e-10)
		}
	}

	// Compute DCT (Discrete Cosine Transform)
	mfcc := &MFCC{
		Coeffs:    make([][]float64, numFrames),
		NumFrames: numFrames,
		NumCoeffs: numCoeffs,
	}

	for i := 0; i < numFrames; i++ {
		mfcc.Coeffs[i] = make([]float64, numCoeffs)
		for k := 0; k < numCoeffs; k++ {
			sum := 0.0
			for j := 0; j < numFilters; j++ {
				theta := math.Pi * float64(k) * (float64(j) + 0.5) / float64(numFilters)
				sum += melSpec[i][j] * math.Cos(theta)
			}
			mfcc.Coeffs[i][k] = sum * math.Sqrt(2.0/float64(numFilters))
		}
	}

	return mfcc
}
