package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"time"
)

// Keep these in sync with the format emitted by ExternalRecorder
// (pcm_s16le / 16000 Hz / mono) so concatenation is always homogeneous
// across mic and output capture sources. The recorder's output format is the
// single source of truth; if it ever changes these constants must change too.
const (
	concatSampleRateHz  uint32 = 16000
	concatChannels      uint16 = 1
	concatBitsPerSample uint16 = 16
)

// ConcatSampleRateHz exposes the canonical PCM sample rate written by the
// recorder backends so callers (the segmenter) can keep their analysis in sync
// without duplicating the magic number.
func ConcatSampleRateHz() uint32 { return concatSampleRateHz }

// ConcatChannels exposes the canonical channel count written by the recorder
// backends. The segmenter uses it to keep its energy analysis aligned with the
// recorder's actual PCM layout.
func ConcatChannels() uint16 { return concatChannels }

type wavFormat struct {
	audioFormat   uint16
	channels      uint16
	sampleRateHz  uint32
	bitsPerSample uint16
}

// RMSLevelFromWAV parses a PCM WAV file and returns a normalized RMS level in [0, 1].
func RMSLevelFromWAV(path string) (float64, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer file.Close()
	return RMSLevelFromWAVReader(file)
}

// RMSLevelFromWAVReader parses a PCM WAV stream and returns a normalized RMS level in [0, 1].
func RMSLevelFromWAVReader(r io.Reader) (float64, error) {
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return 0, fmt.Errorf("read WAV header: %w", err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return 0, errors.New("not a RIFF/WAVE file")
	}

	var format wavFormat
	haveFormat := false
	for {
		var header [8]byte
		_, err := io.ReadFull(r, header[:])
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return 0, fmt.Errorf("read WAV chunk header: %w", err)
		}
		chunkID := string(header[0:4])
		chunkSize := binary.LittleEndian.Uint32(header[4:8])
		limited := &io.LimitedReader{R: r, N: int64(chunkSize)}

		switch chunkID {
		case "fmt ":
			parsed, err := readWAVFormat(limited)
			if err != nil {
				return 0, err
			}
			format = parsed
			haveFormat = true
		case "data":
			if !haveFormat {
				return 0, errors.New("WAV data chunk appeared before fmt chunk")
			}
			level, err := pcmRMSLevel(limited, format)
			if err != nil {
				return 0, err
			}
			return level, consumeWAVPadding(r, limited, chunkSize)
		}

		if err := consumeWAVPadding(r, limited, chunkSize); err != nil {
			return 0, err
		}
	}
	return 0, errors.New("WAV data chunk not found")
}

func readWAVFormat(r io.Reader) (wavFormat, error) {
	var data [16]byte
	if _, err := io.ReadFull(r, data[:]); err != nil {
		return wavFormat{}, fmt.Errorf("read WAV fmt chunk: %w", err)
	}
	format := wavFormat{
		audioFormat:   binary.LittleEndian.Uint16(data[0:2]),
		channels:      binary.LittleEndian.Uint16(data[2:4]),
		sampleRateHz:  binary.LittleEndian.Uint32(data[4:8]),
		bitsPerSample: binary.LittleEndian.Uint16(data[14:16]),
	}
	if format.channels == 0 {
		return wavFormat{}, errors.New("WAV channel count is zero")
	}
	return format, nil
}

func consumeWAVPadding(r io.Reader, limited *io.LimitedReader, chunkSize uint32) error {
	if limited.N > 0 {
		if _, err := io.Copy(io.Discard, limited); err != nil {
			return err
		}
	}
	if chunkSize%2 == 1 {
		var pad [1]byte
		_, err := io.ReadFull(r, pad[:])
		return err
	}
	return nil
}

func pcmRMSLevel(r io.Reader, format wavFormat) (float64, error) {
	data, err := io.ReadAll(r)
	if err != nil {
		return 0, err
	}
	if len(data) == 0 {
		return 0, errors.New("WAV data chunk is empty")
	}

	var sumSquares float64
	var samples int
	switch {
	case format.audioFormat == 1 && format.bitsPerSample == 8:
		for _, sample := range data {
			normalized := (float64(sample) - 128) / 128
			sumSquares += normalized * normalized
			samples++
		}
	case format.audioFormat == 1 && format.bitsPerSample == 16:
		if len(data)%2 != 0 {
			return 0, errors.New("WAV PCM16 data length is not sample-aligned")
		}
		for i := 0; i < len(data); i += 2 {
			sample := int16(binary.LittleEndian.Uint16(data[i : i+2]))
			normalized := float64(sample) / 32768
			sumSquares += normalized * normalized
			samples++
		}
	case format.audioFormat == 1 && format.bitsPerSample == 24:
		if len(data)%3 != 0 {
			return 0, errors.New("WAV PCM24 data length is not sample-aligned")
		}
		for i := 0; i < len(data); i += 3 {
			sample := int32(data[i]) | int32(data[i+1])<<8 | int32(data[i+2])<<16
			if sample&0x800000 != 0 {
				sample |= ^0xffffff
			}
			normalized := float64(sample) / 8388608
			sumSquares += normalized * normalized
			samples++
		}
	case format.audioFormat == 1 && format.bitsPerSample == 32:
		if len(data)%4 != 0 {
			return 0, errors.New("WAV PCM32 data length is not sample-aligned")
		}
		for i := 0; i < len(data); i += 4 {
			sample := int32(binary.LittleEndian.Uint32(data[i : i+4]))
			normalized := float64(sample) / 2147483648
			sumSquares += normalized * normalized
			samples++
		}
	case format.audioFormat == 3 && format.bitsPerSample == 32:
		if len(data)%4 != 0 {
			return 0, errors.New("WAV float32 data length is not sample-aligned")
		}
		for i := 0; i < len(data); i += 4 {
			normalized := float64(math.Float32frombits(binary.LittleEndian.Uint32(data[i : i+4])))
			sumSquares += normalized * normalized
			samples++
		}
	default:
		return 0, fmt.Errorf("unsupported WAV format %d with %d bits per sample", format.audioFormat, format.bitsPerSample)
	}
	if samples == 0 {
		return 0, errors.New("WAV data has no samples")
	}
	level := math.Sqrt(sumSquares / float64(samples))
	if level < 0 {
		return 0, nil
	}
	if level > 1 {
		return 1, nil
	}
	return level, nil
}

// PCM16SamplesFromWAV reads a PCM16 WAV file and returns the mono 16-bit PCM
// samples as int16 values. It only supports the homogeneous PCM16 16kHz mono
// format produced by this app's recorder backends.
func PCM16SamplesFromWAV(r io.Reader) ([]int16, error) {
	var riff [12]byte
	if _, err := io.ReadFull(r, riff[:]); err != nil {
		return nil, fmt.Errorf("read WAV header: %w", err)
	}
	if string(riff[0:4]) != "RIFF" || string(riff[8:12]) != "WAVE" {
		return nil, errors.New("not a RIFF/WAVE file")
	}
	var format wavFormat
	haveFormat := false
	for {
		var header [8]byte
		_, err := io.ReadFull(r, header[:])
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read WAV chunk header: %w", err)
		}
		chunkID := string(header[0:4])
		chunkSize := binary.LittleEndian.Uint32(header[4:8])
		limited := &io.LimitedReader{R: r, N: int64(chunkSize)}
		switch chunkID {
		case "fmt ":
			parsed, err := readWAVFormat(limited)
			if err != nil {
				return nil, err
			}
			format = parsed
			haveFormat = true
		case "data":
			if !haveFormat {
				return nil, errors.New("WAV data chunk appeared before fmt chunk")
			}
			payload, err := io.ReadAll(limited)
			if err != nil {
				return nil, err
			}
			if err := consumeWAVPadding(r, limited, chunkSize); err != nil {
				return nil, fmt.Errorf("consume WAV data padding: %w", err)
			}
			if format.audioFormat != 1 || format.bitsPerSample != 16 {
				return nil, fmt.Errorf("unsupported WAV format %d with %d bits per sample (only PCM 16-bit supported)", format.audioFormat, format.bitsPerSample)
			}
			if format.channels != concatChannels || format.sampleRateHz != concatSampleRateHz {
				return nil, fmt.Errorf("unsupported WAV layout: %d Hz, %d channels (expected %d Hz, %d channel)", format.sampleRateHz, format.channels, concatSampleRateHz, concatChannels)
			}
			if len(payload)%2 != 0 {
				return nil, errors.New("WAV PCM16 data length is not sample-aligned")
			}
			samples := make([]int16, len(payload)/2)
			for i := 0; i < len(payload); i += 2 {
				samples[i/2] = int16(binary.LittleEndian.Uint16(payload[i : i+2]))
			}
			return samples, nil
		}
		if err := consumeWAVPadding(r, limited, chunkSize); err != nil {
			return nil, err
		}
	}
	return nil, errors.New("WAV data chunk not found")
}

// FramesWithEnergy splits PCM samples into fixed-duration frames and reports
// the RMS amplitude of each frame. sampleRateHz and channels must match the
// recorder output (16000 Hz / mono). framesPerSecond is the desired analysis
// cadence (e.g. 10 = one 100ms frame). Callers translate a threshold in the
// legacy signed-int16 amplitude scale to a normalized [0, 1] RMS threshold before
// passing it in (see Segmenter.IsLoud).
func FramesWithEnergy(samples []int16, sampleRateHz uint32, channels uint16, framesPerSecond int) []float64 {
	if framesPerSecond <= 0 {
		return nil
	}
	frameSamples := int(sampleRateHz) * int(channels) / framesPerSecond
	if frameSamples <= 0 {
		return nil
	}
	var frames []float64
	for start := 0; start+frameSamples <= len(samples); start += frameSamples {
		var sumSquares float64
		for j := start; j < start+frameSamples; j++ {
			normalized := float64(samples[j]) / 32768
			sumSquares += normalized * normalized
		}
		frames = append(frames, math.Sqrt(sumSquares/float64(frameSamples)))
	}
	return frames
}

// TrailingSilentFrames counts how many of the most recent energy frames are
// below the threshold. This drives the "have we reached the silence limit"
// check in the segmenter.
func TrailingSilentFrames(energy []float64, threshold float64) int {
	count := 0
	for i := len(energy) - 1; i >= 0; i-- {
		if energy[i] < threshold {
			count++
			continue
		}
		break
	}
	return count
}

// WritePCM16WAV serializes a PCM16 mono WAV file to w with the canonical
// 44-byte header used by this app. It is the inverse of PCM16SamplesFromWAV
// for the recorder's homogeneous format.
func WritePCM16WAV(w io.Writer, samples []int16, sampleRateHz uint32) error {
	const (
		channelCount  = concatChannels
		bitsPerSample = concatBitsPerSample
	)
	byteRate := sampleRateHz * uint32(channelCount) * uint32(bitsPerSample) / 8
	blockAlign := channelCount * uint16(bitsPerSample) / 8
	dataSize := uint32(len(samples) * int(bitsPerSample/8))
	header := struct {
		Riff          [4]byte
		ChunkSize     uint32
		Wave          [4]byte
		Fmt           [4]byte
		SubchunkSize  uint32
		AudioFormat   uint16
		Channels      uint16
		SampleRate    uint32
		ByteRate      uint32
		BlockAlign    uint16
		BitsPerSample uint16
		Data          [4]byte
		DataSize      uint32
	}{
		Riff:          [4]byte{'R', 'I', 'F', 'F'},
		ChunkSize:     36 + dataSize,
		Wave:          [4]byte{'W', 'A', 'V', 'E'},
		Fmt:           [4]byte{'f', 'm', 't', ' '},
		SubchunkSize:  16,
		AudioFormat:   1,
		Channels:      channelCount,
		SampleRate:    sampleRateHz,
		ByteRate:      byteRate,
		BlockAlign:    blockAlign,
		BitsPerSample: bitsPerSample,
		Data:          [4]byte{'d', 'a', 't', 'a'},
		DataSize:      dataSize,
	}
	if err := binary.Write(w, binary.LittleEndian, header); err != nil {
		return err
	}
	for _, sample := range samples {
		if err := binary.Write(w, binary.LittleEndian, sample); err != nil {
			return err
		}
	}
	return nil
}

// ConcatenatePCM16WAV reads each WAV path, decodes its PCM16 samples, and
// writes the concatenation of those samples into a single new PCM16 WAV file
// at destinationPath. Each input file must be PCM16 mono; channel/sample-rate
// mismatches produce an error so callers never silently mix formats.
func ConcatenatePCM16WAV(sourcePaths []string, destinationPath string) error {
	if len(sourcePaths) == 0 {
		return errors.New("concatenate requires at least one source WAV")
	}
	var combined []int16
	for _, src := range sourcePaths {
		file, err := os.Open(src)
		if err != nil {
			return err
		}
		samples, err := PCM16SamplesFromWAV(file)
		file.Close()
		if err != nil {
			return fmt.Errorf("read %s: %w", src, err)
		}
		combined = append(combined, samples...)
	}
	if err := os.MkdirAll(filepath.Dir(destinationPath), 0o700); err != nil {
		return err
	}
	out, err := os.OpenFile(destinationPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	defer out.Close()
	return WritePCM16WAV(out, combined, concatSampleRateHz)
}

// SampleDurationOf returns the wall-clock playback duration encoded by count
// PCM16 samples at the recorder's canonical sample rate / channel count.
func SampleDurationOf(sampleCount int) time.Duration {
	if sampleCount <= 0 {
		return 0
	}
	seconds := float64(sampleCount) / float64(concatSampleRateHz*uint32(concatChannels))
	return time.Duration(seconds * float64(time.Second))
}
