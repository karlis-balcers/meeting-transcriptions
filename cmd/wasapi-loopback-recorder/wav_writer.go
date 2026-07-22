package main

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
)

type captureSampleKind int

const (
	captureSamplePCM captureSampleKind = iota + 1
	captureSampleFloat
)

type captureFormat struct {
	SampleRate    int
	Channels      int
	BitsPerSample int
	BlockAlign    int
	SampleKind    captureSampleKind
}

type pcm16WAVWriter struct {
	file       *os.File
	sampleRate int
	channels   int
	dataBytes  uint32
	closed     bool
}

func newPCM16WAVWriter(path string, sampleRate int, channels int) (*pcm16WAVWriter, error) {
	if sampleRate <= 0 {
		return nil, errors.New("sample rate must be positive")
	}
	if channels <= 0 {
		return nil, errors.New("channel count must be positive")
	}
	file, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	writer := &pcm16WAVWriter{file: file, sampleRate: sampleRate, channels: channels}
	if err := writer.writeHeader(); err != nil {
		_ = file.Close()
		return nil, err
	}
	return writer, nil
}

func (w *pcm16WAVWriter) writeHeader() error {
	byteRate := uint32(w.sampleRate * w.channels * 2)
	blockAlign := uint16(w.channels * 2)
	header := make([]byte, 44)
	copy(header[0:4], "RIFF")
	copy(header[8:12], "WAVE")
	copy(header[12:16], "fmt ")
	binary.LittleEndian.PutUint32(header[16:20], 16)
	binary.LittleEndian.PutUint16(header[20:22], 1)
	binary.LittleEndian.PutUint16(header[22:24], uint16(w.channels))
	binary.LittleEndian.PutUint32(header[24:28], uint32(w.sampleRate))
	binary.LittleEndian.PutUint32(header[28:32], byteRate)
	binary.LittleEndian.PutUint16(header[32:34], blockAlign)
	binary.LittleEndian.PutUint16(header[34:36], 16)
	copy(header[36:40], "data")
	_, err := w.file.Write(header)
	return err
}

func (w *pcm16WAVWriter) WriteSamples(samples []int16) error {
	if w.closed {
		return errors.New("write to closed WAV writer")
	}
	if len(samples) == 0 {
		return nil
	}
	if len(samples)%w.channels != 0 {
		return fmt.Errorf("sample count %d is not aligned to %d channel(s)", len(samples), w.channels)
	}
	if uint64(w.dataBytes)+uint64(len(samples)*2) > math.MaxUint32 {
		return errors.New("WAV data exceeds RIFF 4 GiB limit")
	}
	buf := make([]byte, len(samples)*2)
	for i, sample := range samples {
		binary.LittleEndian.PutUint16(buf[i*2:i*2+2], uint16(sample))
	}
	if _, err := w.file.Write(buf); err != nil {
		return err
	}
	w.dataBytes += uint32(len(buf))
	return nil
}

func (w *pcm16WAVWriter) WriteSilence(frames int) error {
	if frames <= 0 {
		return nil
	}
	return w.WriteSamples(make([]int16, frames*w.channels))
}

func (w *pcm16WAVWriter) FramesWritten() int {
	if w.channels <= 0 {
		return 0
	}
	return int(w.dataBytes) / (2 * w.channels)
}

func (w *pcm16WAVWriter) Close() error {
	if w.closed {
		return nil
	}
	w.closed = true
	var firstErr error
	riffSize := uint32(36) + w.dataBytes
	if _, err := w.file.Seek(4, 0); err != nil {
		firstErr = err
	} else {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], riffSize)
		if _, err := w.file.Write(buf[:]); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	if _, err := w.file.Seek(40, 0); err != nil && firstErr == nil {
		firstErr = err
	} else if firstErr == nil {
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], w.dataBytes)
		if _, err := w.file.Write(buf[:]); err != nil {
			firstErr = err
		}
	}
	if err := w.file.Close(); err != nil && firstErr == nil {
		firstErr = err
	}
	return firstErr
}

type pcm16Converter struct {
	input       captureFormat
	outRate     int
	outChannels int
	resampler   linearResampler
}

func newPCM16Converter(input captureFormat, outRate int, outChannels int) (*pcm16Converter, error) {
	if err := validateCaptureFormat(input); err != nil {
		return nil, err
	}
	if outRate <= 0 {
		return nil, errors.New("target sample rate must be positive")
	}
	if outChannels <= 0 {
		return nil, errors.New("target channel count must be positive")
	}
	return &pcm16Converter{
		input:       input,
		outRate:     outRate,
		outChannels: outChannels,
		resampler:   newLinearResampler(input.SampleRate, outRate),
	}, nil
}

func validateCaptureFormat(format captureFormat) error {
	if format.SampleRate <= 0 {
		return errors.New("capture sample rate is zero")
	}
	if format.Channels <= 0 {
		return errors.New("capture channel count is zero")
	}
	if format.BitsPerSample <= 0 || format.BitsPerSample%8 != 0 {
		return fmt.Errorf("unsupported capture bits per sample %d", format.BitsPerSample)
	}
	minimumBlockAlign := format.Channels * (format.BitsPerSample / 8)
	if format.BlockAlign < minimumBlockAlign {
		return fmt.Errorf("capture block align %d is smaller than channel sample width %d", format.BlockAlign, minimumBlockAlign)
	}
	switch format.SampleKind {
	case captureSamplePCM:
		switch format.BitsPerSample {
		case 8, 16, 24, 32:
			return nil
		default:
			return fmt.Errorf("unsupported PCM bit depth %d", format.BitsPerSample)
		}
	case captureSampleFloat:
		if format.BitsPerSample == 32 {
			return nil
		}
		return fmt.Errorf("unsupported float bit depth %d", format.BitsPerSample)
	default:
		return errors.New("unsupported capture sample format")
	}
}

func (c *pcm16Converter) ConvertPacket(data []byte, frames uint32, silent bool) ([]int16, error) {
	if frames == 0 {
		return nil, nil
	}
	mono, err := c.decodeMono(data, int(frames), silent)
	if err != nil {
		return nil, err
	}
	return c.encodeMono(c.resampler.Append(mono)), nil
}

func (c *pcm16Converter) Flush() []int16 {
	return c.encodeMono(c.resampler.Flush())
}

func (c *pcm16Converter) decodeMono(data []byte, frames int, silent bool) ([]float64, error) {
	if silent {
		return make([]float64, frames), nil
	}
	wantBytes := frames * c.input.BlockAlign
	if len(data) < wantBytes {
		return nil, fmt.Errorf("capture packet has %d byte(s), expected at least %d", len(data), wantBytes)
	}
	mono := make([]float64, frames)
	sampleBytes := c.input.BitsPerSample / 8
	for frame := 0; frame < frames; frame++ {
		frameOffset := frame * c.input.BlockAlign
		var sum float64
		for channel := 0; channel < c.input.Channels; channel++ {
			sampleOffset := frameOffset + channel*sampleBytes
			sum += decodeSample(data[sampleOffset:sampleOffset+sampleBytes], c.input)
		}
		mono[frame] = sum / float64(c.input.Channels)
	}
	return mono, nil
}

func decodeSample(data []byte, format captureFormat) float64 {
	switch format.SampleKind {
	case captureSampleFloat:
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(data[:4])))
	case captureSamplePCM:
		switch format.BitsPerSample {
		case 8:
			return (float64(data[0]) - 128) / 128
		case 16:
			return float64(int16(binary.LittleEndian.Uint16(data[:2]))) / 32768
		case 24:
			sample := int32(data[0]) | int32(data[1])<<8 | int32(data[2])<<16
			if sample&0x800000 != 0 {
				sample |= ^0xffffff
			}
			return float64(sample) / 8388608
		case 32:
			return float64(int32(binary.LittleEndian.Uint32(data[:4]))) / 2147483648
		}
	}
	return 0
}

func (c *pcm16Converter) encodeMono(samples []float64) []int16 {
	if len(samples) == 0 {
		return nil
	}
	encoded := make([]int16, 0, len(samples)*c.outChannels)
	for _, sample := range samples {
		pcm := floatToPCM16(sample)
		for channel := 0; channel < c.outChannels; channel++ {
			encoded = append(encoded, pcm)
		}
	}
	return encoded
}

func floatToPCM16(sample float64) int16 {
	if sample > 1 {
		sample = 1
	} else if sample < -1 {
		sample = -1
	}
	if sample >= 0 {
		return int16(math.Round(sample * 32767))
	}
	return int16(math.Round(sample * 32768))
}

type linearResampler struct {
	inRate float64
	step   float64
	buffer []float64
	pos    float64
}

func newLinearResampler(inRate int, outRate int) linearResampler {
	return linearResampler{inRate: float64(inRate), step: float64(inRate) / float64(outRate)}
}

func (r *linearResampler) Append(samples []float64) []float64 {
	if len(samples) == 0 {
		return nil
	}
	r.buffer = append(r.buffer, samples...)
	var out []float64
	for r.pos+1 < float64(len(r.buffer)) {
		index := int(r.pos)
		frac := r.pos - float64(index)
		left := r.buffer[index]
		right := r.buffer[index+1]
		out = append(out, left+(right-left)*frac)
		r.pos += r.step
	}
	r.discardConsumed()
	return out
}

func (r *linearResampler) Flush() []float64 {
	if len(r.buffer) == 0 {
		return nil
	}
	var out []float64
	for r.pos < float64(len(r.buffer)) {
		index := int(r.pos)
		if index >= len(r.buffer) {
			break
		}
		out = append(out, r.buffer[index])
		r.pos += r.step
	}
	r.buffer = nil
	r.pos = 0
	return out
}

func (r *linearResampler) discardConsumed() {
	consumed := int(r.pos)
	if consumed <= 0 {
		return
	}
	if consumed >= len(r.buffer) {
		consumed = len(r.buffer) - 1
	}
	r.buffer = r.buffer[consumed:]
	r.pos -= float64(consumed)
}
