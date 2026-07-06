package main

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestPCM16WAVWriterFinalizesRIFFSizes(t *testing.T) {
	path := filepath.Join(t.TempDir(), "capture.wav")
	writer, err := newPCM16WAVWriter(path, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteSamples([]int16{0, 1000, -1000, 32767}); err != nil {
		t.Fatal(err)
	}
	if got := writer.FramesWritten(); got != 4 {
		t.Fatalf("unexpected frame count before close: got %d want 4", got)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data[0:4]) != "RIFF" || string(data[8:12]) != "WAVE" || string(data[36:40]) != "data" {
		t.Fatalf("invalid WAV header: %q %q %q", data[0:4], data[8:12], data[36:40])
	}
	if got, want := binary.LittleEndian.Uint32(data[4:8]), uint32(44+8-8); got != want {
		t.Fatalf("unexpected RIFF size: got %d want %d", got, want)
	}
	if got, want := binary.LittleEndian.Uint32(data[40:44]), uint32(8); got != want {
		t.Fatalf("unexpected data size: got %d want %d", got, want)
	}
}

func TestPCM16ConverterDownmixesFloatStereoAndResamples(t *testing.T) {
	format := captureFormat{SampleRate: 48000, Channels: 2, BitsPerSample: 32, BlockAlign: 8, SampleKind: captureSampleFloat}
	converter, err := newPCM16Converter(format, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	frames := make([]float32, 48*2)
	for frame := 0; frame < 48; frame++ {
		frames[frame*2] = 0.5
		frames[frame*2+1] = -0.25
	}
	packet := make([]byte, len(frames)*4)
	for i, sample := range frames {
		binary.LittleEndian.PutUint32(packet[i*4:i*4+4], math.Float32bits(sample))
	}
	samples, err := converter.ConvertPacket(packet, 48, false)
	if err != nil {
		t.Fatal(err)
	}
	samples = append(samples, converter.Flush()...)
	if len(samples) < 15 || len(samples) > 17 {
		t.Fatalf("48 frames at 48k should become about 16 frames at 16k, got %d", len(samples))
	}
	for _, sample := range samples {
		if sample < 4000 || sample > 4100 {
			t.Fatalf("downmixed sample should be around 0.125 PCM16, got %d (all=%v)", sample, samples)
		}
	}
}

func TestPCM16ConverterWritesSilentPacket(t *testing.T) {
	format := captureFormat{SampleRate: 16000, Channels: 1, BitsPerSample: 16, BlockAlign: 2, SampleKind: captureSamplePCM}
	converter, err := newPCM16Converter(format, 16000, 1)
	if err != nil {
		t.Fatal(err)
	}
	samples, err := converter.ConvertPacket(nil, 4, true)
	if err != nil {
		t.Fatal(err)
	}
	samples = append(samples, converter.Flush()...)
	if len(samples) != 4 {
		t.Fatalf("expected four silent samples, got %d", len(samples))
	}
	for _, sample := range samples {
		if sample != 0 {
			t.Fatalf("silent sample should be zero, got %d", sample)
		}
	}
}
