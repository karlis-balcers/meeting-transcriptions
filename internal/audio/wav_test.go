package audio

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestRMSLevelFromWAVReaderPCM16(t *testing.T) {
	wav := pcm16WAV([]int16{0, 32767, -32768, 0})
	level, err := RMSLevelFromWAVReader(bytes.NewReader(wav))
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(level-math.Sqrt(0.5)) > 0.001 {
		t.Fatalf("unexpected RMS level: got %.6f want about %.6f", level, math.Sqrt(0.5))
	}
}

func TestRMSLevelFromWAVReaderSilence(t *testing.T) {
	wav := pcm16WAV([]int16{0, 0, 0, 0})
	level, err := RMSLevelFromWAVReader(bytes.NewReader(wav))
	if err != nil {
		t.Fatal(err)
	}
	if level != 0 {
		t.Fatalf("silence should have zero level, got %.6f", level)
	}
}

func TestRMSLevelFromWAVReaderRejectsNonWAV(t *testing.T) {
	_, err := RMSLevelFromWAVReader(bytes.NewReader([]byte("not a wave")))
	if err == nil {
		t.Fatal("expected invalid WAV error")
	}
}

func pcm16WAV(samples []int16) []byte {
	var data bytes.Buffer
	for _, sample := range samples {
		_ = binary.Write(&data, binary.LittleEndian, sample)
	}
	dataSize := uint32(data.Len())

	var wav bytes.Buffer
	wav.WriteString("RIFF")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(36)+dataSize)
	wav.WriteString("WAVE")
	wav.WriteString("fmt ")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(1))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(1))
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16000))
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16000*2))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(2))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(16))
	wav.WriteString("data")
	_ = binary.Write(&wav, binary.LittleEndian, dataSize)
	wav.Write(data.Bytes())
	return wav.Bytes()
}
