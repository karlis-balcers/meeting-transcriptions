package audio

import (
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

type wavFormat struct {
	audioFormat   uint16
	channels      uint16
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
