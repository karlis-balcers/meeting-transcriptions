//go:build windows

package main

import (
	"encoding/binary"
	"testing"
	"unsafe"
)

func TestParseMixFormatExtensibleFloat32(t *testing.T) {
	data := make([]byte, 40)
	binary.LittleEndian.PutUint16(data[0:2], waveFormatTagExtensible)
	binary.LittleEndian.PutUint16(data[2:4], 2)
	binary.LittleEndian.PutUint32(data[4:8], 48000)
	binary.LittleEndian.PutUint32(data[8:12], 48000*2*4)
	binary.LittleEndian.PutUint16(data[12:14], 8)
	binary.LittleEndian.PutUint16(data[14:16], 32)
	binary.LittleEndian.PutUint16(data[16:18], 22)
	binary.LittleEndian.PutUint16(data[18:20], 32)
	binary.LittleEndian.PutUint32(data[20:24], 3)
	writeGUID(data[24:40], subtypeIEEEFloat)

	format, err := parseMixFormat(unsafe.Pointer(&data[0]))
	if err != nil {
		t.Fatal(err)
	}
	if format.SampleRate != 48000 || format.Channels != 2 || format.BitsPerSample != 32 || format.BlockAlign != 8 || format.SampleKind != captureSampleFloat {
		t.Fatalf("unexpected parsed format: %+v", format)
	}
}

func writeGUID(dst []byte, value guid) {
	binary.LittleEndian.PutUint32(dst[0:4], value.Data1)
	binary.LittleEndian.PutUint16(dst[4:6], value.Data2)
	binary.LittleEndian.PutUint16(dst[6:8], value.Data3)
	copy(dst[8:16], value.Data4[:])
}
