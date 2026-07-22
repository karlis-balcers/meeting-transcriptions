package main

import (
	"strings"
	"testing"
	"time"
)

func TestParseRecordOptionsAcceptsSecondsAndDefaultsDevice(t *testing.T) {
	options, err := parseRecordOptions([]string{"--output-file", "out.wav", "--duration", "1.250"})
	if err != nil {
		t.Fatal(err)
	}
	if options.OutputFile != "out.wav" || options.Duration != 1250*time.Millisecond || options.DeviceID != "default" || options.TargetSampleRate != 16000 || options.TargetChannels != 1 {
		t.Fatalf("unexpected parsed options: %+v", options)
	}
}

func TestParseRecordOptionsRejectsMissingOutput(t *testing.T) {
	_, err := parseRecordOptions([]string{"--duration", "2"})
	if err == nil || !strings.Contains(err.Error(), "--output-file") {
		t.Fatalf("expected missing output-file error, got %v", err)
	}
}

func TestParseRecordOptionsAllowsDeviceNameWithoutDeviceID(t *testing.T) {
	options, err := parseRecordOptions([]string{"--output-file", "out.wav", "--duration", "2", "--device-name", "Speakers (Realtek(R) Audio) [Loopback]"})
	if err != nil {
		t.Fatal(err)
	}
	if options.DeviceID != "" || options.DeviceName != "Speakers (Realtek(R) Audio) [Loopback]" {
		t.Fatalf("device-name-only selection should not be overwritten by default ID, got %+v", options)
	}
}

func TestStripLoopbackDisplaySuffix(t *testing.T) {
	got := stripLoopbackDisplaySuffix("Speakers (Realtek(R) Audio) [Loopback]")
	if got != "Speakers (Realtek(R) Audio)" {
		t.Fatalf("unexpected suffix stripping: %q", got)
	}
}
