package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"
)

const helperVersion = "wasapi-loopback-recorder protocol v1"

type recordOptions struct {
	OutputFile       string
	Duration         time.Duration
	DeviceID         string
	DeviceName       string
	TargetSampleRate int
	TargetChannels   int
}

func parseRecordOptions(args []string) (recordOptions, error) {
	fs := flag.NewFlagSet("record", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	outputFile := fs.String("output-file", "", "WAV file path to write")
	durationValue := fs.String("duration", "", "bounded capture duration in seconds")
	deviceID := fs.String("device-id", "", "Windows render endpoint ID, friendly name, or default")
	deviceName := fs.String("device-name", "", "Windows render endpoint display name")
	sampleRateValue := fs.String("sample-rate", "16000", "target output sample rate")
	channelsValue := fs.String("channels", "1", "target output channel count")
	if err := fs.Parse(args); err != nil {
		return recordOptions{}, err
	}
	if fs.NArg() > 0 {
		return recordOptions{}, fmt.Errorf("unexpected positional arguments: %s", strings.Join(fs.Args(), " "))
	}
	missing := missingFlags([]namedValue{
		{name: "--output-file", value: *outputFile},
		{name: "--duration", value: *durationValue},
	})
	if len(missing) > 0 {
		return recordOptions{}, fmt.Errorf("missing required %s", strings.Join(missing, ", "))
	}
	duration, err := parseDuration(*durationValue)
	if err != nil {
		return recordOptions{}, fmt.Errorf("invalid --duration %q: %w", *durationValue, err)
	}
	sampleRate, err := strconv.Atoi(strings.TrimSpace(*sampleRateValue))
	if err != nil || sampleRate <= 0 {
		return recordOptions{}, fmt.Errorf("invalid --sample-rate %q", *sampleRateValue)
	}
	channels, err := strconv.Atoi(strings.TrimSpace(*channelsValue))
	if err != nil || channels <= 0 {
		return recordOptions{}, fmt.Errorf("invalid --channels %q", *channelsValue)
	}
	selectedDeviceID := strings.TrimSpace(*deviceID)
	selectedDeviceName := strings.TrimSpace(*deviceName)
	if selectedDeviceID == "" && selectedDeviceName == "" {
		selectedDeviceID = "default"
	}
	return recordOptions{
		OutputFile:       strings.TrimSpace(*outputFile),
		Duration:         duration,
		DeviceID:         selectedDeviceID,
		DeviceName:       selectedDeviceName,
		TargetSampleRate: sampleRate,
		TargetChannels:   channels,
	}, nil
}

func parseDuration(value string) (time.Duration, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return 0, errors.New("empty duration")
	}
	if parsed, err := time.ParseDuration(trimmed); err == nil {
		if parsed <= 0 {
			return 0, errors.New("duration must be positive")
		}
		return parsed, nil
	}
	seconds, err := strconv.ParseFloat(trimmed, 64)
	if err != nil {
		return 0, err
	}
	if seconds <= 0 {
		return 0, errors.New("duration must be positive")
	}
	return time.Duration(seconds * float64(time.Second)), nil
}

type namedValue struct {
	name  string
	value string
}

func missingFlags(values []namedValue) []string {
	var missing []string
	for _, value := range values {
		if strings.TrimSpace(value.value) == "" {
			missing = append(missing, value.name)
		}
	}
	return missing
}

func isDefaultDeviceSelector(value string) bool {
	selector := strings.ToLower(strings.TrimSpace(value))
	return selector == "" || selector == "default" || selector == "default render" || selector == "default output"
}

func loopbackDisplayName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return ""
	}
	if strings.Contains(strings.ToLower(trimmed), "loopback") {
		return trimmed
	}
	return trimmed + " [Loopback]"
}

func stripLoopbackDisplaySuffix(name string) string {
	trimmed := strings.TrimSpace(name)
	return strings.TrimSpace(strings.TrimSuffix(trimmed, " [Loopback]"))
}
