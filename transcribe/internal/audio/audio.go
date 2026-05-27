package audio

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"
)

type Source string

const (
	SourceMic    Source = "mic"
	SourceOutput Source = "output"
)

type Device struct {
	ID      string
	Name    string
	Source  Source
	Default bool
	Backend string
}

func (d Device) DisplayName() string {
	if strings.TrimSpace(d.Name) != "" {
		return d.Name
	}
	return d.ID
}

type Preferences struct {
	MicDeviceID      string
	MicDeviceName    string
	OutputDeviceID   string
	OutputDeviceName string
}

type Selection struct {
	Mic    Device
	Output Device
}

type Discoverer interface {
	ListDevices(ctx context.Context) ([]Device, []string, error)
}

type Recorder interface {
	Validate(ctx context.Context, selected Selection) error
	RecordChunk(ctx context.Context, request ChunkRequest) (Chunk, error)
}

type ChunkRequest struct {
	Device   Device
	Source   Source
	TempDir  string
	Duration time.Duration
	Sequence uint64
	Started  time.Time
	Speaker  string
}

type Chunk struct {
	Source   Source
	Device   Device
	FilePath string
	Started  time.Time
	Ended    time.Time
	Sequence uint64
	Speaker  string
}

func SelectDevices(devices []Device, prefs Preferences) (Selection, []string, error) {
	mic, micWarnings, micErr := selectOne(devices, SourceMic, prefs.MicDeviceID, prefs.MicDeviceName)
	output, outputWarnings, outputErr := selectOne(devices, SourceOutput, prefs.OutputDeviceID, prefs.OutputDeviceName)
	warnings := append(micWarnings, outputWarnings...)
	if micErr != nil || outputErr != nil {
		var messages []string
		if micErr != nil {
			messages = append(messages, micErr.Error())
		}
		if outputErr != nil {
			messages = append(messages, outputErr.Error())
		}
		return Selection{}, warnings, errors.New(strings.Join(messages, "; "))
	}
	return Selection{Mic: mic, Output: output}, warnings, nil
}

func selectOne(devices []Device, source Source, preferredID, preferredName string) (Device, []string, error) {
	var candidates []Device
	for _, device := range devices {
		if device.Source == source {
			candidates = append(candidates, device)
		}
	}
	if len(candidates) == 0 {
		if source == SourceOutput {
			return Device{}, nil, errors.New("no system-output capture device found; configure a Pulse/PipeWire monitor source, WASAPI loopback, or virtual loopback device")
		}
		return Device{}, nil, errors.New("no microphone capture device found; connect a microphone or configure audio.mic_device_id")
	}

	if strings.TrimSpace(preferredID) != "" {
		for _, candidate := range candidates {
			if candidate.ID == preferredID {
				return candidate, nil, nil
			}
		}
		for _, candidate := range candidates {
			if normalize(candidate.ID) == normalize(preferredID) || normalize(candidate.Name) == normalize(preferredID) {
				return candidate, nil, nil
			}
		}
	}
	if strings.TrimSpace(preferredName) != "" {
		for _, candidate := range candidates {
			if normalize(candidate.Name) == normalize(preferredName) {
				return candidate, nil, nil
			}
		}
	}
	for _, candidate := range candidates {
		if candidate.Default {
			warnings := stalePreferenceWarnings(source, preferredID, preferredName, candidate)
			return candidate, warnings, nil
		}
	}
	warnings := stalePreferenceWarnings(source, preferredID, preferredName, candidates[0])
	return candidates[0], warnings, nil
}

func stalePreferenceWarnings(source Source, preferredID, preferredName string, selected Device) []string {
	if strings.TrimSpace(preferredID) == "" && strings.TrimSpace(preferredName) == "" {
		return nil
	}
	return []string{fmt.Sprintf("configured %s device was not found; using %q", source, selected.DisplayName())}
}

func normalize(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	replacer := strings.NewReplacer("_", " ", "-", " ", ".", " ", ":", " ", "\t", " ")
	value = replacer.Replace(value)
	return strings.Join(strings.Fields(value), " ")
}
