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
	Aliases []string
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
	if sameDevice(mic, output) {
		if configured := firstNonEmpty(prefs.OutputDeviceID, prefs.OutputDeviceName); configured != "" && !isDefaultPreference(configured) {
			return Selection{}, warnings, errors.New("microphone and output capture resolved to the same device; choose a distinct output-capture loopback/monitor device with --output or transcribe --list-devices")
		}
		replacement, replaceWarnings, ok := selectDistinctOutputCandidate(devices, mic)
		if !ok {
			return Selection{}, warnings, errors.New("microphone and output capture resolved to the same device; no distinct output-capture device was available")
		}
		warnings = append(warnings, replaceWarnings...)
		output = replacement
	}
	return Selection{Mic: mic, Output: output}, warnings, nil
}

func SameDevice(left Device, right Device) bool {
	return sameDevice(left, right)
}

func sameDevice(left Device, right Device) bool {
	leftKeys := append([]string{left.ID}, left.Aliases...)
	rightKeys := append([]string{right.ID}, right.Aliases...)
	for _, leftKey := range leftKeys {
		leftKey = normalize(leftKey)
		if leftKey == "" {
			continue
		}
		for _, rightKey := range rightKeys {
			if leftKey == normalize(rightKey) {
				return true
			}
		}
	}
	return false
}

func selectDistinctOutputCandidate(devices []Device, avoid Device) (Device, []string, bool) {
	var outputCandidates []Device
	for _, device := range devices {
		if device.Source == SourceOutput {
			outputCandidates = append(outputCandidates, device)
		}
	}
	for _, candidate := range outputCandidates {
		if candidate.Default && !sameDevice(candidate, avoid) {
			return candidate, []string{fmt.Sprintf("output capture initially matched the microphone; using %q instead", candidate.DisplayName())}, true
		}
	}
	for _, candidate := range outputCandidates {
		if !sameDevice(candidate, avoid) {
			return candidate, []string{fmt.Sprintf("output capture initially matched the microphone; using %q instead", candidate.DisplayName())}, true
		}
	}
	return Device{}, nil, false
}

func OutputCaptureCandidates(devices []Device, current Device) []Device {
	var outputCandidates []Device
	for _, device := range devices {
		if device.Source == SourceOutput {
			outputCandidates = append(outputCandidates, device)
		}
	}
	if len(outputCandidates) == 0 {
		if current.Source == SourceOutput && strings.TrimSpace(current.ID) != "" {
			return []Device{current}
		}
		return nil
	}

	start := -1
	for i, candidate := range outputCandidates {
		if sameDevice(candidate, current) {
			start = i
			break
		}
	}

	var ordered []Device
	if start >= 0 {
		ordered = append(ordered, outputCandidates[start:]...)
	} else {
		if current.Source == SourceOutput && strings.TrimSpace(current.ID) != "" {
			ordered = append(ordered, current)
		}
		ordered = append(ordered, outputCandidates...)
	}
	return dedupeDevices(ordered)
}

func dedupeDevices(devices []Device) []Device {
	result := make([]Device, 0, len(devices))
	for _, device := range devices {
		found := false
		for _, existing := range result {
			if sameDevice(existing, device) {
				found = true
				break
			}
		}
		if !found {
			result = append(result, device)
		}
	}
	return result
}

func selectOne(devices []Device, source Source, preferredID, preferredName string) (Device, []string, error) {
	var candidates []Device
	for _, device := range devices {
		if device.Source == source {
			candidates = append(candidates, device)
		}
	}
	if len(candidates) == 0 {
		return Device{}, nil, missingDeviceError(source, preferredID, preferredName)
	}

	if strings.TrimSpace(preferredID) != "" {
		for _, candidate := range candidates {
			if candidate.ID == strings.TrimSpace(preferredID) {
				return candidate, nil, nil
			}
		}
		for _, candidate := range candidates {
			if candidateMatches(candidate, preferredID) {
				return candidate, nil, nil
			}
		}
	}
	if strings.TrimSpace(preferredName) != "" {
		for _, candidate := range candidates {
			if candidateMatches(candidate, preferredName) {
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

func candidateMatches(candidate Device, value string) bool {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return false
	}
	if candidate.ID == trimmed || candidate.Name == trimmed {
		return true
	}
	for _, alias := range candidate.Aliases {
		if alias == trimmed {
			return true
		}
	}
	normalized := normalize(trimmed)
	if normalize(candidate.ID) == normalized || normalize(candidate.Name) == normalized {
		return true
	}
	for _, alias := range candidate.Aliases {
		if normalize(alias) == normalized {
			return true
		}
	}
	return false
}

func missingDeviceError(source Source, preferredID, preferredName string) error {
	configured := firstNonEmpty(preferredID, preferredName)
	selectionHint := fmt.Sprintf("run transcribe --list-devices and choose a displayed %s device name/id", source)
	if source == SourceOutput {
		selectionHint = "run transcribe --list-devices and choose a displayed output-capture device name/id; on Windows this must resolve to a WASAPI loopback or virtual output device such as Stereo Mix, virtual-audio-capturer, VB-CABLE, or another loopback/monitor source"
	}

	message := fmt.Sprintf("no %s capture device found; %s", source, selectionHint)
	if isDefaultPreference(configured) {
		message += fmt.Sprintf("; configured %s device %q was unresolved because no concrete device was enumerated", source, configured)
	} else if configured != "" {
		message += fmt.Sprintf("; configured %s device %q was not found", source, configured)
	}
	return errors.New(message)
}

func stalePreferenceWarnings(source Source, preferredID, preferredName string, selected Device) []string {
	configured := firstNonEmpty(preferredID, preferredName)
	if configured == "" {
		return nil
	}
	if isDefaultPreference(configured) {
		return []string{fmt.Sprintf("configured %s device %q was not listed as a concrete device; using %q. Run transcribe --list-devices to choose a displayed device name/id if this is not correct", source, configured, selected.DisplayName())}
	}
	return []string{fmt.Sprintf("configured %s device was not found; using %q", source, selected.DisplayName())}
}

func isDefaultPreference(value string) bool {
	switch normalize(value) {
	case "default", "default windows microphone", "windows default microphone", "virtual audio capturer", "windows system audio capture virtual audio capturer wasapi compatible":
		return true
	default:
		return false
	}
}

func normalize(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	replacer := strings.NewReplacer("_", " ", "-", " ", ".", " ", ":", " ", "\t", " ")
	value = replacer.Replace(value)
	return strings.Join(strings.Fields(value), " ")
}
