package audio

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

type SystemDiscoverer struct {
	Preferences Preferences
}

func (d SystemDiscoverer) ListDevices(ctx context.Context) ([]Device, []string, error) {
	switch runtime.GOOS {
	case "linux":
		return d.listLinux(ctx)
	case "darwin":
		return d.listDarwin(), nil, nil
	case "windows":
		return d.listWindows(), nil, nil
	default:
		devices := configuredDevices(d.Preferences, "manual")
		return devices, []string{fmt.Sprintf("%s audio device discovery is not implemented; configure audio device IDs manually", runtime.GOOS)}, nil
	}
}

func (d SystemDiscoverer) listLinux(ctx context.Context) ([]Device, []string, error) {
	var warnings []string
	info, infoErr := exec.CommandContext(ctx, "pactl", "info").Output()
	sources, sourcesErr := exec.CommandContext(ctx, "pactl", "list", "short", "sources").Output()
	if infoErr != nil || sourcesErr != nil {
		devices := configuredDevices(d.Preferences, "pulse")
		if !hasSource(devices, SourceMic) {
			devices = append(devices, Device{ID: "default", Name: "PulseAudio default microphone", Source: SourceMic, Default: true, Backend: "pulse"})
		}
		warnings = append(warnings, "pactl is unavailable or PulseAudio/PipeWire is not running; using generic/default devices where possible")
		return devices, warnings, nil
	}
	devices := ParsePactlSources(string(sources), ParsePactlInfo(string(info)))
	devices = appendConfiguredIfMissing(devices, d.Preferences, "pulse")
	if !hasSource(devices, SourceMic) {
		devices = append(devices, Device{ID: "default", Name: "PulseAudio default microphone", Source: SourceMic, Default: true, Backend: "pulse"})
		warnings = append(warnings, "no non-monitor PulseAudio source was detected; using PulseAudio default microphone")
	}
	return devices, warnings, nil
}

func (d SystemDiscoverer) listDarwin() []Device {
	devices := []Device{{ID: ":0", Name: "Default AVFoundation microphone", Source: SourceMic, Default: true, Backend: "avfoundation"}}
	return appendConfiguredIfMissing(devices, d.Preferences, "avfoundation")
}

func (d SystemDiscoverer) listWindows() []Device {
	devices := []Device{
		{ID: "default", Name: "Default Windows microphone", Source: SourceMic, Default: true, Backend: "dshow"},
		{ID: "virtual-audio-capturer", Name: "Windows system audio capture (virtual-audio-capturer/WASAPI-compatible)", Source: SourceOutput, Default: true, Backend: "dshow"},
	}
	return appendConfiguredIfMissing(devices, d.Preferences, "dshow")
}

type PactlDefaults struct {
	DefaultSource string
	DefaultSink   string
}

func ParsePactlInfo(info string) PactlDefaults {
	var defaults PactlDefaults
	for _, line := range strings.Split(info, "\n") {
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		switch strings.TrimSpace(key) {
		case "Default Source":
			defaults.DefaultSource = strings.TrimSpace(value)
		case "Default Sink":
			defaults.DefaultSink = strings.TrimSpace(value)
		}
	}
	return defaults
}

func ParsePactlSources(sources string, defaults PactlDefaults) []Device {
	var devices []Device
	for _, rawLine := range strings.Split(sources, "\n") {
		line := strings.TrimSpace(rawLine)
		if line == "" {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		id := fields[1]
		name := id
		if len(fields) > 2 {
			name = id
		}
		isMonitor := strings.Contains(strings.ToLower(id), ".monitor") || strings.Contains(strings.ToLower(id), "monitor")
		source := SourceMic
		defaultDevice := id == defaults.DefaultSource
		if isMonitor {
			source = SourceOutput
			defaultDevice = id == defaults.DefaultSink+".monitor" || id == defaults.DefaultSource
		}
		devices = append(devices, Device{ID: id, Name: name, Source: source, Default: defaultDevice, Backend: "pulse"})
	}
	return devices
}

func configuredDevices(prefs Preferences, backend string) []Device {
	var devices []Device
	if strings.TrimSpace(prefs.MicDeviceID) != "" || strings.TrimSpace(prefs.MicDeviceName) != "" {
		devices = append(devices, Device{ID: firstNonEmpty(prefs.MicDeviceID, prefs.MicDeviceName), Name: firstNonEmpty(prefs.MicDeviceName, prefs.MicDeviceID), Source: SourceMic, Default: true, Backend: backend})
	}
	if strings.TrimSpace(prefs.OutputDeviceID) != "" || strings.TrimSpace(prefs.OutputDeviceName) != "" {
		devices = append(devices, Device{ID: firstNonEmpty(prefs.OutputDeviceID, prefs.OutputDeviceName), Name: firstNonEmpty(prefs.OutputDeviceName, prefs.OutputDeviceID), Source: SourceOutput, Default: true, Backend: backend})
	}
	return devices
}

func appendConfiguredIfMissing(devices []Device, prefs Preferences, backend string) []Device {
	for _, configured := range configuredDevices(prefs, backend) {
		found := false
		for _, existing := range devices {
			if existing.Source == configured.Source && normalize(existing.ID) == normalize(configured.ID) {
				found = true
				break
			}
		}
		if !found {
			devices = append(devices, configured)
		}
	}
	return devices
}

func hasSource(devices []Device, source Source) bool {
	for _, device := range devices {
		if device.Source == source {
			return true
		}
	}
	return false
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func FormatDevices(devices []Device) string {
	var buf bytes.Buffer
	for _, device := range devices {
		defaultMarker := ""
		if device.Default {
			defaultMarker = "\tdefault"
		}
		fmt.Fprintf(&buf, "%s\t%s\t%s\t%s%s\n", device.Source, device.Backend, device.ID, device.DisplayName(), defaultMarker)
	}
	return buf.String()
}
