package audio

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

type SystemDiscoverer struct {
	Preferences Preferences
	FFmpegPath  string
}

func (d SystemDiscoverer) ListDevices(ctx context.Context) ([]Device, []string, error) {
	switch runtime.GOOS {
	case "linux":
		return d.listLinux(ctx)
	case "darwin":
		return d.listDarwin(), nil, nil
	case "windows":
		return d.listWindows(ctx)
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

func (d SystemDiscoverer) listWindows(ctx context.Context) ([]Device, []string, error) {
	path, err := d.ffmpegPath()
	if err != nil {
		return nil, nil, err
	}
	cmd := exec.CommandContext(ctx, path, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy")
	output, cmdErr := cmd.CombinedOutput()
	if ctx.Err() != nil {
		return nil, nil, ctx.Err()
	}
	listed := ParseDShowAudioDevices(string(output))
	devices, warnings := devicesFromDShowAudioDevices(listed)
	devices = appendWindowsConfiguredIfMissing(devices, d.Preferences)
	if len(listed) == 0 {
		message := "ffmpeg did not list any DirectShow audio devices"
		if cmdErr != nil {
			message = fmt.Sprintf("ffmpeg DirectShow device discovery failed: %v", cmdErr)
		}
		if trimmed := strings.TrimSpace(string(output)); trimmed != "" {
			message += ": " + firstLine(trimmed)
		}
		return devices, warnings, errors.New(message)
	}
	return devices, warnings, nil
}

func (d SystemDiscoverer) ffmpegPath() (string, error) {
	if strings.TrimSpace(d.FFmpegPath) != "" {
		return d.FFmpegPath, nil
	}
	path, err := exec.LookPath("ffmpeg")
	if err != nil {
		return "", errors.New("ffmpeg was not found in PATH; install ffmpeg before listing or recording Windows DirectShow audio devices")
	}
	return path, nil
}

type DShowAudioDevice struct {
	Name       string
	Alternates []string
}

func ParseDShowAudioDevices(output string) []DShowAudioDevice {
	var devices []DShowAudioDevice
	section := ""
	current := -1
	for _, rawLine := range strings.Split(output, "\n") {
		line := stripDShowPrefix(rawLine)
		if line == "" {
			continue
		}
		lower := strings.ToLower(line)
		switch {
		case strings.Contains(lower, "directshow video devices"):
			section = "video"
			current = -1
			continue
		case strings.Contains(lower, "directshow audio devices"):
			section = "audio"
			current = -1
			continue
		}
		if section != "audio" {
			continue
		}
		if strings.Contains(lower, "alternative name") {
			if current >= 0 {
				if alt := extractQuoted(line); alt != "" {
					devices[current].Alternates = append(devices[current].Alternates, alt)
				}
			}
			continue
		}
		name := extractQuoted(line)
		if name == "" {
			continue
		}
		devices = append(devices, DShowAudioDevice{Name: name})
		current = len(devices) - 1
	}
	return devices
}

func devicesFromDShowAudioDevices(listed []DShowAudioDevice) ([]Device, []string) {
	var devices []Device
	if len(listed) == 0 {
		return devices, nil
	}
	defaultOutput := defaultDShowOutputIndex(listed)
	var warnings []string
	if defaultOutput < 0 {
		defaultOutput = 0
		warnings = append(warnings, "DirectShow lists audio capture devices but does not identify system-output loopback reliably; output settings show all DirectShow audio candidates. Choose Stereo Mix, virtual-audio-capturer, VB-CABLE, or another loopback/virtual device for system audio.")
	}
	for i, listedDevice := range listed {
		id := listedDevice.Name
		if len(listedDevice.Alternates) > 0 {
			id = listedDevice.Alternates[0]
		}
		devices = append(devices, Device{
			ID:      id,
			Name:    listedDevice.Name,
			Aliases: append([]string(nil), listedDevice.Alternates...),
			Source:  SourceMic,
			Default: i == 0,
			Backend: "dshow",
		})
		devices = append(devices, Device{
			ID:      id,
			Name:    listedDevice.Name,
			Aliases: append([]string(nil), listedDevice.Alternates...),
			Source:  SourceOutput,
			Default: i == defaultOutput,
			Backend: "dshow",
		})
	}
	return devices, warnings
}

func defaultDShowOutputIndex(devices []DShowAudioDevice) int {
	for i, device := range devices {
		if likelyDShowLoopback(device.Name) {
			return i
		}
		for _, alias := range device.Alternates {
			if likelyDShowLoopback(alias) {
				return i
			}
		}
	}
	return -1
}

func likelyDShowLoopback(value string) bool {
	normalized := normalize(value)
	for _, token := range []string{"stereo mix", "what u hear", "loopback", "virtual audio capturer", "vb audio", "vb cable", "cable output", "voicemeeter", "blackhole", "soundflower", "monitor", "wasapi"} {
		if strings.Contains(normalized, token) {
			return true
		}
	}
	return false
}

func appendWindowsConfiguredIfMissing(devices []Device, prefs Preferences) []Device {
	for _, configured := range configuredDevices(prefs, "dshow") {
		if isSyntheticWindowsConfiguredDevice(configured) {
			continue
		}
		found := false
		for _, existing := range devices {
			if existing.Source == configured.Source && candidateMatches(existing, configured.ID) {
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

func isSyntheticWindowsConfiguredDevice(device Device) bool {
	if isDefaultPreference(device.ID) || isDefaultPreference(device.Name) {
		return true
	}
	return normalize(device.ID) == "virtual audio capturer" || normalize(device.Name) == "virtual audio capturer" || strings.Contains(normalize(device.Name), "windows system audio capture")
}

func stripDShowPrefix(line string) string {
	line = strings.TrimSpace(line)
	if strings.HasPrefix(line, "[") {
		if end := strings.Index(line, "]"); end >= 0 {
			line = strings.TrimSpace(line[end+1:])
		}
	}
	return line
}

func extractQuoted(line string) string {
	start := strings.Index(line, "\"")
	if start < 0 {
		return ""
	}
	end := strings.LastIndex(line[start+1:], "\"")
	if end < 0 {
		return ""
	}
	return strings.TrimSpace(line[start+1 : start+1+end])
}

func firstLine(value string) string {
	line := strings.TrimSpace(strings.Split(value, "\n")[0])
	if len(line) > 240 {
		return line[:240] + "..."
	}
	return line
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
		aliases := aliasesForDisplay(device)
		aliasText := ""
		if len(aliases) > 0 {
			aliasText = "\taliases=" + strings.Join(aliases, ",")
		}
		fmt.Fprintf(&buf, "%s\t%s\t%s\t%s%s%s\n", device.Source, device.Backend, device.ID, device.DisplayName(), defaultMarker, aliasText)
	}
	return buf.String()
}

func aliasesForDisplay(device Device) []string {
	var aliases []string
	seen := map[string]struct{}{}
	for _, alias := range device.Aliases {
		if alias == "" {
			continue
		}
		if _, ok := seen[alias]; ok {
			continue
		}
		seen[alias] = struct{}{}
		aliases = append(aliases, alias)
	}
	return aliases
}
