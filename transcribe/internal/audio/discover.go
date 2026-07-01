package audio

import (
	"bytes"
	"context"
	"encoding/json"
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
	if len(listed) == 0 {
		fallbackEndpoints, fallbackWarnings, fallbackErr := d.listWindowsAudioEndpoints(ctx)
		if fallbackErr == nil && len(fallbackEndpoints) > 0 {
			devices, warnings := devicesFromWindowsAudioEndpoints(fallbackEndpoints)
			devices = appendWindowsConfiguredIfMissing(devices, d.Preferences)
			warnings = append(warnings, fallbackWarnings...)
			warnings = append(warnings, "ffmpeg did not list any DirectShow audio devices; using Windows audio endpoints from PowerShell as a fallback discovery source")
			return devices, warnings, nil
		}
		message := "ffmpeg did not list any DirectShow audio devices"
		if cmdErr != nil {
			message = fmt.Sprintf("ffmpeg DirectShow device discovery failed: %v", cmdErr)
		}
		if trimmed := strings.TrimSpace(string(output)); trimmed != "" {
			message += ": " + firstLine(trimmed)
		}
		if fallbackErr != nil {
			message += "; Windows audio endpoint fallback failed: " + firstLine(fallbackErr.Error())
		}
		devices, warnings := devicesFromDShowAudioDevices(listed)
		devices = appendWindowsConfiguredIfMissing(devices, d.Preferences)
		return devices, warnings, errors.New(message)
	}
	devices, warnings := devicesFromDShowAudioDevices(listed)
	if !hasSource(devices, SourceOutput) {
		fallbackEndpoints, fallbackWarnings, fallbackErr := d.listWindowsAudioEndpoints(ctx)
		if fallbackErr == nil && len(fallbackEndpoints) > 0 {
			var supplemented bool
			devices, warnings, supplemented = supplementWindowsOutputCapture(devices, warnings, fallbackEndpoints)
			if supplemented {
				warnings = append(warnings, fallbackWarnings...)
			}
		}
	}
	devices = appendWindowsConfiguredIfMissing(devices, d.Preferences)
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

type windowsAudioEndpoint struct {
	FriendlyName string `json:"FriendlyName"`
	InstanceID   string `json:"InstanceId"`
}

const windowsOutputCaptureWarning = "Windows audio discovery did not identify a WASAPI loopback/monitor output-capture device; output capture requires Stereo Mix, virtual-audio-capturer, VB-CABLE, or another loopback/virtual device"

func windowsAudioEndpointLooksLikeOutput(endpoint windowsAudioEndpoint) bool {
	name := endpoint.DisplayName()
	if strings.TrimSpace(name) == "" {
		return false
	}
	if likelyDShowLoopback(name) {
		return true
	}
	normalized := normalize(name)
	return normalizedContainsAnyToken(normalized, "speaker", "speakers", "headphone", "headphones", "earphone", "earphones", "line out", "audio output", "digital audio", "digital output", "s/pdif", "spdif", "playback", "render")
}

func windowsLoopbackDisplayName(name string) string {
	trimmed := strings.TrimSpace(name)
	if trimmed == "" {
		return ""
	}
	if strings.Contains(normalize(trimmed), "loopback") {
		return trimmed
	}
	return trimmed + " [Loopback]"
}

func defaultWindowsAudioEndpointMicIndex(endpoints []windowsAudioEndpoint) int {
	for i, endpoint := range endpoints {
		if !windowsAudioEndpointLooksLikeOutput(endpoint) {
			return i
		}
	}
	return -1
}

func defaultWindowsAudioEndpointOutputIndex(endpoints []windowsAudioEndpoint) int {
	for i, endpoint := range endpoints {
		if windowsAudioEndpointLooksLikeOutput(endpoint) {
			return i
		}
	}
	return -1
}

func normalizedContainsAnyToken(normalized string, tokens ...string) bool {
	for _, token := range tokens {
		if normalizedContainsToken(normalized, token) {
			return true
		}
	}
	return false
}

func normalizedContainsToken(normalized string, token string) bool {
	normalized = strings.TrimSpace(normalized)
	token = strings.TrimSpace(normalize(token))
	if normalized == "" || token == "" {
		return false
	}
	return strings.Contains(" "+normalized+" ", " "+token+" ")
}

func windowsOutputDevicesFromEndpoints(endpoints []windowsAudioEndpoint) []Device {
	if len(endpoints) == 0 {
		return nil
	}
	devices, _ := devicesFromWindowsAudioEndpoints(endpoints)
	outputs := make([]Device, 0, len(devices))
	for _, device := range devices {
		if device.Source == SourceOutput {
			outputs = append(outputs, device)
		}
	}
	return outputs
}

func supplementWindowsOutputCapture(devices []Device, warnings []string, endpoints []windowsAudioEndpoint) ([]Device, []string, bool) {
	if hasSource(devices, SourceOutput) {
		return devices, warnings, false
	}
	outputDevices := windowsOutputDevicesFromEndpoints(endpoints)
	if len(outputDevices) == 0 {
		return devices, warnings, false
	}
	devices = appendDevicesIfMissing(devices, outputDevices)
	warnings = removeWarning(warnings, windowsOutputCaptureWarning)
	warnings = append(warnings, "ffmpeg DirectShow listed no output-capture devices; using Windows audio endpoints from PowerShell to supplement output capture")
	return devices, warnings, true
}

func (e windowsAudioEndpoint) DisplayName() string {
	if name := strings.TrimSpace(e.FriendlyName); name != "" {
		return name
	}
	return strings.TrimSpace(e.InstanceID)
}

func (d SystemDiscoverer) listWindowsAudioEndpoints(ctx context.Context) ([]windowsAudioEndpoint, []string, error) {
	command := `Get-PnpDevice -Class AudioEndpoint -ErrorAction SilentlyContinue | Where-Object { $_.FriendlyName -and $_.InstanceId } | Sort-Object FriendlyName | Select-Object FriendlyName, InstanceId | ConvertTo-Json -Compress -Depth 2`
	output, err := exec.CommandContext(ctx, "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command).CombinedOutput()
	if ctx.Err() != nil {
		return nil, nil, ctx.Err()
	}
	endpoints, parseErr := parseWindowsAudioEndpoints(string(output))
	if parseErr != nil {
		message := parseErr.Error()
		if trimmed := strings.TrimSpace(string(output)); trimmed != "" {
			message += ": " + firstLine(trimmed)
		}
		return nil, nil, errors.New(message)
	}
	if len(endpoints) == 0 {
		message := "Windows audio endpoint discovery did not return any devices"
		if err != nil {
			message = fmt.Sprintf("Windows audio endpoint discovery failed: %v", err)
		}
		if trimmed := strings.TrimSpace(string(output)); trimmed != "" {
			message += ": " + firstLine(trimmed)
		}
		return nil, nil, errors.New(message)
	}
	var warnings []string
	if err != nil {
		warnings = append(warnings, fmt.Sprintf("PowerShell audio endpoint query returned %v; using returned Windows audio endpoints anyway", err))
	}
	return endpoints, warnings, nil
}

func parseWindowsAudioEndpoints(output string) ([]windowsAudioEndpoint, error) {
	trimmed := strings.TrimSpace(output)
	if trimmed == "" {
		return nil, nil
	}
	var endpoints []windowsAudioEndpoint
	if err := json.Unmarshal([]byte(trimmed), &endpoints); err == nil {
		return normalizeWindowsAudioEndpoints(endpoints), nil
	}
	var single windowsAudioEndpoint
	if err := json.Unmarshal([]byte(trimmed), &single); err == nil {
		return normalizeWindowsAudioEndpoints([]windowsAudioEndpoint{single}), nil
	}
	return nil, errors.New("parse Windows audio endpoint JSON")
}

func normalizeWindowsAudioEndpoints(endpoints []windowsAudioEndpoint) []windowsAudioEndpoint {
	result := make([]windowsAudioEndpoint, 0, len(endpoints))
	for _, endpoint := range endpoints {
		name := strings.TrimSpace(endpoint.FriendlyName)
		instanceID := strings.TrimSpace(endpoint.InstanceID)
		if name == "" && instanceID == "" {
			continue
		}
		result = append(result, windowsAudioEndpoint{FriendlyName: name, InstanceID: instanceID})
	}
	return result
}

type DShowAudioDevice struct {
	Name       string
	Alternates []string
}

func ParseDShowAudioDevices(output string) []DShowAudioDevice {
	var rawDevices []rawDShowDevice
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
		if strings.Contains(lower, "alternative name") {
			if current >= 0 {
				if alt := extractQuoted(line); alt != "" {
					rawDevices[current].Alternates = append(rawDevices[current].Alternates, alt)
				}
			}
			continue
		}
		name, suffix, ok := splitDShowQuotedLine(line)
		if !ok {
			continue
		}
		rawDevices = append(rawDevices, rawDShowDevice{Name: name, Section: section, Kind: dshowLineKind(suffix)})
		current = len(rawDevices) - 1
	}
	devices := make([]DShowAudioDevice, 0, len(rawDevices))
	for _, rawDevice := range rawDevices {
		if !rawDevice.isAudio() {
			continue
		}
		devices = append(devices, DShowAudioDevice{Name: rawDevice.Name, Alternates: append([]string(nil), rawDevice.Alternates...)})
	}
	return devices
}

type rawDShowDevice struct {
	Name       string
	Alternates []string
	Section    string
	Kind       string
}

func (d rawDShowDevice) isAudio() bool {
	switch d.Kind {
	case "audio":
		return true
	case "video":
		return false
	}
	if d.hasAudioSignal() {
		return true
	}
	return d.Section == "audio"
}

func (d rawDShowDevice) hasAudioSignal() bool {
	if likelyDShowAudioCaptureName(d.Name) {
		return true
	}
	for _, alt := range d.Alternates {
		if likelyDShowAudioCaptureName(alt) {
			return true
		}
		normalized := normalize(alt)
		if strings.Contains(normalized, "@device cm") || strings.Contains(normalized, "wave") {
			return true
		}
	}
	return false
}

func likelyDShowAudioCaptureName(value string) bool {
	normalized := normalize(value)
	for _, token := range []string{"microphone", "mic", "headset", "stereo mix", "what u hear", "loopback", "virtual audio capturer", "vb audio", "vb cable", "cable output", "voicemeeter", "blackhole", "soundflower", "monitor", "speaker", "line in"} {
		if strings.Contains(normalized, token) {
			return true
		}
	}
	return false
}

func splitDShowQuotedLine(line string) (string, string, bool) {
	start := strings.Index(line, "\"")
	if start < 0 {
		return "", "", false
	}
	end := strings.Index(line[start+1:], "\"")
	if end < 0 {
		return "", "", false
	}
	end += start + 1
	return strings.TrimSpace(line[start+1 : end]), strings.TrimSpace(line[end+1:]), true
}

func dshowLineKind(suffix string) string {
	lower := strings.ToLower(suffix)
	switch {
	case strings.Contains(lower, "audio"):
		return "audio"
	case strings.Contains(lower, "video"):
		return "video"
	default:
		return ""
	}
}

func devicesFromDShowAudioDevices(listed []DShowAudioDevice) ([]Device, []string) {
	var devices []Device
	if len(listed) == 0 {
		return devices, nil
	}
	defaultMic := defaultDShowMicIndex(listed)
	defaultOutput := defaultDShowOutputIndex(listed)
	var warnings []string
	outputCount := 0
	for i, listedDevice := range listed {
		id := listedDevice.Name
		aliases := append([]string(nil), listedDevice.Alternates...)
		devices = append(devices, Device{
			ID:      id,
			Name:    listedDevice.Name,
			Aliases: aliases,
			Source:  SourceMic,
			Default: i == defaultMic,
			Backend: "dshow",
		})
		if dshowDeviceLooksLikeLoopback(listedDevice) {
			devices = append(devices, Device{
				ID:      id,
				Name:    listedDevice.Name,
				Aliases: aliases,
				Source:  SourceOutput,
				Default: i == defaultOutput,
				Backend: "dshow",
			})
			outputCount++
		}
	}
	if outputCount == 0 {
		warnings = append(warnings, windowsOutputCaptureWarning)
	}
	return devices, warnings
}

func devicesFromWindowsAudioEndpoints(endpoints []windowsAudioEndpoint) ([]Device, []string) {
	if len(endpoints) == 0 {
		return nil, nil
	}
	defaultMic := defaultWindowsAudioEndpointMicIndex(endpoints)
	defaultOutput := defaultWindowsAudioEndpointOutputIndex(endpoints)
	var devices []Device
	outputCount := 0
	for i, endpoint := range endpoints {
		rawName := endpoint.DisplayName()
		if rawName == "" {
			continue
		}
		aliases := []string(nil)
		if instanceID := strings.TrimSpace(endpoint.InstanceID); instanceID != "" {
			aliases = append(aliases, instanceID)
		}
		if windowsAudioEndpointLooksLikeOutput(endpoint) {
			devices = append(devices, Device{
				ID:      rawName,
				Name:    windowsLoopbackDisplayName(rawName),
				Aliases: append([]string(nil), aliases...),
				Source:  SourceOutput,
				Default: i == defaultOutput,
				Backend: "dshow",
			})
			outputCount++
			continue
		}
		devices = append(devices, Device{
			ID:      rawName,
			Name:    rawName,
			Aliases: append([]string(nil), aliases...),
			Source:  SourceMic,
			Default: i == defaultMic,
			Backend: "dshow",
		})
	}
	warnings := []string(nil)
	if outputCount == 0 {
		warnings = append(warnings, windowsOutputCaptureWarning)
	}
	return devices, warnings
}

func defaultDShowMicIndex(devices []DShowAudioDevice) int {
	for i, device := range devices {
		if !dshowDeviceLooksLikeLoopback(device) {
			return i
		}
	}
	return 0
}

func defaultDShowOutputIndex(devices []DShowAudioDevice) int {
	for i, device := range devices {
		if dshowDeviceLooksLikeLoopback(device) {
			return i
		}
	}
	return -1
}

func dshowDeviceLooksLikeLoopback(device DShowAudioDevice) bool {
	if likelyDShowLoopback(device.Name) {
		return true
	}
	for _, alias := range device.Alternates {
		if likelyDShowLoopback(alias) {
			return true
		}
	}
	return false
}

func appendDevicesIfMissing(devices []Device, candidates []Device) []Device {
	for _, candidate := range candidates {
		found := false
		for _, existing := range devices {
			if existing.Source == candidate.Source && candidateMatches(existing, candidate.ID) {
				found = true
				break
			}
		}
		if !found {
			devices = append(devices, candidate)
		}
	}
	return devices
}

func removeWarning(warnings []string, target string) []string {
	if len(warnings) == 0 {
		return warnings
	}
	filtered := make([]string, 0, len(warnings))
	for _, warning := range warnings {
		if warning == target {
			continue
		}
		filtered = append(filtered, warning)
	}
	return filtered
}

func likelyDShowLoopback(value string) bool {
	normalized := normalize(value)
	for _, token := range []string{"stereo mix", "what u hear", "loopback", "virtual audio capturer", "vb audio", "vb cable", "cable output", "voicemeeter", "blackhole", "soundflower", "monitor"} {
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
		if configured.Source == SourceOutput && !isWindowsOutputCaptureDevice(configured) {
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
			devices = append(devices, normalizeWindowsConfiguredDevice(configured))
		}
	}
	return devices
}

// normalizeWindowsConfiguredDevice ensures a configured device uses a ffmpeg-openable
// friendly name as its ID. DirectShow alternative names (e.g. "@device_cm_{...}\wave_{...}")
// are accepted by ffmpeg -list_devices but cannot be opened as capture inputs, so when a
// saved preference stores the alternative name as the ID, promote the friendly name to the
// ID and demote the alternative name to an alias.
func normalizeWindowsConfiguredDevice(device Device) Device {
	if !looksLikeDShowAlternativeName(device.ID) {
		return device
	}
	friendlyName := strings.TrimSpace(device.Name)
	if friendlyName == "" || friendlyName == device.ID {
		return device
	}
	aliases := append([]string{}, device.Aliases...)
	if !containsString(aliases, device.ID) {
		aliases = append(aliases, device.ID)
	}
	return Device{
		ID:      friendlyName,
		Name:    friendlyName,
		Aliases: aliases,
		Source:  device.Source,
		Default: device.Default,
		Backend: device.Backend,
	}
}

func looksLikeDShowAlternativeName(value string) bool {
	normalized := normalize(value)
	return strings.Contains(normalized, "@device cm") || strings.Contains(normalized, "wave")
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func isSyntheticWindowsConfiguredDevice(device Device) bool {
	if isDefaultPreference(device.ID) || isDefaultPreference(device.Name) {
		return true
	}
	return normalize(device.ID) == "virtual audio capturer" || normalize(device.Name) == "virtual audio capturer" || strings.Contains(normalize(device.Name), "windows system audio capture")
}

func isWindowsOutputCaptureDevice(device Device) bool {
	if device.Source != SourceOutput {
		return true
	}
	if windowsAudioEndpointLooksLikeOutput(windowsAudioEndpoint{FriendlyName: device.ID}) || windowsAudioEndpointLooksLikeOutput(windowsAudioEndpoint{FriendlyName: device.Name}) {
		return true
	}
	for _, alias := range device.Aliases {
		if windowsAudioEndpointLooksLikeOutput(windowsAudioEndpoint{FriendlyName: alias}) {
			return true
		}
	}
	return false
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
	quoted, _, ok := splitDShowQuotedLine(line)
	if !ok {
		return ""
	}
	return quoted
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
		index := fields[0]
		id := fields[1]
		name := id
		aliases := []string(nil)
		if strings.TrimSpace(index) != "" && index != id {
			aliases = append(aliases, index)
		}
		isMonitor := strings.Contains(strings.ToLower(id), ".monitor") || strings.Contains(strings.ToLower(id), "monitor")
		source := SourceMic
		defaultDevice := id == defaults.DefaultSource
		if isMonitor {
			source = SourceOutput
			defaultDevice = id == defaults.DefaultSink+".monitor" || id == defaults.DefaultSource
		}
		devices = append(devices, Device{ID: id, Name: name, Aliases: aliases, Source: source, Default: defaultDevice, Backend: "pulse"})
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
