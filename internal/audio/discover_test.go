package audio

import (
	"strings"
	"testing"
)

func TestParsePactlSourcesClassifiesMonitorAndDefaults(t *testing.T) {
	info := `Server String: /run/user/1000/pulse/native
Default Sink: alsa_output.pci-0000_00_1f.3.analog-stereo
Default Source: alsa_input.pci-0000_00_1f.3.analog-stereo`
	sources := `42	alsa_input.pci-0000_00_1f.3.analog-stereo	module-alsa-card.c	s16le 2ch 48000Hz
43	alsa_output.pci-0000_00_1f.3.analog-stereo.monitor	module-alsa-card.c	s16le 2ch 48000Hz`
	devices := ParsePactlSources(sources, ParsePactlInfo(info))
	if len(devices) != 2 {
		t.Fatalf("expected two devices, got %d", len(devices))
	}
	if devices[0].Source != SourceMic || !devices[0].Default {
		t.Fatalf("first device should be default mic: %+v", devices[0])
	}
	if devices[1].Source != SourceOutput || !devices[1].Default {
		t.Fatalf("second device should be default output monitor: %+v", devices[1])
	}
}

func TestSelectDevicesPreferenceDefaultFallback(t *testing.T) {
	devices := []Device{
		{ID: "mic-a", Name: "Laptop Mic", Source: SourceMic},
		{ID: "mic-b", Name: "USB Mic", Source: SourceMic, Default: true},
		{ID: "out-a", Name: "Monitor A", Source: SourceOutput, Default: true},
	}
	selection, warnings, err := SelectDevices(devices, Preferences{MicDeviceName: "USB_Mic"})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Mic.ID != "mic-b" || selection.Output.ID != "out-a" {
		t.Fatalf("unexpected selection: %+v", selection)
	}
	if len(warnings) != 0 {
		t.Fatalf("unexpected warnings: %v", warnings)
	}

	selection, warnings, err = SelectDevices(devices, Preferences{MicDeviceID: "stale"})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Mic.ID != "mic-b" || len(warnings) != 1 {
		t.Fatalf("expected fallback warning, got selection=%+v warnings=%v", selection, warnings)
	}
}

func TestParsePactlSourcesKeepsNumericIndicesForSelection(t *testing.T) {
	info := `Server String: /run/user/1000/pulse/native
Default Sink: alsa_output.pci-0000_00_1f.3.analog-stereo
Default Source: alsa_input.usb-Blue_Mic-00.analog-stereo`
	sources := `42	alsa_input.pci-0000_00_1f.3.analog-stereo	module-alsa-card.c	s16le 2ch 48000Hz
43	alsa_input.usb-Blue_Mic-00.analog-stereo	module-alsa-card.c	s16le 2ch 48000Hz
44	alsa_output.pci-0000_00_1f.3.analog-stereo.monitor	module-alsa-card.c	s16le 2ch 48000Hz
45	alsa_output.usb-BT_Dongle-00.analog-stereo.monitor	module-alsa-card.c	s16le 2ch 48000Hz`
	devices := ParsePactlSources(sources, ParsePactlInfo(info))
	if len(devices) != 4 {
		t.Fatalf("expected four PulseAudio devices, got %d: %+v", len(devices), devices)
	}
	for i, wantAlias := range []string{"42", "43", "44", "45"} {
		if len(devices[i].Aliases) != 1 || devices[i].Aliases[0] != wantAlias {
			t.Fatalf("device %d should keep its numeric index alias %q, got %+v", i, wantAlias, devices[i].Aliases)
		}
	}
	formatted := FormatDevices(devices)
	if !containsAll(formatted, "aliases=42", "aliases=43", "aliases=44", "aliases=45") {
		t.Fatalf("formatted devices should expose PulseAudio indices as aliases, got:\n%s", formatted)
	}
	selection, warnings, err := SelectDevices(devices, Preferences{MicDeviceID: "43", OutputDeviceID: "45"})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Mic.ID != "alsa_input.usb-Blue_Mic-00.analog-stereo" || selection.Output.ID != "alsa_output.usb-BT_Dongle-00.analog-stereo.monitor" {
		t.Fatalf("numeric index selection picked the wrong devices: %+v", selection)
	}
	if len(warnings) != 0 {
		t.Fatalf("unexpected warnings when selecting by index: %v", warnings)
	}
}

func TestPulseFallbackUsesConcreteDefaultsAmongMultipleDevices(t *testing.T) {
	devices := []Device{
		{ID: "alsa_input.pci-0000_00_1f.3.analog-stereo", Name: "Built-in Mic", Aliases: []string{"42"}, Source: SourceMic, Backend: "pulse"},
		{ID: "alsa_input.usb-Blue_Mic-00.analog-stereo", Name: "USB Mic", Aliases: []string{"43"}, Source: SourceMic, Default: true, Backend: "pulse"},
		{ID: "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor", Name: "Built-in Speakers", Aliases: []string{"44"}, Source: SourceOutput, Default: true, Backend: "pulse"},
		{ID: "alsa_output.usb-BT_Dongle-00.analog-stereo.monitor", Name: "Bluetooth Dongle Monitor", Aliases: []string{"45"}, Source: SourceOutput, Backend: "pulse"},
	}
	selection, warnings, err := SelectDevices(devices, Preferences{MicDeviceID: "stale-mic", OutputDeviceID: "stale-output"})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Mic.ID != "alsa_input.usb-Blue_Mic-00.analog-stereo" {
		t.Fatalf("mic fallback should use the concrete default device, got %+v", selection.Mic)
	}
	if selection.Output.ID != "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor" {
		t.Fatalf("output fallback should use the concrete default device, got %+v", selection.Output)
	}
	if len(warnings) != 2 {
		t.Fatalf("expected stale-selection warnings for mic and output, got %v", warnings)
	}
	if !containsAll(warnings[0], "mic", "not found", "using") || !containsAll(warnings[1], "output", "not found", "using") {
		t.Fatalf("fallback warnings should explain the concrete default selection, got %v", warnings)
	}
}

func TestSelectDevicesRequiresOutputCapture(t *testing.T) {
	_, _, err := SelectDevices([]Device{{ID: "mic", Name: "Mic", Source: SourceMic, Default: true}}, Preferences{})
	if err == nil {
		t.Fatal("expected missing output error")
	}
}

func TestSelectDevicesSwitchesToDistinctOutputWhenDefaultMatchesMic(t *testing.T) {
	devices := []Device{
		{ID: "headset", Name: "Headset", Source: SourceMic, Default: true},
		{ID: "headset", Name: "Headset", Source: SourceOutput, Default: true},
		{ID: "stereo-mix", Name: "Stereo Mix", Source: SourceOutput},
	}
	selection, warnings, err := SelectDevices(devices, Preferences{})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Mic.ID != "headset" {
		t.Fatalf("unexpected mic selection: %+v", selection.Mic)
	}
	if selection.Output.ID != "stereo-mix" {
		t.Fatalf("output should switch to the distinct device instead of duplicating the mic, got %+v", selection.Output)
	}
	if len(warnings) == 0 || !containsAll(strings.Join(warnings, " "), "using", "Stereo Mix") {
		t.Fatalf("expected a warning about switching to the distinct output device, got %v", warnings)
	}
}

func TestSelectDevicesFailsWhenNoDistinctOutputDeviceExists(t *testing.T) {
	_, _, err := SelectDevices([]Device{
		{ID: "shared", Name: "Headset", Source: SourceMic, Default: true},
		{ID: "shared", Name: "Headset", Source: SourceOutput, Default: true},
	}, Preferences{})
	if err == nil {
		t.Fatal("expected identical mic/output selection to fail when no distinct output exists")
	}
	if !containsAll(err.Error(), "same device", "output") {
		t.Fatalf("duplicate-device error should explain the failure, got %q", err.Error())
	}
}

func TestSelectDevicesRejectsExplicitDuplicateOutputSelection(t *testing.T) {
	_, _, err := SelectDevices([]Device{
		{ID: "headset", Name: "Headset", Source: SourceMic, Default: true},
		{ID: "headset", Name: "Headset", Source: SourceOutput, Default: true},
		{ID: "stereo-mix", Name: "Stereo Mix", Source: SourceOutput},
	}, Preferences{OutputDeviceID: "headset"})
	if err == nil {
		t.Fatal("expected explicit duplicate output selection to fail")
	}
	if !containsAll(err.Error(), "same device", "output") {
		t.Fatalf("explicit duplicate output error should explain the failure, got %q", err.Error())
	}
}

func TestParseDShowAudioDevicesIgnoresVideoAndKeepsAlternates(t *testing.T) {
	devices := ParseDShowAudioDevices(sampleDShowDeviceList)
	if len(devices) != 3 {
		t.Fatalf("expected three DirectShow audio devices, got %d: %+v", len(devices), devices)
	}
	for _, device := range devices {
		if device.Name == "Integrated Camera" {
			t.Fatalf("video device was misclassified as audio: %+v", devices)
		}
		if len(device.Alternates) != 1 {
			t.Fatalf("expected one alternative name for %q, got %+v", device.Name, device.Alternates)
		}
	}
	if devices[0].Name != "Microphone Array (Realtek(R) Audio)" || devices[0].Alternates[0] != `@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{11111111-1111-1111-1111-111111111111}` {
		t.Fatalf("first audio device/alias not parsed correctly: %+v", devices[0])
	}
}

func TestParseDShowAudioDevicesKeepsAudioRowsWhenVideoComesFirst(t *testing.T) {
	devices := ParseDShowAudioDevices(sampleDShowDeviceListVideoFirst)
	if len(devices) != 3 {
		t.Fatalf("expected three audio devices despite the camera row, got %d: %+v", len(devices), devices)
	}
	for _, device := range devices {
		if device.Name == "Integrated Camera" {
			t.Fatalf("video device should be ignored when audio rows are present, got %+v", devices)
		}
	}
	if devices[0].Name != "Microphone Array (Realtek(R) Audio)" || devices[0].Alternates[0] != `@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{11111111-1111-1111-1111-111111111111}` {
		t.Fatalf("audio rows should still surface in order after a video-first listing, got %+v", devices[0])
	}
	parsed, warnings := devicesFromDShowAudioDevices(devices)
	if len(warnings) != 0 {
		t.Fatalf("video-first audio rows should not trigger loopback warnings, got %v", warnings)
	}
	var mics, outputs []Device
	for _, device := range parsed {
		switch device.Source {
		case SourceMic:
			mics = append(mics, device)
		case SourceOutput:
			outputs = append(outputs, device)
		}
	}
	if len(mics) != 3 || len(outputs) != 1 {
		t.Fatalf("expected microphones plus a single loopback output, got mics=%d outputs=%d devices=%+v", len(mics), len(outputs), parsed)
	}
	if !mics[0].Default || !outputs[0].Default || outputs[0].Name != "Stereo Mix (Realtek(R) Audio)" {
		t.Fatalf("mic/output defaults should still be assigned from the recovered audio rows, got mics=%+v outputs=%+v", mics, outputs)
	}
}

func TestWindowsDShowDevicesExposeLoopbackOutputOnly(t *testing.T) {
	listed := ParseDShowAudioDevices(sampleDShowDeviceList)
	devices, warnings := devicesFromDShowAudioDevices(listed)
	if len(warnings) != 0 {
		t.Fatalf("Stereo Mix should be recognized as an output candidate without warning, got %v", warnings)
	}
	var mics, outputs []Device
	for _, device := range devices {
		switch device.Source {
		case SourceMic:
			mics = append(mics, device)
		case SourceOutput:
			outputs = append(outputs, device)
		}
	}
	if len(mics) != 3 || len(outputs) != 1 {
		t.Fatalf("expected microphones plus a single loopback output, got mics=%d outputs=%d devices=%+v", len(mics), len(outputs), devices)
	}
	if mics[0].ID == "default" || mics[0].ID == "" {
		t.Fatalf("mic ID should be a concrete DirectShow name/alternative, got %+v", mics[0])
	}
	if outputs[0].Name != "Stereo Mix (Realtek(R) Audio)" || !outputs[0].Default {
		t.Fatalf("Stereo Mix should be preferred for output capture, got %+v", outputs[0])
	}
}

func TestWindowsDShowMicDefaultSkipsLoopbackDevices(t *testing.T) {
	listed := ParseDShowAudioDevices(sampleDShowDeviceListLoopbackFirst)
	devices, warnings := devicesFromDShowAudioDevices(listed)
	if len(warnings) != 0 {
		t.Fatalf("loopback-first list should not warn when a concrete loopback output is available, got %v", warnings)
	}
	selection, warnings, err := SelectDevices(devices, Preferences{})
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Fatalf("unexpected warnings when selecting default DirectShow devices: %v", warnings)
	}
	if selection.Mic.Name != "Microphone Array (Realtek(R) Audio)" {
		t.Fatalf("mic default should prefer the non-loopback microphone, got %+v", selection.Mic)
	}
	if selection.Output.Name != "Stereo Mix (Realtek(R) Audio)" {
		t.Fatalf("output default should still prefer Stereo Mix, got %+v", selection.Output)
	}
}

func TestParseWindowsAudioEndpointsJSON(t *testing.T) {
	endpoints, err := parseWindowsAudioEndpoints(`[
{"FriendlyName":"Microphone Array (Realtek(R) Audio)","InstanceId":"SWD\\MMDEVAPI\\{11111111-1111-1111-1111-111111111111}"},
{"FriendlyName":"Stereo Mix (Realtek(R) Audio)","InstanceId":"SWD\\MMDEVAPI\\{33333333-3333-3333-3333-333333333333}"}
]`)
	if err != nil {
		t.Fatal(err)
	}
	if len(endpoints) != 2 {
		t.Fatalf("expected two Windows audio endpoints, got %d: %+v", len(endpoints), endpoints)
	}
	if endpoints[0].DisplayName() != "Microphone Array (Realtek(R) Audio)" || endpoints[1].DisplayName() != "Stereo Mix (Realtek(R) Audio)" {
		t.Fatalf("Windows audio endpoint names were not preserved: %+v", endpoints)
	}
}

func TestWindowsAudioEndpointRenderCandidatesBecomeWasapiLoopbackOutputs(t *testing.T) {
	const (
		micID        = `SWD\MMDEVAPI\{11111111-1111-1111-1111-111111111111}`
		speakersID   = `SWD\MMDEVAPI\{33333333-3333-3333-3333-333333333333}`
		headphonesID = `SWD\MMDEVAPI\{44444444-4444-4444-4444-444444444444}`
	)
	devices, warnings := devicesFromWindowsAudioEndpoints([]windowsAudioEndpoint{
		{FriendlyName: "Microphone Array (Realtek(R) Audio)", InstanceID: micID},
		{FriendlyName: "Speakers (Realtek(R) Audio)", InstanceID: speakersID},
		{FriendlyName: "Headphones (Jabra Link 380)", InstanceID: headphonesID},
	})
	if len(warnings) != 0 {
		t.Fatalf("render endpoints should not warn about missing output capture, got %v", warnings)
	}
	var mics, outputs []Device
	for _, device := range devices {
		switch device.Source {
		case SourceMic:
			mics = append(mics, device)
		case SourceOutput:
			outputs = append(outputs, device)
		}
	}
	if len(mics) != 1 || len(outputs) != 2 {
		t.Fatalf("expected one microphone and two loopback outputs, got mics=%d outputs=%d devices=%+v", len(mics), len(outputs), devices)
	}
	if mics[0].ID != "Microphone Array (Realtek(R) Audio)" || mics[0].Name != "Microphone Array (Realtek(R) Audio)" {
		t.Fatalf("microphone endpoint should stay mic-only, got %+v", mics[0])
	}
	wantOutputs := map[string]string{
		"Speakers (Realtek(R) Audio)": "Speakers (Realtek(R) Audio) [Loopback]",
		"Headphones (Jabra Link 380)": "Headphones (Jabra Link 380) [Loopback]",
	}
	for _, device := range outputs {
		if device.Source != SourceOutput {
			t.Fatalf("render endpoint should be surfaced as output capture, got %+v", device)
		}
		if device.Backend != BackendWasapiLoopback {
			t.Fatalf("render endpoint should be prepared for the WASAPI sidecar backend, got %+v", device)
		}
		if want, ok := wantOutputs[device.ID]; !ok || device.Name != want {
			t.Fatalf("output endpoint should use a synthesized loopback display name, got %+v", device)
		}
		if len(device.Aliases) != 1 {
			t.Fatalf("output endpoint should preserve its Windows instance ID alias, got %+v", device.Aliases)
		}
	}
	if !outputs[0].Default || outputs[1].Default {
		t.Fatalf("first render endpoint should be the default output, got outputs=%+v", outputs)
	}
	selection, warnings, err := SelectDevices(devices, Preferences{})
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Fatalf("unexpected warnings when selecting render endpoints: %v", warnings)
	}
	if selection.Mic.ID != mics[0].ID {
		t.Fatalf("default mic should prefer the real microphone, got %+v", selection.Mic)
	}
	if selection.Output.ID != outputs[0].ID || selection.Output.Name != outputs[0].Name {
		t.Fatalf("default output should prefer the first render endpoint, got %+v", selection.Output)
	}
	selection, warnings, err = SelectDevices(devices, Preferences{OutputDeviceName: outputs[0].DisplayName()})
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Fatalf("selecting by display name should not warn, got %v", warnings)
	}
	if selection.Output.ID != outputs[0].ID {
		t.Fatalf("display-name selection should resolve to the first render endpoint, got %+v", selection.Output)
	}
	selection, warnings, err = SelectDevices(devices, Preferences{OutputDeviceID: outputs[0].Aliases[0]})
	if err != nil {
		t.Fatal(err)
	}
	if len(warnings) != 0 {
		t.Fatalf("selecting by alias should not warn, got %v", warnings)
	}
	if selection.Output.ID != outputs[0].ID {
		t.Fatalf("alias selection should resolve to the first render endpoint, got %+v", selection.Output)
	}
	formatted := FormatDevices(devices)
	if !containsAll(formatted, BackendWasapiLoopback, "Speakers (Realtek(R) Audio) [Loopback]", "Headphones (Jabra Link 380) [Loopback]", `aliases=`+speakersID, `aliases=`+headphonesID) {
		t.Fatalf("formatted fallback devices should expose loopback render endpoints and aliases, got:\n%s", formatted)
	}
}

func TestWindowsDShowMicOnlyListDoesNotExposeOutputCapture(t *testing.T) {
	listed := ParseDShowAudioDevices(sampleDShowDeviceListMicOnly)
	devices, warnings := devicesFromDShowAudioDevices(listed)
	if len(warnings) != 1 {
		t.Fatalf("mic-only Windows discovery should emit exactly one missing-output warning, got %v", warnings)
	}
	// The warning must be the actionable remediation, not a generic "no loopback"
	// note: it names Stereo Mix / a virtual loopback device and points at
	// `--list-devices` so a stock Windows user knows how to enable output capture.
	if warnings[0] != windowsOutputCaptureWarning {
		t.Fatalf("expected the canonical windowsOutputCaptureWarning, got %q", warnings[0])
	}
	if !containsAll(warnings[0], "loopback", "output-capture", "Stereo Mix", "VB-CABLE", "Voicemeeter", "--list-devices") {
		t.Fatalf("missing-loopback warning must be actionable (name Stereo Mix / a virtual loopback device and the --list-devices follow-up), got %q", warnings[0])
	}
	var outputs []Device
	for _, device := range devices {
		if device.Source == SourceOutput {
			outputs = append(outputs, device)
		}
	}
	if len(outputs) != 0 {
		t.Fatalf("plain microphones must not be exposed as output capture candidates, got %+v", outputs)
	}
	if _, _, err := SelectDevices(devices, Preferences{}); err == nil || !containsAll(err.Error(), "no output capture device") {
		t.Fatalf("mic-only Windows discovery should fail clearly when selecting output capture, got err=%v", err)
	}
}

func TestWindowsDirectShowMicOnlyCanSupplementLoopbackEndpointOutput(t *testing.T) {
	devices, warnings := devicesFromDShowAudioDevices(ParseDShowAudioDevices(sampleDShowDeviceListMicOnly))
	if len(warnings) != 1 || !containsAll(warnings[0], "loopback", "output-capture") {
		t.Fatalf("mic-only DirectShow discovery should warn about missing loopback output capture, got %v", warnings)
	}
	supplemented, supplementedWarnings, ok := supplementWindowsOutputCapture(devices, warnings, []windowsAudioEndpoint{
		{FriendlyName: "Microphone Array (Realtek(R) Audio)", InstanceID: `SWD\MMDEVAPI\{11111111-1111-1111-1111-111111111111}`},
		{FriendlyName: "Speakers (Realtek(R) Audio)", InstanceID: `@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{39D41DF1-9D14-414E-BBE8-2FFD637045DC}`},
	})
	if !ok {
		t.Fatal("expected Windows endpoint fallback to append a loopback output device")
	}
	var mics, outputs []Device
	for _, device := range supplemented {
		switch device.Source {
		case SourceMic:
			mics = append(mics, device)
		case SourceOutput:
			outputs = append(outputs, device)
		}
	}
	if len(mics) != 2 || len(outputs) != 1 {
		t.Fatalf("expected the mic-only DirectShow list plus one loopback output, got mics=%d outputs=%d devices=%+v", len(mics), len(outputs), supplemented)
	}
	if outputs[0].ID != "Speakers (Realtek(R) Audio)" || outputs[0].Name != "Speakers (Realtek(R) Audio) [Loopback]" || outputs[0].Backend != BackendWasapiLoopback || len(outputs[0].Aliases) != 1 {
		t.Fatalf("fallback output should keep the concrete loopback endpoint identity, got %+v", outputs[0])
	}
	if strings.Contains(strings.Join(supplementedWarnings, " "), windowsOutputCaptureWarning) {
		t.Fatalf("fallback supplementation should replace the stale output warning, got %v", supplementedWarnings)
	}
	if !containsAll(strings.Join(supplementedWarnings, " "), "PowerShell", "supplement output capture") {
		t.Fatalf("fallback supplementation warning should explain the PowerShell output source, got %v", supplementedWarnings)
	}
	selection, selectWarnings, err := SelectDevices(supplemented, Preferences{OutputDeviceID: `@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{39D41DF1-9D14-414E-BBE8-2FFD637045DC}`})
	if err != nil {
		t.Fatal(err)
	}
	if len(selectWarnings) != 0 {
		t.Fatalf("selecting the supplemented loopback output should not warn, got %v", selectWarnings)
	}
	if selection.Output.ID != outputs[0].ID {
		t.Fatalf("supplemented output selection should resolve to the loopback endpoint, got %+v", selection.Output)
	}
}

func TestFormatDevicesIncludesDirectShowAlternativeNames(t *testing.T) {
	devices, _ := devicesFromDShowAudioDevices(ParseDShowAudioDevices(sampleDShowDeviceList))
	formatted := FormatDevices(devices)
	if !containsAll(formatted, "Microphone Array (Realtek(R) Audio)", "aliases=@device_cm_", "Stereo Mix (Realtek(R) Audio)") {
		t.Fatalf("formatted devices should expose concrete DirectShow names and aliases, got:\n%s", formatted)
	}
	if strings.Contains(formatted, "audio=default") {
		t.Fatalf("device listing must not expose synthetic audio=default, got:\n%s", formatted)
	}
}

func TestWindowsDefaultPreferenceResolvesToConcreteDShowDevice(t *testing.T) {
	devices, _ := devicesFromDShowAudioDevices(ParseDShowAudioDevices(sampleDShowDeviceList))
	selection, warnings, err := SelectDevices(devices, Preferences{MicDeviceID: "default", OutputDeviceID: "default"})
	if err != nil {
		t.Fatal(err)
	}
	if selection.Mic.ID == "default" || selection.Output.ID == "default" {
		t.Fatalf("default preferences must resolve to concrete DirectShow devices, got %+v", selection)
	}
	if len(warnings) != 2 {
		t.Fatalf("expected unresolved-default warnings for mic/output, got %v", warnings)
	}
	for _, warning := range warnings {
		if !containsAll(warning, "default", "concrete", "--list-devices") {
			t.Fatalf("warning should explain unresolved default and list-devices hint, got %q", warning)
		}
	}
}

func TestMissingDShowDefaultErrorIsActionable(t *testing.T) {
	_, _, err := SelectDevices([]Device{{ID: "out", Name: "Output", Source: SourceOutput, Default: true}}, Preferences{MicDeviceID: "default"})
	if err == nil {
		t.Fatal("expected missing mic error")
	}
	if !containsAll(err.Error(), "--list-devices", "default", "unresolved") {
		t.Fatalf("missing device error should mention list-devices and unresolved default, got %q", err.Error())
	}
}

func containsAll(value string, needles ...string) bool {
	for _, needle := range needles {
		if !strings.Contains(value, needle) {
			return false
		}
	}
	return true
}

func TestOutputCaptureCandidatesExcludesSynthesizedRenderEndpoints(t *testing.T) {
	// These two Speakers/Headphones devices mirror exactly what
	// devicesFromWindowsAudioEndpoints emits (raw render endpoint name as the ID +
	// a "[Loopback]" display name + an InstanceID alias). They are display-only:
	// ffmpeg cannot open `audio=Speakers (...)`. The runtime fallback chain must
	// never fall through to them. Stereo Mix is a real ffmpeg-openable source and
	// must be the reachable fallback target. (The failing/current device may stay
	// at index 0 so nextOutputCaptureCandidate can advance past it via [1], but
	// the actual fallback targets candidates[1:] must be real, buildable devices.)
	speakers := Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{33333333-3333-3333-3333-333333333333}`}, Source: SourceOutput, Default: true, Backend: "dshow"}
	headphones := Device{ID: "Headphones (JBL Flip 6)", Name: "Headphones (JBL Flip 6) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{44444444-4444-4444-4444-444444444444}`}, Source: SourceOutput, Backend: "dshow"}
	stereoMix := Device{ID: "Stereo Mix (Realtek(R) Audio)", Name: "Stereo Mix (Realtek(R) Audio)", Source: SourceOutput, Backend: "dshow"}
	mic := Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Default: true, Backend: "dshow"}
	devices := []Device{mic, speakers, headphones, stereoMix}

	candidates := OutputCaptureCandidates(devices, speakers)
	if len(candidates) == 0 {
		t.Fatalf("fallback chain should at least keep the current device, got empty")
	}
	// Every actual fallback target (after the current device at index 0) must be
	// a real, runtime-capable source — synthesized render endpoints may never be
	// reached via a fallback attempt.
	for _, candidate := range candidates[1:] {
		if candidate.Backend == "dshow" && SynthesizedDirectShowRenderEndpoint(candidate) {
			t.Fatalf("runtime fallback must never reach a synthesized render endpoint, got %+v", candidate)
		}
		if candidate.ID == speakers.ID || candidate.ID == headphones.ID {
			t.Fatalf("synthesized %q reached as a fallback target: %+v", candidate.ID, candidates)
		}
	}
	// The first real fallback target (what nextOutputCaptureCandidate would pick)
	// must be the real Stereo Mix loopback, not a synthesized render endpoint.
	next := candidates[1]
	if next.ID != stereoMix.ID {
		t.Fatalf("first fallback target should be the real Stereo Mix source, got %+v (chain=%+v)", next, candidates)
	}
}

func TestOutputCaptureCandidatesKeepsSelfWhenOnlySynthesizedDevicesRemain(t *testing.T) {
	// Regression for the cascading "falling back -> failed -> falling back" churn:
	// when every discovered output device is synthesized, the fallback chain must
	// collapse to just the failing device (self) so the caller's len(candidates)
	// check fails and the session surfaces a single clear error instead of
	// cycling through ffmpeg-unopenable render endpoints one by one.
	speakers := Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{33333333-3333-3333-3333-333333333333}`}, Source: SourceOutput, Default: true, Backend: "dshow"}
	headphones := Device{ID: "Headphones (JBL Flip 6)", Name: "Headphones (JBL Flip 6) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{44444444-4444-4444-4444-444444444444}`}, Source: SourceOutput, Backend: "dshow"}
	mic := Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Default: true, Backend: "dshow"}
	devices := []Device{mic, speakers, headphones}

	candidates := OutputCaptureCandidates(devices, speakers)
	if len(candidates) != 1 || !sameDevice(candidates[0], speakers) {
		t.Fatalf("fallback chain should collapse to only self when no runtime-capable candidate exists, got %+v", candidates)
	}
}

func TestOutputCaptureCandidatesKeepsNonDShowBackends(t *testing.T) {
	// Non-Windows backends (pulse, avfoundation) and test fakes must keep flowing
	// through the runtime fallback chain unchanged — the synthesized-render-endpoint
	// filter is a Windows-only DirectShow concern.
	pulseMonitor := Device{ID: "alsa_output.pci.monitor", Name: "Built-in Monitor", Source: SourceOutput, Default: true, Backend: "pulse"}
	fakeOutput := Device{ID: "out-b", Name: "QA Output B", Source: SourceOutput, Backend: "fake"}
	mic := Device{ID: "mic", Name: "Mic", Source: SourceMic, Default: true, Backend: "pulse"}
	devices := []Device{mic, pulseMonitor, fakeOutput}

	candidates := OutputCaptureCandidates(devices, pulseMonitor)
	if len(candidates) != 2 {
		t.Fatalf("non-dshow output devices should be preserved in fallback chain, got %+v", candidates)
	}
	for _, candidate := range candidates {
		if SynthesizedDirectShowRenderEndpoint(candidate) {
			t.Fatalf("non-dshow device should never classify as synthesized, got %+v", candidate)
		}
	}
}

func TestOutputCaptureCandidatesKeepsWasapiSidecarCandidatesAsRuntimeTargets(t *testing.T) {
	// WASAPI loopback render endpoints are runtime-capable through the native
	// helper sidecar and may be used as fallback targets after another output
	// capture attempt fails. A real DirectShow loopback source such as Stereo Mix
	// remains in the chain too.
	speakers := Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{33333333-3333-3333-3333-333333333333}`}, Source: SourceOutput, Default: true, Backend: BackendWasapiLoopback}
	headphones := Device{ID: "Headphones (JBL Flip 6)", Name: "Headphones (JBL Flip 6) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{44444444-4444-4444-4444-444444444444}`}, Source: SourceOutput, Backend: BackendWasapiLoopback}
	stereoMix := Device{ID: "Stereo Mix (Realtek(R) Audio)", Name: "Stereo Mix (Realtek(R) Audio)", Source: SourceOutput, Backend: BackendDirectShow}
	mic := Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Default: true, Backend: BackendDirectShow}

	candidates := OutputCaptureCandidates([]Device{mic, speakers, headphones, stereoMix}, speakers)
	if len(candidates) != 3 {
		t.Fatalf("expected WASAPI sidecar endpoints plus DirectShow loopback in fallback chain, got %+v", candidates)
	}
	if candidates[0].ID != speakers.ID || candidates[1].ID != headphones.ID || candidates[2].ID != stereoMix.ID {
		t.Fatalf("unexpected fallback order for WASAPI sidecar endpoints, got %+v", candidates)
	}
}

func TestOutputCaptureCandidatesKeepsWasapiSidecarOnlySet(t *testing.T) {
	speakers := Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{33333333-3333-3333-3333-333333333333}`}, Source: SourceOutput, Default: true, Backend: BackendWasapiLoopback}
	headphones := Device{ID: "Headphones (JBL Flip 6)", Name: "Headphones (JBL Flip 6) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{44444444-4444-4444-4444-444444444444}`}, Source: SourceOutput, Backend: BackendWasapiLoopback}
	mic := Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Default: true, Backend: BackendDirectShow}

	candidates := OutputCaptureCandidates([]Device{mic, speakers, headphones}, speakers)
	if len(candidates) != 2 || !sameDevice(candidates[0], speakers) || !sameDevice(candidates[1], headphones) {
		t.Fatalf("WASAPI sidecar endpoints should stay runtime-capable when no DirectShow fallback exists, got %+v", candidates)
	}
}

func TestSynthesizedRenderEndpointChecksIDNotPoisonedDisplayName(t *testing.T) {
	// The display name is poisoned with a literal "[Loopback]" suffix by
	// devicesFromWindowsAudioEndpoints, so the classifier must inspect the ID and
	// aliases (which carry the real loopback signature) — never the display name.
	// A real Stereo Mix device keeps its friendly name as the ID, so it must NOT
	// be flagged synthesized even though a naive name match might misfire.
	realStereoMix := Device{ID: "Stereo Mix (Realtek(R) Audio)", Name: "Stereo Mix (Realtek(R) Audio) [Loopback]", Source: SourceOutput, Backend: "dshow"}
	if SynthesizedDirectShowRenderEndpoint(realStereoMix) {
		t.Fatalf("real loopback source should not be flagged synthesized just because its display name carries [Loopback], got %+v", realStereoMix)
	}
	synthesized := Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Source: SourceOutput, Backend: "dshow"}
	if !SynthesizedDirectShowRenderEndpoint(synthesized) {
		t.Fatalf("render endpoint without a loopback alias should be flagged synthesized, got %+v", synthesized)
	}
	synthesizedWithRealAlias := Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio)", Aliases: []string{"Stereo Mix"}, Source: SourceOutput, Backend: "dshow"}
	if SynthesizedDirectShowRenderEndpoint(synthesizedWithRealAlias) {
		t.Fatalf("device whose alias carries a real loopback signature should NOT be flagged synthesized, got %+v", synthesizedWithRealAlias)
	}
}

func TestOutputCaptureUnavailableGuidanceFiresForSynthesizedOnlySet(t *testing.T) {
	mic := Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Default: true, Backend: "dshow"}
	speakers := Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{33333333-3333-3333-3333-333333333333}`}, Source: SourceOutput, Backend: "dshow"}
	guidance := OutputCaptureUnavailableGuidance([]Device{mic, speakers})
	if guidance == "" {
		t.Fatalf("guidance should fire when only synthesized render endpoints remain")
	}
	if guidance != windowsOutputCaptureWarning {
		t.Fatalf("guidance should be the canonical windowsOutputCaptureWarning, got %q", guidance)
	}
	if !containsAll(guidance, "Stereo Mix", "VB-CABLE", "Voicemeeter", "--list-devices") {
		t.Fatalf("guidance must name the concrete remediation, got %q", guidance)
	}
}

func TestOutputCaptureUnavailableGuidanceSilentWhenRealLoopbackExists(t *testing.T) {
	mic := Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Default: true, Backend: "dshow"}
	stereoMix := Device{ID: "Stereo Mix (Realtek(R) Audio)", Name: "Stereo Mix (Realtek(R) Audio)", Source: SourceOutput, Backend: "dshow"}
	if guidance := OutputCaptureUnavailableGuidance([]Device{mic, stereoMix}); guidance != "" {
		t.Fatalf("guidance should stay silent when a real loopback source exists, got %q", guidance)
	}
	if guidance := OutputCaptureUnavailableGuidance([]Device{mic}); guidance != "" {
		t.Fatalf("guidance should stay silent when there are no output devices at all (discovery warns), got %q", guidance)
	}
}

func TestOutputCaptureUnavailableGuidanceSilentForWasapiSidecarOnlySet(t *testing.T) {
	mic := Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Default: true, Backend: BackendDirectShow}
	speakers := Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{33333333-3333-3333-3333-333333333333}`}, Source: SourceOutput, Backend: BackendWasapiLoopback}
	if guidance := OutputCaptureUnavailableGuidance([]Device{mic, speakers}); guidance != "" {
		t.Fatalf("WASAPI sidecar endpoints are runtime-capable, so guidance should stay silent, got %q", guidance)
	}
}

const sampleDShowDeviceList = `[dshow @ 000001a29acef880] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000001a29acef880]  "Integrated Camera"
[dshow @ 000001a29acef880]     Alternative name "@device_pnp_\\?\usb#vid_5986&pid_2115#camera"
[dshow @ 000001a29acef880] DirectShow audio devices
[dshow @ 000001a29acef880]  "Microphone Array (Realtek(R) Audio)"
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{11111111-1111-1111-1111-111111111111}"
[dshow @ 000001a29acef880]  "Headset Microphone (Jabra Link 380)"
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{22222222-2222-2222-2222-222222222222}"
[dshow @ 000001a29acef880]  "Stereo Mix (Realtek(R) Audio)"
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{33333333-3333-3333-3333-333333333333}"
dummy: Immediate exit requested`

const sampleDShowDeviceListVideoFirst = `[dshow @ 000001a29acef880] DirectShow video devices (some may be both video and audio devices)
[dshow @ 000001a29acef880]  "Integrated Camera" (video)
[dshow @ 000001a29acef880]     Alternative name "@device_pnp_\\?\usb#vid_5986&pid_2115#camera"
[dshow @ 000001a29acef880]  "Microphone Array (Realtek(R) Audio)" (audio)
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{11111111-1111-1111-1111-111111111111}"
[dshow @ 000001a29acef880]  "Headset Microphone (Jabra Link 380)" (audio)
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{22222222-2222-2222-2222-222222222222}"
[dshow @ 000001a29acef880]  "Stereo Mix (Realtek(R) Audio)" (audio)
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{33333333-3333-3333-3333-333333333333}"
dummy: Immediate exit requested`

const sampleDShowDeviceListLoopbackFirst = `[dshow @ 000001a29acef880] DirectShow audio devices
[dshow @ 000001a29acef880]  "Stereo Mix (Realtek(R) Audio)"
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{33333333-3333-3333-3333-333333333333}"
[dshow @ 000001a29acef880]  "Microphone Array (Realtek(R) Audio)"
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{11111111-1111-1111-1111-111111111111}"
dummy: Immediate exit requested`

const sampleDShowDeviceListMicOnly = `[dshow @ 000001a29acef880] DirectShow audio devices
[dshow @ 000001a29acef880]  "Microphone Array (Realtek(R) Audio)"
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{11111111-1111-1111-1111-111111111111}"
[dshow @ 000001a29acef880]  "Headset Microphone (Jabra Link 380)"
[dshow @ 000001a29acef880]     Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{22222222-2222-2222-2222-222222222222}"
dummy: Immediate exit requested`
