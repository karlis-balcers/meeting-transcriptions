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

func TestSelectDevicesRequiresOutputCapture(t *testing.T) {
	_, _, err := SelectDevices([]Device{{ID: "mic", Name: "Mic", Source: SourceMic, Default: true}}, Preferences{})
	if err == nil {
		t.Fatal("expected missing output error")
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

func TestWindowsDShowDevicesSurfaceAllAudioForMicAndOutput(t *testing.T) {
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
	if len(mics) != 3 || len(outputs) != 3 {
		t.Fatalf("expected every DirectShow audio device in both settings roles, got mics=%d outputs=%d devices=%+v", len(mics), len(outputs), devices)
	}
	if mics[0].ID == "default" || mics[0].ID == "" {
		t.Fatalf("mic ID should be a concrete DirectShow name/alternative, got %+v", mics[0])
	}
	var defaultOutput Device
	for _, output := range outputs {
		if output.Default {
			defaultOutput = output
			break
		}
	}
	if defaultOutput.Name != "Stereo Mix (Realtek(R) Audio)" {
		t.Fatalf("Stereo Mix should be preferred for output capture, got %+v", defaultOutput)
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
