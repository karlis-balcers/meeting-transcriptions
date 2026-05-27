package audio

import "testing"

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
