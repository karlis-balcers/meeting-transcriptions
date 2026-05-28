package audio

import (
	"context"
	"reflect"
	"strings"
	"testing"
)

func TestExternalRecorderRejectsUnresolvedDShowDefaultsBeforeCapture(t *testing.T) {
	selection := Selection{
		Mic: Device{
			ID:      "default",
			Name:    "Default Windows microphone",
			Source:  SourceMic,
			Backend: "dshow",
		},
		Output: Device{
			ID:      `@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{33333333-3333-3333-3333-333333333333}`,
			Name:    "Stereo Mix (Realtek(R) Audio)",
			Source:  SourceOutput,
			Backend: "dshow",
		},
	}

	err := (ExternalRecorder{}).Validate(context.Background(), selection)
	if err == nil {
		t.Fatal("expected unresolved DirectShow default to fail validation before ffmpeg capture")
	}
	if !containsAll(err.Error(), "default", "DirectShow", "--list-devices") {
		t.Fatalf("error should explain unresolved DirectShow default and list-devices hint, got %q", err.Error())
	}
}

func TestDShowInputArgsUseConcreteAlternativeName(t *testing.T) {
	deviceID := `@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{11111111-1111-1111-1111-111111111111}`
	got := inputArgs(Device{ID: deviceID, Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Backend: "dshow"})
	want := []string{"-f", "dshow", "-i", "audio=" + deviceID}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected DirectShow input args: got %#v want %#v", got, want)
	}
	if strings.Contains(strings.Join(got, " "), "audio=default") {
		t.Fatalf("DirectShow input args must not use synthetic audio=default: %#v", got)
	}
}
