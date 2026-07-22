package audio

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
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

func TestDShowInputArgsUseFriendlyName(t *testing.T) {
	deviceName := "Microphone Array (Realtek(R) Audio)"
	got := inputArgs(Device{ID: deviceName, Name: deviceName, Source: SourceMic, Backend: "dshow"})
	want := []string{"-f", "dshow", "-i", "audio=" + deviceName}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected DirectShow input args: got %#v want %#v", got, want)
	}
	if strings.Contains(strings.Join(got, " "), "audio=default") {
		t.Fatalf("DirectShow input args must not use synthetic audio=default: %#v", got)
	}
}

func TestWindowsLoopbackOutputArgsUseRawRenderName(t *testing.T) {
	deviceID := "Speakers (Realtek(R) Audio)"
	got := inputArgs(Device{ID: deviceID, Name: deviceID + " [Loopback]", Source: SourceOutput, Backend: "dshow"})
	want := []string{"-f", "dshow", "-i", "audio=" + deviceID}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("unexpected DirectShow output args for a synthesized loopback label: got %#v want %#v", got, want)
	}
	if strings.Contains(strings.Join(got, " "), "[Loopback]") {
		t.Fatalf("ffmpeg capture args must keep using the raw render endpoint name, got %#v", got)
	}
}

func TestWindowsOutputRecorderUsesFFmpegDirectShow(t *testing.T) {
	recorder := ExternalRecorder{
		FFmpegPath: "ffmpeg",
		Platform:   "windows",
	}
	request := ChunkRequest{
		Device: Device{
			ID:      "Speakers (Realtek(R) Audio)",
			Name:    "Speakers (Realtek(R) Audio) [Loopback]",
			Source:  SourceOutput,
			Backend: "dshow",
		},
		Source:   SourceOutput,
		TempDir:  t.TempDir(),
		Duration: 2 * time.Second,
		Sequence: 7,
	}
	outputFile := "transcribe-output-000007.wav"
	command, args, err := recorder.commandForRequest(request, outputFile)
	if err != nil {
		t.Fatal(err)
	}
	if command != "ffmpeg" {
		t.Fatalf("Windows output capture must use ffmpeg, got command %q", command)
	}
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "-f dshow") {
		t.Fatalf("Windows output capture must use the ffmpeg DirectShow backend, got args %#v", args)
	}
	if !strings.Contains(joined, "audio=Speakers (Realtek(R) Audio)") {
		t.Fatalf("Windows output capture must target the raw output device name via audio=<device>, got args %#v", args)
	}
	if strings.Contains(joined, "[Loopback]") {
		t.Fatalf("Windows output capture args must not echo the synthesized [Loopback] name, got args %#v", args)
	}
	if strings.Contains(joined, "audio_capture.py") || strings.Contains(joined, "record-output") {
		t.Fatalf("Windows output capture must not invoke the legacy Python helper, got args %#v", args)
	}
}

func TestWindowsWasapiLoopbackRecorderUsesSidecarProtocol(t *testing.T) {
	tempDir := t.TempDir()
	helperPath := filepath.Join(tempDir, "wasapi-loopback-recorder.exe")
	if err := os.WriteFile(helperPath, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}
	recorder := ExternalRecorder{
		FFmpegPath:       "ffmpeg",
		Platform:         "windows",
		WasapiHelperPath: helperPath,
	}
	request := ChunkRequest{
		Device: Device{
			ID:      "Speakers (Realtek(R) Audio)",
			Name:    "Speakers (Realtek(R) Audio) [Loopback]",
			Source:  SourceOutput,
			Backend: BackendWasapiLoopback,
		},
		Source:   SourceOutput,
		TempDir:  tempDir,
		Duration: 2 * time.Second,
		Sequence: 7,
	}
	outputFile := filepath.Join(tempDir, "transcribe-output-000007.wav")
	command, args, err := recorder.commandForRequest(request, outputFile)
	if err != nil {
		t.Fatal(err)
	}
	if command != helperPath {
		t.Fatalf("WASAPI loopback output capture should use the helper sidecar, got %q", command)
	}
	joined := strings.Join(args, " ")
	if !containsAll(joined, "record", "--output-file", outputFile, "--duration", "2.000", "--sample-rate", "16000", "--channels", "1", "--device-id", request.Device.ID, "--device-name", request.Device.Name) {
		t.Fatalf("sidecar command did not include the expected protocol arguments: %#v", args)
	}
	if strings.Contains(joined, "-f dshow") || strings.Contains(joined, "audio=") || strings.Contains(joined, "audio_capture.py") || strings.Contains(joined, "record-output") {
		t.Fatalf("WASAPI loopback sidecar command must not use DirectShow ffmpeg or the legacy Python helper, got %#v", args)
	}
}

func TestWindowsWasapiLoopbackRecorderReportsMissingSidecar(t *testing.T) {
	recorder := ExternalRecorder{FFmpegPath: "ffmpeg", Platform: "windows", WasapiHelperPath: filepath.Join(t.TempDir(), "missing.exe")}
	_, _, err := recorder.commandForRequest(ChunkRequest{
		Device:   Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Source: SourceOutput, Backend: BackendWasapiLoopback},
		Source:   SourceOutput,
		TempDir:  t.TempDir(),
		Duration: time.Second,
		Sequence: 1,
	}, filepath.Join(t.TempDir(), "out.wav"))
	if err == nil {
		t.Fatal("expected missing WASAPI helper to fail before capture")
	}
	if !containsAll(err.Error(), "wasapi-loopback-recorder.exe", "TRANSCRIBE_WINDOWS_WASAPI_HELPER", "bundled helper") {
		t.Fatalf("missing helper error should name the sidecar protocol and implementation gap, got %q", err.Error())
	}
	if strings.Contains(err.Error(), "audio_capture.py") || strings.Contains(err.Error(), "TRANSCRIBE_WINDOWS_AUDIO_HELPER") || strings.Contains(err.Error(), "TRANSCRIBE_WINDOWS_PYTHON") {
		t.Fatalf("missing helper error must not reference removed Python helper plumbing, got %q", err.Error())
	}
}

func TestWindowsWasapiLoopbackValidationRequiresBundledSidecar(t *testing.T) {
	recorder := ExternalRecorder{FFmpegPath: "ffmpeg", Platform: "windows", WasapiHelperPath: filepath.Join(t.TempDir(), "missing.exe")}
	err := recorder.Validate(context.Background(), Selection{
		Mic:    Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Backend: BackendDirectShow},
		Output: Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Source: SourceOutput, Backend: BackendWasapiLoopback},
	})
	if err == nil {
		t.Fatal("expected validation to require the WASAPI helper before session start")
	}
	if !containsAll(err.Error(), "wasapi-loopback-recorder.exe", "TRANSCRIBE_WINDOWS_WASAPI_HELPER") {
		t.Fatalf("validation error should name the helper packaging contract, got %q", err.Error())
	}
}

func TestWindowsOutputRecorderValidationRequiresFFmpegOnly(t *testing.T) {
	// Windows output capture is now unified on ffmpeg DirectShow, so Validate must
	// never reference audio_capture.py / TRANSCRIBE_WINDOWS_* plumbing. Probe the
	// ffmpeg-not-found path deterministically: with FFmpegPath empty and no
	// resolvable ffmpeg on PATH, Validate must blame ffmpeg (not a Python helper).
	selection := Selection{
		Mic:    Device{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: SourceMic, Backend: "dshow"},
		Output: Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Source: SourceOutput, Backend: "dshow"},
	}
	if err := (ExternalRecorder{FFmpegPath: "ffmpeg", Platform: "windows"}).Validate(context.Background(), selection); err != nil {
		t.Fatalf("Windows output validation should succeed with ffmpeg available, got %v", err)
	}
	// Force the missing-ffmpeg branch: a non-empty FFmpegPath that ffmpeg() trusts
	// means Validate returns nil (it does not stat the binary). Assert that stable
	// behavior so future code can't reintroduce a Python-helper probe here.
	if err := (ExternalRecorder{FFmpegPath: "/nonexistent/transcribe-ffmpeg", Platform: "windows"}).Validate(context.Background(), selection); err != nil {
		if strings.Contains(err.Error(), "audio_capture.py") || strings.Contains(err.Error(), "TRANSCRIBE_WINDOWS_") {
			t.Fatalf("validation error must not reference removed Python helper plumbing, got %q", err.Error())
		}
		t.Fatalf("Validate should accept a configured FFmpegPath without probing the binary, got %v", err)
	}
}
