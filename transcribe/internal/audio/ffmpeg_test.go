package audio

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"time"
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

func TestWindowsOutputRecorderUsesPythonWASAPIHelper(t *testing.T) {
	tempDir := t.TempDir()
	pythonPath := filepath.Join(tempDir, "python.exe")
	helperPath := filepath.Join(tempDir, "audio_capture.py")
	if err := os.WriteFile(pythonPath, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(helperPath, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}
	recorder := ExternalRecorder{Platform: "windows", PythonPath: pythonPath, HelperScriptPath: helperPath}
	request := ChunkRequest{
		Device: Device{
			ID:      "Speakers (Realtek(R) Audio)",
			Name:    "Speakers (Realtek(R) Audio) [Loopback]",
			Source:  SourceOutput,
			Backend: "dshow",
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
	if command != pythonPath {
		t.Fatalf("Windows output capture should use Python, got %q", command)
	}
	joined := strings.Join(args, " ")
	if !containsAll(joined, helperPath, "record-output", "--output-file", outputFile, "--device-id", request.Device.ID, "--device-name", request.Device.Name) {
		t.Fatalf("helper command did not include the expected WASAPI arguments: %#v", args)
	}
	if strings.Contains(joined, "-f dshow") || strings.Contains(joined, "audio=") {
		t.Fatalf("Windows output helper command must not fall back to DirectShow ffmpeg args, got %#v", args)
	}
}

func TestWindowsOutputRecorderValidationRequiresHelperScript(t *testing.T) {
	tempDir := t.TempDir()
	pythonPath := filepath.Join(tempDir, "python.exe")
	if err := os.WriteFile(pythonPath, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}
	recorder := ExternalRecorder{FFmpegPath: "ffmpeg", Platform: "windows", PythonPath: pythonPath, HelperScriptPath: filepath.Join(tempDir, "missing", "audio_capture.py")}
	selection := Selection{
		Mic: Device{ID: "mic", Name: "Mic", Source: SourceMic, Backend: "dshow"},
		Output: Device{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Source: SourceOutput, Backend: "dshow"},
	}
	if err := recorder.Validate(context.Background(), selection); err == nil || !containsAll(err.Error(), "audio_capture.py", "TRANSCRIBE_WINDOWS_AUDIO_HELPER") {
		t.Fatalf("expected helper validation to fail clearly, got %v", err)
	}
}

func TestWindowsPythonPathPrefersBuildLocalVenv(t *testing.T) {
	tempDir := t.TempDir()
	pythonPath := filepath.Join(tempDir, ".venv", "Scripts", "python.exe")
	if err := os.MkdirAll(filepath.Dir(pythonPath), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(pythonPath, []byte(""), 0o600); err != nil {
		t.Fatal(err)
	}

	oldWd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(tempDir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(oldWd)
	})

	got, err := (ExternalRecorder{Platform: "windows"}).windowsPythonPath()
	if err != nil {
		t.Fatal(err)
	}
	want, err := filepath.Abs(pythonPath)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("windowsPythonPath should prefer the build-local venv: got %q want %q", got, want)
	}
}
