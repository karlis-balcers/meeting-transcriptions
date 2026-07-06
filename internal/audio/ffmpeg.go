package audio

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

const wasapiLoopbackHelperName = "wasapi-loopback-recorder.exe"

type ExternalRecorder struct {
	FFmpegPath       string
	Platform         string
	WasapiHelperPath string
}

func (r ExternalRecorder) platform() string {
	if value := strings.TrimSpace(r.Platform); value != "" {
		return strings.ToLower(value)
	}
	return runtime.GOOS
}

// commandForRequest builds the recorder command for the given chunk. The main
// app consumes only bounded PCM WAV chunks after this point; platform-specific
// capture details stay inside this command selection layer.
func (r ExternalRecorder) commandForRequest(request ChunkRequest, filePath string) (string, []string, error) {
	if r.usesWasapiLoopbackSidecar(request) {
		return r.wasapiLoopbackCommand(request, filePath)
	}
	path, err := r.ffmpeg()
	if err != nil {
		return "", nil, err
	}
	args := []string{"-nostdin", "-hide_banner", "-loglevel", "error", "-y"}
	args = append(args, inputArgs(request.Device)...)
	args = append(args,
		"-t", fmt.Sprintf("%.3f", request.Duration.Seconds()),
		"-acodec", "pcm_s16le",
		"-ar", "16000",
		"-ac", "1",
		filePath,
	)
	return path, args, nil
}

func (r ExternalRecorder) Validate(_ context.Context, selected Selection) error {
	if selected.Mic.ID == "" {
		return errors.New("microphone device is not selected")
	}
	if isUnresolvedDShowDevice(selected.Mic) {
		return errors.New("microphone device \"default\" was unresolved for DirectShow; run transcribe --list-devices and choose a displayed Windows microphone device name/id")
	}
	if selected.Output.ID == "" {
		return errors.New("system-output capture device is not selected")
	}
	if isUnresolvedDShowDevice(selected.Output) {
		return errors.New("system-output DirectShow device was unresolved; run transcribe --list-devices and choose a displayed output-capture device name/id")
	}
	if _, err := r.ffmpeg(); err != nil {
		return errors.New("ffmpeg was not found in PATH; install ffmpeg or configure a recorder backend before starting")
	}
	if r.platform() == "windows" && selected.Output.Source == SourceOutput && selected.Output.Backend == BackendWasapiLoopback {
		if _, err := r.wasapiHelper(); err != nil {
			return err
		}
	}
	return nil
}

func (r ExternalRecorder) RecordChunk(ctx context.Context, request ChunkRequest) (Chunk, error) {
	if request.Duration <= 0 {
		return Chunk{}, errors.New("chunk duration must be positive")
	}
	if err := os.MkdirAll(request.TempDir, 0o700); err != nil {
		return Chunk{}, fmt.Errorf("create temp directory: %w", err)
	}
	started := request.Started
	if started.IsZero() {
		started = time.Now()
	}
	filePath := filepath.Join(request.TempDir, fmt.Sprintf("transcribe-%s-%06d.wav", request.Source, request.Sequence))
	path, args, err := r.commandForRequest(request, filePath)
	if err != nil {
		return Chunk{}, err
	}
	backendName := "ffmpeg"
	if r.usesWasapiLoopbackSidecar(request) {
		backendName = "Windows WASAPI loopback sidecar"
	}
	cmd := exec.CommandContext(ctx, path, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			_ = os.Remove(filePath)
			return Chunk{}, ctx.Err()
		}
		return Chunk{}, fmt.Errorf("%s failed for %s device %q: %w: %s", backendName, request.Source, request.Device.DisplayName(), err, string(output))
	}
	info, err := os.Stat(filePath)
	if err != nil {
		return Chunk{}, fmt.Errorf("recorded chunk missing: %w", err)
	}
	if info.Size() == 0 {
		return Chunk{}, fmt.Errorf("recorded chunk for %s is empty", request.Source)
	}
	return Chunk{
		Source:   request.Source,
		Device:   request.Device,
		FilePath: filePath,
		Started:  started,
		Ended:    time.Now(),
		Sequence: request.Sequence,
		Speaker:  request.Speaker,
	}, nil
}

func (r ExternalRecorder) ffmpeg() (string, error) {
	if r.FFmpegPath != "" {
		return r.FFmpegPath, nil
	}
	return exec.LookPath("ffmpeg")
}

func (r ExternalRecorder) usesWasapiLoopbackSidecar(request ChunkRequest) bool {
	return r.platform() == "windows" && request.Source == SourceOutput && request.Device.Backend == BackendWasapiLoopback
}

func (r ExternalRecorder) wasapiLoopbackCommand(request ChunkRequest, filePath string) (string, []string, error) {
	path, err := r.wasapiHelper()
	if err != nil {
		return "", nil, err
	}
	args := []string{
		"record",
		"--output-file", filePath,
		"--duration", fmt.Sprintf("%.3f", request.Duration.Seconds()),
		"--sample-rate", "16000",
		"--channels", "1",
	}
	if deviceID := strings.TrimSpace(request.Device.ID); deviceID != "" {
		args = append(args, "--device-id", deviceID)
	}
	if deviceName := strings.TrimSpace(request.Device.DisplayName()); deviceName != "" {
		args = append(args, "--device-name", deviceName)
	}
	return path, args, nil
}

func (r ExternalRecorder) wasapiHelper() (string, error) {
	for _, candidate := range []string{r.WasapiHelperPath, strings.TrimSpace(os.Getenv("TRANSCRIBE_WINDOWS_WASAPI_HELPER"))} {
		if path := normalizeExecutableCandidate(candidate); path != "" {
			if resolved, err := exec.LookPath(path); err == nil {
				return resolved, nil
			}
			if resolved, err := resolveExistingFile(path); err == nil {
				return resolved, nil
			}
		}
	}
	if exe, err := os.Executable(); err == nil {
		if resolved, err := resolveExistingFile(filepath.Join(filepath.Dir(exe), wasapiLoopbackHelperName)); err == nil {
			return resolved, nil
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		if resolved, err := resolveExistingFile(filepath.Join(cwd, wasapiLoopbackHelperName)); err == nil {
			return resolved, nil
		}
	}
	return "", fmt.Errorf("Windows WASAPI loopback capture requires %s beside transcribe.exe or TRANSCRIBE_WINDOWS_WASAPI_HELPER pointing to the bundled helper", wasapiLoopbackHelperName)
}

func inputArgs(device Device) []string {
	backend := device.Backend
	if backend == "" {
		switch runtime.GOOS {
		case "linux":
			backend = BackendPulse
		case "darwin":
			backend = BackendAVFoundation
		case "windows":
			backend = BackendDirectShow
		}
	}
	switch backend {
	case BackendPulse:
		return []string{"-f", "pulse", "-i", device.ID}
	case BackendAVFoundation:
		return []string{"-f", "avfoundation", "-i", device.ID}
	case BackendDirectShow:
		return []string{"-f", "dshow", "-i", "audio=" + device.ID}
	default:
		return []string{"-i", device.ID}
	}
}

func isUnresolvedDShowDevice(device Device) bool {
	if device.Backend != BackendDirectShow {
		return false
	}
	if normalize(device.ID) == "default" && normalize(device.Name) != "default" {
		return true
	}
	if normalize(device.ID) == "default" && strings.TrimSpace(device.Name) == "" {
		return true
	}
	return normalize(device.ID) == "virtual audio capturer" && strings.Contains(normalize(device.Name), "windows system audio capture")
}

func normalizeExecutableCandidate(value string) string {
	return strings.Trim(strings.TrimSpace(value), `"`)
}

func resolveExistingFile(path string) (string, error) {
	if info, err := os.Stat(path); err == nil && !info.IsDir() {
		if abs, absErr := filepath.Abs(path); absErr == nil {
			return abs, nil
		}
		return path, nil
	}
	return "", os.ErrNotExist
}
