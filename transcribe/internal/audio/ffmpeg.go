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

type ExternalRecorder struct {
	FFmpegPath       string
	PythonPath       string
	HelperScriptPath string
	Platform         string
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
	if r.platform() == "windows" && selected.Output.Source == SourceOutput {
		if _, _, err := r.commandForRequest(ChunkRequest{
			Device:   selected.Output,
			Source:   SourceOutput,
			TempDir:  os.TempDir(),
			Duration: time.Second,
			Sequence: 1,
		}, filepath.Join(os.TempDir(), "transcribe-output-validation.wav")); err != nil {
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
	cmd := exec.CommandContext(ctx, path, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			_ = os.Remove(filePath)
			return Chunk{}, ctx.Err()
		}
		backendName := "ffmpeg"
		if r.platform() == "windows" && request.Source == SourceOutput {
			backendName = "Windows output helper"
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

func inputArgs(device Device) []string {
	backend := device.Backend
	if backend == "" {
		switch runtime.GOOS {
		case "linux":
			backend = "pulse"
		case "darwin":
			backend = "avfoundation"
		case "windows":
			backend = "dshow"
		}
	}
	switch backend {
	case "pulse":
		return []string{"-f", "pulse", "-i", device.ID}
	case "avfoundation":
		return []string{"-f", "avfoundation", "-i", device.ID}
	case "dshow":
		return []string{"-f", "dshow", "-i", "audio=" + device.ID}
	default:
		return []string{"-i", device.ID}
	}
}

func isUnresolvedDShowDevice(device Device) bool {
	if device.Backend != "dshow" {
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
