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
	FFmpegPath string
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
	return nil
}

func (r ExternalRecorder) RecordChunk(ctx context.Context, request ChunkRequest) (Chunk, error) {
	path, err := r.ffmpeg()
	if err != nil {
		return Chunk{}, err
	}
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
	args := []string{"-nostdin", "-hide_banner", "-loglevel", "error", "-y"}
	args = append(args, inputArgs(request.Device)...)
	args = append(args,
		"-t", fmt.Sprintf("%.3f", request.Duration.Seconds()),
		"-acodec", "pcm_s16le",
		"-ar", "16000",
		"-ac", "1",
		filePath,
	)
	cmd := exec.CommandContext(ctx, path, args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		if ctx.Err() != nil {
			_ = os.Remove(filePath)
			return Chunk{}, ctx.Err()
		}
		return Chunk{}, fmt.Errorf("ffmpeg failed for %s device %q: %w: %s", request.Source, request.Device.DisplayName(), err, string(output))
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
