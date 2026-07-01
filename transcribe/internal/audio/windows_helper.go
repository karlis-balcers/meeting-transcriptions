package audio

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

func (r ExternalRecorder) platform() string {
	if value := strings.TrimSpace(r.Platform); value != "" {
		return strings.ToLower(value)
	}
	return runtime.GOOS
}

func (r ExternalRecorder) commandForRequest(request ChunkRequest, filePath string) (string, []string, error) {
	if r.platform() == "windows" && request.Source == SourceOutput {
		return r.windowsOutputHelperCommand(request, filePath)
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

func (r ExternalRecorder) windowsOutputHelperCommand(request ChunkRequest, filePath string) (string, []string, error) {
	pythonPath, err := r.windowsPythonPath()
	if err != nil {
		return "", nil, err
	}
	helperPath, err := r.windowsHelperScriptPath()
	if err != nil {
		return "", nil, err
	}
	args := []string{helperPath, "record-output", "--output-file", filePath, "--duration", fmt.Sprintf("%.3f", request.Duration.Seconds())}
	if deviceID := strings.TrimSpace(request.Device.ID); deviceID != "" {
		args = append(args, "--device-id", deviceID)
	}
	if deviceName := strings.TrimSpace(request.Device.DisplayName()); deviceName != "" {
		args = append(args, "--device-name", deviceName)
	}
	return pythonPath, args, nil
}

func (r ExternalRecorder) windowsPythonPath() (string, error) {
	for _, candidate := range []string{r.PythonPath, strings.TrimSpace(os.Getenv("TRANSCRIBE_WINDOWS_PYTHON"))} {
		if path := normalizeExecutableCandidate(candidate); path != "" {
			if resolved, err := exec.LookPath(path); err == nil {
				return resolved, nil
			}
			if resolved, err := resolveExistingFile(path); err == nil {
				return resolved, nil
			}
		}
	}
	for _, candidate := range windowsPythonCandidates() {
		if candidate == "" {
			continue
		}
		if resolved, err := exec.LookPath(candidate); err == nil {
			return resolved, nil
		}
		if resolved, err := resolveExistingFile(candidate); err == nil {
			return resolved, nil
		}
	}
	return "", errors.New("Windows output capture requires a Python interpreter; set TRANSCRIBE_WINDOWS_PYTHON or keep .venv\\Scripts\\python.exe beside transcribe.exe")
}

func (r ExternalRecorder) windowsHelperScriptPath() (string, error) {
	for _, candidate := range []string{r.HelperScriptPath, strings.TrimSpace(os.Getenv("TRANSCRIBE_WINDOWS_AUDIO_HELPER"))} {
		if path := normalizeExecutableCandidate(candidate); path != "" {
			if resolved, err := resolveExistingFile(path); err == nil {
				return resolved, nil
			}
		}
	}
	if exe, err := os.Executable(); err == nil {
		if resolved, err := resolveExistingFile(filepath.Join(filepath.Dir(exe), "audio_capture.py")); err == nil {
			return resolved, nil
		}
	}
	if cwd, err := os.Getwd(); err == nil {
		if resolved, err := resolveExistingFile(filepath.Join(cwd, "audio_capture.py")); err == nil {
			return resolved, nil
		}
	}
	return "", errors.New("Windows output capture requires audio_capture.py; copy it beside transcribe.exe or set TRANSCRIBE_WINDOWS_AUDIO_HELPER")
}

func windowsPythonCandidates() []string {
	candidates := []string{}
	if cwd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(cwd, ".venv", "Scripts", "python.exe"))
	}
	if exe, err := os.Executable(); err == nil {
		candidates = append(candidates, filepath.Join(filepath.Dir(exe), ".venv", "Scripts", "python.exe"))
	}
	candidates = append(candidates, "python")
	return candidates
}

func normalizeExecutableCandidate(value string) string {
	trimmed := strings.Trim(strings.TrimSpace(value), `"`)
	if trimmed == "" {
		return ""
	}
	return trimmed
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
