package logging

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

var (
	mu      sync.RWMutex
	logger  *log.Logger
	logFile *os.File
	logPath string
)

// EnableCurrentDir opens transcribe.log in the current working directory and
// routes logging output there until Close is called.
func EnableCurrentDir() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("determine current directory for log file: %w", err)
	}
	return Enable(filepath.Join(cwd, "transcribe.log"))
}

// Enable opens the given log file path and enables package logging helpers.
func Enable(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return "", fmt.Errorf("log file path cannot be empty")
	}
	mu.Lock()
	defer mu.Unlock()
	if logFile != nil {
		return logPath, nil
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return "", fmt.Errorf("open log file %s: %w", path, err)
	}
	logFile = f
	logPath = path
	logger = log.New(f, "", log.LstdFlags|log.Lmicroseconds)
	return logPath, nil
}

// Close disables package logging helpers and closes the file handle.
func Close() error {
	mu.Lock()
	defer mu.Unlock()
	if logFile == nil {
		logger = nil
		logPath = ""
		return nil
	}
	err := logFile.Close()
	logFile = nil
	logger = nil
	logPath = ""
	return err
}

// Printf writes formatted output to the active log file when logging is enabled.
func Printf(format string, args ...any) {
	mu.RLock()
	l := logger
	mu.RUnlock()
	if l == nil {
		return
	}
	l.Printf(format, args...)
}
