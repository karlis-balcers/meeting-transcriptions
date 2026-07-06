package cli

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/logging"
)

func TestSetupLoggingWritesTranscribeLogInCurrentDirectory(t *testing.T) {
	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tempDir := t.TempDir()
	if err := os.Chdir(tempDir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(oldWD)
	})

	cleanup, err := setupLogging(1)
	if err != nil {
		t.Fatal(err)
	}
	if cleanup != nil {
		defer cleanup()
	}

	logging.Printf("startup complete")
	if cleanup != nil {
		cleanup()
	}

	data, err := os.ReadFile(filepath.Join(tempDir, "transcribe.log"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "startup complete") {
		t.Fatalf("log file does not contain startup message: %s", string(data))
	}
}