package logging

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEnableCurrentDirWritesTranscribeLogInCwd(t *testing.T) {
	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tempDir := t.TempDir()
	if err := os.Chdir(tempDir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = Close()
		_ = os.Chdir(oldWD)
	})

	path, err := EnableCurrentDir()
	if err != nil {
		t.Fatal(err)
	}
	if want := filepath.Join(tempDir, "transcribe.log"); path != want {
		t.Fatalf("unexpected log path: got %q want %q", path, want)
	}
	Printf("hello %s", "world")
	if err := Close(); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(filepath.Join(tempDir, "transcribe.log"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(content), "hello world") {
		t.Fatalf("log file did not contain expected message: %q", string(content))
	}
}
