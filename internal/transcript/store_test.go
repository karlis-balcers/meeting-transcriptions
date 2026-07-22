package transcript

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStoreOrdersEventsAndRendersPlainTranscript(t *testing.T) {
	dir := t.TempDir()
	created := time.Date(2026, 5, 27, 15, 0, 0, 0, time.UTC)
	store, err := NewFileStore(dir, func() time.Time { return created }, Metadata{Language: "en", Model: "model"})
	if err != nil {
		t.Fatal(err)
	}
	later := created.Add(2 * time.Second)
	earlier := created.Add(time.Second)
	if err := store.Add(Event{Speaker: "B", Text: "second", Start: later}); err != nil {
		t.Fatal(err)
	}
	if err := store.Add(Event{Speaker: "A", Text: "first\\nline", Start: earlier}); err != nil {
		t.Fatal(err)
	}
	snapshot := store.Snapshot()
	if snapshot[0].Speaker != "A" || snapshot[1].Speaker != "B" {
		t.Fatalf("events not sorted: %+v", snapshot)
	}
	plain := RenderPlain(snapshot)
	if plain != "A: first\nline\n\nB: second\n\n" {
		t.Fatalf("unexpected plain transcript: %q", plain)
	}
	content, err := os.ReadFile(filepath.Join(dir, "transcription-20260527_150000.md"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(content)
	if !strings.Contains(text, "# Transcription Log") || strings.Index(text, "A: first") > strings.Index(text, "B: second") {
		t.Fatalf("unexpected markdown content:\n%s", text)
	}
}

func TestRenderPlainSilentContractHasNoMetadata(t *testing.T) {
	created := time.Date(2026, 5, 27, 15, 0, 0, 0, time.UTC)
	out := RenderPlain([]Event{{Speaker: "You", Text: "hello", Start: created}})
	if out != "You: hello\n\n" {
		t.Fatalf("stdout transcript should contain only transcript lines, got %q", out)
	}
	if strings.Contains(out, "Created") || strings.Contains(out, "Transcription Log") {
		t.Fatalf("silent output leaked metadata: %q", out)
	}
}
