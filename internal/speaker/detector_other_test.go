//go:build !windows

package speaker

import (
	"context"
	"strings"
	"testing"
)

func TestNonWindowsDetectorWarnsAndFallsBack(t *testing.T) {
	detector := New(true)
	if got := detector.Current(); got != "Person_?" {
		t.Fatalf("fallback current speaker = %q, want Person_?", got)
	}
	events := detector.Start(context.Background())
	event, ok := <-events
	if !ok {
		t.Fatal("expected one warning event")
	}
	if !strings.Contains(event.Warning, "only available on Windows") {
		t.Fatalf("unexpected warning: %q", event.Warning)
	}
	if _, ok := <-events; ok {
		t.Fatal("non-Windows detector should close after warning")
	}
}
