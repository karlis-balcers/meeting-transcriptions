package transcript

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

type Event struct {
	Source  string
	Speaker string
	Text    string
	Start   time.Time
	End     time.Time
}

type Metadata struct {
	CreatedAt    time.Time
	MicDevice    string
	OutputDevice string
	Language     string
	Model        string
}

type Store struct {
	mu       sync.Mutex
	path     string
	metadata Metadata
	events   []Event
}

func NewFileStore(outputDir string, clock func() time.Time, metadata Metadata) (*Store, error) {
	if strings.TrimSpace(outputDir) == "" {
		return nil, fmt.Errorf("output directory cannot be empty")
	}
	if err := os.MkdirAll(outputDir, 0o755); err != nil {
		return nil, fmt.Errorf("create transcript output directory: %w", err)
	}
	now := clock()
	if now.IsZero() {
		now = time.Now()
	}
	metadata.CreatedAt = now
	path := filepath.Join(outputDir, fmt.Sprintf("transcription-%s.md", now.Format("20060102_150405")))
	store := &Store{path: path, metadata: metadata}
	if err := store.rewriteLocked(); err != nil {
		return nil, err
	}
	return store, nil
}

func (s *Store) Path() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.path
}

func (s *Store) Add(event Event) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	event.Text = normalizeText(event.Text)
	event.Speaker = strings.TrimSpace(event.Speaker)
	if event.Speaker == "" {
		event.Speaker = "Person_?"
	}
	if strings.TrimSpace(event.Text) == "" {
		return nil
	}
	s.events = append(s.events, event)
	s.sortLocked()
	return s.rewriteLocked()
}

func (s *Store) Snapshot() []Event {
	s.mu.Lock()
	defer s.mu.Unlock()
	copyEvents := append([]Event(nil), s.events...)
	return copyEvents
}

func (s *Store) sortLocked() {
	sort.SliceStable(s.events, func(i, j int) bool {
		return s.events[i].Start.Before(s.events[j].Start)
	})
}

func (s *Store) rewriteLocked() error {
	var buf bytes.Buffer
	buf.WriteString("# Transcription Log\n\n")
	buf.WriteString(fmt.Sprintf("**Created:** %s\n", s.metadata.CreatedAt.Format("2006-01-02 15:04:05")))
	if s.metadata.MicDevice != "" || s.metadata.OutputDevice != "" || s.metadata.Language != "" || s.metadata.Model != "" {
		buf.WriteString("\n")
		if s.metadata.MicDevice != "" {
			buf.WriteString(fmt.Sprintf("**Microphone:** %s\n", s.metadata.MicDevice))
		}
		if s.metadata.OutputDevice != "" {
			buf.WriteString(fmt.Sprintf("**Output capture:** %s\n", s.metadata.OutputDevice))
		}
		if s.metadata.Language != "" {
			buf.WriteString(fmt.Sprintf("**Language:** %s\n", s.metadata.Language))
		}
		if s.metadata.Model != "" {
			buf.WriteString(fmt.Sprintf("**Model:** %s\n", s.metadata.Model))
		}
	}
	buf.WriteString("\n")
	buf.WriteString(RenderPlain(s.events))
	tmp := s.path + ".tmp"
	if err := os.WriteFile(tmp, buf.Bytes(), 0o644); err != nil {
		return fmt.Errorf("write transcript file: %w", err)
	}
	if err := os.Rename(tmp, s.path); err != nil {
		return fmt.Errorf("publish transcript file: %w", err)
	}
	return nil
}

func RenderPlain(events []Event) string {
	var buf strings.Builder
	for _, event := range events {
		text := strings.TrimSpace(normalizeText(event.Text))
		if text == "" {
			continue
		}
		speaker := strings.TrimSpace(event.Speaker)
		if speaker == "" {
			speaker = "Person_?"
		}
		buf.WriteString(speaker)
		buf.WriteString(": ")
		buf.WriteString(text)
		buf.WriteString("\n\n")
	}
	return buf.String()
}

func normalizeText(text string) string {
	text = strings.ReplaceAll(text, "\\n", "\n")
	return strings.TrimSpace(text)
}
