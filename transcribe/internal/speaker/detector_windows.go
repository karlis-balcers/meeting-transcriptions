//go:build windows

package speaker

import (
	"context"
	"os/exec"
	"strings"
	"sync"
	"time"
)

func New(enabled bool) Detector {
	if !enabled {
		return &noopDetector{current: "Person_?"}
	}
	return &windowsDetector{current: "Person_?", interval: 2 * time.Second}
}

type windowsDetector struct {
	mu       sync.Mutex
	current  string
	interval time.Duration
}

func (d *windowsDetector) Start(ctx context.Context) <-chan Event {
	events := make(chan Event, 4)
	go func() {
		defer close(events)
		d.send(ctx, events, Event{Warning: "MS Teams speaker recognition is best-effort via window-title polling; using Person_? until a speaker is detected"})
		d.poll(ctx, events)
		ticker := time.NewTicker(d.interval)
		defer ticker.Stop()
		warnedCommandFailure := false
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				if err := d.poll(ctx, events); err != nil && !warnedCommandFailure {
					warnedCommandFailure = true
					d.send(ctx, events, Event{Warning: "MS Teams window-title polling failed; using Person_? until Teams titles can be read: " + err.Error()})
				}
			}
		}
	}()
	return events
}

func (d *windowsDetector) Current() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	if d.current == "" {
		return "Person_?"
	}
	return d.current
}

func (d *windowsDetector) Close() error { return nil }

func (d *windowsDetector) poll(ctx context.Context, events chan<- Event) error {
	titles, err := teamsWindowTitles(ctx)
	if err != nil {
		return err
	}
	speaker := DetectSpeakerFromTitles(titles)
	if speaker == "" {
		return nil
	}
	d.mu.Lock()
	changed := d.current != speaker
	d.current = speaker
	d.mu.Unlock()
	if changed {
		d.send(ctx, events, Event{Speaker: speaker})
	}
	return nil
}

func (d *windowsDetector) send(ctx context.Context, events chan<- Event, event Event) {
	select {
	case <-ctx.Done():
	case events <- event:
	}
}

func teamsWindowTitles(ctx context.Context) ([]string, error) {
	queryCtx, cancel := context.WithTimeout(ctx, 1500*time.Millisecond)
	defer cancel()
	command := strings.Join([]string{
		"Get-Process |",
		"Where-Object { $_.MainWindowTitle -and ($_.ProcessName -match '^(Teams|MSTeams|ms-teams)$' -or $_.MainWindowTitle -match 'Teams|Microsoft Teams') } |",
		"ForEach-Object { $_.MainWindowTitle }",
	}, " ")
	output, err := exec.CommandContext(queryCtx, "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command).Output()
	if err != nil {
		return nil, err
	}
	var titles []string
	for _, line := range strings.Split(string(output), "\n") {
		line = strings.TrimSpace(line)
		if line != "" {
			titles = append(titles, line)
		}
	}
	return titles, nil
}
