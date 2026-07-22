package speaker

import "context"

type Event struct {
	Speaker string
	Warning string
}

type Detector interface {
	Start(ctx context.Context) <-chan Event
	Current() string
	Close() error
}

type noopDetector struct {
	warning string
	current string
}

func (d *noopDetector) Start(ctx context.Context) <-chan Event {
	events := make(chan Event, 1)
	if d.warning != "" {
		events <- Event{Warning: d.warning}
	}
	close(events)
	return events
}

func (d *noopDetector) Current() string {
	if d.current == "" {
		return "Person_?"
	}
	return d.current
}

func (d *noopDetector) Close() error { return nil }
