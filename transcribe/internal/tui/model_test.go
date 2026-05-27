package tui

import (
	"context"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/app"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
)

func TestShortcutsSettingsAndBottomChrome(t *testing.T) {
	mic := audio.Device{ID: "mic-1", Name: "QA Mic", Source: audio.SourceMic, Default: true}
	output := audio.Device{ID: "out-1", Name: "QA Output", Source: audio.SourceOutput, Default: true}
	controller := &fakeController{mic: mic, output: output, events: make(chan app.Event)}
	model := New(controller, []audio.Device{mic, output}, nil)

	model = pressAndRun(t, model, runeKey('p'))
	if controller.pauseCalls != 1 || model.status != "Paused" {
		t.Fatalf("P should pause, calls=%d status=%q", controller.pauseCalls, model.status)
	}

	model = pressAndRun(t, model, runeKey('r'))
	if controller.resumeCalls != 1 || model.status != "Recording" {
		t.Fatalf("R should resume, calls=%d status=%q", controller.resumeCalls, model.status)
	}

	model = pressAndRun(t, model, runeKey('m'))
	if controller.muteCalls != 1 || model.status != "Microphone muted" {
		t.Fatalf("M should mute in main mode, calls=%d status=%q", controller.muteCalls, model.status)
	}

	model = pressAndRun(t, model, runeKey('u'))
	if controller.unmuteCalls != 1 || model.status != "Microphone unmuted" {
		t.Fatalf("U should unmute, calls=%d status=%q", controller.unmuteCalls, model.status)
	}

	updated, cmd := model.Update(runeKey('s'))
	model = updated.(Model)
	if cmd != nil || model.mode != ModeSettings {
		t.Fatalf("S should open settings without a command, mode=%v cmd=%v", model.mode, cmd)
	}

	updated, cmd = model.Update(runeKey('m'))
	model = updated.(Model)
	if cmd != nil || model.mode != ModeMicSelect || controller.muteCalls != 1 {
		t.Fatalf("M in settings should open mic selection, mode=%v muteCalls=%d", model.mode, controller.muteCalls)
	}
	model = pressAndRun(t, model, tea.KeyMsg{Type: tea.KeyEnter})
	if controller.mic.ID != "mic-1" {
		t.Fatalf("Enter in mic selection should persist selected mic, got %+v", controller.mic)
	}

	updated, cmd = model.Update(tea.KeyMsg{Type: tea.KeyEsc})
	model = updated.(Model)
	if cmd != nil || model.mode != ModeSettings {
		t.Fatalf("Esc from mic selection should return to settings, mode=%v", model.mode)
	}
	updated, cmd = model.Update(runeKey('o'))
	model = updated.(Model)
	if cmd != nil || model.mode != ModeOutputSelect {
		t.Fatalf("O in settings should open output selection, mode=%v", model.mode)
	}
	model = pressAndRun(t, model, tea.KeyMsg{Type: tea.KeyEnter})
	if controller.output.ID != "out-1" {
		t.Fatalf("Enter in output selection should persist selected output, got %+v", controller.output)
	}

	updated, _ = model.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	model = updated.(Model)
	view := model.View()
	for _, want := range []string{"Mic     ", "Output  ", "P Pause  R Resume  M Mute mic  U Unmute  S Settings  Q Quit"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view missing %q:\n%s", want, view)
		}
	}
}

func pressAndRun(t *testing.T, model Model, key tea.KeyMsg) Model {
	t.Helper()
	updated, cmd := model.Update(key)
	model = updated.(Model)
	if cmd == nil {
		return model
	}
	msg := cmd()
	updated, _ = model.Update(msg)
	return updated.(Model)
}

func runeKey(r rune) tea.KeyMsg {
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}}
}

type fakeController struct {
	mic         audio.Device
	output      audio.Device
	events      chan app.Event
	pauseCalls  int
	resumeCalls int
	muteCalls   int
	unmuteCalls int
	stopCalls   int
	startedAt   time.Time
}

func (c *fakeController) Pause() error {
	c.pauseCalls++
	return nil
}

func (c *fakeController) Resume() error {
	c.resumeCalls++
	return nil
}

func (c *fakeController) MuteMic() error {
	c.muteCalls++
	return nil
}

func (c *fakeController) UnmuteMic() error {
	c.unmuteCalls++
	return nil
}

func (c *fakeController) Stop(context.Context) error {
	c.stopCalls++
	return nil
}

func (c *fakeController) SetMicDevice(device audio.Device) error {
	c.mic = device
	return nil
}

func (c *fakeController) SetOutputDevice(device audio.Device) error {
	c.output = device
	return nil
}

func (c *fakeController) Snapshot() app.Snapshot {
	if c.startedAt.IsZero() {
		c.startedAt = time.Now()
	}
	return app.Snapshot{
		Status:         "Recording",
		StartedAt:      c.startedAt,
		Mic:            c.mic,
		Output:         c.output,
		TranscriptPath: "/tmp/transcription.md",
	}
}

func (c *fakeController) Events() <-chan app.Event { return c.events }
