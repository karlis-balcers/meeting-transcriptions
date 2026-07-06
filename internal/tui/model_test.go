package tui

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/muesli/termenv"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/app"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
)

func init() {
	lipgloss.SetColorProfile(termenv.ANSI256)
}

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
	settingsView := model.View()
	if !strings.Contains(settingsView, "╭") || !strings.Contains(settingsView, "╰") {
		t.Fatalf("settings should render in a framed block, got:\n%s", settingsView)
	}
	if strings.LastIndex(settingsView, "M choose microphone") < strings.LastIndex(settingsView, "Transcript will appear here as chunks finish transcribing.") {
		t.Fatalf("settings menu should appear below the transcript area, got:\n%s", settingsView)
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
	if model.mode != ModeSettings {
		t.Fatalf("Enter in mic selection should return to settings, mode=%v", model.mode)
	}
	view := model.View()
	for _, want := range []string{"Selected mic: QA Mic", "Selected output: QA Output"} {
		if !strings.Contains(view, want) {
			t.Fatalf("settings view should show updated device summary %q, got:\n%s", want, view)
		}
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
	if model.mode != ModeSettings {
		t.Fatalf("Enter in output selection should return to settings, mode=%v", model.mode)
	}

	updated, _ = model.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	model = updated.(Model)
	view = model.View()
	for _, want := range []string{"Mic     ", "Output  ", "P Pause", "R Record", "M Mute mic", "U Unmute", "S Settings", "Q Quit"} {
		if !strings.Contains(view, want) {
			t.Fatalf("view missing %q:\n%s", want, view)
		}
	}
	if strings.Contains(view, "R Resume") {
		t.Fatalf("footer should say R Record, got:\n%s", view)
	}
}

func TestFooterViewHighlightsCurrentSessionState(t *testing.T) {
	mic := audio.Device{ID: "mic-1", Name: "QA Mic", Source: audio.SourceMic, Default: true}
	output := audio.Device{ID: "out-1", Name: "QA Output", Source: audio.SourceOutput, Default: true}
	controller := &fakeController{mic: mic, output: output, events: make(chan app.Event)}
	model := New(controller, []audio.Device{mic, output}, nil)

	view := model.View()
	assertFooterShortcut(t, view, "R Record", true)
	assertFooterShortcut(t, view, "P Pause", false)
	assertFooterShortcut(t, view, "U Unmute", true)
	assertFooterShortcut(t, view, "M Mute mic", false)

	model = pressAndRun(t, model, runeKey('p'))
	view = model.View()
	assertFooterShortcut(t, view, "P Pause", true)
	assertFooterShortcut(t, view, "R Record", false)

	model = pressAndRun(t, model, runeKey('m'))
	view = model.View()
	assertFooterShortcut(t, view, "M Mute mic", true)
	assertFooterShortcut(t, view, "U Unmute", false)
}

func TestRenderFooterShortcutUsesGreenStyleWhenActive(t *testing.T) {
	if got, want := renderFooterShortcut(footerShortcut{label: "R Record", active: true}), statusStyle.Render("R Record"); got != want {
		t.Fatalf("active shortcut should use green status style, got %q want %q", got, want)
	}
	if got, want := renderFooterShortcut(footerShortcut{label: "R Record", active: false}), shortcutStyle.Render("R Record"); got != want {
		t.Fatalf("inactive shortcut should use neutral style, got %q want %q", got, want)
	}
}

func TestDevicePickerShowsAndSelectsMultipleDevices(t *testing.T) {
	mic1 := audio.Device{ID: "mic-1", Name: "Built-in Mic", Source: audio.SourceMic, Default: true}
	mic2 := audio.Device{ID: "mic-2", Name: "USB Mic", Source: audio.SourceMic}
	out1 := audio.Device{ID: "out-1", Name: "Speakers", Source: audio.SourceOutput, Default: true}
	out2 := audio.Device{ID: "out-2", Name: "Headphones", Source: audio.SourceOutput}
	controller := &fakeController{mic: mic1, output: out1, events: make(chan app.Event)}
	model := New(controller, []audio.Device{mic1, mic2, out1, out2}, nil)

	updated, _ := model.Update(runeKey('s'))
	model = updated.(Model)
	updated, _ = model.Update(runeKey('m'))
	model = updated.(Model)
	view := model.View()
	for _, want := range []string{"Built-in Mic", "USB Mic"} {
		if !strings.Contains(view, want) {
			t.Fatalf("mic picker should list %q, got:\n%s", want, view)
		}
	}
	model = pressAndRun(t, model, runeKey('j'))
	model = pressAndRun(t, model, tea.KeyMsg{Type: tea.KeyEnter})
	if controller.mic.ID != mic2.ID {
		t.Fatalf("mic picker should allow selecting the second microphone, got %+v", controller.mic)
	}
	if model.mode != ModeSettings {
		t.Fatalf("mic picker should return to settings after selection, mode=%v", model.mode)
	}

	updated, _ = model.Update(runeKey('o'))
	model = updated.(Model)
	view = model.View()
	for _, want := range []string{"Speakers", "Headphones"} {
		if !strings.Contains(view, want) {
			t.Fatalf("output picker should list %q, got:\n%s", want, view)
		}
	}
	model = pressAndRun(t, model, runeKey('j'))
	model = pressAndRun(t, model, tea.KeyMsg{Type: tea.KeyEnter})
	if controller.output.ID != out2.ID {
		t.Fatalf("output picker should allow selecting the second output device, got %+v", controller.output)
	}
	if model.mode != ModeSettings {
		t.Fatalf("output picker should return to settings after selection, mode=%v", model.mode)
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
	paused      bool
	mutedMic    bool
	pauseCalls  int
	resumeCalls int
	muteCalls   int
	unmuteCalls int
	stopCalls   int
	startedAt   time.Time
}

func (c *fakeController) Pause() error {
	c.pauseCalls++
	c.paused = true
	return nil
}

func (c *fakeController) Resume() error {
	c.resumeCalls++
	c.paused = false
	return nil
}

func (c *fakeController) MuteMic() error {
	c.muteCalls++
	c.mutedMic = true
	return nil
}

func (c *fakeController) UnmuteMic() error {
	c.unmuteCalls++
	c.mutedMic = false
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
	status := "Recording"
	if c.paused {
		status = "Paused"
	}
	return app.Snapshot{
		Status:         status,
		Paused:         c.paused,
		MutedMic:       c.mutedMic,
		StartedAt:      c.startedAt,
		Mic:            c.mic,
		Output:         c.output,
		TranscriptPath: "/tmp/transcription.md",
	}
}

func assertFooterShortcut(t *testing.T, view, label string, active bool) {
	t.Helper()
	want := shortcutStyle.Render(label)
	if active {
		want = statusStyle.Render(label)
	}
	if !strings.Contains(view, want) {
		t.Fatalf("footer missing %q (active=%t):\n%s", label, active, view)
	}
}

func (c *fakeController) Events() <-chan app.Event { return c.events }
