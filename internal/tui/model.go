package tui

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/progress"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/app"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/transcript"
)

type Controller interface {
	Pause() error
	Resume() error
	MuteMic() error
	UnmuteMic() error
	Stop(context.Context) error
	SetMicDevice(audio.Device) error
	SetOutputDevice(audio.Device) error
	Snapshot() app.Snapshot
	Events() <-chan app.Event
}

type Mode int

const (
	ModeMain Mode = iota
	ModeSettings
	ModeMicSelect
	ModeOutputSelect
)

type Model struct {
	controller Controller
	devices    []audio.Device
	warnings   []string
	viewport   viewport.Model
	micMeter   progress.Model
	outMeter   progress.Model
	mode       Mode
	selected   int
	status     string
	errorLine  string
	micLevel   float64
	outLevel   float64
	width      int
	height     int
	lastTick   time.Time
	quitting   bool
}

type eventMsg app.Event
type tickMsg time.Time
type stoppedMsg struct{ err error }
type deviceAppliedMsg struct{ err error }

type footerShortcut struct {
	label  string
	active bool
}

var (
	titleStyle         = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	errorStyle         = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	warnStyle          = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	statusStyle        = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	shortcutStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	settingsFrameStyle = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("63")).
				Padding(0, 1)
	settingsTitleStyle  = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("63"))
	settingsHintStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	settingsCursorStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("63")).Bold(true)
)

func New(controller Controller, devices []audio.Device, warnings []string) Model {
	vp := viewport.New(80, 12)
	return Model{
		controller: controller,
		devices:    devices,
		warnings:   warnings,
		viewport:   vp,
		micMeter:   progress.New(progress.WithDefaultGradient()),
		outMeter:   progress.New(progress.WithDefaultGradient()),
		status:     "Recording",
		lastTick:   time.Now(),
	}
}

func Run(ctx context.Context, controller Controller, devices []audio.Device, warnings []string) error {
	program := tea.NewProgram(New(controller, devices, warnings), tea.WithContext(ctx), tea.WithAltScreen())
	_, err := program.Run()
	return err
}

func (m Model) Init() tea.Cmd {
	return tea.Batch(m.waitEvent(), tick())
}

func (m Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width = msg.Width
		m.height = msg.Height
		m.viewport.Width = max(20, msg.Width-2)
		m.micMeter.Width = max(10, msg.Width/2-12)
		m.outMeter.Width = max(10, msg.Width/2-12)
		return m.applyLayout(), nil
	case tickMsg:
		m.lastTick = time.Time(msg)
		m.refreshTranscript()
		return m, tick()
	case eventMsg:
		event := app.Event(msg)
		switch event.Kind {
		case app.EventStatus:
			m.status = event.Message
		case app.EventWarning:
			m.warnings = append(m.warnings, event.Message)
		case app.EventError:
			m.errorLine = event.Message
		case app.EventTranscript:
			m.refreshTranscript()
		case app.EventLoudness:
			if event.Source == audio.SourceMic {
				m.micLevel = clampLevel(event.MicLevel)
			}
			if event.Source == audio.SourceOutput {
				m.outLevel = clampLevel(event.OutputLevel)
			}
		}
		return m, m.waitEvent()
	case stoppedMsg:
		if msg.err != nil {
			m.errorLine = msg.err.Error()
			return m, nil
		}
		return m, tea.Quit
	case deviceAppliedMsg:
		if msg.err != nil {
			m.errorLine = msg.err.Error()
			return m, m.waitEvent()
		}
		m.errorLine = ""
		m.mode = ModeSettings
		m.selected = 0
		return m.applyLayout(), m.waitEvent()
	case tea.KeyMsg:
		return m.handleKey(msg)
	}
	var cmd tea.Cmd
	m.viewport, cmd = m.viewport.Update(msg)
	return m, cmd
}

func (m Model) View() string {
	snap := m.controller.Snapshot()
	if m.quitting {
		return "Stopping recorder and restoring terminal...\n"
	}
	var b strings.Builder
	b.WriteString(titleStyle.Render("transcribe"))
	b.WriteString("  ")
	b.WriteString(statusStyle.Render(snap.Status))
	if !snap.StartedAt.IsZero() {
		b.WriteString(fmt.Sprintf("  %s", time.Since(snap.StartedAt).Truncate(time.Second)))
	}
	b.WriteString("\n")
	b.WriteString(fmt.Sprintf("Selected mic: %s\n", snap.Mic.DisplayName()))
	b.WriteString(fmt.Sprintf("Selected output: %s\n", snap.Output.DisplayName()))
	b.WriteString(fmt.Sprintf("Transcript: %s\n", snap.TranscriptPath))
	if len(m.warnings) > 0 {
		b.WriteString(warnStyle.Render("Warning: " + m.warnings[len(m.warnings)-1]))
		b.WriteString("\n")
	}
	if m.errorLine != "" || snap.Error != "" {
		line := m.errorLine
		if line == "" {
			line = snap.Error
		}
		b.WriteString(errorStyle.Render("Error: " + line))
		b.WriteString("\n")
	}
	b.WriteString(fmt.Sprintf("Mic     %s\n", m.micMeter.ViewAs(m.micLevel)))
	b.WriteString(fmt.Sprintf("Output  %s\n\n", m.outMeter.ViewAs(m.outLevel)))
	b.WriteString(m.viewport.View())
	if m.mode == ModeSettings || m.mode == ModeMicSelect || m.mode == ModeOutputSelect {
		b.WriteString("\n")
		b.WriteString(m.settingsView())
	}
	b.WriteString("\n")
	b.WriteString(footerView(snap))
	return b.String()
}

func (m Model) handleKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	key := strings.ToLower(msg.String())
	if m.mode == ModeSettings {
		switch key {
		case "m":
			m.mode = ModeMicSelect
			m.selected = 0
			return m.applyLayout(), nil
		case "o":
			m.mode = ModeOutputSelect
			m.selected = 0
			return m.applyLayout(), nil
		case "esc", "s":
			m.mode = ModeMain
			return m.applyLayout(), nil
		}
	}
	if m.mode == ModeMicSelect || m.mode == ModeOutputSelect {
		return m.handleDeviceKey(key)
	}
	switch key {
	case "p":
		return m, func() tea.Msg { _ = m.controller.Pause(); return eventMsg{Kind: app.EventStatus, Message: "Paused"} }
	case "r":
		return m, func() tea.Msg {
			_ = m.controller.Resume()
			return eventMsg{Kind: app.EventStatus, Message: "Recording"}
		}
	case "m":
		return m, func() tea.Msg {
			_ = m.controller.MuteMic()
			return eventMsg{Kind: app.EventStatus, Message: "Microphone muted"}
		}
	case "u":
		return m, func() tea.Msg {
			_ = m.controller.UnmuteMic()
			return eventMsg{Kind: app.EventStatus, Message: "Microphone unmuted"}
		}
	case "s":
		m.mode = ModeSettings
		return m.applyLayout(), nil
	case "q", "ctrl+c":
		m.quitting = true
		return m, func() tea.Msg {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			return stoppedMsg{err: m.controller.Stop(ctx)}
		}
	}
	return m, nil
}

func (m Model) handleDeviceKey(key string) (tea.Model, tea.Cmd) {
	devices := m.devicesForMode()
	if len(devices) == 0 {
		m.errorLine = "No devices available for this source"
		m.mode = ModeSettings
		return m.applyLayout(), nil
	}
	switch key {
	case "up", "k":
		if m.selected > 0 {
			m.selected--
		}
	case "down", "j":
		if m.selected < len(devices)-1 {
			m.selected++
		}
	case "enter":
		device := devices[m.selected]
		return m, func() tea.Msg {
			var err error
			if m.mode == ModeMicSelect {
				err = m.controller.SetMicDevice(device)
			} else {
				err = m.controller.SetOutputDevice(device)
			}
			return deviceAppliedMsg{err: err}
		}
	case "esc", "s":
		m.mode = ModeSettings
	}
	return m.applyLayout(), nil
}

func (m Model) settingsView() string {
	if m.mode == ModeSettings {
		content := []string{
			settingsTitleStyle.Render("▌ Settings"),
			"  M choose microphone",
			"  O choose output device",
			"  Esc close",
		}
		return settingsFrameStyle.Width(max(24, m.width-2)).Render(strings.Join(content, "\n"))
	}
	devices := m.devicesForMode()
	title := "Microphones"
	if m.mode == ModeOutputSelect {
		title = "Output capture devices"
	}
	var lines []string
	lines = append(lines, settingsTitleStyle.Render("▌ "+title))
	lines = append(lines, settingsHintStyle.Render("  ↑/↓ move • Enter select • Esc back"))
	for i, device := range devices {
		cursor := "  "
		if i == m.selected {
			cursor = settingsCursorStyle.Render("▸ ")
		}
		line := cursor + settingsDeviceLabel(device)
		if device.Default {
			line += " [default]"
		}
		lines = append(lines, line)
	}
	return settingsFrameStyle.Width(max(24, m.width-2)).Render(strings.Join(lines, "\n"))
}

func (m Model) applyLayout() Model {
	if m.height <= 0 {
		return m
	}
	reserved := 12
	switch m.mode {
	case ModeSettings:
		reserved = 16
	case ModeMicSelect, ModeOutputSelect:
		reserved = 18
	}
	m.viewport.Height = max(6, m.height-reserved)
	return m
}

func (m Model) devicesForMode() []audio.Device {
	source := audio.SourceMic
	if m.mode == ModeOutputSelect {
		source = audio.SourceOutput
	}
	var result []audio.Device
	for _, device := range m.devices {
		if device.Source == source {
			result = append(result, device)
		}
	}
	return result
}

func settingsDeviceLabel(device audio.Device) string {
	label := device.DisplayName()
	var details []string
	if strings.TrimSpace(device.Backend) != "" {
		details = append(details, device.Backend)
	}
	if device.Source == audio.SourceOutput && device.Backend == audio.BackendDirectShow {
		details = append(details, "DirectShow audio capture candidate")
	}
	if device.Source == audio.SourceOutput && device.Backend == audio.BackendWasapiLoopback {
		details = append(details, "WASAPI loopback sidecar candidate")
	}
	if strings.TrimSpace(device.ID) != "" && device.ID != device.DisplayName() {
		details = append(details, "id: "+device.ID)
	}
	if len(details) > 0 {
		label += " (" + strings.Join(details, "; ") + ")"
	}
	return label
}

func footerView(snap app.Snapshot) string {
	shortcuts := []footerShortcut{
		{label: "P Pause", active: snap.Paused},
		{label: "R Record", active: !snap.Paused},
		{label: "M Mute mic", active: snap.MutedMic},
		{label: "U Unmute", active: !snap.MutedMic},
		{label: "S Settings"},
		{label: "Q Quit"},
	}
	var rendered []string
	for _, shortcut := range shortcuts {
		rendered = append(rendered, renderFooterShortcut(shortcut))
	}
	return strings.Join(rendered, "  ")
}

func renderFooterShortcut(shortcut footerShortcut) string {
	style := shortcutStyle
	if shortcut.active {
		style = statusStyle
	}
	return style.Render(shortcut.label)
}

func (m *Model) refreshTranscript() {
	m.viewport.SetContent(renderTranscript(m.controller.Snapshot().Transcript))
	m.viewport.GotoBottom()
}

func (m Model) waitEvent() tea.Cmd {
	return func() tea.Msg {
		event, ok := <-m.controller.Events()
		if !ok {
			return eventMsg{Kind: app.EventStatus, Message: "Stopped"}
		}
		return eventMsg(event)
	}
}

func tick() tea.Cmd {
	return tea.Tick(time.Second, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func renderTranscript(events []transcript.Event) string {
	if len(events) == 0 {
		return "Transcript will appear here as chunks finish transcribing."
	}
	return strings.TrimSpace(transcript.RenderPlain(events))
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func clampLevel(level float64) float64 {
	if level < 0 {
		return 0
	}
	if level > 1 {
		return 1
	}
	return level
}
