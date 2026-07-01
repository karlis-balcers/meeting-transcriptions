package app

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"time"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/config"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/filter"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/logging"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/transcript"
)

type EventKind string

const (
	EventStatus     EventKind = "status"
	EventWarning    EventKind = "warning"
	EventError      EventKind = "error"
	EventTranscript EventKind = "transcript"
	EventLoudness   EventKind = "loudness"
)

type Event struct {
	Kind        EventKind
	Source      audio.Source
	Message     string
	Transcript  transcript.Event
	MicLevel    float64
	OutputLevel float64
}

type Snapshot struct {
	Status         string
	Error          string
	Paused         bool
	MutedMic       bool
	StartedAt      time.Time
	Mic            audio.Device
	Output         audio.Device
	Transcript     []transcript.Event
	TranscriptPath string
}

type Session struct {
	prepared *Prepared
	store    *transcript.Store
	filter   *filter.Filter
	events   chan Event
	chunks   chan audio.Chunk
	controls chan struct{}

	ctx              context.Context
	cancel           context.CancelFunc
	transcribeCtx    context.Context
	cancelTranscribe context.CancelFunc
	wg               sync.WaitGroup

	mu                    sync.Mutex
	status                string
	errText               string
	paused                bool
	mutedMic              bool
	outputCaptureDisabled bool
	started               bool
	startedAt             time.Time
	selection             audio.Selection
	cfg                   config.Config

	micCaptureCancel    *captureCancel
	outputCaptureCancel *captureCancel

	sequence atomic.Uint64
}

type captureCancel struct {
	cancel context.CancelFunc
}

func NewSession(prepared *Prepared) (*Session, error) {
	store, err := transcript.NewFileStore(prepared.OutputDir, prepared.Dependencies.Clock, transcript.Metadata{
		MicDevice:    prepared.Selection.Mic.DisplayName(),
		OutputDevice: prepared.Selection.Output.DisplayName(),
		Language:     prepared.Config.Language,
		Model:        prepared.Config.OpenAI.Model,
	})
	if err != nil {
		return nil, err
	}
	flt, filterWarnings := filter.New(filterConfig(prepared.Config))
	session := &Session{
		prepared:  prepared,
		store:     store,
		filter:    flt,
		events:    make(chan Event, 128),
		chunks:    make(chan audio.Chunk, 32),
		controls:  make(chan struct{}, 1),
		status:    "Starting",
		selection: prepared.Selection,
		cfg:       prepared.Config,
	}
	for _, warning := range filterWarnings {
		session.emit(Event{Kind: EventWarning, Message: warning})
	}
	return session, nil
}

func (s *Session) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return nil
	}
	s.ctx, s.cancel = context.WithCancel(ctx)
	s.transcribeCtx, s.cancelTranscribe = context.WithCancel(context.Background())
	s.started = true
	s.startedAt = s.prepared.Dependencies.Clock()
	s.status = "Recording"
	s.mu.Unlock()
	logging.Printf("session started: mic=%q output=%q transcript=%q", s.selection.Mic.DisplayName(), s.selection.Output.DisplayName(), s.store.Path())

	s.emit(Event{Kind: EventStatus, Message: "Recording"})
	for _, warning := range s.prepared.Warnings {
		s.emit(Event{Kind: EventWarning, Message: warning})
	}
	s.wg.Add(1)
	go s.speakerLoop()
	s.wg.Add(1)
	go s.captureLoops()
	s.wg.Add(1)
	go s.transcribeLoop()
	return nil
}

func (s *Session) Stop(ctx context.Context) error {
	s.mu.Lock()
	if !s.started {
		s.mu.Unlock()
		return nil
	}
	s.status = "Stopping"
	if s.cancel != nil {
		s.cancel()
	}
	s.mu.Unlock()
	logging.Printf("session stopping")

	done := make(chan struct{})
	go func() {
		s.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-ctx.Done():
		if s.cancelTranscribe != nil {
			s.cancelTranscribe()
		}
		return ctx.Err()
	}
	if s.cancelTranscribe != nil {
		s.cancelTranscribe()
	}
	s.prepared.Dependencies.Speaker.Close()
	s.mu.Lock()
	s.status = "Stopped"
	s.started = false
	s.mu.Unlock()
	s.emit(Event{Kind: EventStatus, Message: "Stopped"})
	logging.Printf("session stopped")
	return nil
}

func (s *Session) Events() <-chan Event { return s.events }

func (s *Session) Snapshot() Snapshot {
	s.mu.Lock()
	defer s.mu.Unlock()
	return Snapshot{
		Status:         s.status,
		Error:          s.errText,
		Paused:         s.paused,
		MutedMic:       s.mutedMic,
		StartedAt:      s.startedAt,
		Mic:            s.selection.Mic,
		Output:         s.selection.Output,
		Transcript:     s.store.Snapshot(),
		TranscriptPath: s.store.Path(),
	}
}

func (s *Session) Pause() error {
	s.mu.Lock()
	s.paused = true
	s.status = "Paused"
	s.mu.Unlock()
	logging.Printf("session paused")
	s.cancelCapture(audio.SourceMic)
	s.cancelCapture(audio.SourceOutput)
	s.notifyControl()
	s.emit(Event{Kind: EventStatus, Message: "Paused"})
	return nil
}

func (s *Session) Resume() error {
	s.mu.Lock()
	s.paused = false
	s.status = "Recording"
	s.mu.Unlock()
	logging.Printf("session resumed")
	s.notifyControl()
	s.emit(Event{Kind: EventStatus, Message: "Recording"})
	return nil
}

func (s *Session) MuteMic() error {
	s.mu.Lock()
	s.mutedMic = true
	s.mu.Unlock()
	logging.Printf("microphone muted")
	s.cancelCapture(audio.SourceMic)
	s.notifyControl()
	s.emit(Event{Kind: EventStatus, Message: "Microphone muted"})
	return nil
}

func (s *Session) UnmuteMic() error {
	s.mu.Lock()
	s.mutedMic = false
	s.mu.Unlock()
	logging.Printf("microphone unmuted")
	s.notifyControl()
	s.emit(Event{Kind: EventStatus, Message: "Microphone unmuted"})
	return nil
}

func (s *Session) SetMicDevice(device audio.Device) error {
	s.mu.Lock()
	currentOutput := s.selection.Output
	cfg := s.cfg
	path := s.prepared.ConfigPath
	s.mu.Unlock()
	if audio.SameDevice(device, currentOutput) {
		return errors.New("microphone and output capture cannot be the same device; choose a distinct output-capture loopback/monitor device")
	}
	cfg.Audio.MicDeviceID = device.ID
	cfg.Audio.MicDeviceName = device.Name
	if path != "" {
		if err := config.Save(path, cfg); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.selection.Mic = device
	s.cfg = cfg
	s.status = "Microphone device changed; restarting capture"
	s.mu.Unlock()
	logging.Printf("microphone device changed: %q (%s)", device.DisplayName(), device.ID)
	s.cancelCapture(audio.SourceMic)
	s.notifyControl()
	s.emit(Event{Kind: EventStatus, Message: "Microphone device changed; restarting capture"})
	return nil
}

func (s *Session) SetOutputDevice(device audio.Device) error {
	s.mu.Lock()
	currentMic := s.selection.Mic
	cfg := s.cfg
	path := s.prepared.ConfigPath
	s.mu.Unlock()
	if audio.SameDevice(currentMic, device) {
		return errors.New("microphone and output capture cannot be the same device; choose a distinct output-capture loopback/monitor device")
	}
	cfg.Audio.OutputDeviceID = device.ID
	cfg.Audio.OutputDeviceName = device.Name
	if path != "" {
		if err := config.Save(path, cfg); err != nil {
			return err
		}
	}
	s.mu.Lock()
	s.selection.Output = device
	s.outputCaptureDisabled = false
	s.cfg = cfg
	s.status = "Output device changed; restarting capture"
	s.mu.Unlock()
	logging.Printf("output device changed: %q (%s)", device.DisplayName(), device.ID)
	s.cancelCapture(audio.SourceOutput)
	s.notifyControl()
	s.emit(Event{Kind: EventStatus, Message: "Output device changed; restarting capture"})
	return nil
}

func (s *Session) captureLoops() {
	defer s.wg.Done()
	var sourceWG sync.WaitGroup
	sourceWG.Add(2)
	go s.sourceLoop(&sourceWG, audio.SourceMic)
	go s.sourceLoop(&sourceWG, audio.SourceOutput)
	sourceWG.Wait()
	close(s.chunks)
}

func (s *Session) sourceLoop(sourceWG *sync.WaitGroup, source audio.Source) {
	defer sourceWG.Done()
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
		}
		skip, device, speakerName := s.captureState(source)
		if skip {
			if !s.waitForControlOrStop() {
				return
			}
			continue
		}
		seq := s.sequence.Add(1)
		started := s.prepared.Dependencies.Clock()
		duration := s.captureDuration()
		logging.Printf("%s capture starting: device=%q sequence=%d duration=%s", source, device.DisplayName(), seq, duration)
		chunkCtx, cancel := context.WithCancel(s.ctx)
		token := s.setCaptureCancel(source, cancel)
		chunk, err := s.prepared.Dependencies.Recorder.RecordChunk(chunkCtx, audio.ChunkRequest{
			Device:   device,
			Source:   source,
			TempDir:  s.prepared.TempDir,
			Duration: duration,
			Sequence: seq,
			Started:  started,
			Speaker:  speakerName,
		})
		s.clearCaptureCancel(source, token)
		cancel()
		if err != nil {
			if s.ctx.Err() != nil {
				return
			}
			if errors.Is(err, context.Canceled) {
				continue
			}
			if source == audio.SourceOutput {
				if next, ok := s.nextOutputCaptureCandidate(device); ok {
					s.setRuntimeOutputSelection(next)
					warning := fmt.Sprintf("output capture failed for %s; falling back to %s", device.DisplayName(), next.DisplayName())
					logging.Printf(warning)
					s.emit(Event{Kind: EventWarning, Message: warning})
					continue
				}
				message := fmt.Sprintf("output capture failed for %s: %v", device.DisplayName(), err)
				logging.Printf(message)
				s.setError(message)
				if s.cancel != nil {
					s.cancel()
				}
				return
			}
			logging.Printf("%s capture failed: %v", source, err)
			s.setError(fmt.Sprintf("%s capture failed: %v", source, err))
			if s.cancel != nil {
				s.cancel()
			}
			return
		}
		logging.Printf("%s chunk recorded: %s", source, chunk.FilePath)
		level, err := audio.RMSLevelFromWAV(chunk.FilePath)
		if err != nil {
			s.emit(Event{Kind: EventWarning, Message: fmt.Sprintf("could not compute %s loudness from captured audio: %v", source, err)})
		} else {
			s.emitLoudness(source, level)
		}
		select {
		case <-s.ctx.Done():
			_ = os.Remove(chunk.FilePath)
			return
		case s.chunks <- chunk:
		}
	}
}

func (s *Session) transcribeLoop() {
	defer s.wg.Done()
	for chunk := range s.chunks {
		logging.Printf("sending %s chunk %d to OpenAI: %s", chunk.Source, chunk.Sequence, chunk.FilePath)
		text, err := s.prepared.Dependencies.Transcriber.Transcribe(s.transcribeCtx, chunk.FilePath, openAIOptions(s.currentConfig()))
		_ = os.Remove(chunk.FilePath)
		if err != nil {
			if s.transcribeCtx.Err() != nil {
				return
			}
			logging.Printf("OpenAI transcription failed for %s chunk %d: %v", chunk.Source, chunk.Sequence, err)
			s.emit(Event{Kind: EventError, Message: fmt.Sprintf("OpenAI transcription failed: %v", err)})
			continue
		}
		if !s.filter.Keep(text) {
			logging.Printf("filtered %s chunk %d after transcription", chunk.Source, chunk.Sequence)
			continue
		}
		logging.Printf("transcribed %s chunk %d: %d characters kept", chunk.Source, chunk.Sequence, len(text))
		event := transcript.Event{Source: string(chunk.Source), Speaker: chunk.Speaker, Text: text, Start: chunk.Started, End: chunk.Ended}
		if err := s.store.Add(event); err != nil {
			s.emit(Event{Kind: EventError, Message: err.Error()})
			continue
		}
		s.emit(Event{Kind: EventTranscript, Transcript: event})
	}
}

func (s *Session) speakerLoop() {
	defer s.wg.Done()
	for event := range s.prepared.Dependencies.Speaker.Start(s.ctx) {
		if event.Warning != "" {
			s.emit(Event{Kind: EventWarning, Message: event.Warning})
		}
	}
}

func (s *Session) captureState(source audio.Source) (bool, audio.Device, string) {
	s.mu.Lock()
	selection := s.selection
	cfg := s.cfg
	skip := s.paused || (source == audio.SourceMic && s.mutedMic) || (source == audio.SourceOutput && s.outputCaptureDisabled)
	s.mu.Unlock()
	if source == audio.SourceMic {
		return skip, selection.Mic, cfg.UserName
	}
	return skip, selection.Output, s.prepared.Dependencies.Speaker.Current()
}

func (s *Session) nextOutputCaptureCandidate(current audio.Device) (audio.Device, bool) {
	candidates := audio.OutputCaptureCandidates(s.prepared.Devices, current)
	if len(candidates) < 2 {
		return audio.Device{}, false
	}
	return candidates[1], true
}

func (s *Session) setRuntimeOutputSelection(device audio.Device) {
	s.mu.Lock()
	s.selection.Output = device
	s.outputCaptureDisabled = false
	s.mu.Unlock()
}

func (s *Session) disableOutputCapture() {
	s.mu.Lock()
	s.outputCaptureDisabled = true
	s.mu.Unlock()
}

func (s *Session) currentConfig() config.Config {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.cfg
}

func (s *Session) captureDuration() time.Duration {
	cfg := s.currentConfig()
	duration := cfg.Audio.CaptureChunkDuration.Duration
	maxDuration := cfg.Audio.MaxSegmentDuration.Duration
	if duration <= 0 || duration > maxDuration {
		return maxDuration
	}
	return duration
}

func (s *Session) waitForControlOrStop() bool {
	select {
	case <-s.ctx.Done():
		return false
	case <-s.controls:
		return true
	case <-time.After(250 * time.Millisecond):
		return true
	}
}

func (s *Session) setCaptureCancel(source audio.Source, cancel context.CancelFunc) *captureCancel {
	token := &captureCancel{cancel: cancel}
	s.mu.Lock()
	if source == audio.SourceMic {
		s.micCaptureCancel = token
	} else {
		s.outputCaptureCancel = token
	}
	s.mu.Unlock()
	return token
}

func (s *Session) clearCaptureCancel(source audio.Source, token *captureCancel) {
	s.mu.Lock()
	if source == audio.SourceMic {
		if s.micCaptureCancel == token {
			s.micCaptureCancel = nil
		}
	} else if s.outputCaptureCancel == token {
		s.outputCaptureCancel = nil
	}
	s.mu.Unlock()
}

func (s *Session) cancelCapture(source audio.Source) {
	s.mu.Lock()
	var token *captureCancel
	if source == audio.SourceMic {
		token = s.micCaptureCancel
	} else {
		token = s.outputCaptureCancel
	}
	s.mu.Unlock()
	if token != nil {
		token.cancel()
	}
}

func (s *Session) notifyControl() {
	select {
	case s.controls <- struct{}{}:
	default:
	}
}

func (s *Session) setError(message string) {
	s.mu.Lock()
	s.errText = message
	s.status = "Error"
	s.mu.Unlock()
	logging.Printf("session error: %s", message)
	s.emit(Event{Kind: EventError, Message: message})
}

func (s *Session) emit(event Event) {
	select {
	case s.events <- event:
	default:
	}
}

func (s *Session) emitLoudness(source audio.Source, level float64) {
	if level < 0 {
		level = 0
	}
	if level > 1 {
		level = 1
	}
	event := Event{Kind: EventLoudness, Source: source}
	if source == audio.SourceMic {
		event.MicLevel = level
	} else {
		event.OutputLevel = level
	}
	s.emit(event)
}
