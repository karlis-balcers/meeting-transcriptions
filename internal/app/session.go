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

	micSegmenter    *Segmenter
	outputSegmenter *Segmenter

	lastMicSpeaker    string
	lastOutputSpeaker string

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
	session.micSegmenter = NewSegmenter(segmenterConfig(prepared.Config, audio.SourceMic, prepared.TempDir))
	session.outputSegmenter = NewSegmenter(segmenterConfig(prepared.Config, audio.SourceOutput, prepared.TempDir))
	for _, warning := range filterWarnings {
		session.emit(Event{Kind: EventWarning, Message: warning})
	}
	return session, nil
}

// segmenterConfig builds the per-source silence-gating configuration from the
// app config. The recorder writes PCM16 mono 16kHz WAVs, so the segmenter's
// sample-rate / channel constants must match those exactly. Legacy int16
// silence_threshold values are scaled into the normalized [0, 1]
// RMS domain so an RMS-based detector speaks the same language as the old
// Python amplitude check. Values already in [0, 1] are treated as normalized
// RMS thresholds.
func segmenterConfig(cfg config.Config, source audio.Source, tempDir string) SegmentConfig {
	return SegmentConfig{
		Source:             source,
		TempDir:            tempDir,
		FrameDurationMS:    cfg.Audio.FrameDurationMS,
		SampleRateHz:       audio.ConcatSampleRateHz(),
		Channels:           audio.ConcatChannels(),
		SilenceThreshold:   normalizeSilenceThreshold(cfg.Audio.SilenceThreshold),
		SilenceDuration:    cfg.Audio.SilenceDuration.Duration,
		MinSegmentDuration: minSegmentDurationFor(cfg, source),
		MaxSegmentDuration: cfg.Audio.MaxSegmentDuration.Duration,
	}
}

// normalizeSilenceThreshold maps the legacy signed-int16 amplitude scalar
// into the normalized [0, 1] RMS domain used by FramesWithEnergy. Values
// already in [0, 1] (e.g. a YAML author writes `0.02` directly) pass through
// unchanged so operators can tune the detector with a precise RMS target if
// they prefer.
func normalizeSilenceThreshold(raw float64) float64 {
	switch {
	case raw <= 0:
		// A small non-zero default so a literal 0 still gates near-silence;
		// absolute digital zero is rarely useful.
		return 0.01
	case raw <= 1:
		return raw
	default:
		return raw / 32768.0
	}
}

// minSegmentDurationFor returns the minimum amount of speech a segment must
// accumulate before a trailing silence is allowed to end it. The frame
// duration bounds it so we never wait an entire segment for a single frame of
// audio; everything else comes from the configured capture chunk duration,
// which the recorder emits as one analysis window.
func minSegmentDurationFor(cfg config.Config, source audio.Source) time.Duration {
	frame := time.Duration(cfg.Audio.FrameDurationMS) * time.Millisecond
	if frame <= 0 {
		frame = 100 * time.Millisecond
	}
	return frame
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
	// When the source loop exits (session stop, fatal capture error), flush any
	// speech the segmenter is still holding so the final partial utterance is
	// transcribed rather than stranded. Only this goroutine mutates the
	// segmenter, so the deferred flush is race-free.
	defer s.flushPendingSegment(source)
	for {
		select {
		case <-s.ctx.Done():
			return
		default:
		}
		skip, device, speakerName := s.captureState(source)
		if skip {
			// A pause, mute, device restart, or disabled output source is a
			// segment boundary. Flush here, inside the source loop that owns
			// the segmenter, rather than mutating it from the control method.
			s.flushPendingSegment(source)
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
				if guidance := audio.OutputCaptureUnavailableGuidance(s.prepared.Devices); guidance != "" {
					warning := fmt.Sprintf("output capture unavailable for %s: %v — output capture disabled; microphone transcription will continue. %s", device.DisplayName(), err, guidance)
					logging.Printf(warning)
					s.disableOutputCapture()
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
		// Hand the raw frame to the per-source segmenter. It either discards
		// silence, holds the frame, or signals that enough speech has
		// accumulated (silence after speech, max segment length) to flush a
		// single transcription-ready WAV. Speaker changes are handled here too:
		// any pending speech under the previous speaker is flushed first so the
		// new speaker's audio is not mis-attributed.
		segmenter := s.segmenterFor(source)
		if segmenter == nil {
			// Defensive: should never happen; degrade to the legacy per-frame
			// behavior so audio still reaches the transcriber.
			logging.Printf("%s segmenter missing; falling back to direct transcription", source)
			s.sendChunk(chunk)
			continue
		}
		segmentStart := chunk.Started
		if segmentStart.IsZero() {
			segmentStart = started
		}
		segmentEnd := chunk.Ended
		if segmentEnd.IsZero() {
			segmentEnd = s.prepared.Dependencies.Clock()
		}

		// Speaker change: flush whatever we have under the previous speaker
		// before accumulating the new frame.
		if chunk.Speaker != "" {
			if prev, ok := s.lastSpeakerFor(source); ok && prev != chunk.Speaker && segmenter.HasPending() {
				if segment, flushed, flushErr := segmenter.Flush(segmentStart, prev); flushErr != nil {
					logging.Printf("%s segmenter flush on speaker change failed: %v", source, flushErr)
					s.emit(Event{Kind: EventWarning, Message: fmt.Sprintf("could not finalize %s segment on speaker change: %v", source, flushErr)})
				} else if flushed {
					logging.Printf("%s segment flushed for speaker change %q -> %q: %s", source, prev, chunk.Speaker, segment.FilePath)
					s.sendSegmentChunk(segment, source, chunk.Device, seq)
				}
			}
			s.setLastSpeakerFor(source, chunk.Speaker)
		}

		loud, energy, loudErr := segmenter.IsLoud(chunk.FilePath)
		if loudErr != nil {
			logging.Printf("%s loudness analysis failed: %v", source, loudErr)
			_ = os.Remove(chunk.FilePath)
			continue
		}
		decision := segmenter.Accumulate(chunk.FilePath, loud, energy, func() time.Time { return segmentStart })
		if decision.Discarded {
			logging.Printf("%s frame discarded (silent, no speech started): %s", source, chunk.FilePath)
			continue
		}
		if decision.Flush {
			speaker := chunk.Speaker
			if segment, flushed, flushErr := segmenter.Flush(segmentEnd, speaker); flushErr != nil {
				logging.Printf("%s segmenter flush failed: %v", source, flushErr)
				s.emit(Event{Kind: EventWarning, Message: fmt.Sprintf("could not finalize %s segment: %v", source, flushErr)})
			} else if flushed {
				logging.Printf("%s segment flushed (%s): %s", source, decision.Reason, segment.FilePath)
				s.sendSegmentChunk(segment, source, chunk.Device, seq)
			}
		} else {
			logging.Printf("%s frame held (%s, speech=%s trailing-silence=%s)", source, decision.Reason, decision.ElapsedSpeech, decision.TrailingSilent)
		}
	}
}

// segmenterFor returns the per-source Segmenter that accumulates recorded
// frames into silence-gated segments, or nil if the source has no segmenter.
func (s *Session) segmenterFor(source audio.Source) *Segmenter {
	switch source {
	case audio.SourceMic:
		return s.micSegmenter
	case audio.SourceOutput:
		return s.outputSegmenter
	default:
		return nil
	}
}

// sendChunk enqueues a captured frame/segment WAV for transcription, honoring
// the stop context. It is the legacy direct-send path used when the segmenter
// is not configured (defensive) and remains the chokepoint all flushes funnel
// through as sendSegmentChunk.
func (s *Session) sendChunk(chunk audio.Chunk) {
	select {
	case <-s.ctx.Done():
		_ = os.Remove(chunk.FilePath)
	case s.chunks <- chunk:
	}
}

// sendFinalChunk enqueues a finalized segment even after capture has been
// canceled. Capture cancellation happens before chunks is closed, while the
// transcription loop remains alive until it drains that channel. Checking the
// transcription context prevents a shutdown timeout from leaving a source
// goroutine blocked forever after the consumer has been canceled.
func (s *Session) sendFinalChunk(chunk audio.Chunk) {
	if s.transcribeCtx == nil {
		_ = os.Remove(chunk.FilePath)
		return
	}
	select {
	case s.chunks <- chunk:
	case <-s.transcribeCtx.Done():
		_ = os.Remove(chunk.FilePath)
	}
}

// sendSegmentChunk wraps a finalized Segment in an audio.Chunk carrying the
// correct start/end timestamps, speaker, and source/device so the downstream
// transcription loop and transcript store see all the metadata they expect
// from a single concatenated utterance WAV.
func (s *Session) sendSegmentChunk(segment Segment, source audio.Source, device audio.Device, seq uint64) {
	chunk := audio.Chunk{
		Source:   source,
		Device:   device,
		FilePath: segment.FilePath,
		Started:  segment.Started,
		Ended:    segment.Ended,
		Sequence: seq,
		Speaker:  segment.Speaker,
	}
	// A finalized segment is safe to drain after capture cancellation. Using
	// the stop-aware path here also closes the race where Stop happens between
	// Accumulate/Flush and enqueueing a segment.
	s.sendFinalChunk(chunk)
}

// lastSpeakerFor returns the most recent detected speaker label for the source
// and whether one has been recorded yet. Speaker-change flushes use it to
// attribute pending speech to the previous speaker before accumulating audio
// under the new speaker label.
func (s *Session) lastSpeakerFor(source audio.Source) (string, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch source {
	case audio.SourceMic:
		return s.lastMicSpeaker, s.lastMicSpeaker != ""
	case audio.SourceOutput:
		return s.lastOutputSpeaker, s.lastOutputSpeaker != ""
	default:
		return "", false
	}
}

// setLastSpeakerFor records the latest detected speaker label for the source.
func (s *Session) setLastSpeakerFor(source audio.Source, speaker string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch source {
	case audio.SourceMic:
		s.lastMicSpeaker = speaker
	case audio.SourceOutput:
		s.lastOutputSpeaker = speaker
	}
}

// flushPendingSegment finalizes any accumulated speech for the source so a
// pause/mute/device-change/stop does not strand half an utterance in the
// segmenter's pending buffer. Speaker comes from the last detected label for
// the source so the partial WAV is transcribed under the correct speaker. It
// is a no-op when the segmenter has no pending speech and never returns an
// error to the caller; failures are surfaced as a warning event.
func (s *Session) flushPendingSegment(source audio.Source) {
	segmenter := s.segmenterFor(source)
	if segmenter == nil || !segmenter.HasPending() {
		return
	}
	speaker := ""
	if last, ok := s.lastSpeakerFor(source); ok {
		speaker = last
	}
	segment, flushed, err := segmenter.Flush(s.prepared.Dependencies.Clock(), speaker)
	if err != nil {
		logging.Printf("%s flush pending segment failed: %v", source, err)
		s.emit(Event{Kind: EventWarning, Message: fmt.Sprintf("could not finalize pending %s segment: %v", source, err)})
		return
	}
	if !flushed {
		return
	}
	device := s.deviceFor(source)
	logging.Printf("%s segment flushed on interrupt: %s", source, segment.FilePath)
	s.sendSegmentChunk(segment, source, device, s.sequence.Add(1))
}

// resetPendingSegment drops any accumulated-but-unflushed frames for a source
// without transcribing them. Used when a session is stopping abruptly and the
// partial segment is either too short to transcribe or timing-critical.
func (s *Session) resetPendingSegment(source audio.Source) {
	if segmenter := s.segmenterFor(source); segmenter != nil {
		segmenter.Reset()
	}
}

// deviceFor returns the currently selected capture device for a source.
func (s *Session) deviceFor(source audio.Source) audio.Device {
	s.mu.Lock()
	defer s.mu.Unlock()
	if source == audio.SourceMic {
		return s.selection.Mic
	}
	return s.selection.Output
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
