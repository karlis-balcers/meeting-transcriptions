package app

import (
	"bytes"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/config"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/logging"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/openai"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/speaker"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/transcript"
)

func TestPrepareRejectsMissingAPIKeyBeforeDeviceDiscovery(t *testing.T) {
	discoverer := &countingDiscoverer{}
	_, err := Prepare(context.Background(), config.Defaults(), "", "", Dependencies{Discoverer: discoverer})
	if err == nil || !strings.Contains(err.Error(), "OPENAI_API_KEY") {
		t.Fatalf("expected missing API key error, got %v", err)
	}
	if discoverer.calls.Load() != 0 {
		t.Fatalf("device discovery should not run without an API key, got %d calls", discoverer.calls.Load())
	}
}

func TestPauseCancelsActiveRecorderChunks(t *testing.T) {
	cfg := config.Defaults()
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	cfg.Audio.CaptureChunkDuration = config.NewDuration(5 * time.Second)
	recorder := newBlockingRecorder()
	prepared := prepareTestSession(t, cfg, filepath.Join(t.TempDir(), "config.yaml"), recorder)
	session, err := NewSession(prepared)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := session.Start(ctx); err != nil {
		t.Fatal(err)
	}
	started := waitForSources(t, recorder.started, audio.SourceMic, audio.SourceOutput)
	for source, request := range started {
		if request.Duration != 5*time.Second {
			t.Fatalf("%s should record short cancellable chunks, got %s", source, request.Duration)
		}
	}

	if err := session.Pause(); err != nil {
		t.Fatal(err)
	}
	waitForSources(t, recorder.canceled, audio.SourceMic, audio.SourceOutput)
	if snap := session.Snapshot(); !snap.Paused || snap.Status != "Paused" {
		t.Fatalf("session should be paused after cancellation, got %+v", snap)
	}
	stopSession(t, session)
}

func TestSetMicDevicePersistsConfigAndRestartsMicCapture(t *testing.T) {
	cfg := config.Defaults()
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	cfg.Audio.CaptureChunkDuration = config.NewDuration(5 * time.Second)
	configPath := filepath.Join(t.TempDir(), "config.yaml")
	recorder := newBlockingRecorder()
	prepared := prepareTestSession(t, cfg, configPath, recorder)
	session, err := NewSession(prepared)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := session.Start(ctx); err != nil {
		t.Fatal(err)
	}
	waitForRequest(t, recorder.started, audio.SourceMic)

	newMic := audio.Device{ID: "mic-2", Name: "Backup Mic", Source: audio.SourceMic, Backend: "fake"}
	if err := session.SetMicDevice(newMic); err != nil {
		t.Fatal(err)
	}
	waitForRequest(t, recorder.canceled, audio.SourceMic)
	restarted := waitForRequest(t, recorder.started, audio.SourceMic)
	if restarted.Device.ID != newMic.ID {
		t.Fatalf("mic capture should restart with new device, got %+v", restarted.Device)
	}
	content, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(content), "mic_device_id: mic-2") || !strings.Contains(string(content), "mic_device_name: Backup Mic") {
		t.Fatalf("config did not persist selected mic:\n%s", string(content))
	}
	stopSession(t, session)
}

func TestSetOutputDeviceRejectsDuplicateMicSelection(t *testing.T) {
	cfg := config.Defaults()
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	cfg.Audio.CaptureChunkDuration = config.NewDuration(5 * time.Second)
	recorder := newBlockingRecorder()
	prepared := prepareTestSession(t, cfg, filepath.Join(t.TempDir(), "config.yaml"), recorder)
	session, err := NewSession(prepared)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := session.Start(ctx); err != nil {
		t.Fatal(err)
	}
	waitForRequest(t, recorder.started, audio.SourceMic)
	if err := session.SetOutputDevice(audio.Device{ID: "mic", Name: "QA Mic", Source: audio.SourceOutput, Backend: "fake"}); err == nil || !strings.Contains(err.Error(), "same device") {
		t.Fatalf("expected duplicate output selection to be rejected, got %v", err)
	}
	stopSession(t, session)
}

func TestSetMicDeviceRejectsDuplicateOutputSelection(t *testing.T) {
	cfg := config.Defaults()
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	cfg.Audio.CaptureChunkDuration = config.NewDuration(5 * time.Second)
	recorder := newBlockingRecorder()
	prepared := prepareTestSession(t, cfg, filepath.Join(t.TempDir(), "config.yaml"), recorder)
	session, err := NewSession(prepared)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := session.Start(ctx); err != nil {
		t.Fatal(err)
	}
	waitForRequest(t, recorder.started, audio.SourceMic)
	if err := session.SetMicDevice(audio.Device{ID: "out", Name: "QA Output", Source: audio.SourceMic, Backend: "fake"}); err == nil || !strings.Contains(err.Error(), "same device") {
		t.Fatalf("expected duplicate microphone selection to be rejected, got %v", err)
	}
	stopSession(t, session)
}

func TestOutputCaptureFallsBackToNextDeviceAndKeepsMicRecording(t *testing.T) {
	base := time.Date(2026, 5, 29, 12, 0, 0, 0, time.UTC)
	cfg := config.Defaults()
	cfg.UserName = "You"
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	segmenterFlushConfig(&cfg, 50*time.Millisecond)
	cfg.Audio.MicDeviceID = "mic-1"
	cfg.Audio.MicDeviceName = "QA Mic"
	cfg.Audio.OutputDeviceID = "out-a"
	cfg.Audio.OutputDeviceName = "QA Output A"

	configPath := filepath.Join(t.TempDir(), "config.yaml")
	if err := config.Save(configPath, cfg); err != nil {
		t.Fatal(err)
	}
	originalConfig, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	recorder := newFallbackRecorder(base)
	prepared, err := Prepare(ctx, cfg, configPath, "test-key", Dependencies{
		Discoverer: fakeDiscoverer{devices: []audio.Device{
			{ID: "mic-1", Name: "QA Mic", Source: audio.SourceMic, Default: true, Backend: "fake"},
			{ID: "out-a", Name: "QA Output A", Source: audio.SourceOutput, Default: true, Backend: "fake"},
			{ID: "out-b", Name: "QA Output B", Source: audio.SourceOutput, Backend: "fake"},
		}},
		Recorder:    recorder,
		Speaker:     fixedSpeaker{current: "Alice"},
		Transcriber: &cancelAfterTranscripts{cancel: cancel, want: 2},
		Clock:       func() time.Time { return base },
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := NewSession(prepared)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Start(ctx); err != nil {
		t.Fatal(err)
	}

	select {
	case <-ctx.Done():
	case <-time.After(3 * time.Second):
		t.Fatal("timed out waiting for fallback session to complete")
	}

	stopSession(t, session)

	snap := session.Snapshot()
	if snap.Error != "" {
		t.Fatalf("session should keep running after output fallback, got error %q", snap.Error)
	}
	if !transcriptContainsSource(snap.Transcript, string(audio.SourceMic)) {
		t.Fatalf("expected microphone transcript after fallback, got %+v", snap.Transcript)
	}
	if !transcriptContainsSource(snap.Transcript, string(audio.SourceOutput)) {
		t.Fatalf("expected output transcript after fallback, got %+v", snap.Transcript)
	}

	events := drainEvents(session.Events())
	if !hasWarningContaining(events, "falling back to") {
		t.Fatalf("expected output fallback warning, got events %+v", events)
	}
	if hasWarningContaining(events, "output capture unavailable") {
		t.Fatalf("output capture should have fallen back to another device, got events %+v", events)
	}

	outputOrder := outputRequestOrder(recorder.RecordedRequests())
	if len(outputOrder) < 2 || outputOrder[0] != "out-a" || outputOrder[1] != "out-b" {
		t.Fatalf("expected deterministic output fallback order out-a -> out-b, got %v", outputOrder)
	}

	afterConfig, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(afterConfig, originalConfig) {
		t.Fatalf("config file was rewritten during runtime fallback\nbefore:\n%s\nafter:\n%s", string(originalConfig), string(afterConfig))
	}
}

func TestOutputCaptureFailureStopsSessionInsteadOfMicOnlyFallback(t *testing.T) {
	cfg := config.Defaults()
	cfg.UserName = "You"
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	segmenterFlushConfig(&cfg, 50*time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	recorder, failed := newOutputFailureRecorder()
	prepared, err := Prepare(ctx, cfg, filepath.Join(t.TempDir(), "config.yaml"), "test-key", Dependencies{
		Discoverer: fakeDiscoverer{devices: []audio.Device{
			{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: audio.SourceMic, Default: true, Backend: "dshow"},
			{ID: "Stereo Mix (Realtek(R) Audio)", Name: "Stereo Mix (Realtek(R) Audio)", Source: audio.SourceOutput, Default: true, Backend: "dshow"},
		}},
		Recorder:    recorder,
		Speaker:     fixedSpeaker{current: "Alice"},
		Transcriber: &cancelAfterTranscripts{cancel: cancel, want: 1000},
		Clock:       time.Now,
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := NewSession(prepared)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Start(ctx); err != nil {
		t.Fatal(err)
	}

	select {
	case <-failed:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for the output recorder to fail")
	}

	deadline := time.After(2 * time.Second)
	for {
		snap := session.Snapshot()
		if snap.Status == "Error" {
			if !strings.Contains(strings.ToLower(snap.Error), "output capture failed") {
				t.Fatalf("expected output capture failure in session error, got %+v", snap)
			}
			break
		}
		select {
		case <-deadline:
			t.Fatalf("expected output capture failure to stop the session, got %+v", snap)
		case <-time.After(10 * time.Millisecond):
		}
	}

	micRequests, outputRequests := recorder.counts()
	if outputRequests == 0 {
		t.Fatal("expected the output recorder to be exercised")
	}
	if micRequests > 3 {
		t.Fatalf("output failure should not degrade into mic-only recording, got %d microphone requests", micRequests)
	}

	stopSession(t, session)
}

// TestSynthesizedRenderEndpointFailureDisablesOutputAndKeepsMicRecording
// reproduces the Windows display-only render endpoint case: when only
// synthesized Speakers/Headphones [Loopback] outputs exist, output capture is
// unavailable to ffmpeg DirectShow. The session should warn, disable output
// capture, and keep microphone transcription alive instead of failing or
// cascading through more synthesized endpoints.
func TestSynthesizedRenderEndpointFailureDisablesOutputAndKeepsMicRecording(t *testing.T) {
	cfg := config.Defaults()
	cfg.UserName = "You"
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	segmenterFlushConfig(&cfg, 50*time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	recorder, failed := newOutputFailureRecorder()
	prepared, err := Prepare(ctx, cfg, filepath.Join(t.TempDir(), "config.yaml"), "test-key", Dependencies{
		Discoverer: fakeDiscoverer{devices: []audio.Device{
			{ID: "Microphone Array (Realtek(R) Audio)", Name: "Microphone Array (Realtek(R) Audio)", Source: audio.SourceMic, Default: true, Backend: "dshow"},
			{ID: "Speakers (JBL Flip 6)", Name: "Speakers (JBL Flip 6) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{22222222-2222-2222-2222-222222222222}`}, Source: audio.SourceOutput, Default: true, Backend: "dshow"},
			{ID: "Speakers (Realtek(R) Audio)", Name: "Speakers (Realtek(R) Audio) [Loopback]", Aliases: []string{`SWD\MMDEVAPI\{33333333-3333-3333-3333-333333333333}`}, Source: audio.SourceOutput, Backend: "dshow"},
		}},
		Recorder:    recorder,
		Speaker:     fixedSpeaker{current: "Alice"},
		Transcriber: &cancelAfterTranscripts{cancel: cancel, want: 2},
		Clock:       time.Now,
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := NewSession(prepared)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Start(ctx); err != nil {
		t.Fatal(err)
	}

	select {
	case <-failed:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for the output recorder to fail")
	}

	select {
	case <-ctx.Done():
	case <-time.After(2 * time.Second):
		t.Fatalf("timed out waiting for microphone transcription to continue after output capture was disabled; snapshot=%+v", session.Snapshot())
	}

	stopSession(t, session)

	snap := session.Snapshot()
	if snap.Error != "" {
		t.Fatalf("synthesized output unavailability should be non-fatal, got error %q", snap.Error)
	}
	if !transcriptContainsSource(snap.Transcript, string(audio.SourceMic)) {
		t.Fatalf("expected microphone transcript after disabling output capture, got %+v", snap.Transcript)
	}
	if transcriptContainsSource(snap.Transcript, string(audio.SourceOutput)) {
		t.Fatalf("did not expect output transcripts after output capture was disabled, got %+v", snap.Transcript)
	}

	events := drainEvents(session.Events())
	if !hasWarningContaining(events, "output capture disabled") || !hasWarningContaining(events, "Stereo Mix") || !hasWarningContaining(events, "VB-CABLE") || !hasWarningContaining(events, "--list-devices") {
		t.Fatalf("expected non-fatal output-disabled warning with actionable guidance, got events %+v", events)
	}
	if hasWarningContaining(events, "falling back to") {
		t.Fatalf("synthesized render endpoints should not trigger fallback warnings, got events %+v", events)
	}

	micRequests, outputRequests := recorder.counts()
	if micRequests < 2 {
		t.Fatalf("expected repeated microphone recording after output capture was disabled, got %d mic requests", micRequests)
	}
	if outputRequests != 1 {
		t.Fatalf("output capture should be exercised once then disabled, got %d output requests", outputRequests)
	}
}

func TestFilteredChunkIsLoggedAndNotStored(t *testing.T) {
	oldWD, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	logDir := t.TempDir()
	if err := os.Chdir(logDir); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(oldWD)
		_ = logging.Close()
	})
	if _, err := logging.EnableCurrentDir(); err != nil {
		t.Fatal(err)
	}

	base := time.Date(2026, 5, 28, 16, 0, 0, 0, time.UTC)
	cfg := config.Defaults()
	cfg.UserName = "You"
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	segmenterFlushConfig(&cfg, 50*time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	prepared, err := Prepare(ctx, cfg, filepath.Join(t.TempDir(), "config.yaml"), "test-key", Dependencies{
		Discoverer: fakeDiscoverer{devices: []audio.Device{
			{ID: "mic", Name: "QA Mic", Source: audio.SourceMic, Default: true, Backend: "fake"},
			{ID: "out", Name: "QA Output", Source: audio.SourceOutput, Default: true, Backend: "fake"},
		}},
		Recorder:    &oneChunkPerSourceRecorder{base: base},
		Speaker:     fixedSpeaker{current: "Alice"},
		Transcriber: &cancelAfterTranscripts{cancel: cancel, want: 1, text: "LAMPA"},
		Clock:       func() time.Time { return base },
	})
	if err != nil {
		t.Fatal(err)
	}

	session, err := NewSession(prepared)
	if err != nil {
		t.Fatal(err)
	}
	if err := session.Start(ctx); err != nil {
		t.Fatal(err)
	}
	select {
	case <-ctx.Done():
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for filtered chunk to be processed")
	}
	stopSession(t, session)

	content, err := os.ReadFile(filepath.Join(logDir, "transcribe.log"))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(content), "filtered") {
		t.Fatalf("expected filtered chunk to be logged, got:\n%s", string(content))
	}
	if got := session.Snapshot().Transcript; len(got) != 0 {
		t.Fatalf("filtered chunk should not be stored, got %+v", got)
	}
}

func TestRunSilentWritesOnlyFinalTranscriptToStdout(t *testing.T) {
	base := time.Date(2026, 5, 27, 16, 0, 0, 0, time.UTC)
	cfg := config.Defaults()
	cfg.UserName = "You"
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	segmenterFlushConfig(&cfg, 50*time.Millisecond)

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	recorder := &oneChunkPerSourceRecorder{base: base}
	transcriber := &cancelAfterTranscripts{cancel: cancel, want: 2}
	prepared, err := Prepare(ctx, cfg, filepath.Join(t.TempDir(), "config.yaml"), "test-key", Dependencies{
		Discoverer: fakeDiscoverer{devices: []audio.Device{
			{ID: "mic", Name: "QA Mic", Source: audio.SourceMic, Default: true, Backend: "fake"},
			{ID: "out", Name: "QA Output", Source: audio.SourceOutput, Default: true, Backend: "fake"},
		}},
		Recorder:    recorder,
		Speaker:     fixedSpeaker{current: "Alice"},
		Transcriber: transcriber,
		Clock:       func() time.Time { return base },
	})
	if err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	if err := RunSilent(ctx, prepared, &stdout, &stderr); err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("RunSilent returned error: %v", err)
	}

	got := stdout.String()
	want := "You: mic transcript\n\nAlice: output transcript\n\n"
	if got != want {
		t.Fatalf("silent stdout mismatch\ngot:  %q\nwant: %q\nstderr: %q", got, want, stderr.String())
	}
	for _, forbidden := range []string{"microphone:", "output capture:", "transcript file:", "# Transcription Log", "Created:"} {
		if strings.Contains(got, forbidden) {
			t.Fatalf("silent stdout leaked diagnostic/metadata %q in %q", forbidden, got)
		}
	}
	if !strings.Contains(stderr.String(), "microphone: QA Mic") || !strings.Contains(stderr.String(), "output capture: QA Output") {
		t.Fatalf("silent diagnostics should go to stderr, got %q", stderr.String())
	}
}

type countingDiscoverer struct {
	calls atomic.Int32
}

func (d *countingDiscoverer) ListDevices(context.Context) ([]audio.Device, []string, error) {
	d.calls.Add(1)
	return nil, nil, errors.New("should not discover")
}

type fakeDiscoverer struct {
	devices  []audio.Device
	warnings []string
	err      error
}

func (d fakeDiscoverer) ListDevices(context.Context) ([]audio.Device, []string, error) {
	return d.devices, d.warnings, d.err
}

type oneChunkPerSourceRecorder struct {
	mu   sync.Mutex
	seen map[audio.Source]int
	base time.Time
}

func (r *oneChunkPerSourceRecorder) Validate(context.Context, audio.Selection) error {
	return nil
}

func (r *oneChunkPerSourceRecorder) RecordChunk(ctx context.Context, request audio.ChunkRequest) (audio.Chunk, error) {
	r.mu.Lock()
	if r.seen == nil {
		r.seen = map[audio.Source]int{}
	}
	r.seen[request.Source]++
	seen := r.seen[request.Source]
	r.mu.Unlock()

	if seen > 1 {
		<-ctx.Done()
		return audio.Chunk{}, ctx.Err()
	}

	if err := os.MkdirAll(request.TempDir, 0o700); err != nil {
		return audio.Chunk{}, err
	}
	filePath := filepath.Join(request.TempDir, fmt.Sprintf("%s-%d.wav", request.Source, seen))
	if err := writeLoudWAV(filePath, int(request.Duration.Milliseconds())); err != nil {
		return audio.Chunk{}, err
	}
	started := r.base
	if request.Source == audio.SourceOutput {
		started = started.Add(time.Second)
	}
	return audio.Chunk{
		Source:   request.Source,
		Device:   request.Device,
		FilePath: filePath,
		Started:  started,
		Ended:    started.Add(50 * time.Millisecond),
		Sequence: request.Sequence,
		Speaker:  request.Speaker,
	}, nil
}

type cancelAfterTranscripts struct {
	cancel context.CancelFunc
	want   int32
	text   string
	calls  atomic.Int32
}

func (t *cancelAfterTranscripts) Transcribe(_ context.Context, filePath string, _ openai.Options) (string, error) {
	defer func() {
		if t.calls.Add(1) >= t.want {
			t.cancel()
		}
	}()
	if t.text != "" {
		return t.text, nil
	}
	if strings.Contains(filepath.Base(filePath), string(audio.SourceMic)) {
		return "mic transcript", nil
	}
	return "output transcript", nil
}

type fixedSpeaker struct {
	current string
}

func (s fixedSpeaker) Start(context.Context) <-chan speaker.Event {
	events := make(chan speaker.Event)
	close(events)
	return events
}

func (s fixedSpeaker) Current() string { return s.current }

func (s fixedSpeaker) Close() error { return nil }

type blockingRecorder struct {
	started  chan audio.ChunkRequest
	canceled chan audio.ChunkRequest
}

func newBlockingRecorder() *blockingRecorder {
	return &blockingRecorder{
		started:  make(chan audio.ChunkRequest, 16),
		canceled: make(chan audio.ChunkRequest, 16),
	}
}

func (r *blockingRecorder) Validate(context.Context, audio.Selection) error { return nil }

func (r *blockingRecorder) RecordChunk(ctx context.Context, request audio.ChunkRequest) (audio.Chunk, error) {
	select {
	case r.started <- request:
	case <-ctx.Done():
		return audio.Chunk{}, ctx.Err()
	}
	<-ctx.Done()
	select {
	case r.canceled <- request:
	default:
	}
	return audio.Chunk{}, ctx.Err()
}

type fallbackRecorder struct {
	mu       sync.Mutex
	base     time.Time
	attempts map[string]int
	requests []audio.ChunkRequest
}

func newFallbackRecorder(base time.Time) *fallbackRecorder {
	return &fallbackRecorder{base: base, attempts: map[string]int{}}
}

func (r *fallbackRecorder) Validate(context.Context, audio.Selection) error { return nil }

func (r *fallbackRecorder) RecordChunk(ctx context.Context, request audio.ChunkRequest) (audio.Chunk, error) {
	r.mu.Lock()
	if r.attempts == nil {
		r.attempts = map[string]int{}
	}
	r.requests = append(r.requests, request)
	key := string(request.Source) + "|" + request.Device.ID
	attempt := r.attempts[key]
	r.attempts[key] = attempt + 1
	r.mu.Unlock()

	if request.Source == audio.SourceOutput && request.Device.ID == "out-a" && attempt == 0 {
		return audio.Chunk{}, errors.New("ffmpeg failed to open output device out-a")
	}
	if attempt > 0 {
		<-ctx.Done()
		return audio.Chunk{}, ctx.Err()
	}

	if err := os.MkdirAll(request.TempDir, 0o700); err != nil {
		return audio.Chunk{}, err
	}
	filePath := filepath.Join(request.TempDir, fmt.Sprintf("transcribe-%s-%s-%d.wav", request.Source, request.Device.ID, attempt+1))
	if err := writeLoudWAV(filePath, int(request.Duration.Milliseconds())); err != nil {
		return audio.Chunk{}, err
	}
	started := r.base
	if request.Source == audio.SourceOutput {
		started = started.Add(time.Second)
	}
	return audio.Chunk{
		Source:   request.Source,
		Device:   request.Device,
		FilePath: filePath,
		Started:  started,
		Ended:    started.Add(50 * time.Millisecond),
		Sequence: request.Sequence,
		Speaker:  request.Speaker,
	}, nil
}

func (r *fallbackRecorder) RecordedRequests() []audio.ChunkRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]audio.ChunkRequest(nil), r.requests...)
}

type outputFailureRecorder struct {
	mu             sync.Mutex
	once           sync.Once
	micRequests    int
	outputRequests int
	failed         chan struct{}
}

// newOutputFailureRecorder returns the recorder and its failure signal. The
// returned channel is closed exactly once when the first output capture chunk
// fails; callers must read the returned channel (which never races with an
// internal re-assignment) rather than the r.failed field.
func newOutputFailureRecorder() (*outputFailureRecorder, <-chan struct{}) {
	failed := make(chan struct{})
	return &outputFailureRecorder{failed: failed}, failed
}

func (r *outputFailureRecorder) Validate(context.Context, audio.Selection) error { return nil }

func (r *outputFailureRecorder) RecordChunk(ctx context.Context, request audio.ChunkRequest) (audio.Chunk, error) {
	if request.Source == audio.SourceOutput {
		r.mu.Lock()
		r.outputRequests++
		r.mu.Unlock()
		// Close the failure signal exactly once. sync.Once makes the close
		// race-free even though RecordChunk runs on the session goroutine while
		// the test goroutine reads the immutable channel reference.
		r.once.Do(func() { close(r.failed) })
		return audio.Chunk{}, errors.New("ffmpeg failed for output device")
	}

	r.mu.Lock()
	r.micRequests++
	count := r.micRequests
	r.mu.Unlock()

	if err := os.MkdirAll(request.TempDir, 0o700); err != nil {
		return audio.Chunk{}, err
	}
	filePath := filepath.Join(request.TempDir, fmt.Sprintf("mic-%d.wav", count))
	if err := writeLoudWAV(filePath, int(request.Duration.Milliseconds())); err != nil {
		return audio.Chunk{}, err
	}
	started := time.Now()
	return audio.Chunk{
		Source:   request.Source,
		Device:   request.Device,
		FilePath: filePath,
		Started:  started,
		Ended:    started.Add(50 * time.Millisecond),
		Sequence: request.Sequence,
		Speaker:  request.Speaker,
	}, nil
}

func (r *outputFailureRecorder) counts() (int, int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.micRequests, r.outputRequests
}

func prepareTestSession(t *testing.T, cfg config.Config, configPath string, recorder audio.Recorder) *Prepared {
	t.Helper()
	prepared, err := Prepare(context.Background(), cfg, configPath, "test-key", Dependencies{
		Discoverer: fakeDiscoverer{devices: []audio.Device{
			{ID: "mic", Name: "QA Mic", Source: audio.SourceMic, Default: true, Backend: "fake"},
			{ID: "out", Name: "QA Output", Source: audio.SourceOutput, Default: true, Backend: "fake"},
		}},
		Recorder:    recorder,
		Speaker:     fixedSpeaker{current: "Alice"},
		Transcriber: &cancelAfterTranscripts{want: 1000},
		Clock:       time.Now,
	})
	if err != nil {
		t.Fatal(err)
	}
	return prepared
}

func waitForSources(t *testing.T, ch <-chan audio.ChunkRequest, sources ...audio.Source) map[audio.Source]audio.ChunkRequest {
	t.Helper()
	want := make(map[audio.Source]bool, len(sources))
	found := make(map[audio.Source]audio.ChunkRequest, len(sources))
	for _, source := range sources {
		want[source] = true
	}
	deadline := time.After(2 * time.Second)
	for len(want) > 0 {
		select {
		case request := <-ch:
			if want[request.Source] {
				found[request.Source] = request
				delete(want, request.Source)
			}
		case <-deadline:
			t.Fatalf("timed out waiting for sources %v", want)
		}
	}
	return found
}

func waitForRequest(t *testing.T, ch <-chan audio.ChunkRequest, source audio.Source) audio.ChunkRequest {
	t.Helper()
	deadline := time.After(2 * time.Second)
	for {
		select {
		case request := <-ch:
			if request.Source == source {
				return request
			}
		case <-deadline:
			t.Fatalf("timed out waiting for %s request", source)
		}
	}
}

func stopSession(t *testing.T, session *Session) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := session.Stop(ctx); err != nil {
		t.Fatalf("stop session: %v", err)
	}
}

func drainEvents(ch <-chan Event) []Event {
	var events []Event
	for {
		select {
		case event := <-ch:
			events = append(events, event)
		default:
			return events
		}
	}
}

func hasWarningContaining(events []Event, substr string) bool {
	for _, event := range events {
		if event.Kind == EventWarning && strings.Contains(event.Message, substr) {
			return true
		}
	}
	return false
}

func transcriptContainsSource(events []transcript.Event, source string) bool {
	for _, event := range events {
		if event.Source == source {
			return true
		}
	}
	return false
}

func outputRequestOrder(requests []audio.ChunkRequest) []string {
	var order []string
	for _, request := range requests {
		if request.Source == audio.SourceOutput {
			order = append(order, request.Device.ID)
		}
	}
	return order
}

func testPCM16WAV(samples []int16) []byte {
	var data bytes.Buffer
	for _, sample := range samples {
		_ = binary.Write(&data, binary.LittleEndian, sample)
	}
	dataSize := uint32(data.Len())

	var wav bytes.Buffer
	wav.WriteString("RIFF")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(36)+dataSize)
	wav.WriteString("WAVE")
	wav.WriteString("fmt ")
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(1))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(1))
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16000))
	_ = binary.Write(&wav, binary.LittleEndian, uint32(16000*2))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(2))
	_ = binary.Write(&wav, binary.LittleEndian, uint16(16))
	wav.WriteString("data")
	_ = binary.Write(&wav, binary.LittleEndian, dataSize)
	wav.Write(data.Bytes())
	return wav.Bytes()
}

// loudTestWAV writes a PCM16 16kHz mono WAV to dest whose contents are full-
// amplitude alternating samples for roughly durationMs milliseconds. It is the
// realistic-speech fixture used by recorders in session-level tests so the
// segmenter's energy detector actually recognizes the frame as speech (the old
// 4-sample fixtures were shorter than one analysis frame and got discarded by
// the silence gate). The caller owns cleanup of the returned path.
func loudTestWAV(t *testing.T, dest string, durationMs int) {
	t.Helper()
	if err := writeLoudWAV(dest, durationMs); err != nil {
		t.Fatalf("write loud WAV: %v", err)
	}
}

// writeLoudWAV is the non-test helper recorders use (they don't carry a
// *testing.T). It writes the same realistic-speech fixture and returns an
// error the recorder surfaces to the session.
func writeLoudWAV(dest string, durationMs int) error {
	if durationMs <= 0 {
		durationMs = 150
	}
	sampleCount := 16000 * durationMs / 1000
	samples := make([]int16, sampleCount)
	for i := range samples {
		if i%2 == 0 {
			samples[i] = 32767
		} else {
			samples[i] = -32767
		}
	}
	if err := os.MkdirAll(filepath.Dir(dest), 0o700); err != nil {
		return err
	}
	out, err := os.OpenFile(dest, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return err
	}
	defer out.Close()
	return audio.WritePCM16WAV(out, samples, audio.ConcatSampleRateHz())
}

// segmenterFlushConfig returns the audio config tweaks a test needs so a single
// recorded loud chunk is immediately flushed by the segmenter (rather than held
// forever waiting for trailing silence). Tests set CaptureChunkDuration /
// MaxSegmentDuration / FrameDurationMS to these values when they want the legacy
// "one chunk == one transcription" timing without waiting for real silence.
func segmenterFlushConfig(cfg *config.Config, captureChunkDuration time.Duration) {
	cfg.Audio.CaptureChunkDuration = config.NewDuration(captureChunkDuration)
	// Small analysis frames so even short capture chunks yield at least one
	// complete frame; must be <= CaptureChunkDuration.
	cfg.Audio.FrameDurationMS = 20
	// The capture duration is capped by MaxSegmentDuration, so make the cap
	// exactly one complete analysis frame. The resulting request is 20 ms and
	// the first loud frame reaches the cap deterministically.
	cfg.Audio.MaxSegmentDuration = config.NewDuration(20 * time.Millisecond)
	// Disable the trailing-silence rule in tests; we flush via max-duration.
	cfg.Audio.SilenceDuration = config.NewDuration(time.Hour)
	cfg.Audio.SilenceThreshold = 0.02
}
