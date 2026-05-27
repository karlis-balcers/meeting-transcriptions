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
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/openai"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/speaker"
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

func TestRunSilentWritesOnlyFinalTranscriptToStdout(t *testing.T) {
	base := time.Date(2026, 5, 27, 16, 0, 0, 0, time.UTC)
	cfg := config.Defaults()
	cfg.UserName = "You"
	cfg.Paths.OutputDir = t.TempDir()
	cfg.Paths.TempDir = t.TempDir()
	cfg.Audio.MaxSegmentDuration = config.NewDuration(50 * time.Millisecond)

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
	if err := os.WriteFile(filePath, testPCM16WAV([]int16{0, 1000, -1000, 0}), 0o600); err != nil {
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
	calls  atomic.Int32
}

func (t *cancelAfterTranscripts) Transcribe(_ context.Context, filePath string, _ openai.Options) (string, error) {
	defer func() {
		if t.calls.Add(1) >= t.want {
			t.cancel()
		}
	}()
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
