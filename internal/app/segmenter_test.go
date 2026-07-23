package app

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
)

// newTestSegmenter builds a Segmenter with realistic defaults for tests:
// 100ms frames, relaxed silence threshold, short min-segment so a single loud
// frame can already trigger a silence-driven flush.
func newTestSegmenter(t *testing.T, cfg SegmentConfig) (*Segmenter, string) {
	t.Helper()
	tempDir := t.TempDir()
	cfg.TempDir = tempDir
	if cfg.FrameDurationMS == 0 {
		cfg.FrameDurationMS = 100
	}
	if cfg.SampleRateHz == 0 {
		cfg.SampleRateHz = audio.ConcatSampleRateHz()
	}
	if cfg.Channels == 0 {
		cfg.Channels = audio.ConcatChannels()
	}
	if cfg.SilenceThreshold == 0 {
		// ~1% RMS; comfortably above digitized near-silence.
		cfg.SilenceThreshold = 0.02
	}
	if cfg.SilenceDuration == 0 {
		cfg.SilenceDuration = 200 * time.Millisecond
	}
	if cfg.MinSegmentDuration == 0 {
		cfg.MinSegmentDuration = 100 * time.Millisecond
	}
	if cfg.MaxSegmentDuration == 0 {
		cfg.MaxSegmentDuration = 5 * time.Minute
	}
	return NewSegmenter(cfg), tempDir
}

// writeFramesWAV writes a PCM16 16kHz mono WAV file whose contents are the
// concatenation of the given per-frame amplitudes. Each entry produces one
// frameDurationMS-long slice of samples at the given amplitude (0=silence,
// 1.0=full-scale). Useful for constructing deterministic loud/silent frames.
func writeFramesWAV(t *testing.T, path string, frames []float64, frameDurationMS int) {
	t.Helper()
	sampleRate := int(audio.ConcatSampleRateHz())
	samplesPerFrame := sampleRate * frameDurationMS / 1000
	var samples []int16
	for _, amp := range frames {
		amp := amp
		if amp < 0 {
			amp = 0
		}
		if amp > 1 {
			amp = 1
		}
		fullScale := int16(amp * 32767)
		for i := 0; i < samplesPerFrame; i++ {
			// alternate sign so RMS matches the amplitude even though it DC-offsets.
			if i%2 == 0 {
				samples = append(samples, fullScale)
			} else {
				samples = append(samples, -fullScale)
			}
		}
	}
	out, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		t.Fatalf("open %s: %v", path, err)
	}
	defer out.Close()
	if err := audio.WritePCM16WAV(out, samples, audio.ConcatSampleRateHz()); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestSegmenterDiscardsSilentFrameBeforeSpeech(t *testing.T) {
	seg, tempDir := newTestSegmenter(t, SegmentConfig{Source: audio.SourceMic})
	silentPath := filepath.Join(tempDir, "silent.wav")
	writeFramesWAV(t, silentPath, []float64{0.0, 0.0}, 100)

	clock := fixedClock(time.UnixMilli(1000))
	dec := seg.Accumulate(silentPath, false, []float64{0.0, 0.0}, clock)
	if !dec.Discarded {
		t.Fatalf("expected first silent frame to be discarded, got %+v", dec)
	}
	if seg.HasPending() {
		t.Fatalf("segmenter should have no pending frames after discarding silence")
	}
	if _, err := os.Stat(silentPath); !os.IsNotExist(err) {
		t.Fatalf("silent WAV should be deleted after discard, stat err=%v", err)
	}
}

func TestSegmenterFlushesWhenSilenceDurationMet(t *testing.T) {
	seg, tempDir := newTestSegmenter(t, SegmentConfig{Source: audio.SourceMic})
	clock := fixedClock(time.UnixMilli(0))

	// Loud frame first (speech starts).
	loudPath := filepath.Join(tempDir, "loud.wav")
	writeFramesWAV(t, loudPath, []float64{1.0, 1.0}, 100)
	if dec := seg.Accumulate(loudPath, true, []float64{0.5, 0.5}, clock); dec.Flush {
		t.Fatalf("loud-only frame should not flush, got %+v", dec)
	}

	// Silence frame with 3 silent frames = 300ms > 200ms silence-duration => flush.
	silentPath := filepath.Join(tempDir, "silent.wav")
	writeFramesWAV(t, silentPath, []float64{0.0, 0.0, 0.0}, 100)
	dec := seg.Accumulate(silentPath, false, []float64{0.0, 0.0, 0.0}, clock)
	if !dec.Flush {
		t.Fatalf("trailing silence >= silence-duration should flush, got %+v", dec)
	}

	segment, ok, err := seg.Flush(time.UnixMilli(5000), "Alice")
	if err != nil {
		t.Fatalf("flush failed: %v", err)
	}
	if !ok {
		t.Fatalf("expected a flushed segment")
	}
	if segment.Speaker != "Alice" {
		t.Fatalf("expected speaker Alice, got %q", segment.Speaker)
	}
	if segment.Frames != 2 {
		t.Fatalf("expected 2 accumulated frames, got %d", segment.Frames)
	}
	if _, err := os.Stat(segment.FilePath); err != nil {
		t.Fatalf("flushed segment WAV should exist, stat err=%v", err)
	}
}

func TestSegmenterFlushesOnMaxSegmentDuration(t *testing.T) {
	seg, tempDir := newTestSegmenter(t, SegmentConfig{
		Source:             audio.SourceMic,
		MaxSegmentDuration: 300 * time.Millisecond,
	})
	clock := fixedClock(time.UnixMilli(0))

	// 4 loud frames = 400ms of audio and speech, exceeds the 300ms cap.
	loudPath := filepath.Join(tempDir, "loud.wav")
	writeFramesWAV(t, loudPath, []float64{1.0, 1.0, 1.0, 1.0}, 100)
	dec := seg.Accumulate(loudPath, true, []float64{0.5, 0.5, 0.5, 0.5}, clock)
	if !dec.Flush {
		t.Fatalf("exceeding max segment duration should flush, got %+v", dec)
	}

	segment, ok, err := seg.Flush(time.UnixMilli(1000), "Bob")
	if err != nil || !ok {
		t.Fatalf("expected successful flush, ok=%v err=%v", ok, err)
	}
	if segment.Frames != 1 {
		t.Fatalf("expected 1 accumulated frame, got %d", segment.Frames)
	}
	if _, err := os.Stat(segment.FilePath); err != nil {
		t.Fatalf("max-duration flushed segment WAV should exist, stat err=%v", err)
	}
}

func TestSegmenterMaxDurationIncludesTrailingSilence(t *testing.T) {
	seg, tempDir := newTestSegmenter(t, SegmentConfig{
		Source:             audio.SourceMic,
		MaxSegmentDuration: 300 * time.Millisecond,
	})
	clock := fixedClock(time.UnixMilli(0))

	// The retained chunk contains only 100ms of speech followed by 200ms of
	// silence. The hard cap is based on all retained audio, not speech alone.
	path := filepath.Join(tempDir, "speech-and-silence.wav")
	writeFramesWAV(t, path, []float64{1.0, 0.0, 0.0}, 100)
	dec := seg.Accumulate(path, true, []float64{0.5, 0.0, 0.0}, clock)
	if !dec.Flush {
		t.Fatalf("total retained audio should trigger max duration, got %+v", dec)
	}
	if dec.ElapsedSpeech != 100*time.Millisecond {
		t.Fatalf("expected 100ms speech, got %s", dec.ElapsedSpeech)
	}
	if dec.ElapsedAudio != 300*time.Millisecond {
		t.Fatalf("expected 300ms total audio, got %s", dec.ElapsedAudio)
	}
}

func TestSegmenterResetClearsPendingAndDeletesWAVs(t *testing.T) {
	seg, tempDir := newTestSegmenter(t, SegmentConfig{Source: audio.SourceMic})
	clock := fixedClock(time.UnixMilli(0))

	loudPath := filepath.Join(tempDir, "loud.wav")
	writeFramesWAV(t, loudPath, []float64{1.0, 1.0}, 100)
	_ = seg.Accumulate(loudPath, true, []float64{0.5, 0.5}, clock)
	if !seg.HasPending() {
		t.Fatalf("expected pending frame after accumulate")
	}

	seg.Reset()
	if seg.HasPending() {
		t.Fatalf("reset should clear pending frames")
	}
	if _, err := os.Stat(loudPath); !os.IsNotExist(err) {
		t.Fatalf("reset should delete pending WAV frames, stat err=%v", err)
	}
}

func TestSegmenterFlushConcatenatesMultipleFrames(t *testing.T) {
	seg, tempDir := newTestSegmenter(t, SegmentConfig{Source: audio.SourceOutput})
	clock := fixedClock(time.UnixMilli(0))

	// Three loud frames then enough silence to flush.
	p1 := filepath.Join(tempDir, "loud1.wav")
	writeFramesWAV(t, p1, []float64{1.0, 1.0}, 100)
	_ = seg.Accumulate(p1, true, []float64{0.5, 0.5}, clock)
	p2 := filepath.Join(tempDir, "loud2.wav")
	writeFramesWAV(t, p2, []float64{1.0, 1.0}, 100)
	_ = seg.Accumulate(p2, true, []float64{0.5, 0.5}, clock)
	p3 := filepath.Join(tempDir, "loud3.wav")
	writeFramesWAV(t, p3, []float64{1.0, 1.0, 0.0, 0.0, 0.0}, 100)
	// Two loud frames + 3 silent frames (300ms silence) => flush.
	dec := seg.Accumulate(p3, true, []float64{0.5, 0.5, 0.0, 0.0, 0.0}, clock)
	if !dec.Flush {
		t.Fatalf("combined chunk should trigger flush via trailing silence, got %+v", dec)
	}

	segment, ok, err := seg.Flush(time.UnixMilli(2000), "Alice")
	if err != nil || !ok {
		t.Fatalf("flush failed: ok=%v err=%v", ok, err)
	}
	if segment.Frames != 3 {
		t.Fatalf("flushed segment should contain 3 frames, got %d", segment.Frames)
	}

	// Concatenated WAV should be readable and contain all three frames' samples.
	file, err := os.Open(segment.FilePath)
	if err != nil {
		t.Fatalf("open concatenated WAV: %v", err)
	}
	samples, err := audio.PCM16SamplesFromWAV(file)
	file.Close()
	if err != nil {
		t.Fatalf("read concatenated WAV: %v", err)
	}
	// 2+2+5 = 9 frames * 1600 samples/frame = 14400 samples.
	if len(samples) != 14400 {
		t.Fatalf("expected 14400 concatenated samples, got %d", len(samples))
	}

	// Per-frame source WAVs should be cleaned up after flush.
	for _, p := range []string{p1, p2, p3} {
		if _, err := os.Stat(p); !os.IsNotExist(err) {
			t.Fatalf("source frame %s should be deleted after flush, stat err=%v", p, err)
		}
	}
}

func TestSegmenterHoldsWhenSpeechBelowMinSegmentDuration(t *testing.T) {
	seg, tempDir := newTestSegmenter(t, SegmentConfig{
		Source:             audio.SourceMic,
		MinSegmentDuration: 1 * time.Second,
	})
	clock := fixedClock(time.UnixMilli(0))

	// One loud frame (100ms) then trailing silence: below 1s min, should hold.
	loudPath := filepath.Join(tempDir, "loud.wav")
	writeFramesWAV(t, loudPath, []float64{1.0, 0.0, 0.0, 0.0}, 100)
	dec := seg.Accumulate(loudPath, true, []float64{0.5, 0.0, 0.0, 0.0}, clock)
	if dec.Flush {
		t.Fatalf("speech below min segment duration should not flush even with trailing silence, got %+v", dec)
	}
}

func fixedClock(at time.Time) func() time.Time {
	return func() time.Time { return at }
}
