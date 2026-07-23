package app

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/audio"
	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/logging"
)

// SegmentConfig configures how a Segmenter accumulates recorded PCM frames
// into transcription-ready segments. It mirrors the silence-gating behaviour
// of the original Python recorder.
type SegmentConfig struct {
	// Source identifies which capture stream this segmenter serves
	// (microphone vs system output).
	Source audio.Source
	// TempDir is where concatenated segment WAV files are written. It is
	// created on demand and owned by the segmenter (files are removed after
	// transcription succeeds or the segmenter is reset).
	TempDir string
	// FrameDurationMS is the analysis window used to classify silence vs
	// speech. Typical default 100ms (10 frames per second).
	FrameDurationMS int
	// SampleRateHz is the canonical PCM sample rate written by the recorder
	// (16000 Hz). It must match what FramesWithEnergy expects.
	SampleRateHz uint32
	// Channels is the canonical channel count written by the recorder (1).
	Channels uint16
	// SilenceThreshold is the normalized [0, 1] RMS level at or above which a
	// frame is considered speech. Legacy configs expressed this as a 0-255
	// amplitude scalar; the caller is responsible for scaling that into the
	// normalized domain before constructing the config.
	SilenceThreshold float64
	// SilenceDuration is how much contiguous silence (trailing silent frames)
	// ends a segment once speech has started. e.g. 2s ends a segment after a
	// 2-second pause between sentences.
	SilenceDuration time.Duration
	// MinSegmentDuration is the minimum amount of speech (active audio) that
	// must accumulate before a trailing silence is allowed to end a segment.
	// This prevents isolated clicks/coughs in a quiet room from producing
	// noisy one-word segments.
	MinSegmentDuration time.Duration
	// MaxSegmentDuration bounds the length of a single segment regardless of
	// speech/silence activity. When reached, the segment is flushed. Typical
	// default 15 minutes, well under OpenAI's 25 MB file size cap.
	MaxSegmentDuration time.Duration
}

// Segment is a complete, ready-to-transcribe utterance produced by a
// Segmenter. FilePath is a temporary WAV the segmenter concatenated from one
// or more recorded frames; the caller owns cleanup of that file.
type Segment struct {
	FilePath string
	Started  time.Time
	Ended    time.Time
	Speaker  string
	Frames   int
}

// SegmentDecision explains why a Segmenter decided (or did not decide) to
// flush the current accumulator. Useful for tests and diagnostics.
type SegmentDecision struct {
	// Flush is true when the segment should be flushed to disk now. The caller
	// invokes Flush to materialize the WAV; the segmenter itself never writes
	// a concatenated WAV during Accumulate so flush timing is caller-owned.
	Flush bool
	// Reason is a human-readable explanation (silence reached, max duration,
	// discarded, holding, etc.).
	Reason string
	// ElapsedSpeech is the accumulated duration of active audio in the segment.
	ElapsedSpeech time.Duration
	// ElapsedAudio is the accumulated wall-clock duration of all retained audio,
	// including trailing silence. This is the metric used for the hard maximum
	// segment duration so a long pause cannot make a segment exceed the cap.
	ElapsedAudio time.Duration
	// TrailingSilent is the duration of trailing silent frames in the segment.
	TrailingSilent time.Duration
	// Discarded reports that the current frame was thrown away (silent and no
	// speech yet started).
	Discarded bool
}

// Segmenter accumulates recorded PCM frames into speech segments. Each frame
// is analyzed for energy and frames are kept until one of the flush triggers
// fires (silence after speech, max duration reached). Silent frames before
// any speech are discarded so the recorder does not pay OpenAI to transcribe
// room tone — which historically self-amplified the keyword prompt and produced
// repeated keyword hallucinations.
//
// Speaker-change handling is intentionally caller-owned: the source loop calls
// Flush (with the previous speaker) before calling Accumulate with the new
// speaker. This keeps the segmenter's state machine focused on silence/max
// duration rules and makes flush timing deterministic.
//
// Segmenter is safe for use from a single goroutine (the source capture loop
// that owns it). Flush is synchronous and performs WAV concatenation + disk
// I/O before returning.
type Segmenter struct {
	cfg SegmentConfig

	mu                sync.Mutex
	pendingPaths      []string
	pendingSpeechTime time.Duration
	pendingAudioTime  time.Duration
	startedAt         time.Time
	lastEnergy        []float64
	sequence          uint64
}

// NewSegmenter returns a Segmenter configured for the given source. The
// segmenter does not own the recorder; the caller still records individual
// PCM frames with the existing audio.Recorder and feeds their parsed energy
// into the segmenter via Accumulate.
func NewSegmenter(cfg SegmentConfig) *Segmenter {
	return &Segmenter{cfg: cfg}
}

// Reset drops any accumulated-but-unflushed frames without flushing them and
// deletes the underlying frame WAVs. Used when capture is paused/muted/
// switched mid-segment and the partial speech is not worth transcribing.
// Prefer Flush when speech has accumulated and should be preserved.
func (s *Segmenter) Reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.releasePendingLocked()
}

// HasPending reports whether the segmenter is holding frames that have not yet
// been flushed. Speaker changes use this to decide whether a flush is needed.
func (s *Segmenter) HasPending() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.pendingPaths) > 0
}

// PendingSpeechDuration returns the accumulated duration of active speech in
// the current (un-flushed) segment. Used by the session to decide whether a
// pending segment should be flushed on pause/stop.
func (s *Segmenter) PendingSpeechDuration() time.Duration {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.pendingSpeechTime
}

// releasePendingLocked deletes any on-disk frame files the segmenter has not
// yet merged into a sentinel WAV. The caller must hold s.mu.
func (s *Segmenter) releasePendingLocked() {
	for _, path := range s.pendingPaths {
		_ = os.Remove(path)
	}
	s.pendingPaths = nil
	s.pendingSpeechTime = 0
	s.pendingAudioTime = 0
	s.startedAt = time.Time{}
	s.lastEnergy = nil
}

// framesPerSecond derives the analysis cadence from FrameDurationMS. 100ms -> 10fps.
func (s *Segmenter) framesPerSecond() int {
	if s.cfg.FrameDurationMS <= 0 {
		return 10
	}
	return 1000 / s.cfg.FrameDurationMS
}

// frameDuration returns the analysis window as a time.Duration with a safe
// 100ms default when FrameDurationMS is misconfigured.
func (s *Segmenter) frameDuration() time.Duration {
	if s.cfg.FrameDurationMS <= 0 {
		return 100 * time.Millisecond
	}
	return time.Duration(s.cfg.FrameDurationMS) * time.Millisecond
}

// IsLoud decodes the chunk WAV file once into PCM samples and returns whether
// any analysis frame meets the silence threshold, along with the per-frame
// RMS series so callers can reuse it without re-reading disk.
func (s *Segmenter) IsLoud(filePath string) (loud bool, energy []float64, err error) {
	file, err := os.Open(filePath)
	if err != nil {
		return false, nil, err
	}
	defer file.Close()
	samples, err := audio.PCM16SamplesFromWAV(file)
	if err != nil {
		return false, nil, err
	}
	energy = audio.FramesWithEnergy(samples, s.cfg.SampleRateHz, s.cfg.Channels, s.framesPerSecond())
	if len(energy) == 0 {
		return false, energy, nil
	}
	for _, frame := range energy {
		if frame >= s.cfg.SilenceThreshold {
			return true, energy, nil
		}
	}
	return false, energy, nil
}

// speechTimeInChunk estimates how much of the chunk is active speech from its
// per-frame energy series. Loud frames count toward pending speech duration;
// silent frames do not. This is the metric used for MinSegmentDuration gating;
// the hard maximum uses total retained audio instead.
func (s *Segmenter) speechTimeInChunk(energy []float64) time.Duration {
	if len(energy) == 0 {
		return 0
	}
	frameDuration := s.frameDuration()
	loud := 0
	for _, level := range energy {
		if level >= s.cfg.SilenceThreshold {
			loud++
		}
	}
	return time.Duration(loud) * frameDuration
}

// audioTimeInChunk estimates the wall-clock duration represented by the
// complete analysis frames in a recorded WAV. Recorder chunks are aligned to
// the configured frame duration in normal operation, so this is exact for the
// canonical capture path and deliberately ignores any incomplete tail frame.
func (s *Segmenter) audioTimeInChunk(energy []float64) time.Duration {
	return time.Duration(len(energy)) * s.frameDuration()
}

// Accumulate evaluates a freshly recorded frame against the segmenter rules
// and returns a SegmentDecision describing whether the current segment should
// flush now. The caller invokes Flush when Decision.Flush is true.
//
//   - filePath is the per-frame WAV produced by the recorder.
//   - loud/energy come from IsLoud; passing them in avoids re-reading the file.
//   - clock returns the wall time the frame ended (anchors segment StartedAt).
//
// Silent frames are discarded entirely (and the WAV deleted) until the first
// loud frame arrives; after speech starts, frames are kept until silence or
// the max duration fires. Speaker changes are NOT handled here — the caller
// flushes before Accumulating under a new speaker.
func (s *Segmenter) Accumulate(filePath string, loud bool, energy []float64, clock func() time.Time) SegmentDecision {
	s.mu.Lock()
	defer s.mu.Unlock()

	frameDuration := s.frameDuration()

	// Drop silent frames before any speech has started — do not pay OpenAI to
	// transcribe silence, which historically hallucinated keyword repeats.
	if !loud && len(s.pendingPaths) == 0 {
		_ = os.Remove(filePath)
		return SegmentDecision{Reason: "silent frame discarded before speech started", Discarded: true}
	}

	if s.startedAt.IsZero() {
		s.startedAt = clock()
	}
	s.pendingPaths = append(s.pendingPaths, filePath)
	s.lastEnergy = energy
	s.pendingSpeechTime += s.speechTimeInChunk(energy)
	s.pendingAudioTime += s.audioTimeInChunk(energy)

	return s.evaluateLocked(frameDuration)
}

// evaluateLocked applies the silence / max-duration rules to the current
// accumulator and returns a SegmentDecision. The caller holds s.mu.
func (s *Segmenter) evaluateLocked(frameDuration time.Duration) SegmentDecision {
	dec := SegmentDecision{ElapsedSpeech: s.pendingSpeechTime, ElapsedAudio: s.pendingAudioTime}

	if s.cfg.MaxSegmentDuration > 0 && s.pendingAudioTime >= s.cfg.MaxSegmentDuration {
		dec.Flush = true
		dec.Reason = fmt.Sprintf("max segment duration %s reached (audio %s, speech %s)", s.cfg.MaxSegmentDuration, s.pendingAudioTime, s.pendingSpeechTime)
		return dec
	}

	// Need at least MinSegmentDuration of speech before a trailing silence is
	// allowed to end the segment, otherwise brief clicks fragment it.
	if s.cfg.MinSegmentDuration > 0 && s.pendingSpeechTime < s.cfg.MinSegmentDuration {
		dec.Reason = fmt.Sprintf("holding (speech %s below min %s)", s.pendingSpeechTime, s.cfg.MinSegmentDuration)
		return dec
	}

	if s.cfg.SilenceDuration <= 0 {
		dec.Reason = "holding (silence-duration disabled)"
		return dec
	}

	silentFrames := audio.TrailingSilentFrames(s.lastEnergy, s.cfg.SilenceThreshold)
	trailing := time.Duration(silentFrames) * frameDuration
	dec.TrailingSilent = trailing
	if trailing >= s.cfg.SilenceDuration {
		dec.Flush = true
		dec.Reason = fmt.Sprintf("silence %s >= %s after speech %s", trailing, s.cfg.SilenceDuration, s.pendingSpeechTime)
		return dec
	}
	dec.Reason = fmt.Sprintf("holding (trailing silence %s below %s)", trailing, s.cfg.SilenceDuration)
	return dec
}

// flushLocked performs the on-disk WAV concatenation of all accumulated frame
// paths and returns the resulting Segment. The caller holds s.mu.
func (s *Segmenter) flushLocked(now time.Time, speaker string) (Segment, error) {
	if len(s.pendingPaths) == 0 {
		return Segment{}, nil
	}
	s.sequence++
	destination := filepath.Join(s.cfg.TempDir, fmt.Sprintf("segment-%s-%06d.wav", s.cfg.Source, s.sequence))
	if err := audio.ConcatenatePCM16WAV(s.pendingPaths, destination); err != nil {
		_ = os.Remove(destination)
		// Even on failure we must release the pending frames or we leak files.
		s.releasePendingLocked()
		return Segment{}, err
	}
	// Remove the per-frame source WAVs now that they are merged.
	for _, path := range s.pendingPaths {
		_ = os.Remove(path)
	}
	started := s.startedAt
	if started.IsZero() {
		started = now
	}
	segment := Segment{
		FilePath: destination,
		Started:  started,
		Ended:    now,
		Speaker:  speaker,
		Frames:   len(s.pendingPaths),
	}
	frames := segment.Frames
	s.pendingPaths = nil
	s.pendingSpeechTime = 0
	s.pendingAudioTime = 0
	s.startedAt = time.Time{}
	s.lastEnergy = nil
	logging.Printf("%s segmenter: flushed segment %s (frames=%d speaker=%q)", s.cfg.Source, destination, frames, speaker)
	return segment, nil
}

// Flush finalizes any pending frames into a single Segment and returns it.
// ok=false means the segmenter was empty (no flush happened). The caller is
// responsible for transcribing and then deleting the returned FilePath.
func (s *Segmenter) Flush(now time.Time, speaker string) (Segment, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.pendingPaths) == 0 {
		return Segment{}, false, nil
	}
	segment, err := s.flushLocked(now, speaker)
	if err != nil {
		return Segment{}, false, err
	}
	return segment, true, nil
}
