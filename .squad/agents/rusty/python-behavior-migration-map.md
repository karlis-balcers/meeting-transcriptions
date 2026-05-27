# Python Behavior Migration Map for Go TUI Rebuild

**Author:** Rusty  
**Date:** 2026-05-27T15:57:08Z  
**Scope:** Core Python behavior relevant to audio capture, chunking, transcription queueing, transcript storage, config/environment handling, speaker detection, pause/mute semantics, and intentionally droppable behavior. No code was implemented.

## Source Modules Reviewed

- `app.py` delegates directly to `transcribe.main()`.
- `transcribe.py` owns app lifecycle, global runtime state, Tkinter wiring, queue orchestration, start/stop, settings persistence, and assistant integration points.
- `audio_capture.py` owns device cataloging, capture loops, silence/manual/speaker-change chunking, WAV writing, and temp-file cleanup.
- `openai_transcribe.py` owns OpenAI transcription calls, retry/backoff, prompt keywords, and transcription artifact suppression.
- `speaker_detection.py` owns Windows-only Teams speaker/title scraping.
- `transcript_store.py` owns thread-safe in-memory transcript ordering and Markdown append persistence.
- `transcript_filter.py` owns configurable text artifact filtering.
- `env_utils.py`, `env.sample`, and `README.md` define reusable config semantics and documented user behavior.
- Tests confirm device-selection, environment parsing, transcript filtering, summary filename, and assistant queue/concurrency behavior.

## Current Runtime Shape

Startup side effects:

1. Configure rotating console/file logging under `output/logs`.
2. Load `.env` via `python-dotenv` and parse runtime globals.
3. Initialize assistant if an API key is configured.
4. Create output, summary, and temp directories.
5. Delete stale `.wav` files from the temp directory.
6. Create an initial transcript Markdown file immediately, before the first `Start`.
7. Create a shared `PyAudio` instance and sample-size value.
8. Create a Windows Teams speaker detector object; it remains disconnected on non-Windows.

Session start behavior:

1. No-op if already recording.
2. If multiple `LANGUAGE` candidates exist and start is manual, prompt user to select one. Auto-start uses the first candidate.
3. Initialize preferred/default microphone and output-capture devices.
4. Clear the shared stop event.
5. Drain microphone chunk queue, output chunk queue, and transcript display/assistant-answer queue.
6. Reset manual/speaker-change flush markers.
7. Create a fresh transcript file and clear in-memory transcript state. This is the real session boundary.
8. Start six daemon threads:
   - microphone WAV/transcription worker
   - output WAV/transcription worker
   - microphone capture/chunking worker
   - output capture/chunking worker
   - Teams speaker detection worker
   - transcript display/store queue consumer

Session stop behavior:

1. No-op if not recording.
2. Set the shared stop event.
3. Join each session thread with a two-second timeout.
4. Mark recording stopped and update controls/status.
5. If auto-summary is enabled, schedule summary generation.

Migration implication: the Go core should model this as a session supervisor with explicit source workers, chunk workers, transcription workers, and one ordered transcript sink. It should not rely on a single global stop flag causing all workers to exit at once; final chunks and transcript entries need an orderly drain path.

## Audio Capture and Device Semantics

Current device behavior:

- Windows imports `pyaudiowpatch`; other platforms import standard `pyaudio`.
- Device catalog splits candidates into:
  - microphone inputs
  - output-capture devices
- On Windows, output capture means WASAPI loopback devices with input channels.
- On non-Windows, devices with output channels are listed as output candidates, but capture still opens them with `input=True` and requires `maxInputChannels > 0`. In practice, non-Windows system output capture is only viable if an audio backend exposes a monitor/loopback as an input-capable device.
- Start currently fails if no microphone device or no output device can be selected, even though non-Windows output capture may later no-op when opened.
- Preferred devices are selected by saved index first, then normalized saved name, then default/fallback, then first available.
- Device preferences are persisted as:
  - `AUDIO_INPUT_DEVICE_INDEX`
  - `AUDIO_INPUT_DEVICE_NAME`
  - `AUDIO_OUTPUT_DEVICE_INDEX`
  - `AUDIO_OUTPUT_DEVICE_NAME`

Go migration map:

- Introduce a capture backend abstraction with source capabilities: `mic`, `system_output`, maybe future `file`.
- Preserve preferred-device selection order: saved index, saved normalized name, backend default, first available.
- Treat output capture as optional capability. The TUI should be able to start mic-only when system output capture is not supported or not configured.
- Keep source IDs independent. Do not assume both sources share sample rate/channel count.
- Preserve temp WAV chunk format only if the transcriber still requires file uploads. A streaming transcriber can replace temp files later.

## Silence and Chunking Behavior

Current chunk loop (`audio_capture.collect_from_stream`):

- Opens a PyAudio input stream using selected device index, default sample rate, and `maxInputChannels`.
- Frame size is `defaultSampleRate * FRAME_DURATION_MS / 1000` frames. Default `FRAME_DURATION_MS` is `100`.
- Reads signed 16-bit PCM frames.
- Appends raw frame bytes to an in-memory `frames` buffer.
- Computes volume as RMS over all int16 samples in the most recent frame.
- Uses `SILENCE_THRESHOLD` default `50.0`.
- Uses `SILENCE_DURATION` default `1.0` second.
- Leading silence is not intended to anchor chunk start time. When all buffered frames are silence, `start_time` is updated to current time.
- Once silence has lasted long enough and the buffer contains non-silence, queues the whole buffer, including trailing silence.
- A hard max chunk length is enforced by `RECORD_SECONDS`, default `300` seconds.
- On stop, any remaining buffered frames are queued.

Important current quirks:

- `silence_frame_count` is not reset on non-silent frames. The intended behavior is still clear: split after consecutive silence following speech, and avoid emitting leading-silence-only chunks. The Go implementation should use an explicit state machine rather than preserve this variable-level quirk.
- Stop ordering can lose final transcript entries because all threads observe the same stop event. The transcript consumer exits as soon as the stop event is set, while transcription workers may enqueue results after that. Go should close/drain channels in order: stop capture, flush chunks, finish transcription workers, then close transcript sink.
- Speaker-change chunks for output keep a two-second overlap. If the buffer is shorter than two seconds, current Python can queue an empty chunk. Go should clamp this case.

Go migration map:

- Implement chunking as a per-source state machine:
  - `Idle/LeadingSilence`
  - `SpeechActive`
  - `TrailingSilence`
  - `MaxDurationFlush`
  - `StopFlush`
- Preserve these user-visible rules:
  - no leading-silence-only transcript chunks
  - split after configured trailing silence
  - include some trailing silence unless a future transcriber requires trimming
  - enforce max chunk duration
  - flush residual audio on stop
- Use monotonic timestamps for ordering plus wall-clock timestamps for filenames/display if needed.

## Transcription Queueing and Worker Model

Current queues:

- `g_recordings_in`: microphone chunks waiting for WAV/transcription.
- `g_recordings_out`: output chunks waiting for WAV/transcription.
- `g_transcriptions_in`: final transcript events and assistant answers waiting for storage/display.
- Assistant also has its own internal `messages_in` queue.

Current worker behavior:

- Each source has one store/transcribe worker, so chunks from the same source are transcribed serially.
- Microphone and output transcription can happen concurrently.
- Both transcription workers share the same `OpenAITranscribe` instance.
- Each chunk is written as `{start_time:.2f}-{source}.wav` in `TEMP_DIR`.
- The WAV is deleted after transcription callback returns.
- Transcription errors are logged; failed chunks do not retry outside the provider-level API retry loop.
- `OpenAITranscribe` retries transient API/network failures with exponential backoff and treats auth/permission failures as hard failures.
- Transcription result currently produces simple `Segment(start=0, end=1.0, text=...)` objects because the OpenAI response text is not segment-timestamped.
- Artifact filtering occurs after transcription and before transcript enqueue.

Transcript event mapping:

- Microphone chunks become speaker `YOUR_NAME`.
- Output chunks use:
  - Teams speaker name when available.
  - `Person_X` when manual one-letter label `x` was used.
  - `?` when no label/speaker is known.
- Assistant receives every accepted transcript as a user message in the form `{speaker}: {text}`.
- Assistant live panels only trigger from output-device transcription.
- Assistant responses are enqueued back onto `g_transcriptions_in` as `Agent`, but are not persisted to transcript Markdown.

Go migration map:

- Use per-source chunk channels feeding per-source or bounded shared transcription workers.
- Preserve per-source ordering unless explicitly improved with a sequence number.
- Give each final transcript event:
  - source ID (`mic`/`output`)
  - speaker label
  - text
  - chunk start wall time
  - monotonic ordering timestamp
  - optional provider metadata/error
- Use one transcript sink that remains alive until all transcription workers finish.
- Keep assistant integration out of the capture core. Expose transcript events to an assistant subsystem via subscription or fan-out.

## Transcript Storage Behavior

Current storage:

- Every manual or auto `Start` creates a fresh `output/transcription-YYYYMMDD_HHMMSS.md` file and clears memory.
- File starts with:
  - `# Transcription Log`
  - `**Created:** YYYY-MM-DD HH:MM:SS`
- User transcript entries append immediately as `Speaker: text` followed by a blank line.
- `Agent` entries are kept in memory/display but intentionally not appended to Markdown transcript files.
- In-memory entries are sorted by `start_time` on every add.
- File persistence is arrival-order, not sorted-order.
- Summary generation uses the sorted in-memory snapshot.

Go migration map:

- Preserve one transcript file per started session.
- Drop the startup-created empty transcript file; it is a side effect, not useful behavior.
- Preserve current Markdown shape for compatibility unless the migration owner decides to add timestamps.
- Store transcript events in memory sorted by source timestamp for summary and display.
- If file order matters, prefer writing through the ordered sink rather than direct arrival-order appends.
- Keep assistant-generated responses separate from persisted transcript unless explicitly requested by product.

## Config and Environment Handling

Current config sources:

- `.env` is loaded at import time.
- `env.sample` documents safe schema and defaults.
- Settings UI persists updates back to `.env`.
- API key is only overwritten by settings when the user enters a non-empty new value.

Core parser semantics to preserve:

- Strings are trimmed; empty strings fall back to defaults.
- Integers/floats fall back to defaults and emit warnings on invalid values.
- Booleans accept `1,true,yes,y,on` and `0,false,no,n,off`; invalid values fall back and warn.
- Optional device indexes become `None` when empty or invalid.
- `LANGUAGE` accepts a single code or comma-separated supported codes, lowercases, dedupes, ignores unsupported codes with warnings, and falls back to `en`.
- Escaped multiline prompts use `\n` encoding in env values.

Important compatibility names:

- Preserve the misspelled `INTERUPT_MANUALLY` key for backward compatibility.
- Preserve existing audio segmentation keys:
  - `RECORD_SECONDS`
  - `SILENCE_THRESHOLD`
  - `SILENCE_DURATION`
  - `FRAME_DURATION_MS`
- Preserve output path keys:
  - `OUTPUT_DIR`
  - `TEMP_DIR`
  - `SUMMARIES_DIR`
- Preserve transcription keys:
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL_FOR_TRANSCRIPT`
  - `TRANSCRIBE_API_TIMEOUT_SECONDS`
  - `TRANSCRIBE_API_MAX_RETRIES`
  - `TRANSCRIBE_API_RETRY_BASE_SECONDS`
  - `KEYWORDS`
- Preserve transcript filter keys.

Go migration map:

- Build a typed config struct from environment plus `.env` file loading.
- Return warnings as structured startup diagnostics instead of only logging them.
- Keep safe compatibility with existing `.env` files.
- Consider adding a correctly spelled alias later, but do not remove `INTERUPT_MANUALLY` during migration.

## Speaker Detection Behavior

Current Teams detector:

- Supported only on Windows.
- Uses `pywinauto` UI Automation and connects to title regex `Meeting compact view*`.
- Polls controls every 100 ms.
- Reads meeting title roughly every five seconds.
- Extracts active speaker from menu item text patterns.
- Normalizes names to first name plus second initial when available, otherwise first name.
- Maintains previous/current speaker under lock.
- On speaker change, if there was a previous speaker, both mic and output flush markers are set to `_`.

Speaker-change chunking:

- Microphone ignores `_` speaker-change flushes.
- Output handles `_` by queueing all but the last two seconds of buffered audio with the previous speaker.
- Last two seconds remain buffered as overlap for the next speaker.
- Manual one-letter flushes affect both mic and output.

Go migration map:

- Model speaker detection as an optional event source:
  - `SpeakerChanged(previous, current, timestamp)`
  - `MeetingTitleChanged(title, timestamp)`
- Capture/chunking should consume speaker events without knowing whether they came from Teams, manual TUI commands, or another backend.
- Make Teams UI Automation an optional Windows adapter, not part of core capture.
- If no detector exists, output chunks should still transcribe with speaker `?` or a user-selected fallback label.
- Clamp speaker-change overlap so short buffers do not emit empty chunks.

## Pause, Stop, Mute, and Manual Split Semantics

Current semantics:

- `Start` means new session. It resets transcript state and queues.
- `Stop` means end session. It is not a pause/resume.
- There is no current pause operation.
- Mute is microphone-only:
  - sets `mute_mic_event`
  - microphone capture loop skips reading and appending frames
  - output capture continues
  - muted microphone audio is not queued or transcribed
- Manual split is only bound when assistant is unavailable, to avoid text-entry conflicts.
- Manual split currently maps any `a-z` keypress to both:
  - flush current mic/output buffers
  - label output speaker as `Person_<LETTER>`

Go migration map:

- Preserve `start` as new session and `stop` as end session.
- Add pause only as a new explicit product decision; do not treat current mute as pause.
- Preserve mic-only mute semantics.
- Replace global any-letter manual split with explicit TUI commands such as split-current-chunk and set-speaker-label. The exact TUI command shape belongs to the TUI owner.

## Behavior That Can Be Intentionally Dropped

Safe to drop from the Go core:

- Tkinter windowing, styling, hover effects, icon drawing, dark title bar, widget state details, and layout dimensions.
- Startup-created empty transcript file before the first session starts.
- Any-letter global keyboard split behavior. Replace with explicit TUI commands.
- Coupling manual split to one-letter `Person_X` labels as the only manual speaker path.
- `teams_controls.txt` debug dump as default behavior. Keep only as adapter-level diagnostic if needed.
- Commented local Whisper/Torch/FastWhisper paths that are not active current behavior.
- Live assistant panel UI behavior from the capture core. Keep transcript event fan-out so another owner can rebuild assistant UX.
- Windows compact-view title string as a core assumption. Keep it only in a Windows Teams adapter.

Should not drop without product/team decision:

- Fresh transcript per `Start`.
- Mic and output as separate sources with independent buffering.
- Silence-duration chunking and max-duration chunking.
- Mic-only mute semantics.
- Transcript filtering and keyword override behavior.
- Existing `.env` compatibility, especially output paths, language, OpenAI model, and misspelled `INTERUPT_MANUALLY`.
- Markdown transcript output.
- Optional summary context/title inputs, if summary remains in scope for the TUI rebuild.

## Risks and Open Questions for Other Owners

- TUI architecture and command names belong to Livingston.
- Migration sequencing and MVP scope belong to Danny.
- Assistant UX/live panels/custom prompt behavior likely needs an assistant/product owner; Rusty only mapped queue touchpoints.
- Windows Teams speaker detection needs a Windows/UI Automation owner if it remains in scope.
- Go audio backend choice needs validation on target OSes. Current Python behavior is strongest on Windows because of WASAPI loopback.

## Recommended Go Core Contracts

Minimum interfaces for the rebuild:

- `CaptureBackend`: list devices, open source stream, read PCM frames, close stream.
- `Chunker`: consume PCM frames and control/speaker events, emit audio chunks.
- `Transcriber`: consume audio chunks, emit text segments or errors.
- `TranscriptSink`: consume transcript events, maintain sorted memory, persist Markdown.
- `SpeakerEventSource`: optional producer of speaker/title events.
- `ConfigLoader`: read `.env`/environment into typed config plus warnings.

Minimum event types:

- `AudioFrame(source, pcm, sampleRate, channels, capturedAt)`
- `AudioChunk(source, speaker, startedAt, endedAt, pcmOrFile)`
- `SpeakerChanged(previous, current, at)`
- `ControlEvent(start, stop, muteMic, unmuteMic, split, setSpeaker)`
- `TranscriptEvent(source, speaker, text, startedAt, providerMetadata)`

Core invariant for Go: transcript sink must outlive capture and transcription workers, and shutdown must drain final chunks before closing the session.
