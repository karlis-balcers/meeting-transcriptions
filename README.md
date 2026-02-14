# Meeting Transcription Service

Real-time meeting transcription desktop app with a built-in AI meeting assistant.

It captures microphone and system audio, transcribes speech, detects active speakers in Microsoft Teams (Windows), and can generate assistant replies or meeting summaries using OpenAI.

## What it does

- Live transcription from:
  - Microphone input
  - System output (WASAPI loopback on Windows)
- Speaker tracking from Microsoft Teams window metadata (Windows)
- Tkinter UI with Start/Stop workflow and live status line
- AI assistant answers to user-entered prompts during meetings
- Optional internet search for assistant custom prompts (configurable)
- Optional auto-summary generation when transcription stops
- Transcript filtering to suppress common artifacts/hallucinations
- Markdown output for transcripts and summaries

## Current UI behavior

- `Start` begins a **fully new transcription session**:
  - clears in-memory transcript view
  - creates a fresh transcript file
  - resets pending queues/buffers
- `Stop` ends active transcription.
- There is **no "Reset Log" button** (Start already resets).
- There is **no "Generate Summary" button** in the main window.
  - Summaries are generated through auto-summary flow if enabled.

## Architecture at a glance

- `transcribe.py` - app orchestration, UI lifecycle, settings dialog, thread startup/shutdown
- `assistant.py` - assistant messaging, OpenAI Responses API calls, retries/backoff
- `openai_transcribe.py` - OpenAI transcription integration with retry handling
- `audio_capture.py` - audio stream capture and segmentation helpers
- `speaker_detection.py` - Teams speaker/title detection (Windows)
- `transcript_store.py` - thread-safe transcript storage + persistence
- `transcript_filter.py` - configurable filtering logic
- `ui.py` - transcript rendering + status color helpers
- `summary_utils.py` - summary title filename sanitization
- `env_utils.py` - reusable env parsing helpers

## Requirements

- Python `3.11+`
- Poetry
- Windows recommended for full feature set:
  - WASAPI loopback capture
  - Teams speaker detection via `pywinauto`

On Linux/macOS, transcription works with reduced feature set (no Teams speaker integration).

## Installation

1. Clone repository and enter project folder.
2. Run first-time setup script for your OS:
   - Windows: `first_time_install_win.bat`
   - Linux: `first_time_install_linux.sh`
   - macOS: `first_time_install_mac.sh`
3. Create `.env` from `env.sample` and fill required values.

## Running

- Windows: `run_transcribe_win.bat`
- Linux: `run_transcribe_linux.sh`
- macOS: `run_transcribe_mac.sh`

## Configuration (`.env`)

Use `env.sample` as the source of truth. Important settings:

### Required

- `OPENAI_API_KEY` - required for OpenAI transcription and assistant features

### Identity and language

- `YOUR_NAME` - your speaker label in transcript
- `LANGUAGE` - transcription language code (for example `en`, `lv`)

### Session behavior

- `AUTO_START_TRANSCRIPTION` - auto-start on app launch
- `AUTO_SUMMARIZE_ON_STOP` - generate summary when stopping transcription

### Assistant behavior

- `OPENAI_MODEL_FOR_ASSISTANT` - assistant model (default in sample: `gpt-5.2`)
- `OPENAI_VECTOR_STORE_ID_FOR_ANSWERS` - optional file search source
- `ASSISTANT_ENABLE_WEB_SEARCH_FOR_CUSTOM_PROMPTS` - enable/disable internet search tool for user-entered prompts

### Transcription behavior

- `OPENAI_MODEL_FOR_TRANSCRIPT` - OpenAI transcription model
- `KEYWORDS` - domain keywords to improve recognition quality

### Audio segmentation

- `RECORD_SECONDS`
- `SILENCE_THRESHOLD`
- `SILENCE_DURATION`
- `FRAME_DURATION_MS`

### Output paths

- `OUTPUT_DIR` - transcript + runtime output directory
- `SUMMARIES_DIR` - summary markdown output directory

### Reliability and logging

- `ASSISTANT_API_*` timeout/retry settings
- `TRANSCRIBE_API_*` timeout/retry settings
- `LOG_LEVEL`, `LOG_FILE_MAX_MB`, `LOG_FILE_BACKUP_COUNT`

### Filtering

- `TRANSCRIPT_FILTER_MIN_CHARS`
- Optional extension lists: `TRANSCRIPT_FILTER_EXACT`, `TRANSCRIPT_FILTER_PREFIXES`, `TRANSCRIPT_FILTER_CONTAINS`, `TRANSCRIPT_FILTER_REGEX`

## Settings window

The app settings dialog allows runtime updates and persists them to `.env`:

- Auto-start transcription
- Transcription language
- Auto-summarize on stop
- Enable internet search for assistant custom prompts
- Transcriptions folder
- Summaries folder

## Output files

- Transcriptions are written as Markdown:
  - `output/transcription-YYYYMMDD_HHMMSS.md` (or custom `OUTPUT_DIR`)
- Summaries are written as Markdown:
  - `output_summaries/YYYYMMDD_HHMMSS_<ai_title>.md` (or custom `SUMMARIES_DIR`)
- Temporary `.wav` files are cleaned up from output directory.

## Development and testing

Run tests locally:

- `python -m unittest discover -s tests -p "test_*.py"`

CI:

- GitHub Actions workflow at `.github/workflows/tests.yml`
- Runs unit tests on Python `3.11` and `3.12`

## Troubleshooting

- No audio captured:
  - verify input/output devices and OS audio permissions
- Teams speaker name not detected:
  - feature is Windows-only and depends on Teams window state/title patterns
- API failures/timeouts:
  - verify `OPENAI_API_KEY`
  - tune retry/timeout env variables
- Empty/low-value transcript fragments:
  - tune filter settings and `KEYWORDS`

## Security notes

- Do not commit `.env`
- Keep API keys in environment variables only
- Review transcript and summary storage locations for sensitive meeting data

## Project status notes

- This project currently does not include a `LICENSE` file in the repository.

---

If you are actively iterating on features, keep `env.sample` and `README.md` updated together to avoid config drift.
