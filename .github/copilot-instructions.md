# Meeting Transcription Service - GitHub Copilot Instructions

## Project Overview

This is a real-time meeting transcription and AI assistant application designed for virtual meetings (primarily Microsoft Teams). The application captures audio streams, transcribes them using OpenAI's Whisper models, identifies speakers, and can generate intelligent responses using OpenAI's GPT models.

### Key Capabilities
- Real-time audio recording from microphone and system output (WASAPI on Windows)
- Live transcription using OpenAI API or local Whisper models
- Speaker identification from Microsoft Teams windows
- AI assistant integration with knowledge base (vector store)
- Tkinter-based GUI for displaying transcriptions
- Automatic silence detection and audio segmentation
- Multi-threaded architecture for concurrent processing
- AI-powered meeting summary generation with auto-generated titles
- Context-aware summaries using background information from context.md

## Technology Stack

### Core Technologies
- **Python 3.11+** - Primary language
- **Poetry** - Dependency management and packaging
- **PyAudio/PyAudioWPatch** - Audio capture (WASAPI loopback on Windows)
- **OpenAI API** - Transcription (Whisper) and chat completions (GPT)
- **Tkinter** - GUI framework
- **PyWinAuto** - Windows UI automation for Teams speaker detection (Windows-only)
- **NumPy** - Audio processing and signal analysis
- **python-dotenv** - Environment variable management

### Platform Support
- **Primary**: Windows (with WASAPI loopback and Teams integration)
- **Secondary**: Linux/macOS (limited features, no Teams integration)

## Architecture & Design Patterns

### Threading Model
The application uses a producer-consumer pattern with multiple threads:
1. **Audio Recording Thread** (Microphone) - Captures mic input
2. **Audio Recording Thread** (System Output) - Captures system audio
3. **Transcription Threads** - Process audio files asynchronously
4. **Speaker Detection Thread** - Monitors Teams window (Windows only)
5. **Assistant Thread** - Processes messages and generates responses
6. **GUI Thread** - Main thread, handles UI updates

### Key Components

#### `transcribe.py` (Main Application)
- Entry point and orchestration
- Audio recording with silence detection
- Queue-based communication between threads
- GUI management and event handling
- Speaker name extraction from Teams

#### `openai_transcribe.py` (Transcription)
- OpenAI Whisper API integration
- Prompt engineering for better accuracy with keywords
- Segment filtering (removes noise/artifacts)
- Fake keywords (`LAMPA`, `MEMMEE`) for quality control

#### `assistant.py` (AI Assistant)
- OpenAI chat completions integration
- Vector store knowledge base queries
- Concurrent answer computation (ThreadPoolExecutor)
- Message batching and processing
- Custom prompt handling for user-initiated questions
- Meeting summary generation with AI-generated titles

#### `segment.py` (Data Model)
- Simple data class for transcription segments
- Contains start time, end time, and transcribed text

## Configuration

### Environment Variables (.env)
```env
YOUR_NAME=<Your name for speaker identification>
LANGUAGE=en
INTERUPT_MANUALLY=True
OPENAI_API_KEY=<Your OpenAI API key>
OPENAI_VECTOR_STORE_ID_FOR_ANSWERS=vs_* # Optional, for assistant
OPENAI_MODEL_FOR_TRANSCRIPT=gpt-4o-mini-transcribe # or gpt-4o-transcribe
KEYWORDS="Cat, Mouse, Dog" # Domain-specific terms for better transcription
```

### Audio Processing Parameters
- `RECORD_SECONDS = 300` - Maximum recording duration per segment
- `SILENCE_THRESHOLD = 50` - Amplitude threshold for silence detection
- `SILENCE_DURATION = 1.0` - Seconds of silence before auto-split

## Code Style & Conventions

### General Guidelines
1. **Follow existing patterns** - Match the threading and queue-based architecture
2. **Platform-aware code** - Use `sys.platform == "win32"` checks for Windows-specific features
3. **Error handling** - Wrap API calls and file operations in try-except blocks
4. **Logging** - Use `print()` for key events (consider moving to proper logging module)
5. **Thread safety** - Use `Lock()` for shared state, prefer queue-based communication

### Naming Conventions
- Global variables: `g_prefix` (e.g., `g_transcription`, `g_device_in`)
- Constants: `UPPER_CASE` (e.g., `SILENCE_THRESHOLD`, `RECORD_SECONDS`)
- Functions: `snake_case`
- Classes: `PascalCase`
- Event objects: `descriptive_event` (e.g., `stop_event`, `mute_mic_event`)

### Thread Communication
- **Use Queues** for passing data between threads (`Queue.put()` / `Queue.get()`)
- **Use Events** for signaling state changes (`Event.set()` / `Event.is_set()`)
- **Use Locks** for protecting shared mutable state (`Lock()`)

### Audio File Handling
- Store temporary files in `output/` directory
- Use ISO timestamp format: `transcription-YYYYMMDD_HHMMSS.txt`
- Clean up WAV files after transcription
- Keep text transcripts for future reference
- Store meeting summaries in `output_summaries/` as Markdown files
- Summary filenames format: `YYYYMMDD_HHMMSS_ai_generated_title.md`
- Load context from `context.md` for enhanced summary generation

## Common Development Tasks

### Adding a New Transcription Provider
1. Create a new module following the pattern of `openai_transcribe.py`
2. Implement a class with a `transcribe(audio_file_path: str) -> Iterable[Segment]` method
3. Add environment variable for model selection
4. Update `transcribe.py` to conditionally import and use the provider

### Modifying Speaker Detection
1. Edit the regex patterns in `transcribe.py` (search for `re.compile()`)
2. Test with actual Teams windows to validate patterns
3. Handle edge cases (no speaker, multiple matches, non-English names)

### Adjusting Silence Detection
1. Modify `SILENCE_THRESHOLD` and `SILENCE_DURATION` constants
2. Test with different microphone setups and environments
3. Consider making these configurable via `.env`

### Adding GUI Features
1. All GUI code is in `transcribe.py`
2. Use `tkinter` widgets and follow the existing layout pattern
3. Update UI from threads using `text.after()` or queue-based updates
4. Test responsiveness during heavy transcription workloads
5. The GUI includes a custom prompt entry field for user questions during meetings

## API Integration Notes

### OpenAI Whisper API
- **Models**: `gpt-4o-mini-transcribe` (default), `gpt-4o-transcribe`
- **File size limit**: 25 MB (adjust `RECORD_SECONDS` accordingly)
- **Prompt engineering**: Use `keywords` parameter for domain-specific terms
- **Filtering**: Remove artifacts using fake keywords and special patterns

### OpenAI Chat Completions (Assistant)
- **Model**: Currently hardcoded to `gpt-5-mini` in `assistant.py`
- **Vector stores**: Optional knowledge base integration
- **Concurrency**: Up to 5 concurrent answer computations
- **Message batching**: Groups up to 5 messages for efficiency

## Testing & Debugging

### Local Testing
1. Set up `.env` file with your API key
2. Run `run_transcribe_win.bat` (or appropriate OS script)
3. Monitor console output for errors
4. Check `output/` directory for saved transcriptions

### Common Issues
- **No audio capture**: Check device indices with `pyaudio.get_device_info_by_index()`
- **Poor transcription quality**: Adjust `KEYWORDS`, check audio levels
- **Teams speaker detection fails**: Verify regex patterns, check window titles
- **High API costs**: Reduce `RECORD_SECONDS`, use smaller model

### Performance Optimization
- Consider local Whisper model for offline/cost reduction
- Batch audio segments to reduce API calls
- Implement audio compression before transmission
- Cache speaker names to reduce Teams window queries

## Security & Best Practices

### API Key Management
- **Never commit `.env` files** - Use `.env.sample` as template
- Store API keys in environment variables
- Consider using secret management services for production

### Audio Privacy
- Audio files are stored locally and deleted after transcription
- Transcripts are saved in `output/` - ensure proper access controls
- Consider adding encryption for sensitive meetings

### Error Recovery
- Handle API rate limits gracefully
- Implement retry logic for transient failures
- Save partial transcriptions before crashes

## Dependencies & Installation

### Initial Setup
- **Windows**: `first_time_install_win.bat`
- **Linux**: `first_time_install_linux.sh`
- **macOS**: `first_time_install_mac.sh`

### Running the Application
- **Windows**: `run_transcribe_win.bat`
- **Linux**: `run_transcribe_linux.sh`
- **macOS**: `run_transcribe_mac.sh`

### Key Dependencies
- `pyaudio` / `pyaudiowpatch` - Audio I/O
- `openai` - API client
- `tkinter` - GUI (usually bundled with Python)
- `pywinauto` - Windows automation (Windows only)
- `numpy` - Audio processing
- `python-dotenv` - Configuration

## Future Enhancements

### Potential Improvements
1. **Local Whisper support** - Uncomment faster-whisper code for offline use
2. **Better UI** - Consider migrating from Tkinter to PyQt6
3. **Speaker diarization** - Implement ML-based speaker identification
4. **Export formats** - Add JSON, SRT, VTT export options
5. **Cloud storage** - Integration with S3, OneDrive, etc.
6. **Logging** - Replace print statements with proper logging framework
7. **Configuration UI** - GUI for editing settings instead of .env file
8. **Multi-language support** - Better handling of non-English meetings

### Known Limitations
- Windows-specific features (Teams integration, WASAPI loopback)
- No speaker diarization for non-Teams audio
- 25 MB file size limit with OpenAI API
- GUI responsiveness during heavy processing

## Additional Context

### Project Structure
```
meeting-transcriptions/
├── .github/
│   └── copilot-instructions.md    # This file
├── output/                          # Transcription outputs
├── output_summaries/                # AI-generated meeting summaries
├── __pycache__/                     # Python cache
├── __init__.py                      # Package marker
├── assistant.py                     # AI assistant integration
├── fastwhisper_transcribe.py        # Local Whisper (commented out)
├── openai_transcribe.py             # OpenAI API transcription
├── segment.py                       # Data model
├── transcribe.py                    # Main application
├── teams_controls.txt               # Teams keyboard shortcuts
├── first_time_install_*.{bat,sh}   # Setup scripts
├── run_transcribe_*.{bat,sh}       # Run scripts
├── .env                             # Configuration (not committed)
├── env.sample                       # Configuration template
├── pyproject.toml                   # Poetry configuration
└── README.md                        # Documentation
```

### Contributing Guidelines
1. Test on Windows before committing Windows-specific features
2. Maintain backward compatibility with `.env` configuration
3. Update README.md for user-facing changes
4. Preserve thread safety in concurrent code
5. Document new environment variables in `env.sample`

---

**When suggesting code changes, always consider:**
- Thread safety and concurrent access patterns
- Platform compatibility (Windows/Linux/macOS)
- API rate limits and costs
- User experience (GUI responsiveness, error messages)
- Existing configuration and environment variables
