# Live Audio Chat with Transcription and Speaker Identification

This Python application provides real-time audio recording, transcription, and chat integration. It is designed for use in virtual meetings, enabling users to capture audio streams, transcribe them, and display the results in a user-friendly GUI. The application also identifies speakers in Microsoft Teams and integrates with OpenAI's GPT model to generate intelligent responses or questions based on the transcript.

## Features

- **Real-Time Audio Recording**:
  - Captures audio from both the microphone and system output.
  - Supports silence detection and manual splitting of audio streams.

- **Transcription**:
  - Utilizes the `faster_whisper` library for fast and accurate transcription.
  - Supports transcription using OpenAI's API via the `openai_transcribe.py` module.
  - Filters out irrelevant or unwanted segments during transcription.

- **Speaker Identification**:
  - Connects to Microsoft Teams using `pywinauto` to identify the current speaker.
  - Extracts speaker names using regex patterns.

- **Chat Integration**:
  - Displays transcriptions in a Tkinter-based GUI.
  - Integrates with OpenAI's GPT model to generate responses or questions based on the transcript.

- **Threaded Architecture**:
  - Uses multiple threads for audio recording, transcription, and speaker identification to ensure real-time performance.

- **Environment Configuration**:
  - Loads configuration variables (e.g., API keys, language settings) from a `.env` file.

- **Output Management**:
  - Stores temporary audio files in an `output` directory and deletes them after processing.

## Requirements

- Python 3.8 or higher
- CUDA-enabled GPU (for Whisper model)
- Required Python libraries (see `pyproject.toml` or `poetry.lock`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd meeting-transcriptions
   ```

2. Install dependencies using Poetry:
   ```bash
   first_time_install_<os>.bat|sh
   ```

3. Create a `.env` file in the project root and add the following variables:
   ```env
    YOUR_NAME=John
    LANGUAGE=en
    INTERUPT_MANUALLY=True
    OPENAI_API_KEY=<Your Open AI API Key - from the same organization as the Assistant, if you will use it>
    # Comment out if you don't want answers from Assistant. Set the right vector store ID with the right knowledge (should start wtith vs_*)
    OPENAI_VECTOR_STORE_ID_FOR_ANSWERS=
    # Comment out if you don't want to use OpenAI for transcription, but use local model instead
    OPENAI_MODEL_FOR_TRANSCRIPT=gpt-4o-mini-transcribe
    #OPENAI_MODEL_FOR_TRANSCRIPT=gpt-4o-transcribe
    # Complex words to transcribe:
    KEYWORDS="Cat, Mouse, Dog"
   ```

## Usage

1. Run the application:
   ```bash
   run_transcribe_<os>.bat|sh
   ```

2. The GUI will launch, displaying the live transcription and chat interface.

3. Use the following key bindings for manual splitting:
   - Press any letter key (`a-z`) to trigger a manual split.

4. The application will automatically detect silence and split audio streams accordingly.

## File Structure

- `transcribe.py`: Main application file.
- `output/`: Directory for temporary audio files and transcription files.
- `.env`: Configuration file for environment variables.
- `pyproject.toml` and `poetry.lock`: Dependency management files.

## Notes

- The application is optimized for use with Microsoft Teams but can be adapted for other platforms.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- [Faster Whisper](https://github.com/guillaumekln/faster-whisper)
- [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/)
- [PyWinAuto](https://pywinauto.github.io/)
- [OpenAI](https://openai.com/)
