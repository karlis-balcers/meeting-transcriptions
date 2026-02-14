import sys
if sys.platform == "win32":
    import pyaudiowpatch as pyaudio  # WASAPI features on Windows
else:
    import pyaudio  # Standard PyAudio on non-Windows
#import torch
#import torchaudio
import time
import wave
import os
import tkinter as tk
from threading import Thread, Event, Lock
from queue import Queue, Empty
import signal
import logging
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import struct

import assistant
#import fastwhisper_transcribe
import openai_transcribe
import audio_capture as audio_capture_module
import ui as ui_module
from logging_utils import setup_logging
from transcript_store import TranscriptStore
from speaker_detection import TeamsSpeakerDetector

setup_logging(app_name="meeting-transcriptions", log_dir=os.path.join("output", "logs"))
logger = logging.getLogger("transcribe")

logger.info("Loading configuration from '.env' file...")


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int, warnings: list[str]) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        warnings.append(f"Invalid integer for {name}='{raw}'. Using default {default}.")
        return default


def _env_float(name: str, default: float, warnings: list[str]) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        warnings.append(f"Invalid number for {name}='{raw}'. Using default {default}.")
        return default


def _env_bool(name: str, default: bool, warnings: list[str]) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    value = raw.strip().lower()
    truthy = {"1", "true", "yes", "y", "on"}
    falsy = {"0", "false", "no", "n", "off"}

    if value in truthy:
        return True
    if value in falsy:
        return False

    warnings.append(
        f"Invalid boolean for {name}='{raw}'. Using default {default}."
    )
    return default


def _print_startup_summary() -> None:
    logger.info("=== Startup Configuration Summary ===")
    logger.info("Platform: %s", sys.platform)
    logger.info("User name: %s", g_my_name)
    logger.info("Language: %s", g_language)
    logger.info("Manual interrupt enabled: %s", g_interupt_manually)
    logger.info("Transcript model: %s", g_openai_model_for_transcript)
    logger.info("Record seconds: %s", RECORD_SECONDS)
    logger.info("Silence threshold: %s", SILENCE_THRESHOLD)
    logger.info("Silence duration: %s", SILENCE_DURATION)
    logger.info("Frame duration (ms): %s", FRAME_DURATION_MS)
    logger.info("Keywords configured: %s", 'yes' if g_keywords else 'no')
    logger.info("OpenAI API key configured: %s", 'yes' if g_open_api_key else 'no')
    logger.info("Assistant enabled: %s", 'yes' if g_assistant else 'no')
    logger.info("Vector store configured: %s", 'yes' if g_vector_store_id else 'no')
    teams_available = False
    if g_speaker_detector is not None:
        teams_available = g_speaker_detector.is_connected()
    logger.info("Teams integration available: %s", 'yes' if teams_available else 'no')
    if g_startup_warnings:
        logger.warning("Startup warnings detected:")
        for warning in g_startup_warnings:
            logger.warning("  - %s", warning)
    else:
        logger.info("Warnings: none")
    logger.info("=====================================")


# Default constants (can be overridden by .env)
DEFAULT_RECORD_SECONDS = 300
DEFAULT_SILENCE_THRESHOLD = 50.0
DEFAULT_SILENCE_DURATION = 1.0
DEFAULT_FRAME_DURATION_MS = 100

# Global Queues & Event
g_recordings_in = Queue()
g_recordings_out = Queue()
g_transcriptions_in = Queue()
stop_event = Event()
mute_mic_event = Event()  # Event to control microphone muting
g_transcript_store = None
g_status_lock = Lock()
g_status_message = "Ready"
g_status_level = "info"

g_device_in = {}
g_device_out = {}
g_sample_size = 0

# Global flush counter (for manual splits) and its lock
flush_letter_mic = None
flush_letter_out = None
flush_letter_lock = Lock()

root = None
status_label = None
g_speaker_detector = None


def set_status(message: str, level: str = "info"):
    """Update global status and refresh UI label if available."""
    global g_status_message, g_status_level

    if not message:
        return

    with g_status_lock:
        g_status_message = message
        g_status_level = (level or "info").lower()

    if root is None or status_label is None:
        return

    def _apply_status():
        with g_status_lock:
            current_message = g_status_message
            current_level = g_status_level

        color = ui_module.status_color(current_level)
        status_label.config(text=f"Status: {current_message}", fg=color)

    try:
        root.after(0, _apply_status)
    except Exception:
        pass


def _assistant_status_callback(message: str, level: str = "info"):
    set_status(f"Assistant: {message}", level)


def _transcription_status_callback(message: str, level: str = "info"):
    set_status(f"Transcription: {message}", level)

load_dotenv()  # Load variables from .env file

g_startup_warnings: list[str] = []

RECORD_SECONDS = _env_int("RECORD_SECONDS", DEFAULT_RECORD_SECONDS, g_startup_warnings)
if RECORD_SECONDS <= 0:
    g_startup_warnings.append(f"RECORD_SECONDS must be > 0. Using default {DEFAULT_RECORD_SECONDS}.")
    RECORD_SECONDS = DEFAULT_RECORD_SECONDS

SILENCE_THRESHOLD = _env_float("SILENCE_THRESHOLD", DEFAULT_SILENCE_THRESHOLD, g_startup_warnings)
if SILENCE_THRESHOLD < 0:
    g_startup_warnings.append(f"SILENCE_THRESHOLD must be >= 0. Using default {DEFAULT_SILENCE_THRESHOLD}.")
    SILENCE_THRESHOLD = DEFAULT_SILENCE_THRESHOLD

SILENCE_DURATION = _env_float("SILENCE_DURATION", DEFAULT_SILENCE_DURATION, g_startup_warnings)
if SILENCE_DURATION < 0:
    g_startup_warnings.append(f"SILENCE_DURATION must be >= 0. Using default {DEFAULT_SILENCE_DURATION}.")
    SILENCE_DURATION = DEFAULT_SILENCE_DURATION

FRAME_DURATION_MS = _env_int("FRAME_DURATION_MS", DEFAULT_FRAME_DURATION_MS, g_startup_warnings)
if FRAME_DURATION_MS <= 0:
    g_startup_warnings.append(f"FRAME_DURATION_MS must be > 0. Using default {DEFAULT_FRAME_DURATION_MS}.")
    FRAME_DURATION_MS = DEFAULT_FRAME_DURATION_MS

g_my_name = _env_str("YOUR_NAME", "You")
if g_my_name == "You":
    g_startup_warnings.append("YOUR_NAME is missing in .env. Using fallback name 'You'.")
logger.info("Your name is: %s", g_my_name)
AGENT_NAME="Agent"

g_language = _env_str("LANGUAGE", "en")
logger.info("Language: %s", g_language)

g_open_api_key = _env_str("OPENAI_API_KEY")
if not g_open_api_key:
    g_startup_warnings.append(
        "OPENAI_API_KEY is not configured. Transcription/assistant calls to OpenAI may fail."
    )

g_interupt_manually = _env_bool("INTERUPT_MANUALLY", True, g_startup_warnings)

g_assistant = None
g_vector_store_id = _env_str("OPENAI_VECTOR_STORE_ID_FOR_ANSWERS")

if g_open_api_key:
    try:
        g_assistant = assistant.Assistant(
            g_vector_store_id,
            g_my_name,
            agent_name=AGENT_NAME,
            answer_queue=g_transcriptions_in,
            status_callback=_assistant_status_callback,
        )
        if g_vector_store_id:
            logger.info("Assistant configured with vector store ID: %s", g_vector_store_id)
        else:
            logger.info("Assistant configured without vector store.")
    except Exception as e:
        g_startup_warnings.append(f"Assistant initialization failed: {e}")
else:
    g_startup_warnings.append("Assistant disabled because OPENAI_API_KEY is missing.")

assistant_buttons: dict[str, tk.Button] = {}
custom_prompt_entry = None
send_prompt_button = None
summary_button = None

g_keywords = None
g_keywords = _env_str("KEYWORDS")
if g_keywords:
    logger.info("Using keywords for initial prompt: %s", g_keywords)

g_agent_font_size = _env_int("AGENT_FONT_SIZE", 14, g_startup_warnings)
logger.info("Agent font size: %s", g_agent_font_size)

g_default_font_size = _env_int("DEFAULT_FONT_SIZE", 10, g_startup_warnings)
logger.info("Default font size: %s", g_default_font_size)

g_transcript_ai = None
g_openai_model_for_transcript = _env_str("OPENAI_MODEL_FOR_TRANSCRIPT", "gpt-4o-mini-transcribe")
logger.info("Using OpenAI model for transcript: %s", g_openai_model_for_transcript)
try:
    g_transcript_ai = openai_transcribe.OpenAITranscribe(
        model=g_openai_model_for_transcript,
        keywords=g_keywords,
        language=g_language,
        status_callback=_transcription_status_callback,
    )
except Exception as e:
    g_startup_warnings.append(
        f"Failed to initialize OpenAI transcription model '{g_openai_model_for_transcript}': {e}"
    )
    g_openai_model_for_transcript = "gpt-4o-mini-transcribe"
    g_transcript_ai = openai_transcribe.OpenAITranscribe(
        model=g_openai_model_for_transcript,
        keywords=g_keywords,
        language=g_language,
        status_callback=_transcription_status_callback,
    )
    g_startup_warnings.append(
        f"Falling back to transcription model '{g_openai_model_for_transcript}'."
    )
    #print(f"Using Fast Whisper running on your PC for transcript: large-v3")
    #try:
    #    local_transcribe_model = os.environ.get("LOCAL_TRANSCRIBE_MODEL")
    #except:
    #    local_transcribe_model = "tiny.en"
    #try:
    #    local_transcribe_device = os.environ.get("LOCAL_TRANSCRIBE_MODEL_DEVICE")
    #except:
    #    local_transcribe_device = "cpu"
    #g_transcript_ai = fastwhisper_transcribe.FastWhisperTranscribe(model_name=local_transcribe_model, device=local_transcribe_device, keywords=g_keywords, language=g_language)

# Check if output dir exists
g_output_dir = "output"
if not os.path.exists(g_output_dir):
    os.makedirs(g_output_dir)

# Check if output_summaries dir exists
g_summaries_dir = "output_summaries"
if not os.path.exists(g_summaries_dir):
    os.makedirs(g_summaries_dir)

# Delete all wav files in the output directory
for file in os.listdir(g_output_dir):
    if file.endswith(".wav"):
        file_path = os.path.join(g_output_dir, file)
        try:
            os.remove(file_path)
            logger.info("Deleted old file: %s", file_path)
        except Exception as e:
            logger.warning("Error deleting file %s: %s", file_path, e)

# Clear the file and initialize the transcription log
g_trans_file_name = f"{g_output_dir}/transcription-{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
try:
    with open(g_trans_file_name, "w") as f:
        f.write(f"== Transcription Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ==\n\n")
except Exception as e:
    g_startup_warnings.append(f"Failed to create transcription log file '{g_trans_file_name}': {e}")

g_transcript_store = TranscriptStore(g_trans_file_name, AGENT_NAME, logger)

global_audio = pyaudio.PyAudio()
g_sample_size = global_audio.get_sample_size(pyaudio.paInt16)

g_window_name = "Meeting compact view*"
g_speaker_detector = TeamsSpeakerDetector(stop_event=stop_event, logger=logger, window_name=g_window_name)
if not g_speaker_detector.is_connected() and sys.platform == "win32":
    g_startup_warnings.append("Teams integration not connected at startup.")

_print_startup_summary()

def _get_speaker_snapshot():
    return g_speaker_detector.get_speaker_snapshot()


def _set_current_speaker(new_speaker: str | None):
    return g_speaker_detector.set_current_speaker(new_speaker)


def _get_meeting_title_snapshot() -> str | None:
    return g_speaker_detector.get_meeting_title_snapshot()

def get_ms_teams_window_title():
    return g_speaker_detector.get_ms_teams_window_title()

def inspect_ms_teams():
    return g_speaker_detector.inspect_ms_teams()

def get_speaker_name():
    '''Get the current speaker's name from the MS Teams window.'''
    global flush_letter_mic, flush_letter_out

    def on_speaker_changed(previous_speaker, current_speaker):
        if previous_speaker:
            with flush_letter_lock:
                flush_letter_mic = '_'
                flush_letter_out = '_'

    g_speaker_detector.run_detection_loop(on_speaker_changed=on_speaker_changed)

def toggle_mute():
    """Toggle the microphone mute state."""
    global mute_button
    if mute_mic_event.is_set():
        mute_mic_event.clear()
        logger.info("[UI] Microphone unmuted")
        root.title("Live Audio Chat")
        mute_button.config(text="Mute Mic", bg="#ff6b6b")  # Red when ready to mute
    else:
        mute_mic_event.set()
        logger.info("[UI] Microphone muted")
        root.title("Live Audio Chat - MIC MUTED")
        mute_button.config(text="Unmute Mic", bg="#2ecc40")  # Green when muted (ready to unmute)

def reset_log_file():
    """Reset the transcription log file with a new timestamp."""
    global g_trans_file_name
    
    # Create new filename with current timestamp
    new_filename = f"{g_output_dir}/transcription-{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Initialize the new transcription log
    try:
        with open(new_filename, "w") as f:
            f.write(f"== Transcription Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ==\n\n")
        g_trans_file_name = new_filename
        logger.info("[UI] Reset log file to: %s", g_trans_file_name)
        
        # Clear current transcription display
        g_transcript_store.set_file_path(new_filename)
        g_transcript_store.clear()

        if g_assistant:
            g_assistant.start_new_thread()
        root.after(0, update_chat)
        
    except Exception as e:
        logger.error("[UI] Error creating new log file: %s", e)


def load_context_file():
    """Load background context from context.md if it exists."""
    context_file = "context.md"
    if os.path.exists(context_file):
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context = f.read().strip()
            if context:
                logger.info("[Context] Loaded %s characters from %s", len(context), context_file)
                return context
        except Exception as e:
            logger.warning("[Context] Error loading %s: %s", context_file, e)
    else:
        logger.info("[Context] No context file found at %s", context_file)
    return None

def send_custom_prompt():
    """Send a custom prompt from the user to the assistant."""
    if not g_assistant:
        logger.warning("[UI] Assistant is not available; cannot send prompt.")
        return
    
    prompt_text = custom_prompt_entry.get().strip()
    if not prompt_text:
        logger.info("[UI] Empty prompt, nothing to send.")
        return
    
    logger.info("[UI] Sending custom prompt.")
    timestamp = time.time()
    
    # Add the custom prompt as a user message
    g_assistant.add_custom_prompt(timestamp, prompt_text, g_my_name)
    
    # Clear the text field
    custom_prompt_entry.delete(0, tk.END)
    
    # Trigger the assistant to answer
    success = g_assistant.trigger_custom_prompt_answer()
    if not success:
        logger.warning("[UI] Assistant was unable to generate a response to custom prompt.")

def generate_summary():
    """Generate a detailed meeting summary using AI and save to markdown file."""
    global summary_button
    
    if not g_assistant:
        logger.warning("[UI] Assistant is not available; cannot generate summary.")
        return
    
    transcription_snapshot = g_transcript_store.snapshot()

    if not transcription_snapshot:
        logger.info("[UI] No transcription available to summarize.")
        return
    
    logger.info("[UI] Generating meeting summary...")
    if summary_button:
        summary_button.config(state=tk.DISABLED, text="Generating...")
    
    def generate_and_save():
        try:
            # Get the full transcript
            transcript = ""
            for entry in transcription_snapshot:
                user, text, start_time = entry
                transcript += f"{user}: {text}\n"
            
            # Load context file
            context = load_context_file()
            
            # Generate summary with title
            meeting_title_snapshot = _get_meeting_title_snapshot()
            if meeting_title_snapshot:
                logger.info("[UI] Passing meeting title to AI: '%s'", meeting_title_snapshot)
            else:
                logger.info("[UI] No meeting title detected, generating without title context")
            summary_data = g_assistant.generate_meeting_summary(transcript, meeting_title=meeting_title_snapshot, context=context)
            
            if summary_data:
                title = summary_data.get('title', 'Meeting Summary')
                summary = summary_data.get('summary', '')
                
                # Create filename with timestamp and title
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                # Sanitize title for filename
                safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '' for c in title)
                safe_title = safe_title.replace(' ', '_').lower()[:50]  # Limit length
                filename = f"{g_summaries_dir}/{timestamp}_{safe_title}.md"
                
                # Save to file
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"# {title}\n\n")
                    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(summary)
                
                logger.info("[UI] Summary saved to: %s", filename)
            else:
                logger.warning("[UI] Failed to generate summary.")
        except Exception as e:
            logger.exception("[UI] Error generating summary: %s", e)
        finally:
            if summary_button:
                root.after(0, lambda: summary_button.config(state=tk.NORMAL, text="Generate Summary"))
    
    # Run in a separate thread to avoid blocking UI
    Thread(target=generate_and_save, daemon=True).start()


# GUI Setup
def setup_ui():
    global chat_window, root, mute_button, assistant_buttons, custom_prompt_entry, send_prompt_button, summary_button, status_label
    
    root = tk.Tk()
    root.title("Live Audio Chat")
    root.geometry("600x400")
    
    # Create button frame at the top
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
    
    # Create Mute button
    mute_button = tk.Button(button_frame, text="Mute Mic", command=toggle_mute, 
                           bg="#ff6b6b", fg="white", font=("Arial", g_default_font_size, "bold"))
    mute_button.pack(side=tk.LEFT, padx=(0, 5))
    
    # Create Reset button
    reset_button = tk.Button(button_frame, text="Reset Log", command=reset_log_file,
                            bg="#4ecdc4", fg="white", font=("Arial", g_default_font_size, "bold"))
    reset_button.pack(side=tk.LEFT, padx=(0, 5))
    
    # Create Generate Summary button
    if g_assistant:
        summary_button = tk.Button(button_frame, text="Generate Summary", command=generate_summary,
                                bg="#9b59b6", fg="white", font=("Arial", g_default_font_size, "bold"))
        summary_button.pack(side=tk.LEFT, padx=(0, 5))

    if g_assistant:
        # Create custom prompt input field
        prompt_frame = tk.Frame(button_frame)
        prompt_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        custom_prompt_entry = tk.Entry(prompt_frame, font=("Arial", g_default_font_size))
        custom_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        custom_prompt_entry.bind('<Return>', lambda event: send_custom_prompt())
        
        send_prompt_button = tk.Button(
            prompt_frame,
            text="Send to AI",
            command=send_custom_prompt,
            bg="#1f8ef1",
            fg="white",
            font=("Arial", g_default_font_size, "bold"),
        )
        send_prompt_button.pack(side=tk.LEFT)
        
        assistant_buttons = {}  # Keep empty dict for compatibility
    else:
        assistant_buttons = {}
    
    chat_window = tk.Text(root, wrap=tk.WORD, state=tk.DISABLED)
    chat_window.pack(expand=True, fill=tk.BOTH, padx=5, pady=(0, 5))
    chat_window.tag_config("microphone", foreground="blue")
    chat_window.tag_config("output", foreground="green")
    chat_window.tag_config("agent", foreground="gray", font=("Arial", g_agent_font_size, "bold"))

    status_label = tk.Label(
        root,
        text="Status: Ready",
        anchor="w",
        bg="#ecf0f1",
        fg="#2d3436",
        font=("Arial", max(8, g_default_font_size - 1)),
        padx=6,
        pady=3,
    )
    status_label.pack(side=tk.BOTTOM, fill=tk.X)
    set_status("Ready", "info")
    
    # Bind extra events to see when the window is being hidden/destroyed.
    def on_destroy(event):
        logger.debug("[Tkinter] Window destroy event triggered: %s", event)
    def on_unmap(event):
        logger.debug("[Tkinter] Window unmap (hidden) event triggered: %s", event)
    root.bind("<Destroy>", on_destroy)
    root.bind("<Unmap>", on_unmap)
    '''
    # Bind key press events for manual splitting.
    def on_split_key(event):
        global flush_counter
        with flush_counter_lock:
            flush_counter += 1
        print("[UI] Split key pressed. Flushing current audio buffers.")
    root.bind("<KeyPress-s>", on_split_key)
    root.bind("<KeyPress-S>", on_split_key)
    '''
    # Bind key press events for any letter key a-z
    # Only enable manual splits if assistant is not available (to avoid conflicts with text entry)
    def on_key_press(event):
        global flush_letter_mic, flush_letter_out
        with flush_letter_lock:
            flush_letter_mic = event.char
            flush_letter_out = event.char
        logger.debug("[UI] Key '%s' pressed. Flushing current audio buffers.", event.char)

    if g_interupt_manually and not g_assistant:
        for key in "abcdefghijklmnopqrstuvwxyz":
            root.bind(f"<KeyPress-{key}>", on_key_press)
            root.bind(f"<KeyPress-{key.upper()}>", on_key_press)  # Also bind uppercase versions
    #--------------
    root.protocol("WM_DELETE_WINDOW", stop_recording)
    logger.info("[GUI] UI setup complete.")
    
    return root

def update_chat():
    transcription_snapshot = g_transcript_store.snapshot()
    ui_module.render_transcription(chat_window, transcription_snapshot, g_my_name, AGENT_NAME)

# Audio Recording Functions
def store_audio_stream(queue, filename_suffix, device_info, from_microphone):
    audio_capture_module.store_audio_stream(
        queue=queue,
        filename_suffix=filename_suffix,
        device_info=device_info,
        stop_event=stop_event,
        sample_size_getter=lambda: g_sample_size,
        transcribe_callback=transcribe_and_display,
        logger=logger,
    )

def collect_from_stream(queue, input_device, p_instance, from_microphone):
    global flush_letter_mic, flush_letter_out

    def get_flush_letters():
        return flush_letter_mic, flush_letter_out

    def clear_flush_letters(is_microphone: bool):
        global flush_letter_mic, flush_letter_out
        if is_microphone:
            flush_letter_mic = None
        else:
            flush_letter_out = None

    audio_capture_module.collect_from_stream(
        queue=queue,
        input_device=input_device,
        p_instance=p_instance,
        from_microphone=from_microphone,
        stop_event=stop_event,
        mute_mic_event=mute_mic_event,
        flush_lock=flush_letter_lock,
        get_flush_letters=get_flush_letters,
        clear_flush_letters=clear_flush_letters,
        speaker_snapshot_getter=_get_speaker_snapshot,
        speaker_setter=_set_current_speaker,
        frame_duration_ms=FRAME_DURATION_MS,
        silence_threshold=SILENCE_THRESHOLD,
        silence_duration=SILENCE_DURATION,
        record_seconds=RECORD_SECONDS,
        logger=logger,
    )

def transcribe_and_display(file, from_microphone, letter):
    if not letter:
        letter = "?"
    #print(f"[Transcribe] Starting transcription for {file}.")
    file_size = os.path.getsize(file)  # Size in bytes
    logger.debug("[Transcribe] File size: %.2f MB", file_size / (1024 * 1024))
    try:
        start_time = float(file.split("/")[-1].split("-")[0])
        segments = g_transcript_ai.transcribe(file)
        new_segments = False
        #with g_transcription_lock:
        for segment in segments:
            text = segment.text.strip()
            if len(text) > 0:
                #print(f"Segment: {segment}") 
                # FastWhisper often returns the following text values which are not an actual transcriptions but halucinations.
                if text == 'Bye.'  or text == 'Um' or text.startswith("Thanks for watching") or text.startswith("Thank you for watching") or text.startswith("Thanks for listening") or text.startswith("Thank you for joining") \
                        or text.startswith("Thank you very much") or text.startswith("Thank you for tuning in") or text == 'Paldies!' or text =="Thank you." or text == '.' or text == 'You' \
                        or text.find("please subscribe to my channel")>=0 or text.startswith("Thank you guys.") \
                        or text.find("www.NorthstarIT.co.uk")>=0 or text.find("Amara.org")>=0 or text.find("I'll see you in the next video")>=0 or text.find("brandhagen10.com") >=0 or text.find("WWW.ABERCAP.COM")>=0 \
                    or (g_keywords and text.find(g_keywords)>=0):
                    continue
                converted_time = datetime.fromtimestamp(segment.start)
                #print(converted_time)  # Outputs in a readable format
                logger.info("[%s -> %.2f] %s", converted_time, segment.end, segment.text)
                #root.after(0, update_chat, transcription, letter, from_microphone)
                if from_microphone:
                    add_transcription(g_my_name, text, start_time + segment.start)
                else:
                    if len(letter)==1:
                        add_transcription(f"Person_{letter.upper()}", text, start_time + segment.start)
                    else:
                        add_transcription(f"{letter}", text, start_time + segment.start)

    except Exception as e:
        logger.exception("[Transcribe] Transcription error for %s: %s", file, e)

def add_transcription(user, text, start_time):
    """
    Add a transcription entry to the global transcription list.
    """
    g_transcriptions_in.put((user, text, start_time))
    if g_assistant:
        g_assistant.add_message(start_time+1,f"{user}: {text}")

def update_screen_on_new_transcription():
    """
    Update the transcription display in the chat window.
    """
    while not stop_event.is_set():
        try:
            user, text, start_time = g_transcriptions_in.get(block=True, timeout=1)
        except Empty:
            continue
        g_transcript_store.append_to_file_if_user(user, text)
        g_transcript_store.add(user, text, start_time)
        root.after(0, update_chat)

# Stop Recording
def stop_recording():
    logger.info("[Stop] Stop recording triggered!")
    stop_event.set()
    global_audio.terminate()
    root.destroy()
    
# Initialize Recording
def initialize_recording():
    try:
        with pyaudio.PyAudio() as p:
            global g_device_in, g_device_out
            if sys.platform == "win32":
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                g_device_in = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
                g_device_out = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])

                if not g_device_out.get("isLoopbackDevice", False):
                    for loopback in p.get_loopback_device_info_generator():
                        if g_device_out["name"] in loopback.get("name", ""):
                            g_device_out = loopback
                            break
            else:
                # Fallback: Pick default input/output devices
                g_device_in = p.get_default_input_device_info()
                g_device_out = p.get_default_output_device_info()
        logger.info("[Init] Devices initialized successfully. Recording device: %s Output device: %s", g_device_in["name"], g_device_out["name"])
    except Exception as e:
        logger.error("[Init] Error initializing devices: %s", e)
        return False
    return True

def handler(signum, frame):
    logger.info("Ctrl-C was pressed.")
    if g_assistant:
        g_assistant.stop()
    stop_recording()
    exit(1)

def main():
    logger.info("[Main] Starting application...")
    signal.signal(signal.SIGINT, handler)
    os.makedirs("output", exist_ok=True)
    root = setup_ui()
    if initialize_recording():
        logger.info("[Main] Recording initialized. Launching UI.")
        threads = [
            Thread(target=store_audio_stream, args=(g_recordings_in, "in", g_device_in, True)),
            Thread(target=store_audio_stream, args=(g_recordings_out, "out", g_device_out, False)),
            Thread(target=collect_from_stream, args=(g_recordings_in, g_device_in, global_audio, True)),
            Thread(target=collect_from_stream, args=(g_recordings_out, g_device_out, global_audio, False)),
            Thread(target=get_speaker_name),
            Thread(target=update_screen_on_new_transcription),
        ]
        for thread in threads:
            thread.start()
            logger.info("[Main] Started thread: %s", thread.name)

        try:
            logger.info("[Main] Starting Tkinter main loop...")
            root.mainloop()
            logger.info("[Main] Tkinter loop has exited.")
        except Exception as e:
            logger.exception("[Main] Error in Tkinter loop: %s", e)

        for thread in threads:
            thread.join()
            logger.info("[Main] Thread %s joined.", thread.name)
        logger.info("[Main] All threads have finished. Exiting.")
    else:
        logger.error("[Main] Failed to initialize recording. Exiting...")


if __name__ == "__main__":
    main()
