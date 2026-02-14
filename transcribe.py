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
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import struct

if sys.platform == "win32":
    from pywinauto import Application
import re
import assistant
#import fastwhisper_transcribe
import openai_transcribe

print("Loading configuration from '.env' file...")


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
    print("\n=== Startup Configuration Summary ===")
    print(f"Platform: {sys.platform}")
    print(f"User name: {g_my_name}")
    print(f"Language: {g_language}")
    print(f"Manual interrupt enabled: {g_interupt_manually}")
    print(f"Transcript model: {g_openai_model_for_transcript}")
    print(f"Keywords configured: {'yes' if g_keywords else 'no'}")
    print(f"OpenAI API key configured: {'yes' if g_open_api_key else 'no'}")
    print(f"Assistant enabled: {'yes' if g_assistant else 'no'}")
    print(f"Vector store configured: {'yes' if g_vector_store_id else 'no'}")
    print(f"Teams integration available: {'yes' if g_ms_teams_app is not None else 'no'}")
    if g_startup_warnings:
        print("Warnings:")
        for warning in g_startup_warnings:
            print(f"  - {warning}")
    else:
        print("Warnings: none")
    print("=====================================\n")


# Constants
RECORD_SECONDS = 300 
#RECORD_SECONDS = 120  # Updated from 5 to 60 seconds (5 minutes = 55 MB, OpenAPI only support up to 25 MB files)

SILENCE_THRESHOLD = 50  # Adjust this based on your microphone sensitivity
SILENCE_DURATION = 1.0  # seconds

# Global Queues & Event
g_recordings_in = Queue()
g_recordings_out = Queue()
g_transcriptions_in = Queue()
stop_event = Event()
mute_mic_event = Event()  # Event to control microphone muting
g_transcription = []
g_transcription_lock = Lock()

g_device_in = {}
g_device_out = {}
g_sample_size = 0

# Global flush counter (for manual splits) and its lock
flush_letter_mic = None
flush_letter_out = None
flush_letter_lock = Lock()
g_speaker_state_lock = Lock()

load_dotenv()  # Load variables from .env file

g_startup_warnings: list[str] = []

g_my_name = _env_str("YOUR_NAME", "You")
if g_my_name == "You":
    g_startup_warnings.append("YOUR_NAME is missing in .env. Using fallback name 'You'.")
print(f"Your name is: {g_my_name}")
AGENT_NAME="Agent"

g_language = _env_str("LANGUAGE", "en")
print(f"Language: {g_language}")

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
        )
        if g_vector_store_id:
            print(f"Assistant configured with vector store ID: {g_vector_store_id}")
        else:
            print("Assistant configured without vector store.")
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
    print(f"Using keywords for initial prompt: {g_keywords}")

g_agent_font_size = _env_int("AGENT_FONT_SIZE", 14, g_startup_warnings)
print(f"Agent font size: {g_agent_font_size}")

g_default_font_size = _env_int("DEFAULT_FONT_SIZE", 10, g_startup_warnings)
print(f"Default font size: {g_default_font_size}")

g_transcript_ai = None
g_openai_model_for_transcript = _env_str("OPENAI_MODEL_FOR_TRANSCRIPT", "gpt-4o-mini-transcribe")
print(f"Using OpenAI model for transcript: {g_openai_model_for_transcript}")
try:
    g_transcript_ai = openai_transcribe.OpenAITranscribe(
        model=g_openai_model_for_transcript,
        keywords=g_keywords,
        language=g_language,
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
            print(f"Deleted old file: {file_path}")
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

# Clear the file and initialize the transcription log
g_trans_file_name = f"{g_output_dir}/transcription-{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
try:
    with open(g_trans_file_name, "w") as f:
        f.write(f"== Transcription Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ==\n\n")
except Exception as e:
    g_startup_warnings.append(f"Failed to create transcription log file '{g_trans_file_name}': {e}")

global_audio = pyaudio.PyAudio()

g_window_name = "Meeting compact view*"
g_ms_teams_app = None
if sys.platform == "win32":
    try:
        # Connect to Microsoft Teams by title, or you can use process ID or path
        g_ms_teams_app = Application(backend="uia").connect(title_re=g_window_name)
    except Exception as e:
        g_startup_warnings.append(f"Teams integration not connected at startup: {e}")

_print_startup_summary()

g_previous_speaker = None
g_current_speaker = None
g_meeting_title = None


def _get_speaker_snapshot():
    """Return a thread-safe snapshot of speaker state (previous, current)."""
    with g_speaker_state_lock:
        return g_previous_speaker, g_current_speaker


def _set_current_speaker(new_speaker: str | None):
    """Update speaker state atomically and return (previous, current)."""
    global g_previous_speaker, g_current_speaker
    with g_speaker_state_lock:
        g_previous_speaker = g_current_speaker
        g_current_speaker = new_speaker
        return g_previous_speaker, g_current_speaker


def _get_meeting_title_snapshot() -> str | None:
    with g_speaker_state_lock:
        return g_meeting_title

def get_ms_teams_window_title():
    '''Get the MS Teams window title which usually contains the meeting name.'''
    global g_ms_teams_app, g_meeting_title
    if sys.platform != "win32":
        return None
    try:
        if g_ms_teams_app==None:
            g_ms_teams_app = Application(backend="uia").connect(title_re=g_window_name)
        
        teams_window = g_ms_teams_app.window(title_re=g_window_name)
        window_title = teams_window.window_text()
        
        # Clean up the title - remove "Meeting compact view" or similar suffixes
        if window_title:
            # Remove common Teams window suffixes
            title = window_title.replace("Meeting compact view", "").strip()
            title = title.replace(" | Microsoft Teams", "").strip()
            title = title.replace("|", "").strip()
            
            # Remove leading/trailing separators
            title = title.strip(" -|")
            
            with g_speaker_state_lock:
                previous_title = g_meeting_title

            if title and title != previous_title:
                # Only update and log if we got a new non-empty title
                old_title = previous_title
                with g_speaker_state_lock:
                    g_meeting_title = title
                if old_title is None:
                    print(f"[Teams] Meeting title detected: '{title}'")
                else:
                    print(f"[Teams] Meeting title updated: '{old_title}' -> '{title}'")
                return title
            elif title:
                # Same title as before, don't log but return it
                return title
    except Exception as e:
        # Don't log every error, only when debugging
        pass
    
    # Return the last known title even if we can't detect it now
    return _get_meeting_title_snapshot()

def inspect_ms_teams():
    global g_ms_teams_app
    if sys.platform != "win32":
        return None
    try:
        if g_ms_teams_app==None:
            g_ms_teams_app = Application(backend="uia").connect(title_re=g_window_name)

        # Grab the main Teams window (regex to match partial titles)
        teams_window = g_ms_teams_app.window(title_re=g_window_name)

        # Print the hierarchy of controls for debugging
        teams_window.print_control_identifiers(filename="teams_controls.txt")
        with open("teams_controls.txt", "r") as f:
            data = f.read()
            return data
    except:
        pass
    return None

def get_speaker_name():
    '''Get the current speaker's name from the MS Teams window.'''
    global flush_letter_mic, flush_letter_out
    
    title_check_counter = 0
    TITLE_CHECK_INTERVAL = 50  # Check for title every 50 iterations (5 seconds)
    
    while not stop_event.is_set():
        # Periodically check for meeting title (it's only available when Teams is minimized)
        title_check_counter += 1
        if title_check_counter >= TITLE_CHECK_INTERVAL:
            title_check_counter = 0
            new_title = get_ms_teams_window_title()
            previous_title = _get_meeting_title_snapshot()
            if new_title and new_title != previous_title:
                print(f"[Teams] Detected new meeting: {new_title}")
        
        data = inspect_ms_teams()
        if data:
            # Use regex to find the line with the speaker's information.
            # This pattern looks for a line with "MenuItem - '...video is on..." and captures the content within quotes.
            # Regex pattern: look for "Recording started by" followed by the name (letters and spaces) ending with a dot.
            patterns = [r"MenuItem\s*-\s*'([^']*video is on[^']*)'",
                        r"MenuItem\s+-\s+'([^,]+), Context menu is available'"]

            match = False
            name_parts=[]

            for pattern in patterns:
                match = re.search(pattern, data)
                if match:
                    # Extract the full string e.g. "Mathieu Cornille, video is on, Context menu is available"
                    info = match.group(1)
                    # The speaker's name is assumed to be the first comma-separated token.
                    speaker = info.split(",")[0].strip()
                    #print("Current speaker:", speaker)
                    name_parts = speaker.split(" ")
                    break

            if match:
                if len(name_parts)>1 and name_parts[1][0]!='(':
                    detected_speaker = name_parts[0] + " " + name_parts[1][0]  # only the first name and first letter of the last name
                else:
                    detected_speaker = name_parts[0]
            else:
                detected_speaker = None
        else:
            detected_speaker = None

        previous_speaker, current_speaker = _set_current_speaker(detected_speaker)
        if previous_speaker != current_speaker:
            print(f"New speaker: {current_speaker} (was {previous_speaker})")
            if previous_speaker:
                with flush_letter_lock:
                    flush_letter_mic = '_'
                    flush_letter_out = '_'
        time.sleep(0.1)
        #print("Speaker not found.")

def toggle_mute():
    """Toggle the microphone mute state."""
    global mute_button
    if mute_mic_event.is_set():
        mute_mic_event.clear()
        print("[UI] Microphone unmuted")
        root.title("Live Audio Chat")
        mute_button.config(text="Mute Mic", bg="#ff6b6b")  # Red when ready to mute
    else:
        mute_mic_event.set()
        print("[UI] Microphone muted")
        root.title("Live Audio Chat - MIC MUTED")
        mute_button.config(text="Unmute Mic", bg="#2ecc40")  # Green when muted (ready to unmute)

def reset_log_file():
    """Reset the transcription log file with a new timestamp."""
    global g_trans_file_name, g_transcription
    
    # Create new filename with current timestamp
    new_filename = f"{g_output_dir}/transcription-{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Initialize the new transcription log
    try:
        with open(new_filename, "w") as f:
            f.write(f"== Transcription Log ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ==\n\n")
        g_trans_file_name = new_filename
        print(f"[UI] Reset log file to: {g_trans_file_name}")
        
        # Clear current transcription display
        with g_transcription_lock:
            g_transcription = []

        if g_assistant:
            g_assistant.start_new_thread()
        root.after(0, update_chat)
        
    except Exception as e:
        print(f"[UI] Error creating new log file: {e}")


def load_context_file():
    """Load background context from context.md if it exists."""
    context_file = "context.md"
    if os.path.exists(context_file):
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context = f.read().strip()
            if context:
                print(f"[Context] Loaded {len(context)} characters from {context_file}")
                return context
        except Exception as e:
            print(f"[Context] Error loading {context_file}: {e}")
    else:
        print(f"[Context] No context file found at {context_file}")
    return None

def send_custom_prompt():
    """Send a custom prompt from the user to the assistant."""
    if not g_assistant:
        print("[UI] Assistant is not available; cannot send prompt.")
        return
    
    prompt_text = custom_prompt_entry.get().strip()
    if not prompt_text:
        print("[UI] Empty prompt, nothing to send.")
        return
    
    print(f"[UI] Sending custom prompt: {prompt_text}")
    timestamp = time.time()
    
    # Add the custom prompt as a user message
    g_assistant.add_custom_prompt(timestamp, prompt_text, g_my_name)
    
    # Clear the text field
    custom_prompt_entry.delete(0, tk.END)
    
    # Trigger the assistant to answer
    success = g_assistant.trigger_custom_prompt_answer()
    if not success:
        print("[UI] Assistant was unable to generate a response to custom prompt.")

def generate_summary():
    """Generate a detailed meeting summary using AI and save to markdown file."""
    global summary_button
    
    if not g_assistant:
        print("[UI] Assistant is not available; cannot generate summary.")
        return
    
    with g_transcription_lock:
        transcription_snapshot = list(g_transcription)

    if not transcription_snapshot:
        print("[UI] No transcription available to summarize.")
        return
    
    print("[UI] Generating meeting summary...")
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
                print(f"[UI] Passing meeting title to AI: '{meeting_title_snapshot}'")
            else:
                print("[UI] No meeting title detected, generating without title context")
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
                
                print(f"[UI] Summary saved to: {filename}")
            else:
                print("[UI] Failed to generate summary.")
        except Exception as e:
            print(f"[UI] Error generating summary: {e}")
        finally:
            if summary_button:
                root.after(0, lambda: summary_button.config(state=tk.NORMAL, text="Generate Summary"))
    
    # Run in a separate thread to avoid blocking UI
    Thread(target=generate_and_save, daemon=True).start()


# GUI Setup
def setup_ui():
    global chat_window, root, mute_button, assistant_buttons, custom_prompt_entry, send_prompt_button, summary_button
    
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
    
    # Bind extra events to see when the window is being hidden/destroyed.
    def on_destroy(event):
        print("[Tkinter] Window destroy event triggered:", event)
    def on_unmap(event):
        print("[Tkinter] Window unmap (hidden) event triggered:", event)
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
        print(f"[UI] Key '{event.char}' pressed. Flushing current audio buffers.")

    if g_interupt_manually and not g_assistant:
        for key in "abcdefghijklmnopqrstuvwxyz":
            root.bind(f"<KeyPress-{key}>", on_key_press)
            root.bind(f"<KeyPress-{key.upper()}>", on_key_press)  # Also bind uppercase versions
    #--------------
    root.protocol("WM_DELETE_WINDOW", stop_recording)
    print("[GUI] UI setup complete.")
    
    return root

def update_chat():
    with g_transcription_lock:
        transcription_snapshot = list(g_transcription)

    chat_window.config(state=tk.NORMAL)
    chat_window.delete("1.0", tk.END)
    last_user=''
    for entry in transcription_snapshot:
        user, text, start_time = entry
        if user != last_user:
            if user == g_my_name:
                chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "microphone")
            elif user == AGENT_NAME:
                chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "agent")
            else:
                chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "output")

            last_user=user
        else:
            if user == g_my_name:
                chat_window.insert(tk.END, f"{text}\n", "microphone")
            elif user == AGENT_NAME:
                chat_window.insert(tk.END, f"{text}\n", "agent")
            else:
                chat_window.insert(tk.END, f"{text}\n", "output")
 
    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)

# Audio Recording Functions
def store_audio_stream(queue, filename_suffix, device_info, from_microphone):
    print(f"[{filename_suffix}] store_audio_stream started.")
    while not stop_event.is_set():
        try:
            #print(f"[{filename_suffix}] Waiting for frames from queue...")
            frames, letter, start_time = queue.get(block=True, timeout=1)
            print(f"[{filename_suffix}] Got {len(frames)} frames.")
        except Empty:
            continue
        except Exception as e:
            print(f"[{filename_suffix}] Exception while getting frames: {e}")
            continue
        
        filename = f'output/{start_time:.2f}-{filename_suffix}.wav'
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(device_info["maxInputChannels"])
                wf.setsampwidth(g_sample_size)
                wf.setframerate(int(device_info["defaultSampleRate"]))
                wf.writeframes(b"".join(frames))
            print(f"[{filename_suffix}] Wrote audio to {filename}.")
        except Exception as e:
            print(f"[{filename_suffix}] Error writing WAV file: {e}")
            continue

        transcribe_and_display(filename, filename_suffix == "in", letter)
        try:
            os.remove(filename)
            print(f"[{filename_suffix}] Removed temporary file {filename}.")
        except Exception as e:
            print(f"[{filename_suffix}] Error removing file {filename}: {e}")

    print(f"[{filename_suffix}] store_audio_stream exiting.")

def collect_from_stream(queue, input_device, p_instance, from_microphone):
    global g_sample_size
    global flush_letter_mic, flush_letter_out
    print(f"[{input_device['name']}] Starting collect_from_stream...")
    try:
        print(f"[{input_device['name']}] About to open audio stream...")
        frame_rate = int(input_device["defaultSampleRate"])
        print(f"[{input_device['name']}] Opened audio stream at {frame_rate} Hz.")
        print(f"[{input_device['name']}] Input channels: {input_device['maxInputChannels']}")
        print(f"[{input_device['name']}] Sample size (bytes): {p_instance.get_sample_size(pyaudio.paInt16)}")
        print(f"[{input_device['name']}] Sample format: {pyaudio.paInt16}")
        print(f"[{input_device['name']}] Chunk size: {int(frame_rate * 0.1)} samples")
        print(f"[{input_device['name']}] Frames per buffer: {int(frame_rate * 0.1)} samples")
        print(f"[{input_device['name']}] Input device index: {input_device['index']}")
        FRAME_DURATION_MS = 100
        chunk_size = int(frame_rate * FRAME_DURATION_MS / 1000)  # 20 ms worth of samples
        with p_instance.open(format=pyaudio.paInt16,
                             channels=input_device["maxInputChannels"],
                             rate=frame_rate,
                             frames_per_buffer=chunk_size,
                             input=True,
                             input_device_index=input_device["index"]) as stream:
            print(f"[{input_device['name']}] Audio stream opened successfully.")
            g_sample_size = p_instance.get_sample_size(pyaudio.paInt16)
            print(f"Global sample size: {g_sample_size}")
            frames = []
            start_time = time.time()
            print(f"[{input_device['name']}] Starting to read data...")
            silence_start_time = None
            silence_frame_count = 0
            while not stop_event.is_set():
                try:
                    if len(frames) ==silence_frame_count:
                        start_time = time.time()
                    
                    # Check if microphone is muted (only for microphone input)
                    if from_microphone and mute_mic_event.is_set():
                        # Skip reading audio data when muted, but keep the loop running
                        time.sleep(0.001)  # Small sleep to prevent busy waiting
                        continue
                    
                    data = stream.read(chunk_size, exception_on_overflow=False)
                    frames.append(data)
                    
                    
                    with flush_letter_lock:
                        if from_microphone:
                            current_letter = flush_letter_mic
                            flush_letter_mic=None
                        else:
                            current_letter = flush_letter_out
                            flush_letter_out=None
                    if current_letter:
                        last_letter=current_letter
                        if len(frames)>0:
                            if last_letter=='_' and from_microphone==False:
                                SECONDS_TO_GO_BACK = 2.0
                                frames_to_remove = int((1000/FRAME_DURATION_MS) * SECONDS_TO_GO_BACK)
                                frames_to_proess = frames[:-frames_to_remove]
                                previous_speaker_snapshot, _ = _get_speaker_snapshot()
                                queue.put((frames_to_proess.copy(), previous_speaker_snapshot, start_time))
                                frames = frames[-frames_to_remove:]
                                start_time = time.time() - (frames_to_remove*FRAME_DURATION_MS)/1000
                            elif last_letter != '_':
                                print(f"[{input_device['name']}] Manual split triggered; flushing {len(frames)} frames.")
                                _set_current_speaker(last_letter)
                                queue.put((frames.copy(), last_letter, start_time))
                                frames = []
                            silence_frame_count=0
              
                    #-- Silence check
                    # Convert raw bytes to numpy array
                    audio_samples = np.array(struct.unpack(f"{len(data)//2}h", data))
                    # Calculate the volume (RMS)
                    volume = np.sqrt(np.mean(audio_samples**2))

                    # Check if volume is below threshold
                    if volume < SILENCE_THRESHOLD:
                        silence_frame_count+=1
                        if silence_start_time is None:
                            silence_start_time = time.time()  # Mark when silence starts
                        elif time.time() - silence_start_time >= SILENCE_DURATION:
                            #print(f"[{input_device['name']}] Silence detected for {SILENCE_DURATION} seconds.")
                            if len(frames) ==silence_frame_count:
                                #print(f"[{input_device['name']}] No audio detected for {SILENCE_DURATION} seconds. Ignoring.")
                                pass
                            else:
                                _, current_speaker_snapshot = _get_speaker_snapshot()
                                queue.put((frames.copy(), current_speaker_snapshot, start_time))
                            frames = []
                            silence_frame_count=0
                            silence_start_time = None  # Reset silence timer
                    else:
                        silence_start_time = None  # Reset if we detect sound

                    #-- Time check
                    # If we've reached the RECORD_SECONDS duration, flush automatically.
                    if len(frames) >= int((frame_rate * RECORD_SECONDS) / chunk_size):
                        print(f"[{input_device['name']}] Auto split after reaching {RECORD_SECONDS} seconds; queueing {len(frames)} frames.")
                        _, current_speaker_snapshot = _get_speaker_snapshot()
                        queue.put((frames.copy(), current_speaker_snapshot, start_time))
                        frames = []
                        silence_frame_count=0
                except Exception as e:
                    print(f"[{input_device['name']}] Error reading from stream: {e}")
                    break
            print(f"[{input_device['name']}] Exiting reading loop.")
    except Exception as e:
        print(f"[{input_device.get('name', 'Unknown')}] Failed to open audio stream: {e}")
    print(f"[{input_device.get('name', 'Unknown')}] collect_from_stream exiting.")

def transcribe_and_display(file, from_microphone, letter):
    global g_transcription
    if not letter:
        letter = "?"
    #print(f"[Transcribe] Starting transcription for {file}.")
    file_size = os.path.getsize(file)  # Size in bytes
    print(f"[Transcribe] File size: {file_size / (1024 * 1024):.2f} MB")
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
                        or text.find(g_keywords)>=0:
                    continue
                converted_time = datetime.fromtimestamp(segment.start)
                #print(converted_time)  # Outputs in a readable format
                print(f"[{converted_time} -> {segment.end:.2f}] {segment.text}")
                #root.after(0, update_chat, transcription, letter, from_microphone)
                if from_microphone:
                    add_transcription(g_my_name, text, start_time + segment.start)
                else:
                    if len(letter)==1:
                        add_transcription(f"Person_{letter.upper()}", text, start_time + segment.start)
                    else:
                        add_transcription(f"{letter}", text, start_time + segment.start)

    except Exception as e:
        print(f"[Transcribe] Transcription error for {file}: {e}")

def add_transcription(user, text, start_time):
    """
    Add a transcription entry to the global transcription list.
    """
    g_transcriptions_in.put((user, text, start_time))
    if g_assistant:
        g_assistant.add_message(start_time+1,f"{user}: {text}")

def update_screen_on_new_transcription():
    global g_transcription
    """
    Update the transcription display in the chat window.
    """
    while not stop_event.is_set():
        try:
            user, text, start_time = g_transcriptions_in.get(block=True, timeout=1)
        except Empty:
            continue
        try:
            if user != AGENT_NAME:
                with open(g_trans_file_name, "a") as f:
                    f.write(f"{user}: {text}\n\n")
        except:
            pass
        with g_transcription_lock:
            g_transcription.append([user, text, start_time])
            g_transcription = sorted(g_transcription, key=lambda entry: entry[2])
        root.after(0, update_chat)

# Stop Recording
def stop_recording():
    print("[Stop] Stop recording triggered!")
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
        print("[Init] Devices initialized successfully. Recording device:", g_device_in["name"], "Output device:", g_device_out["name"])
    except Exception as e:
        print(f"[Init] Error initializing devices: {e}")
        return False
    return True

def handler(signum, frame):
    print("Ctrl-C was pressed.", flush=True)
    if g_assistant:
        g_assistant.stop()
    stop_recording()
    exit(1)

if __name__ == "__main__":
    print("[Main] Starting application...")
    signal.signal(signal.SIGINT, handler)
    os.makedirs("output", exist_ok=True)
    root = setup_ui()
    if initialize_recording():
        print("[Main] Recording initialized. Launching UI.")
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
            print(f"[Main] Started thread: {thread.name}")

        try:
            print("[Main] Starting Tkinter main loop...")
            root.mainloop()
            print("[Main] Tkinter loop has exited.")
        except Exception as e:
            print(f"[Main] Error in Tkinter loop: {e}")

        for thread in threads:
            thread.join()
            print(f"[Main] Thread {thread.name} joined.")
        print("[Main] All threads have finished. Exiting.")
    else:
        print("[Main] Failed to initialize recording. Exiting...")
