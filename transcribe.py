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
from tkinter import ttk, filedialog
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
from transcript_filter import TranscriptFilter
from summary_utils import sanitize_title_for_filename
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
    logger.info("Auto-start transcription: %s", g_auto_start_transcription)
    logger.info("Auto-summarize on stop: %s", g_auto_summarize_on_stop)
    logger.info("Assistant web search for custom prompts: %s", g_assistant_web_search_for_custom_prompts)
    logger.info("Transcription output dir: %s", g_output_dir)
    logger.info("Temporary audio dir: %s", g_temp_dir)
    logger.info("Summary output dir: %s", g_summaries_dir)
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
g_transcript_filter = None
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
g_recording_threads: list[Thread] = []
g_is_recording = False
g_devices_initialized = False

start_stop_button = None
settings_button = None

g_auto_start_transcription = False
g_auto_summarize_on_stop = False
g_assistant_web_search_for_custom_prompts = True

g_auto_start_var = None
g_auto_summarize_var = None

mic_icon_on = None
mic_icon_off = None
g_is_closing = False


def _normalize_transcription_text(text: str) -> str:
    """Convert escaped newline sequences to real newlines and normalize line endings."""
    if text is None:
        return ""
    normalized = str(text).replace("\\r\\n", "\\n").replace("\\r", "\\n")
    normalized = normalized.replace("\\\\n", "\n")
    return normalized


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


def _save_env_updates(updates: dict[str, str]) -> None:
    env_path = ".env"
    lines: list[str] = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    remaining = dict(updates)
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue

        key = line.split("=", 1)[0].strip()
        if key in remaining:
            value = str(remaining.pop(key))
            new_lines.append(f"{key}={value}\n")
        else:
            new_lines.append(line)

    for key, value in remaining.items():
        new_lines.append(f"{key}={value}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    for key, value in updates.items():
        os.environ[key] = str(value)


def _drain_queue(q: Queue) -> None:
    while True:
        try:
            q.get_nowait()
        except Empty:
            break


def _create_transcript_ai() -> openai_transcribe.OpenAITranscribe:
    return openai_transcribe.OpenAITranscribe(
        model=g_openai_model_for_transcript,
        keywords=g_keywords,
        language=g_language,
        status_callback=_transcription_status_callback,
    )


def _reinitialize_transcript_ai() -> None:
    global g_transcript_ai
    try:
        g_transcript_ai = _create_transcript_ai()
        logger.info("Transcription model reinitialized with language=%s model=%s", g_language, g_openai_model_for_transcript)
    except Exception as e:
        logger.error("Failed to reinitialize transcription model: %s", e)
        set_status("Transcription reinit failed", "error")

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
g_auto_start_transcription = _env_bool("AUTO_START_TRANSCRIPTION", False, g_startup_warnings)
g_auto_summarize_on_stop = _env_bool("AUTO_SUMMARIZE_ON_STOP", False, g_startup_warnings)
g_assistant_web_search_for_custom_prompts = _env_bool("ASSISTANT_ENABLE_WEB_SEARCH_FOR_CUSTOM_PROMPTS", True, g_startup_warnings)

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
        g_assistant.set_custom_prompt_web_search_enabled(g_assistant_web_search_for_custom_prompts)
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

g_transcript_filter = TranscriptFilter(keywords=g_keywords)

g_agent_font_size = _env_int("AGENT_FONT_SIZE", 14, g_startup_warnings)
logger.info("Agent font size: %s", g_agent_font_size)

g_default_font_size = _env_int("DEFAULT_FONT_SIZE", 10, g_startup_warnings)
logger.info("Default font size: %s", g_default_font_size)

g_transcript_ai = None
g_openai_model_for_transcript = _env_str("OPENAI_MODEL_FOR_TRANSCRIPT", "gpt-4o-mini-transcribe")
logger.info("Using OpenAI model for transcript: %s", g_openai_model_for_transcript)
try:
    g_transcript_ai = _create_transcript_ai()
except Exception as e:
    g_startup_warnings.append(
        f"Failed to initialize OpenAI transcription model '{g_openai_model_for_transcript}': {e}"
    )
    g_openai_model_for_transcript = "gpt-4o-mini-transcribe"
    g_transcript_ai = _create_transcript_ai()
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
g_output_dir = _env_str("OUTPUT_DIR", "output")
if not os.path.exists(g_output_dir):
    os.makedirs(g_output_dir)

# Check if output_summaries dir exists
g_summaries_dir = _env_str("SUMMARIES_DIR", "output_summaries")
if not os.path.exists(g_summaries_dir):
    os.makedirs(g_summaries_dir)

# Check if temporary audio dir exists
g_temp_dir = _env_str("TEMP_DIR", "tmp")
if not os.path.exists(g_temp_dir):
    os.makedirs(g_temp_dir)

# Delete all wav files in the temporary audio directory
for file in os.listdir(g_temp_dir):
    if file.endswith(".wav"):
        file_path = os.path.join(g_temp_dir, file)
        try:
            os.remove(file_path)
            logger.info("Deleted old file: %s", file_path)
        except Exception as e:
            logger.warning("Error deleting file %s: %s", file_path, e)

# Clear the file and initialize the transcription log
g_trans_file_name = f"{g_output_dir}/transcription-{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
try:
    with open(g_trans_file_name, "w", encoding="utf-8") as f:
        f.write(f"# Transcription Log\n\n")
        f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
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


def _create_mic_icons():
    theme = ui_module.THEME
    on_bg = theme["accent_red"]
    off_bg = theme["surface_light"]
    size = 28

    on_icon = tk.PhotoImage(width=size, height=size)
    off_icon = tk.PhotoImage(width=size, height=size)
    on_icon.put(on_bg, to=(0, 0, size, size))
    off_icon.put(off_bg, to=(0, 0, size, size))

    mic_on_color = "#ffffff"
    mic_off_color = theme["text_muted"]

    # Microphone body (scaled)
    for x in range(10, 18):
        for y in range(4, 15):
            on_icon.put(mic_on_color, (x, y))
            off_icon.put(mic_off_color, (x, y))
    # Rounded top hint
    for x in range(11, 17):
        on_icon.put(mic_on_color, (x, 3))
        off_icon.put(mic_off_color, (x, 3))
    # Stem and base
    for y in range(15, 22):
        for x in range(13, 15):
            on_icon.put(mic_on_color, (x, y))
            off_icon.put(mic_off_color, (x, y))
    for x in range(8, 20):
        on_icon.put(mic_on_color, (x, 22))
        off_icon.put(mic_off_color, (x, 22))

    # Mute cross-line
    strike = theme["accent_red"]
    for i in range(4, 24):
        off_icon.put(strike, (i, i))
        if i + 1 < size:
            off_icon.put(strike, (i + 1, i))

    return on_icon, off_icon


def _update_start_stop_button():
    if not start_stop_button:
        return
    theme = ui_module.THEME
    if g_is_recording:
        start_stop_button.config(text="\u25A0  Stop", bg=theme["accent_red"])
        ui_module.add_hover_effect(start_stop_button, theme["accent_red"], theme["accent_red_hover"])
    else:
        start_stop_button.config(text="\u25B6  Start", bg=theme["accent_green"])
        ui_module.add_hover_effect(start_stop_button, theme["accent_green"], theme["accent_green_hover"])


def _sync_auto_start_var():
    global g_auto_start_transcription
    if g_auto_start_var is not None:
        g_auto_start_transcription = bool(g_auto_start_var.get())
        _save_env_updates({"AUTO_START_TRANSCRIPTION": "True" if g_auto_start_transcription else "False"})


def _open_settings():
    global g_language, g_auto_summarize_on_stop, g_output_dir, g_summaries_dir
    global g_assistant_web_search_for_custom_prompts
    global g_my_name, g_interupt_manually, g_keywords, g_openai_model_for_transcript
    global g_agent_font_size, g_default_font_size, g_temp_dir

    win = tk.Toplevel(root)
    win.title("Settings")
    win.geometry("720x620")
    win.minsize(640, 500)
    win.transient(root)
    win.grab_set()

    theme = ui_module.THEME
    win.configure(bg=theme["bg"])
    ui_module.apply_dark_title_bar(win)

    # -- shared style dicts --
    _cb_opts = dict(bg=theme["bg"], fg=theme["text_primary"],
                    selectcolor=theme["surface"], activebackground=theme["bg"],
                    activeforeground=theme["text_primary"], highlightthickness=0,
                    font=ui_module.get_font(g_default_font_size))
    _lbl_opts = dict(bg=theme["bg"], fg=theme["text_secondary"],
                     font=ui_module.get_font(g_default_font_size))
    _entry_opts = dict(bg=theme["input_bg"], fg=theme["input_fg"],
                       insertbackground=theme["text_primary"], relief=tk.FLAT,
                       highlightbackground=theme["border"], highlightcolor=theme["accent_blue"],
                       highlightthickness=1, bd=4,
                       font=ui_module.get_font(g_default_font_size))
    _section_opts = dict(bg=theme["bg"], fg=theme["accent_blue"],
                         font=ui_module.get_font(g_default_font_size, "bold"))
    _browse_opts = dict(bg=theme["surface_light"], fg=theme["text_primary"],
                        relief=tk.FLAT, cursor="hand2", bd=0, highlightthickness=0,
                        font=ui_module.get_font(max(8, g_default_font_size - 1)))
    _spin_opts = dict(bg=theme["input_bg"], fg=theme["input_fg"],
                      buttonbackground=theme["surface_light"],
                      relief=tk.FLAT, highlightthickness=1, highlightbackground=theme["border"],
                      font=ui_module.get_font(g_default_font_size))

    # -- scrollable canvas --
    outer = tk.Frame(win, bg=theme["bg"])
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer, bg=theme["bg"], highlightthickness=0, bd=0)
    scrollbar = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview,
                             bg=theme["surface"], troughcolor=theme["bg"])
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame = tk.Frame(canvas, bg=theme["bg"])
    canvas_window = canvas.create_window((0, 0), window=frame, anchor="nw")

    def _on_frame_configure(_event=None):
        canvas.configure(scrollregion=canvas.bbox("all"))
    frame.bind("<Configure>", _on_frame_configure)

    def _on_canvas_configure(event):
        canvas.itemconfig(canvas_window, width=event.width)
    canvas.bind("<Configure>", _on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", _on_mousewheel)

    pad = {"padx": (20, 20)}
    row = 0

    # ── Section: General ───────────────────────────────────────────────────
    tk.Label(frame, text="General", **_section_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(14, 4), **pad)
    row += 1

    tk.Label(frame, text="Your name", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    name_var = tk.StringVar(value=g_my_name or "")
    tk.Entry(frame, textvariable=name_var, width=30, **_entry_opts).grid(row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Language", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    lang_var = tk.StringVar(value=g_language or "en")
    languages = ["en", "lv", "ru", "de", "fr", "es", "it", "pt", "nl", "pl", "sv", "fi", "et", "lt", "ja", "zh", "ko", "ar", "tr"]
    ttk.Combobox(frame, values=languages, textvariable=lang_var, state="readonly", width=12).grid(
        row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Keywords", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    kw_var = tk.StringVar(value=g_keywords or "")
    tk.Entry(frame, textvariable=kw_var, width=60, **_entry_opts).grid(row=row, column=1, columnspan=2, sticky="we", pady=3)
    row += 1

    interrupt_var = tk.BooleanVar(value=g_interupt_manually)
    tk.Checkbutton(frame, text="Enable manual interrupt (key-press splits audio)", variable=interrupt_var, **_cb_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=3, **pad)
    row += 1

    auto_start_var = tk.BooleanVar(value=g_auto_start_transcription)
    tk.Checkbutton(frame, text="Auto-start transcription when app starts", variable=auto_start_var, **_cb_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=3, **pad)
    row += 1

    auto_sum_var = tk.BooleanVar(value=g_auto_summarize_on_stop)
    tk.Checkbutton(frame, text="Automatically summarize transcription when stopped", variable=auto_sum_var, **_cb_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=3, **pad)
    row += 1

    # ── Section: OpenAI ────────────────────────────────────────────────────
    tk.Label(frame, text="OpenAI", **_section_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(14, 4), **pad)
    row += 1

    tk.Label(frame, text="API key", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    api_key_var = tk.StringVar(value="")
    api_key_entry = tk.Entry(frame, textvariable=api_key_var, width=50, show="\u2022", **_entry_opts)
    api_key_entry.grid(row=row, column=1, columnspan=2, sticky="we", pady=3)
    _api_hint = "configured" if g_open_api_key else "not set"
    tk.Label(frame, text=f"({_api_hint} \u2014 leave blank to keep current)",
             bg=theme["bg"], fg=theme["text_muted"],
             font=ui_module.get_font(max(8, g_default_font_size - 1))).grid(
        row=row + 1, column=1, columnspan=2, sticky="w", **pad)
    row += 2

    tk.Label(frame, text="Transcript model", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    transcript_model_var = tk.StringVar(value=g_openai_model_for_transcript or "gpt-4o-mini-transcribe")
    ttk.Combobox(frame, values=["gpt-4o-mini-transcribe", "gpt-4o-transcribe"],
                 textvariable=transcript_model_var, width=28).grid(row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Assistant model", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    assistant_model_var = tk.StringVar(value=_env_str("OPENAI_MODEL_FOR_ASSISTANT", "gpt-5.2"))
    tk.Entry(frame, textvariable=assistant_model_var, width=28, **_entry_opts).grid(
        row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Vector store ID", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    vs_var = tk.StringVar(value=_env_str("OPENAI_VECTOR_STORE_ID_FOR_ANSWERS", "") or "")
    tk.Entry(frame, textvariable=vs_var, width=50, **_entry_opts).grid(
        row=row, column=1, columnspan=2, sticky="we", pady=3)
    row += 1

    web_search_var = tk.BooleanVar(value=g_assistant_web_search_for_custom_prompts)
    tk.Checkbutton(frame, text="Enable internet search for custom prompts", variable=web_search_var, **_cb_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=3, **pad)
    row += 1

    # ── Section: UI ────────────────────────────────────────────────────────
    tk.Label(frame, text="UI", **_section_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(14, 4), **pad)
    row += 1

    tk.Label(frame, text="Default font size", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    default_fs_var = tk.IntVar(value=g_default_font_size)
    tk.Spinbox(frame, from_=8, to=24, textvariable=default_fs_var, width=6, **_spin_opts).grid(
        row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Agent font size", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    agent_fs_var = tk.IntVar(value=g_agent_font_size)
    tk.Spinbox(frame, from_=8, to=30, textvariable=agent_fs_var, width=6, **_spin_opts).grid(
        row=row, column=1, sticky="w", pady=3)
    row += 1

    # ── Section: Audio ─────────────────────────────────────────────────────
    tk.Label(frame, text="Audio", **_section_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(14, 4), **pad)
    row += 1

    tk.Label(frame, text="Record seconds", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    rec_var = tk.IntVar(value=RECORD_SECONDS)
    tk.Spinbox(frame, from_=10, to=3600, textvariable=rec_var, width=8, **_spin_opts).grid(
        row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Silence threshold", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    sil_thresh_var = tk.DoubleVar(value=SILENCE_THRESHOLD)
    tk.Spinbox(frame, from_=0, to=500, increment=5, textvariable=sil_thresh_var, width=8, **_spin_opts).grid(
        row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Silence duration (s)", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    sil_dur_var = tk.DoubleVar(value=SILENCE_DURATION)
    tk.Spinbox(frame, from_=0.1, to=10, increment=0.1, textvariable=sil_dur_var, width=8,
               format="%.1f", **_spin_opts).grid(row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Frame duration (ms)", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    frame_dur_var = tk.IntVar(value=FRAME_DURATION_MS)
    tk.Spinbox(frame, from_=10, to=1000, increment=10, textvariable=frame_dur_var, width=8, **_spin_opts).grid(
        row=row, column=1, sticky="w", pady=3)
    row += 1

    # ── Section: Directories ───────────────────────────────────────────────
    tk.Label(frame, text="Directories", **_section_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(14, 4), **pad)
    row += 1

    tk.Label(frame, text="Transcriptions", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    out_var = tk.StringVar(value=g_output_dir)
    tk.Entry(frame, textvariable=out_var, width=50, **_entry_opts).grid(row=row, column=1, sticky="we", pady=3)
    tk.Button(frame, text="Browse", command=lambda: out_var.set(
        filedialog.askdirectory(initialdir=out_var.get() or ".") or out_var.get()), **_browse_opts).grid(
        row=row, column=2, padx=6)
    row += 1

    tk.Label(frame, text="Summaries", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    sum_var = tk.StringVar(value=g_summaries_dir)
    tk.Entry(frame, textvariable=sum_var, width=50, **_entry_opts).grid(row=row, column=1, sticky="we", pady=3)
    tk.Button(frame, text="Browse", command=lambda: sum_var.set(
        filedialog.askdirectory(initialdir=sum_var.get() or ".") or sum_var.get()), **_browse_opts).grid(
        row=row, column=2, padx=6)
    row += 1

    tk.Label(frame, text="Temp audio", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    tmp_var = tk.StringVar(value=g_temp_dir)
    tk.Entry(frame, textvariable=tmp_var, width=50, **_entry_opts).grid(row=row, column=1, sticky="we", pady=3)
    tk.Button(frame, text="Browse", command=lambda: tmp_var.set(
        filedialog.askdirectory(initialdir=tmp_var.get() or ".") or tmp_var.get()), **_browse_opts).grid(
        row=row, column=2, padx=6)
    row += 1

    # ── Section: API Resilience ────────────────────────────────────────────
    tk.Label(frame, text="API Resilience", **_section_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(14, 4), **pad)
    row += 1

    _resilience_fields = [
        ("Assistant timeout (s)", "ASSISTANT_API_TIMEOUT_SECONDS", 60),
        ("Assistant max retries", "ASSISTANT_API_MAX_RETRIES", 3),
        ("Assistant retry base (s)", "ASSISTANT_API_RETRY_BASE_SECONDS", 1.0),
        ("Summary timeout (s)", "ASSISTANT_SUMMARY_TIMEOUT_SECONDS", 120),
        ("Title timeout (s)", "ASSISTANT_TITLE_TIMEOUT_SECONDS", 30),
        ("Transcribe timeout (s)", "TRANSCRIBE_API_TIMEOUT_SECONDS", 60),
        ("Transcribe max retries", "TRANSCRIBE_API_MAX_RETRIES", 3),
        ("Transcribe retry base (s)", "TRANSCRIBE_API_RETRY_BASE_SECONDS", 1.0),
    ]
    resilience_vars: dict[str, tk.StringVar] = {}
    for label_text, env_key, default_val in _resilience_fields:
        tk.Label(frame, text=label_text, **_lbl_opts).grid(row=row, column=0, sticky="w", pady=2, **pad)
        var = tk.StringVar(value=_env_str(env_key, str(default_val)) or str(default_val))
        resilience_vars[env_key] = var
        tk.Entry(frame, textvariable=var, width=10, **_entry_opts).grid(row=row, column=1, sticky="w", pady=2)
        row += 1

    # ── Section: Transcript Filtering ──────────────────────────────────────
    tk.Label(frame, text="Transcript Filtering", **_section_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(14, 4), **pad)
    row += 1

    tk.Label(frame, text="Min chars", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    filter_min_var = tk.StringVar(value=_env_str("TRANSCRIPT_FILTER_MIN_CHARS", "2") or "2")
    tk.Entry(frame, textvariable=filter_min_var, width=10, **_entry_opts).grid(row=row, column=1, sticky="w", pady=3)
    row += 1

    _filter_fields = [
        ("Exact matches", "TRANSCRIPT_FILTER_EXACT"),
        ("Prefix matches", "TRANSCRIPT_FILTER_PREFIXES"),
        ("Contains matches", "TRANSCRIPT_FILTER_CONTAINS"),
        ("Regex patterns", "TRANSCRIPT_FILTER_REGEX"),
    ]
    filter_vars: dict[str, tk.StringVar] = {}
    for label_text, env_key in _filter_fields:
        tk.Label(frame, text=label_text, **_lbl_opts).grid(row=row, column=0, sticky="w", pady=2, **pad)
        var = tk.StringVar(value=_env_str(env_key, "") or "")
        filter_vars[env_key] = var
        tk.Entry(frame, textvariable=var, width=60, **_entry_opts).grid(row=row, column=1, columnspan=2, sticky="we", pady=2)
        row += 1

    # ── Section: Logging ───────────────────────────────────────────────────
    tk.Label(frame, text="Logging", **_section_opts).grid(
        row=row, column=0, columnspan=3, sticky="w", pady=(14, 4), **pad)
    row += 1

    tk.Label(frame, text="Log level", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    log_level_var = tk.StringVar(value=_env_str("LOG_LEVEL", "INFO") or "INFO")
    ttk.Combobox(frame, values=["DEBUG", "INFO", "WARNING", "ERROR"], textvariable=log_level_var,
                 state="readonly", width=12).grid(row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Log file max MB", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    log_max_var = tk.StringVar(value=_env_str("LOG_FILE_MAX_MB", "5") or "5")
    tk.Entry(frame, textvariable=log_max_var, width=10, **_entry_opts).grid(row=row, column=1, sticky="w", pady=3)
    row += 1

    tk.Label(frame, text="Log backup count", **_lbl_opts).grid(row=row, column=0, sticky="w", pady=3, **pad)
    log_backup_var = tk.StringVar(value=_env_str("LOG_FILE_BACKUP_COUNT", "5") or "5")
    tk.Entry(frame, textvariable=log_backup_var, width=10, **_entry_opts).grid(row=row, column=1, sticky="w", pady=3)
    row += 1

    # bottom padding so scroll ends comfortably
    tk.Label(frame, text="", bg=theme["bg"]).grid(row=row, column=0, pady=10)
    row += 1

    frame.grid_columnconfigure(1, weight=1)

    # ── Save / Cancel (fixed at bottom) ────────────────────────────────────
    btn_bar = tk.Frame(win, bg=theme["bg_secondary"], padx=20, pady=10)
    btn_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_save():
        global g_language, g_auto_summarize_on_stop, g_output_dir, g_summaries_dir
        global g_assistant_web_search_for_custom_prompts
        global g_my_name, g_interupt_manually, g_keywords, g_openai_model_for_transcript
        global g_agent_font_size, g_default_font_size, g_temp_dir
        global RECORD_SECONDS, SILENCE_THRESHOLD, SILENCE_DURATION, FRAME_DURATION_MS

        g_my_name = name_var.get().strip() or "You"
        g_language = (lang_var.get() or "en").strip()
        g_keywords = kw_var.get().strip() or None
        g_interupt_manually = bool(interrupt_var.get())
        g_auto_summarize_on_stop = bool(auto_sum_var.get())
        g_assistant_web_search_for_custom_prompts = bool(web_search_var.get())
        g_openai_model_for_transcript = transcript_model_var.get().strip() or "gpt-4o-mini-transcribe"
        g_agent_font_size = int(agent_fs_var.get())
        g_default_font_size = int(default_fs_var.get())
        g_output_dir = out_var.get().strip() or "output"
        g_summaries_dir = sum_var.get().strip() or "output_summaries"
        g_temp_dir = tmp_var.get().strip() or "tmp"

        try:
            RECORD_SECONDS = int(rec_var.get())
        except (ValueError, tk.TclError):
            pass
        try:
            SILENCE_THRESHOLD = float(sil_thresh_var.get())
        except (ValueError, tk.TclError):
            pass
        try:
            SILENCE_DURATION = float(sil_dur_var.get())
        except (ValueError, tk.TclError):
            pass
        try:
            FRAME_DURATION_MS = int(frame_dur_var.get())
        except (ValueError, tk.TclError):
            pass

        os.makedirs(g_output_dir, exist_ok=True)
        os.makedirs(g_summaries_dir, exist_ok=True)
        os.makedirs(g_temp_dir, exist_ok=True)

        if g_auto_start_var is not None:
            g_auto_start_var.set(bool(auto_start_var.get()))
        _sync_auto_start_var()

        env_updates = {
            "YOUR_NAME": g_my_name,
            "LANGUAGE": g_language,
            "INTERUPT_MANUALLY": "True" if g_interupt_manually else "False",
            "AUTO_START_TRANSCRIPTION": "True" if bool(auto_start_var.get()) else "False",
            "AUTO_SUMMARIZE_ON_STOP": "True" if g_auto_summarize_on_stop else "False",
            "ASSISTANT_ENABLE_WEB_SEARCH_FOR_CUSTOM_PROMPTS": "True" if g_assistant_web_search_for_custom_prompts else "False",
            "OPENAI_MODEL_FOR_TRANSCRIPT": g_openai_model_for_transcript,
            "OPENAI_MODEL_FOR_ASSISTANT": assistant_model_var.get().strip(),
            "OPENAI_VECTOR_STORE_ID_FOR_ANSWERS": vs_var.get().strip(),
            "AGENT_FONT_SIZE": str(g_agent_font_size),
            "DEFAULT_FONT_SIZE": str(g_default_font_size),
            "RECORD_SECONDS": str(RECORD_SECONDS),
            "SILENCE_THRESHOLD": str(SILENCE_THRESHOLD),
            "SILENCE_DURATION": str(SILENCE_DURATION),
            "FRAME_DURATION_MS": str(FRAME_DURATION_MS),
            "OUTPUT_DIR": g_output_dir,
            "SUMMARIES_DIR": g_summaries_dir,
            "TEMP_DIR": g_temp_dir,
            "KEYWORDS": g_keywords or "",
            "LOG_LEVEL": log_level_var.get().strip(),
            "LOG_FILE_MAX_MB": log_max_var.get().strip(),
            "LOG_FILE_BACKUP_COUNT": log_backup_var.get().strip(),
            "TRANSCRIPT_FILTER_MIN_CHARS": filter_min_var.get().strip(),
        }

        # API key — only overwrite when the user typed something new
        new_key = api_key_var.get().strip()
        if new_key:
            env_updates["OPENAI_API_KEY"] = new_key

        for env_key, var in resilience_vars.items():
            val = var.get().strip()
            if val:
                env_updates[env_key] = val
        for env_key, var in filter_vars.items():
            env_updates[env_key] = var.get().strip()

        _save_env_updates(env_updates)

        if g_assistant:
            g_assistant.set_custom_prompt_web_search_enabled(g_assistant_web_search_for_custom_prompts)

        _reinitialize_transcript_ai()
        reset_log_file()
        set_status("Settings saved", "info")
        canvas.unbind_all("<MouseWheel>")
        win.destroy()

    def on_cancel():
        canvas.unbind_all("<MouseWheel>")
        win.destroy()

    ui_module.create_styled_button(btn_bar, text="Save", command=on_save,
                                   bg=theme["accent_green"], hover_bg=theme["accent_green_hover"],
                                   font_size=g_default_font_size).pack(side=tk.RIGHT, padx=(6, 0))
    ui_module.create_styled_button(btn_bar, text="Cancel", command=on_cancel,
                                   bg=theme["surface_light"], hover_bg=theme["surface"],
                                   fg=theme["text_secondary"],
                                   font_size=g_default_font_size).pack(side=tk.RIGHT)


def start_transcription():
    global g_is_recording, g_recording_threads, g_devices_initialized
    global flush_letter_mic, flush_letter_out
    if g_is_recording:
        return

    if not g_devices_initialized:
        g_devices_initialized = initialize_recording()
        if not g_devices_initialized:
            set_status("Failed to initialize recording devices", "error")
            return

    stop_event.clear()
    mute_mic_event.clear()
    _drain_queue(g_recordings_in)
    _drain_queue(g_recordings_out)
    _drain_queue(g_transcriptions_in)
    with flush_letter_lock:
        flush_letter_mic = None
        flush_letter_out = None

    # Every Start begins a brand-new transcript session and file.
    reset_log_file()

    g_recording_threads = [
        Thread(target=store_audio_stream, args=(g_recordings_in, "in", g_device_in, True), daemon=True),
        Thread(target=store_audio_stream, args=(g_recordings_out, "out", g_device_out, False), daemon=True),
        Thread(target=collect_from_stream, args=(g_recordings_in, g_device_in, global_audio, True), daemon=True),
        Thread(target=collect_from_stream, args=(g_recordings_out, g_device_out, global_audio, False), daemon=True),
        Thread(target=get_speaker_name, daemon=True),
        Thread(target=update_screen_on_new_transcription, daemon=True),
    ]
    for t in g_recording_threads:
        t.start()

    g_is_recording = True
    _update_start_stop_button()
    set_status("Transcription started", "info")


def stop_transcription(trigger_auto_summary: bool = True):
    global g_is_recording, g_recording_threads
    if not g_is_recording:
        return

    stop_event.set()
    for t in g_recording_threads:
        t.join(timeout=2.0)
    g_recording_threads = []
    g_is_recording = False
    _update_start_stop_button()
    set_status("Transcription stopped", "info")

    if trigger_auto_summary and g_auto_summarize_on_stop and g_assistant:
        root.after(0, generate_summary)


def toggle_start_stop():
    if g_is_recording:
        stop_transcription()
    else:
        start_transcription()


def _run_summary_before_close(transcription_snapshot: list[list]) -> None:
    theme = ui_module.THEME
    dialog = tk.Toplevel(root)
    dialog.title("Closing")
    dialog.geometry("440x120")
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.grab_set()
    dialog.protocol("WM_DELETE_WINDOW", lambda: None)
    dialog.configure(bg=theme["bg"])
    ui_module.apply_dark_title_bar(dialog)

    tk.Label(
        dialog,
        text="Generating summary before closing.\nPlease wait\u2026",
        justify=tk.CENTER,
        font=ui_module.get_font(max(9, g_default_font_size)),
        bg=theme["bg"],
        fg=theme["text_primary"],
    ).pack(padx=12, pady=(16, 8))

    progress = ttk.Progressbar(dialog, mode="indeterminate", length=360)
    progress.pack(padx=12, pady=(0, 12))
    progress.start(10)

    completed = Event()
    error_holder: list[Exception] = []

    def _worker():
        try:
            _generate_and_save_summary(transcription_snapshot)
        except Exception as e:
            error_holder.append(e)
        finally:
            completed.set()

    worker = Thread(target=_worker, daemon=True)
    worker.start()

    while not completed.is_set():
        try:
            root.update_idletasks()
            root.update()
        except Exception:
            break
        time.sleep(0.05)

    try:
        progress.stop()
        dialog.grab_release()
        dialog.destroy()
    except Exception:
        pass

    if error_holder:
        logger.warning("Error while generating summary during close: %s", error_holder[-1])


def close_app():
    global g_is_closing
    if g_is_closing:
        return
    g_is_closing = True

    logger.info("[UI] Closing application")
    was_recording = g_is_recording
    try:
        stop_transcription(trigger_auto_summary=False)
    except Exception as e:
        logger.warning("Failed to stop transcription during close: %s", e)

    if was_recording and g_auto_summarize_on_stop and g_assistant:
        try:
            set_status("Summarising transcript...", "info")
            transcription_snapshot = g_transcript_store.snapshot()
            if transcription_snapshot:
                _run_summary_before_close(transcription_snapshot)
            else:
                logger.info("[UI] No transcript to summarize during close.")
        except Exception as e:
            logger.warning("Failed to auto-summarize during close: %s", e)
    try:
        if g_assistant:
            g_assistant.stop()
    except Exception as e:
        logger.warning("Failed to stop assistant during close: %s", e)
    try:
        global_audio.terminate()
    except Exception:
        pass
    root.destroy()

def toggle_mute():
    """Toggle the microphone mute state."""
    global mute_button
    theme = ui_module.THEME
    if mute_mic_event.is_set():
        mute_mic_event.clear()
        logger.info("[UI] Microphone unmuted")
        root.title("Live Audio Chat")
        mute_button.config(image=mic_icon_on, bg=theme["accent_red"])
        ui_module.add_hover_effect(mute_button, theme["accent_red"], theme["accent_red_hover"])
    else:
        mute_mic_event.set()
        logger.info("[UI] Microphone muted")
        root.title("Live Audio Chat \u2014 MIC MUTED")
        mute_button.config(image=mic_icon_off, bg=theme["surface_light"])
        ui_module.add_hover_effect(mute_button, theme["surface_light"], theme["surface"])

def reset_log_file():
    """Reset the transcription log file with a new timestamp."""
    global g_trans_file_name
    
    # Create new filename with current timestamp
    new_filename = f"{g_output_dir}/transcription-{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    # Initialize the new transcription log
    try:
        with open(new_filename, "w", encoding="utf-8") as f:
            f.write(f"# Transcription Log\n\n")
            f.write(f"**Created:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
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
            _generate_and_save_summary(transcription_snapshot)
        finally:
            if summary_button:
                root.after(0, lambda: summary_button.config(state=tk.NORMAL, text="Generate Summary"))
    
    # Run in a separate thread to avoid blocking UI
    Thread(target=generate_and_save, daemon=True).start()


def _generate_and_save_summary(transcription_snapshot: list[list]) -> bool:
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
            safe_title = sanitize_title_for_filename(title, max_length=50, default="meeting_summary")
            filename = f"{g_summaries_dir}/{timestamp}_{safe_title}.md"

            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# {title}\n\n")
                f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(summary)

            logger.info("[UI] Summary saved to: %s", filename)
            return True

        logger.warning("[UI] Failed to generate summary.")
        return False
    except Exception as e:
        logger.exception("[UI] Error generating summary: %s", e)
        return False


# GUI Setup
def setup_ui():
    global chat_window, root, mute_button, assistant_buttons, custom_prompt_entry, send_prompt_button, summary_button, status_label
    global start_stop_button, settings_button, g_auto_start_var, mic_icon_on, mic_icon_off
    
    root = tk.Tk()
    root.title("Live Audio Chat")
    root.geometry("700x500")
    root.minsize(600, 400)

    theme = ui_module.THEME
    root.configure(bg=theme["bg"])
    ui_module.apply_dark_title_bar(root)

    # ── Toolbar ────────────────────────────────────────────────────────────
    toolbar = tk.Frame(root, bg=theme["bg_secondary"], padx=10, pady=7)
    toolbar.pack(side=tk.TOP, fill=tk.X)

    left_group = tk.Frame(toolbar, bg=theme["bg_secondary"])
    left_group.pack(side=tk.LEFT)

    start_stop_button = ui_module.create_styled_button(
        left_group,
        text="\u25B6  Start",
        command=toggle_start_stop,
        bg=theme["accent_green"],
        hover_bg=theme["accent_green_hover"],
        font_size=g_default_font_size,
    )
    start_stop_button.pack(side=tk.LEFT, padx=(0, 6))

    g_auto_start_var = tk.BooleanVar(value=g_auto_start_transcription)

    settings_button = ui_module.create_styled_button(
        left_group,
        text="\u2699",
        command=_open_settings,
        bg=theme["accent_purple"],
        hover_bg=theme["accent_purple_hover"],
        font_size=g_default_font_size,
    )
    settings_button.pack(side=tk.LEFT, padx=(0, 6))

    mic_icon_on, mic_icon_off = _create_mic_icons()

    mute_button = tk.Button(
        left_group,
        image=mic_icon_on,
        command=toggle_mute,
        bg=theme["accent_red"],
        activebackground=theme["accent_red_hover"],
        relief=tk.FLAT,
        bd=0,
        highlightthickness=0,
        cursor="hand2",
        padx=8,
        pady=5,
    )
    mute_button.pack(side=tk.LEFT, padx=(0, 6))
    ui_module.add_hover_effect(mute_button, theme["accent_red"], theme["accent_red_hover"])

    if g_assistant:
        # ── Prompt entry (right side of toolbar) ──
        prompt_frame = tk.Frame(toolbar, bg=theme["bg_secondary"])
        prompt_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(14, 0))

        custom_prompt_entry = tk.Entry(
            prompt_frame,
            font=ui_module.get_font(g_default_font_size),
            bg=theme["input_bg"],
            fg=theme["input_fg"],
            insertbackground=theme["text_primary"],
            relief=tk.FLAT,
            highlightbackground=theme["border"],
            highlightcolor=theme["accent_blue"],
            highlightthickness=1,
            bd=6,
        )
        custom_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 6))
        custom_prompt_entry.bind('<Return>', lambda event: send_custom_prompt())

        send_prompt_button = ui_module.create_styled_button(
            prompt_frame,
            text="Ask",
            command=send_custom_prompt,
            bg=theme["accent_blue"],
            hover_bg=theme["accent_blue_hover"],
            font_size=g_default_font_size,
        )
        send_prompt_button.pack(side=tk.RIGHT)

        assistant_buttons = {}
    else:
        assistant_buttons = {}

    # ── Thin separator ─────────────────────────────────────────────────────
    tk.Frame(root, bg=theme["border"], height=1).pack(side=tk.TOP, fill=tk.X)

    # ── Status bar (pack before chat so it stays at the bottom) ────────────
    status_bar = tk.Frame(root, bg=theme["status_bg"])
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    status_label = tk.Label(
        status_bar,
        text="Status: Ready",
        anchor="w",
        bg=theme["status_bg"],
        fg=theme["status_info"],
        font=ui_module.get_font(max(8, g_default_font_size - 1)),
        padx=10,
        pady=4,
    )
    status_label.pack(fill=tk.X)

    # ── Chat area (fills remaining space) ──────────────────────────────────
    chat_frame = tk.Frame(root, bg=theme["bg"], padx=8, pady=6)
    chat_frame.pack(expand=True, fill=tk.BOTH)

    chat_window = tk.Text(
        chat_frame,
        wrap=tk.WORD,
        state=tk.DISABLED,
        bg=theme["surface"],
        fg=theme["text_primary"],
        font=ui_module.get_font(g_default_font_size),
        relief=tk.FLAT,
        padx=12,
        pady=10,
        insertbackground=theme["text_primary"],
        selectbackground=theme["accent_blue"],
        selectforeground=theme["text_primary"],
        highlightthickness=1,
        highlightbackground=theme["border"],
        highlightcolor=theme["border"],
        bd=0,
        spacing1=2,
        spacing3=2,
    )
    chat_window.pack(expand=True, fill=tk.BOTH)
    ui_module.setup_chat_tags(chat_window, g_agent_font_size, g_default_font_size)

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
    root.protocol("WM_DELETE_WINDOW", close_app)
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
        temp_dir=g_temp_dir,
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
    try:
        file_size = os.path.getsize(file)  # Size in bytes
        logger.debug("[Transcribe] File size: %.2f MB", file_size / (1024 * 1024))
        start_time = float(os.path.basename(file).split("-")[0])
        segments = g_transcript_ai.transcribe(file)
        new_segments = False
        #with g_transcription_lock:
        for segment in segments:
            text = segment.text.strip()
            if len(text) > 0:
                should_filter, reason = g_transcript_filter.should_filter(text)
                if should_filter:
                    logger.debug("[Transcribe] Filtered segment (%s): %s", reason, text)
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
    text = _normalize_transcription_text(text)
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
    close_app()
    exit(1)

def main():
    global g_devices_initialized
    logger.info("[Main] Starting application...")
    signal.signal(signal.SIGINT, handler)
    os.makedirs(g_output_dir, exist_ok=True)
    os.makedirs(g_temp_dir, exist_ok=True)
    root = setup_ui()
    g_devices_initialized = initialize_recording()
    if not g_devices_initialized:
        logger.error("[Main] Failed to initialize recording devices. You can still open settings.")

    _update_start_stop_button()
    if g_auto_start_transcription:
        root.after(100, start_transcription)

    try:
        logger.info("[Main] Starting Tkinter main loop...")
        root.mainloop()
        logger.info("[Main] Tkinter loop has exited.")
    except Exception as e:
        logger.exception("[Main] Error in Tkinter loop: %s", e)


if __name__ == "__main__":
    main()
