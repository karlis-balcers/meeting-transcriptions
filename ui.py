from __future__ import annotations

import sys
import tkinter as tk
from typing import Callable


# ── Modern Dark Theme ──────────────────────────────────────────────────────────

THEME = {
    # Base colours
    "bg":                "#1a1b2e",
    "bg_secondary":      "#1e1f36",
    "surface":           "#252642",
    "surface_light":     "#2e2f4a",
    "border":            "#363758",

    # Text colours
    "text_primary":      "#e8eaf0",
    "text_secondary":    "#9396b0",
    "text_muted":        "#6b6e85",

    # Accent colours
    "accent_green":      "#00d4aa",
    "accent_green_hover":"#00eabc",
    "accent_red":        "#ff6b6b",
    "accent_red_hover":  "#ff8585",
    "accent_purple":     "#7c5cfc",
    "accent_purple_hover":"#9478ff",
    "accent_blue":       "#4a9eff",
    "accent_blue_hover": "#6cb3ff",

    # Chat colours
    "chat_mic":          "#60a5fa",
    "chat_speaker":      "#34d399",
    "chat_agent":        "#c084fc",
    "chat_header_mic":   "#93bbf5",
    "chat_header_speaker":"#7ee0bd",
    "chat_header_agent": "#d4aefe",

    # Status bar
    "status_bg":         "#15162a",
    "status_info":       "#9396b0",
    "status_warning":    "#f59e0b",
    "status_error":      "#ef4444",
    "status_success":    "#00d4aa",

    # Input
    "input_bg":          "#2e2f4a",
    "input_border":      "#464868",
    "input_fg":          "#e8eaf0",
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_font(size: int = 10, weight: str = "normal") -> tuple:
    """Return a platform-appropriate font tuple."""
    family = "Segoe UI" if sys.platform == "win32" else "Helvetica"
    return (family, size, weight)


def status_color(level: str) -> str:
    """Return a theme-appropriate status-bar foreground colour."""
    return {
        "info":    THEME["status_info"],
        "warning": THEME["status_warning"],
        "error":   THEME["status_error"],
        "success": THEME["status_success"],
    }.get((level or "info").lower(), THEME["status_info"])


def add_hover_effect(widget: tk.Widget, normal_bg: str, hover_bg: str) -> None:
    """Bind enter/leave events for a background colour hover transition."""
    widget.bind("<Enter>", lambda _e: widget.config(bg=hover_bg))
    widget.bind("<Leave>", lambda _e: widget.config(bg=normal_bg))


def create_styled_button(
    parent: tk.Widget,
    text: str,
    command: Callable,
    bg: str,
    hover_bg: str | None = None,
    fg: str = "#ffffff",
    font_size: int = 10,
    bold: bool = True,
    padx: int = 14,
    pady: int = 5,
    **kwargs,
) -> tk.Button:
    """Create a flat, modern-styled button with optional hover effect."""
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=bg,
        fg=fg,
        font=get_font(font_size, "bold" if bold else "normal"),
        activebackground=hover_bg or bg,
        activeforeground=fg,
        relief=tk.FLAT,
        cursor="hand2",
        padx=padx,
        pady=pady,
        bd=0,
        highlightthickness=0,
        **kwargs,
    )
    if hover_bg:
        add_hover_effect(btn, bg, hover_bg)
    return btn


def apply_dark_title_bar(window: tk.Tk | tk.Toplevel) -> None:
    """Attempt to enable the immersive dark title bar on Windows 10/11."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        DWMWA_USE_IMMERSIVE_DARK_MODE = 20
        value = ctypes.c_int(1)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )
    except Exception:
        pass


# ── Chat-window tag setup ─────────────────────────────────────────────────────

def setup_chat_tags(chat_window: tk.Text, agent_font_size: int = 14,
                    default_font_size: int = 10) -> None:
    """Configure text tags for the chat window using theme colours."""
    chat_window.tag_config(
        "microphone",
        foreground=THEME["chat_mic"],
        font=get_font(default_font_size),
    )
    chat_window.tag_config(
        "output",
        foreground=THEME["chat_speaker"],
        font=get_font(default_font_size),
    )
    chat_window.tag_config(
        "agent",
        foreground=THEME["chat_agent"],
        font=get_font(default_font_size),
    )
    chat_window.tag_config(
        "header_mic",
        foreground=THEME["chat_header_mic"],
        font=get_font(max(8, default_font_size - 1), "bold"),
        spacing1=8,
    )
    chat_window.tag_config(
        "header_speaker",
        foreground=THEME["chat_header_speaker"],
        font=get_font(max(8, default_font_size - 1), "bold"),
        spacing1=8,
    )
    chat_window.tag_config(
        "header_agent",
        foreground=THEME["chat_header_agent"],
        font=get_font(max(8, default_font_size - 1), "bold"),
        spacing1=8,
    )


# ── Render transcription snapshot into chat window ─────────────────────────────

def render_transcription(chat_window: tk.Text, transcription_snapshot: list[list],
                         my_name: str, agent_name: str) -> None:
    chat_window.config(state=tk.NORMAL)
    chat_window.delete("1.0", tk.END)
    last_user = ""

    for entry in transcription_snapshot:
        user, text, _ = entry

        if user == my_name:
            tag = "microphone"
            header_tag = "header_mic"
        elif user == agent_name:
            tag = "agent"
            header_tag = "header_agent"
        else:
            tag = "output"
            header_tag = "header_speaker"

        if user != last_user:
            chat_window.insert(tk.END, f"\n  {user}\n", header_tag)
            last_user = user

        chat_window.insert(tk.END, f"  {text}\n", tag)

    chat_window.config(state=tk.DISABLED)
    chat_window.yview(tk.END)
