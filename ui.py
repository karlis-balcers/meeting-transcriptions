from __future__ import annotations

import tkinter as tk


def status_color(level: str) -> str:
	return {
		"info": "#2d3436",
		"warning": "#d35400",
		"error": "#c0392b",
	}.get((level or "info").lower(), "#2d3436")


def render_transcription(chat_window: tk.Text, transcription_snapshot: list[list], my_name: str, agent_name: str) -> None:
	chat_window.config(state=tk.NORMAL)
	chat_window.delete("1.0", tk.END)
	last_user = ""

	for entry in transcription_snapshot:
		user, text, _ = entry
		if user != last_user:
			if user == my_name:
				chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "microphone")
			elif user == agent_name:
				chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "agent")
			else:
				chat_window.insert(tk.END, f"\n# {user}:\n{text}\n", "output")
			last_user = user
		else:
			if user == my_name:
				chat_window.insert(tk.END, f"{text}\n", "microphone")
			elif user == agent_name:
				chat_window.insert(tk.END, f"{text}\n", "agent")
			else:
				chat_window.insert(tk.END, f"{text}\n", "output")

	chat_window.config(state=tk.DISABLED)
	chat_window.yview(tk.END)
