from __future__ import annotations

import re
import sys
import time
import io
from contextlib import redirect_stdout
from threading import Lock
from typing import Optional
import os


class TeamsSpeakerDetector:
	def __init__(self, stop_event, logger, window_name: str = "Meeting compact view*"):
		self.stop_event = stop_event
		self.logger = logger
		self.window_name = window_name
		self._state_lock = Lock()
		self._previous_speaker: Optional[str] = None
		self._current_speaker: Optional[str] = None
		self._meeting_title: Optional[str] = None
		self._teams_app = None
		self._debug_dump_controls = os.getenv("TEAMS_DEBUG_DUMP_CONTROLS", "").strip().lower() in {"1", "true", "yes", "y", "on"}

	def _is_supported(self) -> bool:
		return sys.platform == "win32"

	def _connect_if_needed(self):
		if not self._is_supported():
			return None

		if self._teams_app is not None:
			return self._teams_app

		try:
			from pywinauto import Application  # lazy import for non-Windows compatibility

			self._teams_app = Application(backend="uia").connect(title_re=self.window_name)
		except Exception as e:
			self.logger.debug("Teams app connect failed: %s", e)
			self._teams_app = None
		return self._teams_app

	def is_connected(self) -> bool:
		if not self._is_supported():
			return False
		return self._connect_if_needed() is not None

	def get_speaker_snapshot(self):
		with self._state_lock:
			return self._previous_speaker, self._current_speaker

	def set_current_speaker(self, speaker: Optional[str]):
		with self._state_lock:
			self._previous_speaker = self._current_speaker
			self._current_speaker = speaker
			return self._previous_speaker, self._current_speaker

	def get_meeting_title_snapshot(self) -> Optional[str]:
		with self._state_lock:
			return self._meeting_title

	def get_ms_teams_window_title(self) -> Optional[str]:
		if not self._is_supported():
			return None

		try:
			app = self._connect_if_needed()
			if app is None:
				return self.get_meeting_title_snapshot()

			teams_window = app.window(title_re=self.window_name)
			window_title = teams_window.window_text()
			if not window_title:
				return self.get_meeting_title_snapshot()

			title = window_title.replace("Meeting compact view", "").strip()
			title = title.replace(" | Microsoft Teams", "").strip()
			title = title.replace("|", "").strip().strip(" -|")

			with self._state_lock:
				old_title = self._meeting_title
				if title and title != self._meeting_title:
					self._meeting_title = title
					if old_title is None:
						self.logger.info("[Teams] Meeting title detected: '%s'", title)
					else:
						self.logger.info("[Teams] Meeting title updated: '%s' -> '%s'", old_title, title)
				return self._meeting_title
		except Exception as e:
			self.logger.debug("Teams window title read failed: %s", e)

		return self.get_meeting_title_snapshot()

	def inspect_ms_teams(self) -> Optional[str]:
		if not self._is_supported():
			return None

		try:
			app = self._connect_if_needed()
			if app is None:
				return None

			teams_window = app.window(title_re=self.window_name)
			buffer = io.StringIO()
			with redirect_stdout(buffer):
				teams_window.print_control_identifiers()
			data = buffer.getvalue()
			if self._debug_dump_controls and data:
				try:
					with open("teams_controls.txt", "w", encoding="utf-8") as f:
						f.write(data)
				except Exception as dump_error:
					self.logger.debug("Failed to write teams_controls.txt debug dump: %s", dump_error)
			return data
		except Exception as e:
			self.logger.debug("inspect_ms_teams failed: %s", e)
			return None

	def run_detection_loop(self, on_speaker_changed=None):
		"""Continuously detect active speaker from Teams UI."""
		title_check_counter = 0
		title_check_interval = 50

		while not self.stop_event.is_set():
			title_check_counter += 1
			if title_check_counter >= title_check_interval:
				title_check_counter = 0
				self.get_ms_teams_window_title()

			data = self.inspect_ms_teams()
			detected_speaker = None

			if data:
				patterns = [
					r"MenuItem\s*-\s*'([^']*video is on[^']*)'",
					r"MenuItem\s+-\s+'([^,]+), Context menu is available'",
				]

				for pattern in patterns:
					match = re.search(pattern, data)
					if not match:
						continue

					info = match.group(1)
					speaker = info.split(",")[0].strip()
					name_parts = speaker.split(" ")
					if len(name_parts) > 1 and name_parts[1] and name_parts[1][0] != "(":
						detected_speaker = name_parts[0] + " " + name_parts[1][0]
					else:
						detected_speaker = name_parts[0]
					break

			previous_speaker, current_speaker = self.set_current_speaker(detected_speaker)
			if previous_speaker != current_speaker:
				self.logger.info("New speaker: %s (was %s)", current_speaker, previous_speaker)
				if on_speaker_changed:
					try:
						on_speaker_changed(previous_speaker, current_speaker)
					except Exception as e:
						self.logger.debug("on_speaker_changed callback failed: %s", e)

			time.sleep(0.1)
