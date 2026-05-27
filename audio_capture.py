from __future__ import annotations

import os
import struct
import sys
import time
import wave
import numpy as np


def clone_device_info(device_info: dict | None) -> dict:
	"""Return a shallow copy of a device info mapping."""
	return dict(device_info or {})


def is_loopback_device(device_info: dict | None) -> bool:
	"""Return True when the supplied device is a WASAPI loopback device."""
	return bool((device_info or {}).get("isLoopbackDevice", False))


def normalize_device_name(name: str | None) -> str:
	"""Normalize device names so render and loopback variants can be matched."""
	value = str(name or "").strip().casefold()
	value = value.replace("[loopback]", " ")
	return " ".join(value.split())


def device_label(device_info: dict | None) -> str:
	"""Create a stable, user-friendly label for UI selection widgets."""
	if not device_info:
		return "Unknown device"

	name = str(device_info.get("name") or "Unknown device").strip()
	index = device_info.get("index")
	channels = int(device_info.get("maxInputChannels", 0) or 0)
	if index is None:
		return f"{name} • {channels}ch"
	return f"{name} • {channels}ch • #{index}"


def build_recording_device_catalog(devices: list[dict], platform_name: str | None = None) -> dict[str, list[dict]]:
	"""Split raw device infos into microphone and output-capture candidates."""
	platform_name = platform_name or sys.platform
	inputs: list[dict] = []
	outputs: list[dict] = []

	for raw_device in devices:
		device = clone_device_info(raw_device)
		input_channels = int(device.get("maxInputChannels", 0) or 0)
		output_channels = int(device.get("maxOutputChannels", 0) or 0)

		if platform_name == "win32":
			if input_channels > 0 and is_loopback_device(device):
				outputs.append(device)
			elif input_channels > 0:
				inputs.append(device)
		else:
			if input_channels > 0:
				inputs.append(device)
			if output_channels > 0:
				outputs.append(device)

	return {"inputs": inputs, "outputs": outputs}


def pick_preferred_device(
	devices: list[dict],
	preferred_index: int | None = None,
	preferred_name: str | None = None,
	fallback_device: dict | None = None,
) -> dict:
	"""Select a device by saved index/name, then fall back to the suggested/default one."""
	for device in devices:
		if preferred_index is not None and device.get("index") == preferred_index:
			return clone_device_info(device)

	needle = normalize_device_name(preferred_name)
	if needle:
		for device in devices:
			if normalize_device_name(device.get("name")) == needle:
				return clone_device_info(device)

	if fallback_device:
		fallback_index = fallback_device.get("index")
		fallback_name = fallback_device.get("name")
		for device in devices:
			if fallback_index is not None and device.get("index") == fallback_index:
				return clone_device_info(device)
		fallback_needle = normalize_device_name(fallback_name)
		if fallback_needle:
			for device in devices:
				if normalize_device_name(device.get("name")) == fallback_needle:
					return clone_device_info(device)

	return clone_device_info(devices[0]) if devices else {}


def enumerate_recording_devices(p_instance, platform_name: str | None = None) -> dict[str, dict | list[dict]]:
	"""Discover recording-capable input and output-capture devices."""
	platform_name = platform_name or sys.platform
	raw_devices: list[dict] = []
	for device_index in range(int(p_instance.get_device_count())):
		try:
			raw_devices.append(clone_device_info(p_instance.get_device_info_by_index(device_index)))
		except Exception:
			continue

	catalog = build_recording_device_catalog(raw_devices, platform_name=platform_name)
	default_input: dict = {}
	default_output: dict = {}

	if platform_name == "win32":
		try:
			wasapi_info = p_instance.get_host_api_info_by_type(p_instance.paWASAPI)
		except Exception:
			wasapi_info = None

		if wasapi_info:
			try:
				default_input = clone_device_info(
					p_instance.get_device_info_by_index(wasapi_info["defaultInputDevice"])
				)
			except Exception:
				default_input = {}

			try:
				if hasattr(p_instance, "get_default_wasapi_loopback"):
					default_output = clone_device_info(p_instance.get_default_wasapi_loopback())
			except Exception:
				default_output = {}

			if not default_output:
				try:
					default_render_device = clone_device_info(
						p_instance.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
					)
					default_output = pick_preferred_device(
						catalog["outputs"],
						preferred_name=default_render_device.get("name"),
					)
				except Exception:
					default_output = {}
	else:
		try:
			default_input = clone_device_info(p_instance.get_default_input_device_info())
		except Exception:
			default_input = {}
		try:
			default_output = clone_device_info(p_instance.get_default_output_device_info())
		except Exception:
			default_output = {}

	if not default_input and catalog["inputs"]:
		default_input = clone_device_info(catalog["inputs"][0])
	if not default_output and catalog["outputs"]:
		default_output = clone_device_info(catalog["outputs"][0])

	return {
		"inputs": catalog["inputs"],
		"outputs": catalog["outputs"],
		"default_input": default_input,
		"default_output": default_output,
	}


def store_audio_stream(
	queue,
	filename_suffix,
	device_info,
	temp_dir,
	stop_event,
	sample_size_getter,
	transcribe_callback,
	logger,
):
	logger.info("[%s] store_audio_stream started.", filename_suffix)
	os.makedirs(temp_dir, exist_ok=True)
	while not stop_event.is_set():
		try:
			frames, letter, start_time = queue.get(block=True, timeout=1)
			logger.debug("[%s] Got %s frames.", filename_suffix, len(frames))
		except Exception as e:
			# queue.Empty and other transient queue errors
			if stop_event.is_set():
				break
			logger.debug("[%s] Queue read skipped: %s", filename_suffix, e)
			continue

		filename = os.path.join(temp_dir, f"{start_time:.2f}-{filename_suffix}.wav")
		try:
			with wave.open(filename, "wb") as wf:
				wf.setnchannels(device_info["maxInputChannels"])
				wf.setsampwidth(sample_size_getter())
				wf.setframerate(int(device_info["defaultSampleRate"]))
				wf.writeframes(b"".join(frames))
			logger.debug("[%s] Wrote audio to %s.", filename_suffix, filename)
		except Exception as e:
			logger.error("[%s] Error writing WAV file: %s", filename_suffix, e)
			continue

		try:
			transcribe_callback(filename, filename_suffix == "in", letter)
		except Exception as e:
			logger.exception("[%s] Transcribe callback failed for %s: %s", filename_suffix, filename, e)
		try:
			os.remove(filename)
			logger.debug("[%s] Removed temporary file %s.", filename_suffix, filename)
		except Exception as e:
			logger.warning("[%s] Error removing file %s: %s", filename_suffix, filename, e)

	logger.info("[%s] store_audio_stream exiting.", filename_suffix)


def collect_from_stream(
	queue,
	input_device,
	p_instance,
	from_microphone,
	stop_event,
	mute_mic_event,
	flush_lock,
	get_flush_letters,
	clear_flush_letters,
	speaker_snapshot_getter,
	speaker_setter,
	frame_duration_ms,
	silence_threshold,
	silence_duration,
	record_seconds,
	logger,
):
	logger.info("[%s] Starting collect_from_stream...", input_device["name"])
	stream = None
	try:
		frame_rate = int(input_device["defaultSampleRate"])
		chunk_size = int(frame_rate * frame_duration_ms / 1000)
		if chunk_size <= 0:
			chunk_size = 1

		input_channels = int(input_device.get("maxInputChannels", 0) or 0)
		if input_channels <= 0:
			logger.warning(
				"[%s] Device has no input channels; skipping capture for this device.",
				input_device.get("name", "Unknown"),
			)
			return

		stream = p_instance.open(
			format=p_instance.get_format_from_width(2),
			channels=input_channels,
			rate=frame_rate,
			frames_per_buffer=chunk_size,
			input=True,
			input_device_index=input_device["index"],
		)
		logger.info("[%s] Audio stream opened successfully.", input_device["name"])
		frames = []
		start_time = time.time()
		silence_start_time = None
		silence_frame_count = 0

		while not stop_event.is_set():
			try:
				if len(frames) == silence_frame_count:
					start_time = time.time()

				if from_microphone and mute_mic_event.is_set():
					time.sleep(0.001)
					continue

				data = stream.read(chunk_size, exception_on_overflow=False)
				frames.append(data)

				with flush_lock:
					flush_mic, flush_out = get_flush_letters()
					current_letter = flush_mic if from_microphone else flush_out
					clear_flush_letters(from_microphone)

				if current_letter:
					if len(frames) > 0:
						if current_letter == "_" and not from_microphone:
							seconds_to_go_back = 2.0
							frames_to_remove = int((1000 / frame_duration_ms) * seconds_to_go_back)
							frames_to_process = frames[:-frames_to_remove]
							previous_speaker, _ = speaker_snapshot_getter()
							queue.put((frames_to_process.copy(), previous_speaker, start_time))
							frames = frames[-frames_to_remove:]
							start_time = time.time() - (frames_to_remove * frame_duration_ms) / 1000
						elif current_letter != "_":
							logger.info("[%s] Manual split triggered; flushing %s frames.", input_device["name"], len(frames))
							speaker_setter(current_letter)
							queue.put((frames.copy(), current_letter, start_time))
							frames = []
						silence_frame_count = 0

				audio_samples = np.array(struct.unpack(f"{len(data)//2}h", data))
				volume = np.sqrt(np.mean(audio_samples ** 2))

				if volume < silence_threshold:
					silence_frame_count += 1
					if silence_start_time is None:
						silence_start_time = time.time()
					elif time.time() - silence_start_time >= silence_duration:
						if len(frames) != silence_frame_count:
							_, current_speaker = speaker_snapshot_getter()
							queue.put((frames.copy(), current_speaker, start_time))
						frames = []
						silence_frame_count = 0
						silence_start_time = None
				else:
					silence_start_time = None

				if len(frames) >= int((frame_rate * record_seconds) / chunk_size):
					logger.info("[%s] Auto split after reaching %s seconds; queueing %s frames.", input_device["name"], record_seconds, len(frames))
					_, current_speaker = speaker_snapshot_getter()
					queue.put((frames.copy(), current_speaker, start_time))
					frames = []
					silence_frame_count = 0
			except Exception as e:
				logger.warning("[%s] Error reading from stream: %s", input_device["name"], e)
				break

		if frames:
			_, current_speaker = speaker_snapshot_getter()
			queue.put((frames.copy(), current_speaker, start_time))
			logger.info(
				"[%s] Flushed %s buffered frames on stop.",
				input_device["name"],
				len(frames),
			)
	except Exception as e:
		logger.error("[%s] Failed to open audio stream: %s", input_device.get("name", "Unknown"), e)
	finally:
		if stream is not None:
			try:
				if stream.is_active():
					stream.stop_stream()
			except Exception:
				pass
			try:
				stream.close()
			except Exception:
				pass

	logger.info("[%s] collect_from_stream exiting.", input_device.get("name", "Unknown"))
