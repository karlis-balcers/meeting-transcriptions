from __future__ import annotations

import os
import struct
import time
import wave
import numpy as np


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
	try:
		frame_rate = int(input_device["defaultSampleRate"])
		chunk_size = int(frame_rate * frame_duration_ms / 1000)
		if chunk_size <= 0:
			chunk_size = 1

		with p_instance.open(
			format=p_instance.get_format_from_width(2),
			channels=input_device["maxInputChannels"],
			rate=frame_rate,
			frames_per_buffer=chunk_size,
			input=True,
			input_device_index=input_device["index"],
		) as stream:
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
	except Exception as e:
		logger.error("[%s] Failed to open audio stream: %s", input_device.get("name", "Unknown"), e)

	logger.info("[%s] collect_from_stream exiting.", input_device.get("name", "Unknown"))
