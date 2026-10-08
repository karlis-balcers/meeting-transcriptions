import logging
import queue
import tempfile
import threading
import unittest
import wave

import numpy as np

from engine.audio_capture import (
    build_recording_device_catalog,
    normalize_device_name,
    pick_preferred_device,
    store_audio_stream,
    to_mono_pcm16,
)


class AudioCaptureDeviceSelectionTests(unittest.TestCase):
    def test_normalize_device_name_ignores_loopback_suffix(self):
        self.assertEqual(
            normalize_device_name("Headphones (OpenRun Pro by Shokz) [Loopback]"),
            normalize_device_name("Headphones (OpenRun Pro by Shokz)"),
        )

    def test_build_recording_device_catalog_separates_windows_inputs_and_loopbacks(self):
        devices = [
            {"index": 1, "name": "USB Mic", "maxInputChannels": 1, "maxOutputChannels": 0},
            {
                "index": 2,
                "name": "Headphones [Loopback]",
                "maxInputChannels": 2,
                "maxOutputChannels": 0,
                "isLoopbackDevice": True,
            },
            {"index": 3, "name": "Headphones", "maxInputChannels": 0, "maxOutputChannels": 2},
        ]

        catalog = build_recording_device_catalog(devices, platform_name="win32")

        self.assertEqual([device["index"] for device in catalog["inputs"]], [1])
        self.assertEqual([device["index"] for device in catalog["outputs"]], [2])

    def test_pick_preferred_device_uses_saved_index_before_name(self):
        devices = [
            {"index": 5, "name": "Desk Mic"},
            {"index": 9, "name": "Headset Mic"},
        ]

        selected = pick_preferred_device(devices, preferred_index=9, preferred_name="Desk Mic")

        self.assertEqual(selected["index"], 9)
        self.assertEqual(selected["name"], "Headset Mic")

    def test_pick_preferred_device_matches_saved_name_to_loopback_variant(self):
        devices = [
            {"index": 12, "name": "Speakers (USB DAC) [Loopback]"},
            {"index": 13, "name": "Headphones (BT) [Loopback]"},
        ]

        selected = pick_preferred_device(devices, preferred_name="Headphones (BT)")

        self.assertEqual(selected["index"], 13)

    def test_pick_preferred_device_falls_back_to_default_when_saved_one_missing(self):
        devices = [
            {"index": 20, "name": "Mic A"},
            {"index": 21, "name": "Mic B"},
        ]

        selected = pick_preferred_device(
            devices,
            preferred_index=99,
            preferred_name="Missing Mic",
            fallback_device={"index": 21, "name": "Mic B"},
        )

        self.assertEqual(selected["index"], 21)


class StoreAudioStreamTests(unittest.TestCase):
    def _run(self, chunks, channels=2, rate=16000):
        q = queue.Queue()
        for frames in chunks:
            q.put((frames, None, 1.0))
        stop = threading.Event()
        stop.set()
        seen = []

        def callback(path, from_mic, letter):
            with wave.open(path, "rb") as wf:
                seen.append((wf.getnchannels(), wf.getnframes()))

        with tempfile.TemporaryDirectory() as tmp:
            store_audio_stream(q, "out", {"maxInputChannels": channels, "defaultSampleRate": rate}, tmp, stop,
                               lambda: 2, callback, logging.getLogger("test"))
        return seen

    def test_short_and_empty_chunks_are_not_sent(self):
        tiny = [np.zeros(2 * 1600, dtype="<i2").tobytes()]  # 0.1s stereo at 16 kHz
        self.assertEqual(self._run([[], tiny]), [])

    def test_chunks_are_written_as_mono(self):
        one_second = [np.zeros(2 * 16000, dtype="<i2").tobytes()]
        self.assertEqual(self._run([one_second]), [(1, 16000)])

    def test_to_mono_averages_channels(self):
        stereo = np.array([100, 300, -50, -150], dtype="<i2").tobytes()
        self.assertEqual(np.frombuffer(to_mono_pcm16(stereo, 2), dtype="<i2").tolist(), [200, -100])


if __name__ == "__main__":
    unittest.main()
