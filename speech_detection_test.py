"""The voice detector against real audio and the real model.

This file is named speech_detection_test.py rather than test_*.py on purpose.
It needs numpy and onnxruntime, which the host python3 used by the unit tier
does not have, so it is kept out of the unittest discover pattern. CI's pytest
step installs requirements.txt in a virtual environment and collects *_test.py,
which is where these cases run.

ffmpeg is required rather than skipped around: the cases assert what it decodes
and what the model hears in it, and a test that steps aside when ffmpeg is
missing reports success for having looked at nothing.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

# --- stub `mcp.server.fastmcp` so `import server` needs no MCP deps -------
#
# The stubbed `mcp` and `mcp.server` are PACKAGES, not plain modules: a plain
# module works only while every import names a key set directly in sys.modules,
# and Python answers "'mcp.server' is not a package" the moment it has to
# resolve a middle name itself.
if "mcp.server.fastmcp" not in sys.modules:
    class _FastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def deco(fn):
                return fn

            return deco

        def run(self, *args, **kwargs):
            pass

    _fastmcp = types.ModuleType("mcp.server.fastmcp")
    _fastmcp.FastMCP = _FastMCP
    _mcp_server = types.ModuleType("mcp.server")
    _mcp_server.__path__ = []
    _mcp = types.ModuleType("mcp")
    _mcp.__path__ = []
    sys.modules.setdefault("mcp", _mcp)
    sys.modules.setdefault("mcp.server", _mcp_server)
    sys.modules["mcp.server.fastmcp"] = _fastmcp

import server  # noqa: E402
import vad  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def _ffmpeg_or_fail() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    raise AssertionError(
        "ffmpeg and ffprobe are required for this file; install ffmpeg (the CI "
        "pytest job does)."
    )


def _lavfi(path: str, source: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", source,
         "-c:a", "libopus", "-b:a", "32k", "-y", path],
        check=True,
    )


def _concat(path: str, sources: list[str]) -> None:
    args = ["ffmpeg", "-nostdin", "-v", "error"]
    for source in sources:
        args += ["-f", "lavfi", "-i", source]
    labels = "".join(f"[{index}:a]" for index in range(len(sources)))
    args += [
        "-filter_complex", labels + f"concat=n={len(sources)}:v=0:a=1",
        "-c:a", "libopus", "-b:a", "32k", "-y", path,
    ]
    subprocess.run(args, check=True)


def _with_tail(path: str, tail: str) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error",
         "-i", os.path.join(FIXTURES, "meeting-speech.webm"),
         "-f", "lavfi", "-i", tail,
         "-filter_complex",
         "[0:a]aresample=48000,aformat=channel_layouts=mono[a0];"
         "[1:a]aresample=48000,aformat=channel_layouts=mono[a1];"
         "[a0][a1]concat=n=2:v=0:a=1",
         "-c:a", "libopus", "-b:a", "32k", "-y", path],
        check=True,
    )


class ModelAndFixturesTest(unittest.TestCase):
    def setUp(self):
        _ffmpeg_or_fail()

    def test_the_model_loads_and_hears_the_speech_fixture(self):
        self.assertIsNotNone(vad.load())
        runs = vad.speech_runs_of(os.path.join(FIXTURES, "meeting-speech.webm"))
        self.assertTrue(runs)

    def test_the_model_hears_quiet_speech_once_it_is_levelled(self):
        runs = vad.speech_runs_of(
            os.path.join(FIXTURES, "meeting-quiet-speech.webm"),
            audio_filter=server._LEVELLER,
        )
        self.assertTrue(runs)


class NoSpeechTest(unittest.TestCase):
    def setUp(self):
        _ffmpeg_or_fail()

    def _runs_for(self, source: str):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audio.webm")
            _lavfi(path, source)
            return vad.speech_runs_of(path, audio_filter=server._LEVELLER)

    def test_digital_silence_holds_no_speech(self):
        self.assertEqual(
            self._runs_for("anullsrc=r=48000:cl=mono:d=30"), []
        )

    def test_pink_noise_holds_no_speech(self):
        self.assertEqual(
            self._runs_for(
                "anoisesrc=color=pink:amplitude=0.02:duration=30:sample_rate=48000"
            ),
            [],
        )

    def test_a_chime_every_five_seconds_holds_no_speech(self):
        sources = []
        for _ in range(6):
            sources.append("anullsrc=r=48000:cl=mono:d=4.2")
            sources.append("sine=frequency=880:duration=0.8:sample_rate=48000")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "chime.webm")
            _concat(path, sources)
            runs = vad.speech_runs_of(path, audio_filter=server._LEVELLER)
        self.assertEqual(runs, [])


class SilentTailTest(unittest.TestCase):
    def setUp(self):
        _ffmpeg_or_fail()

    def _slices_for(self, tail: str):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "meeting-with-tail.webm")
            _with_tail(path, tail)
            with open(path, "rb") as f:
                raw = f.read()
            real_slice = server._slice_mp3
            slices = []

            async def recording_slice(mp3, start, duration):
                slices.append((start, duration))
                return await real_slice(mp3, start, duration)

            async def fake_multimodal(agent_id, content, max_tokens=None):
                return "parole"

            with mock.patch.object(server, "_multimodal", fake_multimodal), \
                 mock.patch.object(server, "_slice_mp3", recording_slice):
                asyncio.run(server.transcribe_audio("agent-1", raw))
        return slices

    def _assert_one_short_slice(self, slices):
        self.assertEqual(len(slices), 1)
        start, duration = slices[0]
        self.assertLess(start + duration, 5.3 + vad.SPEECH_PAD_SECONDS + 0.5)
        self.assertLess(sum(length for _, length in slices), 10.0)

    def test_a_silent_tail_is_not_sliced(self):
        self._assert_one_short_slice(
            self._slices_for("anullsrc=r=48000:cl=mono:d=60")
        )

    def test_a_quiet_noise_tail_is_not_sliced(self):
        self._assert_one_short_slice(
            self._slices_for(
                "anoisesrc=color=pink:amplitude=0.003:duration=60:sample_rate=48000"
            )
        )


class MeetingSpeechTest(unittest.TestCase):
    def setUp(self):
        _ffmpeg_or_fail()

    def test_loud_pink_noise_is_level_sound_but_not_speech(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "noise.webm")
            _lavfi(
                path,
                "anoisesrc=color=pink:amplitude=0.05:duration=30:sample_rate=48000",
            )
            levels = asyncio.run(server._half_second_levels(path))
            self.assertTrue(server._meeting_has_speech(levels))
            speech, seconds = asyncio.run(server._meeting_speech(path, levels))
        self.assertFalse(speech)
        self.assertEqual(seconds, 0.0)

    def test_the_quiet_sentence_is_speech(self):
        path = os.path.join(FIXTURES, "meeting-quiet-speech.webm")
        levels = asyncio.run(server._half_second_levels(path))
        speech, seconds = asyncio.run(server._meeting_speech(path, levels))
        self.assertTrue(speech)
        self.assertGreater(seconds, 0.0)
        self.assertEqual(seconds, round(seconds, 1))


if __name__ == "__main__":
    unittest.main()
