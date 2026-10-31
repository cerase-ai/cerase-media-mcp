"""`_slice_mp3` — a piece holds its lead silence plus its whole requested length.

Every piece is prefixed with a short silence so a cut that opens on a word is
not lost by the model, but the piece itself must still contain all the audio
that was asked for. A slice that is shorter than the lead silence plus the
requested duration silently drops the last words of the recording, so this
file measures both the length of the piece and the tone at its end rather
than trusting what was requested.

Runs on the host python3 (run-tests.sh unit tier). ffmpeg and ffprobe are
required rather than skipped around: the whole subject here is what they
report, and a test that steps aside when they are absent would report success
for having looked at nothing.
"""
from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

# Running this file by path from the repository root does not put its own
# directory on sys.path, so `import server` would fail before a test ran.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


def _ffmpeg_or_fail() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return
    raise AssertionError(
        "ffmpeg and ffprobe are required for this file — it asserts what they "
        "report, and skipping it would report success for having looked at "
        "nothing. Install ffmpeg (the CI unit job does)."
    )


def _recording_with_a_tone_at_the_end(path: str) -> None:
    """About eight seconds: 7.4 s of silence, then a 0.6 s tone.

    The tone lies wholly inside the last second, so a cut that loses the tail
    of the piece takes the whole tone with it.
    """
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error",
         "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono:d=7.4",
         "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.6",
         "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1",
         "-c:a", "libmp3lame", "-ar", "16000", "-ac", "1", "-y", path],
        check=True,
    )


def _seconds(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


def _mean_volume(path: str, seconds: float) -> float:
    """The mean level of the last `seconds` of `path`, decoded, not declared."""
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner",
         "-sseof", f"-{seconds:.3f}", "-i", path,
         "-af", "volumedetect", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?)\s*dB", proc.stderr)
    if not match:
        raise AssertionError(f"volumedetect reported no mean_volume:\n{proc.stderr}")
    return float(match.group(1))


def _whole_recording_slice(directory: str) -> tuple[str, float]:
    """Slice the whole of a source recording, returning the piece and its length."""
    src = os.path.join(directory, "source.mp3")
    out = os.path.join(directory, "piece.mp3")
    _recording_with_a_tone_at_the_end(src)
    with open(src, "rb") as f:
        mp3 = f.read()
    requested = _seconds(src)
    with open(out, "wb") as f:
        f.write(asyncio.run(server._slice_mp3(mp3, 0.0, requested)))
    return out, requested


class Length(unittest.TestCase):
    """A piece is its lead silence plus the audio that was asked for."""

    def setUp(self):
        _ffmpeg_or_fail()

    def test_a_piece_lasts_its_lead_silence_plus_the_whole_request(self):
        with tempfile.TemporaryDirectory() as d:
            out, requested = _whole_recording_slice(d)
            actual = _seconds(out)
        expected = requested + server._CHUNK_LEAD_SILENCE_SECONDS
        self.assertAlmostEqual(
            actual, expected, delta=0.15,
            msg=(
                f"a piece of a {requested:.3f} s recording came back {actual:.3f} s "
                f"long; it should hold its {server._CHUNK_LEAD_SILENCE_SECONDS:.3f} s "
                f"of lead silence plus the whole {requested:.3f} s requested "
                f"({expected:.3f} s)"
            ),
        )

    def test_a_piece_from_the_middle_lasts_the_whole_request_too(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "source.mp3")
            out = os.path.join(d, "piece.mp3")
            _recording_with_a_tone_at_the_end(src)
            with open(src, "rb") as f:
                mp3 = f.read()
            with open(out, "wb") as f:
                f.write(asyncio.run(server._slice_mp3(mp3, 2.0, 3.0)))
            actual = _seconds(out)
        expected = 3.0 + server._CHUNK_LEAD_SILENCE_SECONDS
        self.assertAlmostEqual(
            actual, expected, delta=0.15,
            msg=(
                f"a 3.0 s slice from the middle came back {actual:.3f} s long; "
                f"it should hold its lead silence plus the whole 3.0 s "
                f"({expected:.3f} s)"
            ),
        )


class Tail(unittest.TestCase):
    """The tone at the end of the recording is still in the piece."""

    def setUp(self):
        _ffmpeg_or_fail()

    def test_the_tone_at_the_end_of_the_recording_survives_the_cut(self):
        with tempfile.TemporaryDirectory() as d:
            out, _ = _whole_recording_slice(d)
            mean = _mean_volume(out, 0.5)
        self.assertGreater(
            mean, -40,
            msg=(
                f"the last 0.5 s of the piece measures {mean:.1f} dB mean "
                "volume; the recording's closing tone is missing from it"
            ),
        )


if __name__ == "__main__":
    unittest.main()
