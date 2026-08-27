"""`meeting_normalise` — the artefact is measured, never believed.

The defect this exists for, measured on a live 85-second Teams call: the
capture driver logged `FFmpeg died unexpectedly`, wrote 85 s into the file's
metadata, uploaded it and reported success twice. The audio ended at 6,3 s.
Anything that reads the declared duration agrees with the driver; only
decoding disagrees.

Runs on the host python3 (run-tests.sh unit tier). ffmpeg is required rather
than skipped around: the whole subject here is what ffmpeg reports, and a test
that steps aside when it is absent would report success for having looked at
nothing.
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

if "mcp.server.fastmcp" not in sys.modules:
    class _FastMCP:
        def __init__(self, *args, **kwargs):
            pass

        def tool(self, *args, **kwargs):
            def deco(fn):
                return fn

            return deco

        def run(self):
            pass

    _fastmcp = types.ModuleType("mcp.server.fastmcp")
    _fastmcp.FastMCP = _FastMCP
    _mcp_server = types.ModuleType("mcp.server")
    _mcp = types.ModuleType("mcp")
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


def _tone(path: str, seconds: float) -> None:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi",
         "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", "libopus", "-b:a", "32k", "-y", path],
        check=True,
    )


def _declared_seconds(path: str) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return float(out)


class DecodedSeconds(unittest.TestCase):
    def setUp(self):
        _ffmpeg_or_fail()

    def test_a_truncated_recording_decodes_short_of_what_it_declares(self):
        # The shape of the real failure: the header was written for the whole
        # call and the bytes stop early, because the encoder died mid-write.
        with tempfile.TemporaryDirectory() as d:
            full = os.path.join(d, "full.webm")
            cut = os.path.join(d, "cut.webm")
            _tone(full, 85)
            size = os.path.getsize(full)
            with open(full, "rb") as src, open(cut, "wb") as dst:
                dst.write(src.read(size * 7 // 100))

            declared = _declared_seconds(cut)
            decoded = asyncio.run(server._decoded_seconds(cut))

        # ffprobe reads the header and reports the whole meeting.
        self.assertGreater(declared, 80)
        # Decoding reads the frames and reports where the audio stops.
        self.assertLess(decoded, 15)
        self.assertGreater(decoded, 0)

    def test_a_whole_recording_decodes_to_what_it_declares(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "whole.webm")
            _tone(path, 4)
            decoded = asyncio.run(server._decoded_seconds(path))
        self.assertAlmostEqual(decoded, 4.0, delta=0.5)

    def test_a_file_that_is_not_audio_decodes_to_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "not-audio")
            with open(path, "wb") as f:
                f.write(b"this is not a recording")
            self.assertEqual(asyncio.run(server._decoded_seconds(path)), 0.0)


class Degradation(unittest.TestCase):
    def test_the_measured_case_is_degraded(self):
        degraded, reason = server._meeting_degradation(85.0, 6.3)
        self.assertTrue(degraded)
        self.assertIn("85", reason)
        self.assertIn("6.3", reason)

    def test_a_whole_recording_is_not(self):
        degraded, reason = server._meeting_degradation(85.0, 84.6)
        self.assertFalse(degraded)
        self.assertEqual(reason, "")

    def test_a_driver_that_declared_nothing_gets_the_benefit_of_it(self):
        # Inventing a declared duration would make every such recording a
        # false alarm; the comparison IS the check.
        self.assertEqual(server._meeting_degradation(0.0, 6.3), (False, ""))

    def test_longer_than_declared_is_not_degraded(self):
        self.assertEqual(server._meeting_degradation(85.0, 91.0), (False, ""))


class Normalisation(unittest.TestCase):
    def setUp(self):
        _ffmpeg_or_fail()

    def test_the_output_is_mono_opus_in_ogg(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.webm")
            dst = os.path.join(d, "out.ogg")
            _tone(src, 3)
            asyncio.run(server._to_meeting_opus(src, dst))
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries",
                 "stream=codec_name,channels:format=format_name",
                 "-of", "default=nw=1", dst],
                check=True, capture_output=True, text=True,
            ).stdout
        self.assertIn("codec_name=opus", probe)
        self.assertIn("channels=1", probe)
        self.assertIn("ogg", probe)

    def test_the_sample_rate_is_read_from_the_opus_header_not_from_a_probe(self):
        # ffprobe reports 48000 for every Opus stream ever made: the format is
        # DEFINED to decode at 48 kHz, and the rate the encoder was fed is
        # recorded in OpusHead instead. Asserting the probe would have pinned a
        # property of the codec rather than a property of this pipeline.
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.webm")
            dst = os.path.join(d, "out.ogg")
            _tone(src, 3)
            asyncio.run(server._to_meeting_opus(src, dst))
            with open(dst, "rb") as f:
                head = f.read(4096)
        at = head.find(b"OpusHead")
        self.assertGreater(at, -1, "the output carries no OpusHead")
        self.assertEqual(head[at + 9], 1, "channel count in OpusHead")
        rate = int.from_bytes(head[at + 12:at + 16], "little")
        self.assertEqual(rate, 16000)

    def test_an_hour_of_this_stays_inside_the_published_size(self):
        # The contract publishes ~7 MB/h, and the retention arithmetic and the
        # bucket sizing are both written against it. A tone is the expensive
        # case for a voip-tuned encoder; speech is cheaper.
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in.webm")
            dst = os.path.join(d, "out.ogg")
            _tone(src, 10)
            asyncio.run(server._to_meeting_opus(src, dst))
            per_hour = os.path.getsize(dst) / 10 * 3600
        self.assertLess(per_hour, 10 * 1024 * 1024)

    def test_a_source_that_carries_no_audio_is_a_refusal_not_an_empty_file(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "in")
            dst = os.path.join(d, "out.ogg")
            with open(src, "wb") as f:
                f.write(b"not a recording")
            with self.assertRaises(RuntimeError):
                asyncio.run(server._to_meeting_opus(src, dst))


class BothUrlsAreGuarded(unittest.TestCase):
    """The DESTINATION is new attack surface, and it is a fetch too.

    The source URL has been guarded since the tool that fetches audio existed.
    A presigned PUT address is supplied by the same caller and reaches the same
    network, so guarding one of the two would leave the guard describing a
    contract the code does not keep.
    """

    def test_a_loopback_source_is_refused(self):
        with self.assertRaises(ValueError):
            asyncio.run(server.meeting_normalise(
                "http://127.0.0.1/recording.webm", "https://example.com/put", 0))

    def test_a_loopback_destination_is_refused(self):
        with self.assertRaises(ValueError):
            asyncio.run(server.meeting_normalise(
                "https://example.com/recording.webm", "http://127.0.0.1/put", 0))

    def test_a_file_url_destination_is_refused(self):
        with self.assertRaises(ValueError):
            asyncio.run(server.meeting_normalise(
                "https://example.com/recording.webm", "file:///tmp/put", 0))


if __name__ == "__main__":
    unittest.main()
