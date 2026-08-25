"""media `transcribe(audio_url=…)` SSRF guard.

The `audio_url` fetch must never reach file://, loopback, link-local,
private (RFC1918), reserved, or cloud-metadata targets; unresolvable
hosts fail closed; an optional host allowlist pins the reachable hosts;
the download is size-bounded.

Runs on the host python3 (run-tests.sh unit tier, `unittest discover`) —
the `mcp` import is stubbed and DNS + the httpx transport are mocked, so
no MCP deps and no real network are needed. Mirrors the PHP contract in
`control-plane/app/Support/SafeHttp.php` (`MSecSafeFetch1Test`) and the
docreader twin (`agent-runtime/docreader/test_server.py`).
"""
from __future__ import annotations

import asyncio
import base64
import os
import socket
import sys
import types
import unittest
from unittest import mock

# --- stub `mcp.server.fastmcp` so `import server` needs no MCP deps -------
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

import chunker  # noqa: E402
import server  # noqa: E402

_PUBLIC_IP = "93.184.216.34"


def _addrinfo(ip: str):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    sockaddr = (ip, 80, 0, 0) if family == socket.AF_INET6 else (ip, 80)
    return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]


def _resolve_to(ip: str):
    return mock.patch("socket.getaddrinfo", return_value=_addrinfo(ip))


def _fake_httpx(chunks: list[bytes]):
    """A stand-in `httpx` module: AsyncClient.stream() yields `chunks`.

    The real httpx is not on the host python — and the guard must be
    provable without a network anyway. `_load_audio_bytes` imports httpx
    lazily, so injecting this into sys.modules mocks the transport.
    """
    mod = types.ModuleType("httpx")

    class _Resp:
        def raise_for_status(self):
            pass

        async def aiter_bytes(self):
            for c in chunks:
                yield c

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class AsyncClient:
        calls: list[str] = []

        def __init__(self, *args, **kwargs):
            pass

        def stream(self, method, url):
            AsyncClient.calls.append(str(url))
            return _Resp()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    mod.AsyncClient = AsyncClient
    return mod


def _load(audio_url: str) -> bytes:
    return asyncio.run(server._load_audio_bytes("a1", None, audio_url, None))


class ValidateFetchUrlRejectsTest(unittest.TestCase):
    """Every hostile shape is refused with ValueError (fail closed)."""

    def test_file_scheme_rejected(self):
        with self.assertRaises(ValueError):
            server._validate_fetch_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        with self.assertRaises(ValueError):
            server._validate_fetch_url("ftp://example.com/note.mp3")

    def test_metadata_ip_rejected(self):
        with self.assertRaises(ValueError):
            server._validate_fetch_url("http://169.254.169.254/latest/meta-data/")

    def test_rfc1918_ip_rejected(self):
        for ip in ("10.0.0.8", "172.16.3.4", "192.168.1.10"):
            with self.subTest(ip=ip), self.assertRaises(ValueError):
                server._validate_fetch_url(f"http://{ip}/x")

    def test_loopback_rejected(self):
        with self.assertRaises(ValueError):
            server._validate_fetch_url("http://127.0.0.1:4000/x")

    def test_localhost_hostname_rejected(self):
        with _resolve_to("127.0.0.1"), self.assertRaises(ValueError):
            server._validate_fetch_url("http://localhost/x")

    def test_hostname_resolving_private_rejected(self):
        with _resolve_to("10.9.8.7"), self.assertRaises(ValueError):
            server._validate_fetch_url("http://cerase-litellm/x")

    def test_unresolvable_host_fails_closed(self):
        with mock.patch(
            "socket.getaddrinfo", side_effect=socket.gaierror("NXDOMAIN")
        ), self.assertRaises(ValueError):
            server._validate_fetch_url("https://no-such-host.invalid/a.mp3")

    def test_allowlist_blocks_other_hosts(self):
        env = {"CERASE_FETCH_ALLOWED_HOSTS": "cdn.example.com"}
        with mock.patch.dict(os.environ, env), _resolve_to(_PUBLIC_IP):
            with self.assertRaises(ValueError):
                server._validate_fetch_url("https://evil.example.net/a.mp3")
            server._validate_fetch_url("https://cdn.example.com/a.mp3")


class LoadAudioBytesSinkTest(unittest.TestCase):
    """The audio_url sink routes through the guard + bounds the read."""

    def test_file_url_rejected_before_transport(self):
        fake = _fake_httpx([b"nope"])
        fake.AsyncClient.calls = []
        with mock.patch.dict(sys.modules, {"httpx": fake}):
            with self.assertRaises(ValueError):
                _load("file:///etc/passwd")
        self.assertEqual(fake.AsyncClient.calls, [])

    def test_metadata_url_rejected(self):
        with self.assertRaises(ValueError):
            _load("http://169.254.169.254/latest/meta-data/")

    def test_rfc1918_url_rejected(self):
        with self.assertRaises(ValueError):
            _load("http://10.0.0.8/note.ogg")

    def test_public_https_url_fetches(self):
        fake = _fake_httpx([b"RIFF", b"data"])
        with _resolve_to(_PUBLIC_IP), mock.patch.dict(sys.modules, {"httpx": fake}):
            raw = _load("https://cdn.example.com/note.wav")
        self.assertEqual(raw, b"RIFFdata")

    def test_oversized_download_rejected(self):
        fake = _fake_httpx([b"x" * 10, b"y" * 10])
        with _resolve_to(_PUBLIC_IP), mock.patch.dict(
            sys.modules, {"httpx": fake}
        ), mock.patch.object(server, "_MAX_FETCH_BYTES", 16):
            with self.assertRaises(ValueError):
                _load("https://cdn.example.com/note.wav")


class TranscriptionTokenBudgetTest(unittest.TestCase):
    """The output ceiling is a rule, not a habit.

    Measured on a running appliance: with no `max_tokens`, three
    of five transcriptions returned the model's own 65,535-token ceiling, and
    a five-minute voice note billed 58x a one-minute one. These cases pin the
    ceiling ITSELF rather than a copy of the arithmetic — `transcription_token_budget`
    is public and pure exactly so this test can call the shipped rule.
    """

    def test_short_audio_gets_the_floor(self):
        # Below the floor the rate is meaningless: a 2-second clip would earn
        # 80 tokens and a legitimate answer could not fit.
        self.assertEqual(server.transcription_token_budget(2), server._TRANSCRIBE_MIN_TOKENS)

    def test_unknown_duration_gets_the_floor_rather_than_no_ceiling(self):
        # ffprobe returning nothing must not become "unbounded" — that is the
        # exact defect this milestone exists for.
        for unknown in (0, 0.0, -1):
            self.assertEqual(
                server.transcription_token_budget(unknown), server._TRANSCRIBE_MIN_TOKENS
            )

    def test_budget_scales_with_duration(self):
        five_min = server.transcription_token_budget(300)
        self.assertEqual(five_min, 300 * server._TRANSCRIBE_TOKENS_PER_AUDIO_SECOND)

    def test_long_audio_is_bounded_by_the_hard_cap_not_by_the_rate(self):
        # THE case this whole milestone is about, and the one a per-second rate
        # alone gets wrong: at any generous rate an hour of audio computes a
        # budget ABOVE the model's own 65,535 ceiling, so the model would bind
        # first and the runaway would be back. Written first, and it failed.
        one_hour = server.transcription_token_budget(3600)
        self.assertEqual(one_hour, server._TRANSCRIBE_MAX_TOKENS)
        self.assertLess(one_hour, 65_535)

    def test_budget_is_generous_against_real_speech(self):
        # Dense speech is ~3 words/second and ~1.5 tokens/word, so ~4.5
        # tokens/second, and Italian tokenises worse than that. The budget must
        # sit several times above it or an honest transcript would be cut.
        self.assertGreater(server.transcription_token_budget(600) / 600, 4.5 * 2)
        # And an hour of dense Italian speech (~13,500 tokens) still fits under
        # the hard cap, so the cap bounds runaways rather than real work.
        self.assertGreater(server._TRANSCRIBE_MAX_TOKENS, 13_500 * 2)


class TranscriptionTruncationIsReportedTest(unittest.TestCase):
    """A ceiling that truncates in silence hands back a transcript that LOOKS whole."""

    def _multimodal_with(self, finish_reason, text="ciao"):
        class _Msg:
            content = text

        class _Choice:
            message = _Msg()

        _Choice.finish_reason = finish_reason

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def __init__(self):
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs
                return _Resp()

        completions = _Completions()
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        with mock.patch.object(server, "_client", lambda: client):
            out = asyncio.run(server._multimodal("agent", [], max_tokens=123))
        return out, completions.kwargs

    def test_length_stop_is_marked(self):
        out, _ = self._multimodal_with("length")
        self.assertTrue(out.endswith(server.TRUNCATION_MARKER))

    def test_normal_stop_is_not_marked(self):
        out, _ = self._multimodal_with("stop")
        self.assertNotIn(server.TRUNCATION_MARKER, out)

    def test_max_tokens_is_actually_sent(self):
        # The regression that matters: a budget computed and never passed to
        # the provider is a ceiling that does not exist.
        _, kwargs = self._multimodal_with("stop")
        self.assertEqual(kwargs.get("max_tokens"), 123)

    def test_omitted_budget_sends_no_max_tokens(self):
        # OCR and image description must keep their previous behaviour.
        class _Completions:
            def __init__(self):
                self.kwargs = None

            async def create(self, **kwargs):
                self.kwargs = kwargs
                raise RuntimeError("stop here")

        completions = _Completions()
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=completions)
        )
        with mock.patch.object(server, "_client", lambda: client):
            with self.assertRaises(RuntimeError):
                asyncio.run(server._multimodal("agent", []))
        self.assertNotIn("max_tokens", completions.kwargs)


class TranscribeChunkingTest(unittest.TestCase):
    """Long audio is cut, transcribed in parallel and re-assembled.

    ffmpeg is stubbed here: what these cases pin is the orchestration -- how
    many model calls a recording costs, what each one is allowed to spend, in
    what order the text is assembled and what happens when the caller walks
    away. The audio handling itself is exercised against real audio in the
    live tier, where a model can say whether the cut was in a sensible place.
    """

    def setUp(self):
        self.slices = []

    def _run(self, duration, texts=None, **kwargs):
        """Drive `transcribe_audio` over a recording of `duration` seconds.

        The stand-in model reads back which slice it was handed, so a case can
        assert on the piece a call belongs to rather than on the order the
        calls happened to start in -- they run concurrently.
        """
        self.plan = chunker.plan_chunks(duration, turns=kwargs.get("turns"))
        self.calls = {}

        async def fake_normalise(raw):
            return b"NORMALISED", duration

        async def fake_slice(mp3, start, chunk_duration):
            self.slices.append((start, chunk_duration))
            return f"SLICE:{start}".encode()

        async def fake_multimodal(agent_id, content, max_tokens=None):
            payload = base64.b64decode(content[1]["input_audio"]["data"]).decode()
            start = float(payload.split(":")[1])
            index = [c.start for c in self.plan].index(start)
            self.calls[index] = max_tokens
            if texts is None:
                return f"piece {index}"
            return texts[index]

        with mock.patch.object(server, "_normalise_to_mp3", fake_normalise), \
             mock.patch.object(server, "_slice_mp3", fake_slice), \
             mock.patch.object(server, "_multimodal", fake_multimodal):
            return asyncio.run(server.transcribe_audio("agent-1", b"RAW", **kwargs))

    def test_a_voice_note_is_still_one_model_call(self):
        result = self._run(45.0)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["text"], "piece 0")
        self.assertEqual(len(result["segments"]), 1)

    def test_a_short_recording_is_still_padded_with_the_lead_silence(self):
        # The model drops the opening sentence of audio that starts on a word,
        # so even an uncut recording goes through the slicer to gain it.
        self._run(45.0)
        self.assertEqual(self.slices, [(0.0, 45.0)])

    def test_an_hour_becomes_many_calls_each_bounded_by_its_own_chunk(self):
        self._run(3600.0)
        self.assertEqual(len(self.calls), 30)
        # The ceiling that matters is the chunk's, not the recording's: a
        # budget derived from an hour would be the runaway the cap prevents.
        whole = server.transcription_token_budget(3600.0)
        for index, budget in self.calls.items():
            self.assertLess(budget, whole)
            self.assertEqual(
                budget, server.transcription_token_budget(self.plan[index].duration)
            )

    def test_the_transcript_is_assembled_in_time_order(self):
        result = self._run(300.0, texts=[
            "the surveyor counted the containers twice",
            "counted the containers twice before the tide turned",
            "before the tide turned and the crew went home",
        ])
        self.assertEqual(
            result["text"],
            "the surveyor counted the containers twice before the tide turned"
            " and the crew went home",
        )

    def test_the_segments_put_back_together_are_the_transcript(self):
        # A meeting driver reads the segments, not the assembled text. If they
        # carried what each chunk heard, every seam would be in there twice.
        result = self._run(300.0, texts=[
            "the surveyor counted the containers twice",
            "counted the containers twice before the tide turned",
            "before the tide turned and the crew went home",
        ])
        self.assertEqual(
            "".join(segment["text"] for segment in result["segments"]),
            result["text"],
        )

    def test_a_truncated_chunk_is_reported_for_the_whole_transcript(self):
        result = self._run(
            300.0, texts=["fine", "cut off" + server.TRUNCATION_MARKER, "fine again"]
        )
        self.assertTrue(result["truncated"])

    def test_no_more_than_the_configured_number_of_calls_run_at_once(self):
        live = 0
        peak = 0

        async def fake_normalise(raw):
            return b"NORMALISED", 3600.0

        async def fake_slice(mp3, start, duration):
            return b"SLICE"

        async def fake_multimodal(agent_id, content, max_tokens=None):
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0)
            live -= 1
            return "text"

        with mock.patch.object(server, "_normalise_to_mp3", fake_normalise), \
             mock.patch.object(server, "_slice_mp3", fake_slice), \
             mock.patch.object(server, "_multimodal", fake_multimodal):
            asyncio.run(server.transcribe_audio("agent-1", b"RAW"))
        self.assertLessEqual(peak, server._TRANSCRIBE_CONCURRENCY)

    def test_the_first_piece_is_delivered_before_the_last_one_is_transcribed(self):
        """The point of chunking a meeting: text while it is still being spoken."""
        release = None
        seen = []

        async def fake_normalise(raw):
            return b"NORMALISED", 400.0

        async def fake_slice(mp3, start, duration):
            return b"SLICE"

        async def fake_multimodal(agent_id, content, max_tokens=None):
            if len(seen) == 0:
                seen.append("first call")
                return "opening words"
            await release.wait()
            return "closing words"

        async def drive():
            nonlocal release
            release = asyncio.Event()
            pieces = server.iter_transcription("agent-1", b"RAW")
            first = await pieces.__anext__()
            still_waiting = not release.is_set()
            release.set()
            await pieces.aclose()
            return first, still_waiting

        with mock.patch.object(server, "_normalise_to_mp3", fake_normalise), \
             mock.patch.object(server, "_slice_mp3", fake_slice), \
             mock.patch.object(server, "_multimodal", fake_multimodal):
            first, still_waiting = asyncio.run(drive())
        self.assertEqual(first["delta"], "opening words")
        self.assertTrue(still_waiting)

    def test_a_caller_that_walks_away_cancels_the_calls_it_did_not_wait_for(self):
        """An abandoned stream must not keep billing for text nobody reads."""
        started = []
        cancelled = []

        async def fake_normalise(raw):
            return b"NORMALISED", 3600.0

        async def fake_slice(mp3, start, duration):
            return b"SLICE"

        async def fake_multimodal(agent_id, content, max_tokens=None):
            started.append(1)
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.append(1)
                raise
            return "text"

        async def drive():
            pieces = server.iter_transcription("agent-1", b"RAW")
            task = asyncio.ensure_future(pieces.__anext__())
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await pieces.aclose()
            await asyncio.sleep(0.05)

        with mock.patch.object(server, "_normalise_to_mp3", fake_normalise), \
             mock.patch.object(server, "_slice_mp3", fake_slice), \
             mock.patch.object(server, "_multimodal", fake_multimodal):
            asyncio.run(drive())
        self.assertGreater(len(started), 0)
        self.assertEqual(len(cancelled), len(started))

    def test_a_speaker_timeline_moves_the_cuts_onto_the_turn_changes(self):
        turns = [chunker.Turn(0, 131, "a"), chunker.Turn(131, 400, "b")]
        self._run(400.0, texts=["a", "b", "c", "d"], turns=turns)
        self.assertEqual(self.slices[0], (0.0, 131.0))
        self.assertEqual(self.slices[1][0], 131.0)


if __name__ == "__main__":
    unittest.main()
