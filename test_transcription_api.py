"""The OpenAI-compatible transcription endpoint's decisions.

Every decision a request forces -- is the caller allowed, who is billed, what
shape does the answer take, where are the speaker turns -- is a plain function
in the module under test, so these run on the host python3 with no web server
and no model. What is deliberately NOT covered here is the routing table; that
needs the framework, and the live tier drives it over real HTTP.
"""
from __future__ import annotations

import asyncio
import json
import unittest

import transcription_api as api


class AuthoriseTest(unittest.TestCase):
    def test_the_right_token_passes(self):
        api.authorise("Bearer s3cret", "s3cret")

    def test_a_wrong_token_is_refused(self):
        with self.assertRaises(api.RequestError) as caught:
            api.authorise("Bearer nope", "s3cret")
        self.assertEqual(caught.exception.status, 401)

    def test_a_missing_header_is_refused(self):
        with self.assertRaises(api.RequestError):
            api.authorise(None, "s3cret")

    def test_a_bare_token_without_the_scheme_is_refused(self):
        with self.assertRaises(api.RequestError):
            api.authorise("s3cret", "s3cret")

    def test_an_unconfigured_secret_refuses_everything(self):
        # The dangerous direction is the other one: an empty configured secret
        # matching an empty presented one would serve free transcription to
        # anyone who found the port.
        with self.assertRaises(api.RequestError) as caught:
            api.authorise("Bearer ", "")
        self.assertEqual(caught.exception.status, 503)


class PrincipalTest(unittest.TestCase):
    def test_the_form_field_names_the_billed_agent(self):
        self.assertEqual(api.principal({"agent_id": "agent-7"}, {}), "agent-7")

    def test_the_header_works_too(self):
        self.assertEqual(api.principal({}, {"x-cerase-agent-id": "agent-7"}), "agent-7")

    def test_a_request_that_names_nobody_is_refused(self):
        # An unattributed transcription is spend on nobody's ledger, and a
        # meeting driver is exactly the caller that would send an hour a day.
        with self.assertRaises(api.RequestError) as caught:
            api.principal({}, {})
        self.assertEqual(caught.exception.status, 400)


class ParsingTest(unittest.TestCase):
    def test_the_default_response_shape_is_json(self):
        self.assertEqual(api.parse_response_format(None), "json")

    def test_an_unknown_response_shape_is_refused(self):
        with self.assertRaises(api.RequestError):
            api.parse_response_format("srt")

    def test_stream_accepts_the_spellings_clients_actually_send(self):
        for raw in ("true", "True", "1", "yes", "on", True):
            self.assertTrue(api.parse_bool(raw))
        for raw in ("false", "0", "", None):
            self.assertFalse(api.parse_bool(raw))

    def test_a_speaker_timeline_becomes_turns(self):
        turns = api.parse_turns(
            '[{"start": 12, "end": 30, "speaker": "b"}, {"start": 0, "end": 12, "speaker": "a"}]'
        )
        self.assertEqual([t.speaker for t in turns], ["a", "b"])
        self.assertEqual(turns[1].start, 12.0)

    def test_no_timeline_is_no_turns(self):
        self.assertEqual(api.parse_turns(None), [])
        self.assertEqual(api.parse_turns(""), [])

    def test_a_broken_timeline_is_refused_rather_than_ignored(self):
        # Falling back to blind cuts would produce a slightly worse transcript
        # and no signal at all that the caller's timeline was thrown away.
        for raw in ("{not json", '{"start": 1}', '[{"speaker": "a"}]', '[{"start": "x"}]'):
            with self.subTest(raw=raw):
                with self.assertRaises(api.RequestError):
                    api.parse_turns(raw)


class ResponseShapeTest(unittest.TestCase):
    RESULT = {
        "text": "the whole transcript",
        "truncated": False,
        "duration_seconds": 206.6,
        "segments": [
            {"id": 0, "start": 0.0, "end": 120.0, "text": "the whole"},
            {"id": 1, "start": 114.0, "end": 206.6, "text": "transcript"},
        ],
    }

    def test_json_carries_the_text(self):
        body, media = api.response_body(self.RESULT, "json", "en")
        self.assertEqual(json.loads(body)["text"], "the whole transcript")
        self.assertEqual(media, "application/json")

    def test_text_is_the_transcript_and_nothing_else(self):
        body, media = api.response_body(self.RESULT, "text", None)
        self.assertEqual(body, "the whole transcript")
        self.assertTrue(media.startswith("text/plain"))

    def test_verbose_json_reports_the_chunks_as_segments(self):
        body, _ = api.response_body(self.RESULT, "verbose_json", "it")
        payload = json.loads(body)
        self.assertEqual(payload["duration"], 206.6)
        self.assertEqual(payload["language"], "it")
        self.assertEqual([s["start"] for s in payload["segments"]], [0.0, 114.0])

    def test_a_truncated_transcript_says_so_in_every_shape(self):
        cut = dict(self.RESULT, truncated=True)
        self.assertTrue(json.loads(api.response_body(cut, "json", None)[0])["truncated"])
        self.assertTrue(
            json.loads(api.response_body(cut, "verbose_json", None)[0])["truncated"]
        )


class StreamTest(unittest.TestCase):
    def _events(self, pieces):
        async def source():
            for piece in pieces:
                yield piece

        async def collect():
            return [event async for event in api.stream_events(source())]

        return asyncio.run(collect())

    def _piece(self, index, delta, assembled, truncated=False):
        return {
            "index": index, "start": index * 120.0, "end": (index + 1) * 120.0,
            "delta": delta, "assembled": assembled, "chunks": 2,
            "truncated": truncated, "duration_seconds": 240.0, "text": delta,
        }

    def test_each_chunk_leaves_as_a_delta_while_the_rest_is_still_running(self):
        events = self._events([
            self._piece(0, "opening words", "opening words"),
            self._piece(1, " closing words", "opening words closing words"),
        ])
        first = json.loads(events[0].removeprefix("data: ").strip())
        self.assertEqual(first["type"], "transcript.text.delta")
        self.assertEqual(first["delta"], "opening words")
        self.assertEqual(first["start"], 0.0)

    def test_the_stream_ends_with_the_whole_transcript_and_a_done_marker(self):
        events = self._events([
            self._piece(0, "opening words", "opening words"),
            self._piece(1, " closing words", "opening words closing words"),
        ])
        done = json.loads(events[-2].removeprefix("data: ").strip())
        self.assertEqual(done["type"], "transcript.text.done")
        self.assertEqual(done["text"], "opening words closing words")
        self.assertEqual(events[-1], "data: [DONE]\n\n")

    def test_a_chunk_that_added_nothing_sends_no_delta(self):
        events = self._events([
            self._piece(0, "opening words", "opening words"),
            self._piece(1, "", "opening words"),
        ])
        self.assertEqual(len([e for e in events if "text.delta" in e]), 1)

    def test_truncation_anywhere_reaches_the_end_of_the_stream(self):
        events = self._events([
            self._piece(0, "opening", "opening", truncated=True),
            self._piece(1, " closing", "opening closing"),
        ])
        self.assertTrue(json.loads(events[-2].removeprefix("data: ").strip())["truncated"])

    def test_every_event_is_framed_as_one_server_sent_event(self):
        for event in self._events([self._piece(0, "text", "text")]):
            self.assertTrue(event.startswith("data: "))
            self.assertTrue(event.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
