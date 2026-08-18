"""An OpenAI-compatible transcription endpoint over the chunker.

`POST /v1/audio/transcriptions` is the interface every meeting driver already
speaks, and LiteLLM routes that path but fails on our multimodal alias with
"Unmapped provider passed in" -- so nothing outside the appliance can reach
Cerase transcription at all. This module is that door, and it opens onto the
same chunker, the same audio normalisation and the same per-agent billing the
MCP tool uses: a voice note and an hour-long meeting differ only in how many
chunks they produce.

Everything a request needs decided -- who is calling, in what shape the answer
comes back, where the speaker turns are -- is a plain function here, taking
and returning built-in types. The HTTP framework is imported inside
`create_app()` and nowhere else, so those decisions can be tested on a bare
interpreter without a web server, which is what the media unit tier runs on.

Beyond the OpenAI fields the endpoint accepts one extension, `speaker_timeline`:
a JSON array of `{start, end, speaker}`. When it is present the chunker cuts on
turn changes, where no word can be split and no audio need be transcribed
twice.

The one environment variable this module reads is CERASE_INTERNAL_SECRET, the
bearer every request must present. The port belongs to whoever starts the
server, and the compose service passes it to uvicorn.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Any, AsyncIterable, Mapping

from chunker import Turn

RESPONSE_FORMATS = ("json", "text", "verbose_json")

_TRUTHY = {"1", "true", "yes", "on"}


class RequestError(Exception):
    """A request that cannot be served, with the status the caller gets."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def authorise(header: str | None, secret: str) -> None:
    """Check the bearer token, in constant time.

    An unset secret refuses everything rather than accepting everything: this
    endpoint takes audio from outside the appliance and bills a model for it,
    and a misconfigured deployment must fail loudly rather than serve a free
    transcription service to whoever finds the port.
    """
    if not secret:
        raise RequestError(503, "transcription endpoint is not configured")
    prefix = "Bearer "
    presented = header[len(prefix):] if header and header.startswith(prefix) else ""
    if not presented or not hmac.compare_digest(presented, secret):
        raise RequestError(401, "invalid or missing bearer token")


def parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in _TRUTHY


def parse_response_format(raw: Any) -> str:
    value = (str(raw).strip() if raw is not None else "") or "json"
    if value not in RESPONSE_FORMATS:
        raise RequestError(
            400,
            "response_format must be one of " + ", ".join(RESPONSE_FORMATS),
        )
    return value


def parse_turns(raw: Any) -> list[Turn]:
    """Read a speaker timeline, or nothing at all.

    A malformed timeline is refused rather than ignored. Ignoring it would
    silently fall back to blind cuts and produce a transcript that is merely
    slightly worse, which is the kind of degradation nobody ever notices.
    """
    if raw is None or raw == "":
        return []
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            raw = json.loads(raw)
        except ValueError as err:
            raise RequestError(400, f"speaker_timeline is not valid JSON: {err}")
    if not isinstance(raw, list):
        raise RequestError(400, "speaker_timeline must be a JSON array")
    turns: list[Turn] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or "start" not in entry:
            raise RequestError(
                400, "each speaker_timeline entry needs start, end and speaker"
            )
        try:
            start = float(entry["start"])
            end = float(entry.get("end", start))
        except (TypeError, ValueError):
            raise RequestError(400, "speaker_timeline start and end must be numbers")
        turns.append(Turn(start=start, end=end, speaker=str(entry.get("speaker", ""))))
    turns.sort(key=lambda t: t.start)
    return turns


def principal(form: Mapping[str, Any], headers: Mapping[str, Any]) -> str:
    """The agent every model call in this request is billed to.

    The OpenAI request has no field for it, so it arrives as a form field or a
    header. It is required: an unattributed transcription is spend that lands
    on nobody's ledger, and a meeting driver is exactly the caller that would
    otherwise send an hour of audio a day without one.
    """
    value = str(
        form.get("agent_id")
        or headers.get("x-cerase-agent-id")
        or headers.get("X-Cerase-Agent-Id")
        or ""
    ).strip()
    if not value:
        raise RequestError(
            400, "agent_id is required (form field or X-Cerase-Agent-Id header)"
        )
    return value


def response_body(
    result: Mapping[str, Any], response_format: str, language: str | None
) -> tuple[str, str]:
    """The finished answer and its content type, in the requested shape."""
    if response_format == "text":
        return result["text"], "text/plain; charset=utf-8"
    if response_format == "verbose_json":
        return json.dumps({
            "task": "transcribe",
            "language": language or "",
            "duration": result["duration_seconds"],
            "text": result["text"],
            "truncated": result["truncated"],
            "segments": [
                {
                    "id": segment["id"],
                    "seek": 0,
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                }
                for segment in result["segments"]
            ],
        }), "application/json"
    return json.dumps({
        "text": result["text"],
        "truncated": result["truncated"],
    }), "application/json"


def sse(payload: Mapping[str, Any] | str) -> str:
    """One server-sent event, framed.

    The stream is what makes a chunker worth having on a meeting: the first
    piece of text leaves while the meeting is still being spoken, instead of
    the whole transcript arriving minutes after it ended.
    """
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return "data: " + json.dumps(payload) + "\n\n"


def delta_event(piece: Mapping[str, Any]) -> str:
    return sse({
        "type": "transcript.text.delta",
        "delta": piece["delta"],
        "start": piece["start"],
        "end": piece["end"],
        "index": piece["index"],
        "chunks": piece["chunks"],
    })


def done_event(text: str, truncated: bool) -> str:
    return sse({
        "type": "transcript.text.done",
        "text": text,
        "truncated": truncated,
    })


async def stream_events(pieces: AsyncIterable[Mapping[str, Any]]):
    """The whole event stream for one transcription, from its pieces.

    The endpoint hands its own transcription straight to this, so what a test
    drives here is the code a client receives rather than a second copy of it.
    """
    text = ""
    truncated = False
    async for piece in pieces:
        text = piece["assembled"]
        truncated = truncated or piece["truncated"]
        if piece["delta"]:
            yield delta_event(piece)
    yield done_event(text, truncated)
    yield sse("[DONE]")


def create_app():  # type: ignore[no-untyped-def]
    """The ASGI application. Imports the web stack and the media server here,
    so importing this module costs neither."""
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    import server

    async def transcriptions(request: Request) -> Response:
        try:
            authorise(
                request.headers.get("authorization"),
                os.environ.get("CERASE_INTERNAL_SECRET", ""),
            )
            form = await request.form()
            agent_id = principal(form, request.headers)
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise RequestError(400, "file is required (multipart form field)")
            raw = await upload.read()
            if len(raw) > server._MAX_FETCH_BYTES:
                raise RequestError(413, "audio exceeds the accepted size")
            language = (str(form.get("language") or "").strip()) or None
            response_format = parse_response_format(form.get("response_format"))
            turns = parse_turns(form.get("speaker_timeline"))
            wants_stream = parse_bool(form.get("stream"))
        except RequestError as err:
            return JSONResponse(
                {"error": {"message": err.message, "type": "invalid_request_error"}},
                status_code=err.status,
            )

        if wants_stream:
            return StreamingResponse(
                stream_events(
                    server.iter_transcription(agent_id, raw, language, turns)
                ),
                media_type="text/event-stream",
            )

        try:
            result = await server.transcribe_audio(agent_id, raw, language, turns)
        except Exception as err:
            # A failure upstream is not the caller's mistake, and a bare 500
            # with a traceback in our log tells a meeting driver nothing it can
            # act on. Name what failed, and say it came from behind us.
            logging.getLogger("cerase.transcription").exception("transcription failed")
            return JSONResponse(
                {"error": {
                    "message": f"transcription failed: {type(err).__name__}",
                    "type": "upstream_error",
                }},
                status_code=502,
            )
        body, media_type = response_body(result, response_format, language)
        return Response(body, media_type=media_type)

    async def healthz(_: Request) -> Response:
        return JSONResponse({"status": "ok"})

    return Starlette(routes=[
        Route("/v1/audio/transcriptions", transcriptions, methods=["POST"]),
        Route("/healthz", healthz, methods=["GET"]),
    ])
