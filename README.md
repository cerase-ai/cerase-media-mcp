# cerase-media-mcp

An MCP server for images, audio and meeting recordings. Five tools send the
image or audio to one multimodal model alias (`multimodal`) through an
OpenAI-compatible LiteLLM proxy; the sixth measures and re-encodes a meeting
recording with ffmpeg and calls no model. The same image also serves an
OpenAI-style transcription endpoint over HTTP.

## Tools

| Tool | What it does | Arguments | Returns |
|---|---|---|---|
| `ocr` | Transcribes the text written in an image, in reading order. | `agent_id`; one of `path`, `image_url`, `image_base64`; optional `prompt` replacing the default instruction | `{text, model}` |
| `describe_image` | Describes what an image shows: scene, objects, people, setting. | `agent_id`; one of `path`, `image_url`, `image_base64`; optional `prompt`, e.g. the user's own question | `{description, model}` |
| `analyze_ui` | Audits a UI screenshot: layout, typography, colours, interactive elements, text, visual errors, accessibility, consistency. | `agent_id`; one of `path`, `image_url`, `image_base64` | `{analysis, model}` (Markdown) |
| `compare_screenshots` | Lists the visual differences between a before and an after screenshot. | `agent_id`; one of `path1`, `image1_url`, `image1_base64`; one of `path2`, `image2_url`, `image2_base64` | `{diff, model}` (Markdown) |
| `transcribe` | Transcribes an audio file of any length. | `agent_id`; one of `path`, `audio_url`, `audio_base64`; optional `language` hint such as `it` | `{text, model, truncated, duration_seconds, chunks}` |
| `meeting_normalise` | Downloads a meeting recording from a presigned URL, measures how much audio it really holds, re-encodes it and uploads it to a second presigned URL. | `source_url`, `destination_url`, optional `declared_seconds` | `{decoded_seconds, declared_seconds, bytes, degraded, degraded_reason, loudest_db, silent}` |

Inside Cerase the gateway fills `agent_id` and `agent_binding` (the latter is
accepted by every tool except `meeting_normalise`); the model never sets them.
`agent_id` is required by the five model tools and is sent to LiteLLM as
request metadata (`metadata.cerase_agent_id`), so each call is billed to the
calling assistant.

Inputs:

- A `path` is a file in the calling assistant's workspace. The server reads it
  directly when it exists under `CERASE_TOOL_WORKSPACE_ROOT`; otherwise it
  fetches it from the Cerase control-plane at
  `GET /api/internal/workspace-file/<agent_id>?path=…`, with
  `CERASE_INTERNAL_SECRET` as a bearer and `agent_binding` as
  `X-Cerase-Agent-Binding`. Images read this way reach the model as `data:`
  URLs.
- An `image_url` is passed to the model as given, so the model provider
  fetches it, not this server.
- An `audio_url` is downloaded by this server, only over `http` or `https` and
  only from a host that resolves to a public address (loopback, private,
  link-local, reserved and cloud-metadata addresses are refused), up to
  `CERASE_FETCH_MAX_BYTES`. `meeting_normalise` applies the same check to both
  of its URLs.

`transcribe` converts the audio to mono 16 kHz MP3 with ffmpeg, cuts it into
pieces of `CERASE_TRANSCRIBE_CHUNK_SECONDS`, transcribes up to
`CERASE_TRANSCRIBE_CONCURRENCY` pieces at once and joins the text. Each piece
after the first repeats the last `CERASE_TRANSCRIBE_CHUNK_OVERLAP_SECONDS` of
the one before, and the repeated words are removed when the pieces are joined.
Every piece, including a recording short enough to need no cut, gets
`CERASE_TRANSCRIBE_LEAD_SILENCE_SECONDS` of silence in front, because audio
that starts on a word comes back without its first sentence. Each model call
has an output ceiling of `CERASE_TRANSCRIBE_TOKENS_PER_AUDIO_SECOND` tokens per
second of audio, never below `CERASE_TRANSCRIBE_MIN_TOKENS` or above
`CERASE_TRANSCRIBE_MAX_TOKENS`; a piece that hits it ends with
`[transcription truncated: output ceiling reached]` and the result carries
`truncated: true`. `cerase.json` asks the Cerase gateway to run `transcribe` in
its slow pool with a 900-second timeout.

`meeting_normalise` decodes the whole recording to measure its length instead
of trusting the container metadata, and measures the RMS level of its loudest
half-second. It writes mono 16 kHz Opus in Ogg to `destination_url`. The
result is `degraded` when the decoded length is below
`CERASE_MEETING_MIN_RATIO` of `declared_seconds`, and `silent` when the
loudest half-second is under `CERASE_MEETING_SILENCE_DB`. A recording longer
than `CERASE_MEETING_MAX_SECONDS` once decoded is refused.

## Transcription over HTTP

`transcription_api.py` serves `POST /v1/audio/transcriptions` and
`GET /healthz` over the same chunker. To run the image on it instead of the
MCP server:

```sh
docker run --rm -p 8080:8080 \
  -e CERASE_INTERNAL_SECRET=<bearer> \
  -e LITELLM_BASE_URL=https://<your-openai-compatible-proxy> \
  -e LITELLM_MASTER_KEY=<your-key> \
  --entrypoint python cerase-media-mcp \
  -m uvicorn --app-dir /app --factory transcription_api:create_app --host 0.0.0.0 --port 8080
```

Every request must carry `Authorization: Bearer <CERASE_INTERNAL_SECRET>`; with
the variable unset, every request is refused with 503. The multipart form takes
`file` (required), `language`, `response_format` (`json`, `text` or
`verbose_json`) and `stream`, plus two fields of its own: `agent_id` (or the
`X-Cerase-Agent-Id` header), which is required and is the assistant every model
call is billed to, and `speaker_timeline`, a JSON array of
`{start, end, speaker}` that moves the cuts onto speaker changes. A `model`
field is accepted and ignored. With `stream=true` each piece is sent as a
`transcript.text.delta` server-sent event as soon as it is transcribed, and the
stream ends with `transcript.text.done` and `[DONE]`.

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `LITELLM_BASE_URL` | `http://cerase-litellm:4000` | Base URL of the OpenAI-compatible proxy; the default is the LiteLLM service name inside a Cerase appliance. |
| `LITELLM_MASTER_KEY` | empty | API key sent to the proxy. |
| `CERASE_MULTIMODAL_ALIAS` | `multimodal` | Model name every model call uses. It must accept images and `input_audio`. |
| `CERASE_TOOL_WORKSPACE_ROOT` | `/workspace` | Directory a `path` is read from locally; a path resolving outside it is never opened. |
| `CERASE_CONTROL_PLANE_URL` | none | Control-plane base URL for the `path` form when the file is not local. |
| `CERASE_INTERNAL_SECRET` | none | Bearer token for that control-plane request, and the token the HTTP endpoint requires. |
| `CERASE_FETCH_MAX_BYTES` | `67108864` (64 MiB) | Largest audio or meeting download, and largest upload to the HTTP endpoint. |
| `CERASE_FETCH_ALLOWED_HOSTS` | empty | Comma-separated hostnames; when set, downloads may reach only these. |
| `CERASE_TRANSCRIBE_CHUNK_SECONDS` | `120` | Length of one transcription piece. |
| `CERASE_TRANSCRIBE_CHUNK_OVERLAP_SECONDS` | `6` | Audio each cut repeats from the piece before. |
| `CERASE_TRANSCRIBE_CONCURRENCY` | `4` | Pieces of one recording transcribed at once. |
| `CERASE_TRANSCRIBE_LEAD_SILENCE_SECONDS` | `1` | Silence put in front of every piece. |
| `CERASE_TRANSCRIBE_TOKENS_PER_AUDIO_SECOND` | `20` | Output tokens allowed per second of audio. |
| `CERASE_TRANSCRIBE_MIN_TOKENS` | `512` | Lowest output ceiling for one call. |
| `CERASE_TRANSCRIBE_MAX_TOKENS` | `32768` | Highest output ceiling for one call. |
| `CERASE_MEETING_MAX_SECONDS` | `36000` | Longest meeting `meeting_normalise` accepts. |
| `CERASE_MEETING_MIN_RATIO` | `0.9` | Share of the declared length below which a meeting is `degraded`. |
| `CERASE_MEETING_SILENCE_DB` | `-70` | Level in dBFS below which a meeting is `silent`. |

## Installation

The connector is published in the Cerase Marketplace as
`studio.guidance/cerase-media`
([marketplace page](https://marketplace.cerase.ai/en/p/studio.guidance/cerase-media)).
Every Cerase appliance installs it at boot, so its assistants have it without
an install step.

The image `ghcr.io/cerase-ai/cerase-media-mcp` is built and published from the
copy of these files kept in the Cerase appliance repository, which is private.
This repository carries the same files byte for byte, so a change made only
here does not reach the image.

## Build and run locally

```sh
docker build -t cerase-media-mcp .
docker run --rm -p 3000:3000 \
  -e LITELLM_BASE_URL=https://<your-openai-compatible-proxy> \
  -e LITELLM_MASTER_KEY=<your-key> \
  cerase-media-mcp
```

`server.py` speaks MCP over stdio; the image runs it behind `mcp-proxy`, which
serves Streamable HTTP at `http://localhost:3000/mcp` and SSE at
`http://localhost:3000/sse`. The image's `HEALTHCHECK` runs
`scripts/healthcheck.py`, an MCP client that completes the handshake and lists
the tools over `/mcp`; its `CERASE_HEALTHCHECK_*` variables exist to point it
at a stub in tests.

The unit tests stub the MCP SDK and need the standard library plus `ffmpeg`
and `ffprobe` on the path:

```sh
python3 -m unittest discover -p 'test_*.py'
```

## License

MIT. See [LICENSE](LICENSE).
