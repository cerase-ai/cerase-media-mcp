# cerase-media-mcp

An MCP server for images, audio and meeting recordings. Five tools send the
image or audio to one multimodal model alias (`multimodal`) through an
OpenAI-compatible LiteLLM proxy; the sixth measures and re-encodes a meeting
recording with ffmpeg and calls no model.

## Tools

| Tool | What it does | Arguments | Returns |
|---|---|---|---|
| `ocr` | Transcribes the text written in an image, in reading order. | `agent_id`; one of `path`, `image_url`, `image_base64`; optional `prompt` replacing the default instruction | `{text, model}` |
| `describe_image` | Describes what an image shows: scene, objects, people, setting. | `agent_id`; one of `path`, `image_url`, `image_base64`; optional `prompt`, e.g. the user's own question | `{description, model}` |
| `analyze_ui` | Audits a UI screenshot: layout, typography, colours, interactive elements, text, visual errors, accessibility, consistency. | `agent_id`; one of `path`, `image_url`, `image_base64` | `{analysis, model}` (Markdown) |
| `compare_screenshots` | Lists the visual differences between a before and an after screenshot. | `agent_id`; one of `path1`, `image1_url`, `image1_base64`; one of `path2`, `image2_url`, `image2_base64` | `{diff, model}` (Markdown) |
| `transcribe` | Transcribes an audio file of any length, in the language it is spoken in, levelled first. | `agent_id`; one of `path`, `audio_url`, `audio_base64` | `{text, model, truncated, repeated, duration_seconds, chunks}` |
| `meeting_normalise` | Downloads a meeting recording from a presigned URL, measures how much audio it really holds, re-encodes it and uploads it to a second presigned URL. | `source_url`, `destination_url`, optional `declared_seconds` | `{decoded_seconds, declared_seconds, bytes, degraded, degraded_reason, loudest_db, silent, longest_sound_seconds, speech, speech_seconds}` |

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

`transcribe` converts the audio to mono 16 kHz MP3 with ffmpeg and runs a
voice detector over it, Silero VAD v6.2.3 on the CPU (`vad.py`,
`models/silero_vad.onnx`). Only the stretches where somebody speaks reach the
model: each stretch is padded by `CERASE_TRANSCRIBE_SPEECH_PAD_SECONDS`, and a
silence of more than `CERASE_TRANSCRIBE_SKIPPED_SILENCE_SECONDS` between two
stretches is left out. A recording in which nobody speaks costs no model call
and comes back with empty text. Each stretch is cut into
pieces of `CERASE_TRANSCRIBE_CHUNK_SECONDS`, and the tool transcribes up to
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
`truncated: true`. A piece whose text starts repeating itself (a phrase said
back to back at least three times, over at least 24 words) keeps the first
occurrence, ends with `[transcription cut: the model began repeating itself]`,
and the result carries `repeated: true`. `cerase.json` asks the Cerase gateway to run `transcribe` in
its slow pool with a 900-second timeout.

`meeting_normalise` decodes the whole recording to measure its length instead
of trusting the container metadata, and measures the RMS level of its loudest
half-second. It writes mono 16 kHz Opus in Ogg to `destination_url`. The
result is `degraded` when the decoded length is below
`CERASE_MEETING_MIN_RATIO` of `declared_seconds`, and `silent` when the
loudest half-second is under `CERASE_MEETING_SILENCE_DB`. It holds `speech`
when some stretch of its sound lasts `CERASE_MEETING_SPEECH_SECONDS` and the
voice detector, run on the recording levelled as `transcribe` levels it, hears
somebody speak; `speech_seconds` is how much speech it heard. A recording longer
than `CERASE_MEETING_MAX_SECONDS` once decoded is refused.

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
| `CERASE_TRANSCRIBE_SPEECH_PAD_SECONDS` | `1` | Audio kept on each side of a stretch of speech. |
| `CERASE_TRANSCRIBE_SKIPPED_SILENCE_SECONDS` | `10` | Shortest silence between two stretches of speech that is not sent to the model. |
| `CERASE_MEETING_MAX_SECONDS` | `36000` | Longest meeting `meeting_normalise` accepts. |
| `CERASE_MEETING_MIN_RATIO` | `0.9` | Share of the declared length below which a meeting is `degraded`. |
| `CERASE_MEETING_SILENCE_DB` | `-70` | Level in dBFS below which a meeting is `silent`. |
| `CERASE_MEETING_SPEECH_SECONDS` | `2` | Shortest unbroken stretch of sound a meeting needs before it can hold `speech`. |

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

`speech_detection_test.py` runs the voice detector on real audio, so it needs
the image's own dependencies (`numpy`, `onnxruntime`) besides ffmpeg:

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt pytest
.venv/bin/python -m pytest -q
```

## License

MIT. See [LICENSE](LICENSE). The voice detector model, Silero VAD, is MIT as
well: [models/silero_vad.LICENSE](models/silero_vad.LICENSE).
