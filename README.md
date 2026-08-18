# cerase-media MCP

First-party multimodal understanding (M-MEDIA-1 = the merge of the former
`cerase-ocr` + `cerase-transcriber`): five **async** tools over the
`multimodal` tool-model alias through cerase-litellm, billed per-agent.
The last two (`analyze_ui`, `compare_screenshots`) are the UX/UI screenshot
pair added by M-CERASE-MEDIA-UX — same `multimodal` endpoint, specialised
prompts, no extra dependency.

| Tool | Question it answers | Returns |
|---|---|---|
| `ocr` | what is WRITTEN in this image? | `{text, model}` |
| `describe_image` | what does this image SHOW? | `{description, model}` |
| `transcribe` | what does this audio say? | `{text, model}` |
| `analyze_ui` | what's in this UI screenshot? — structured audit of layout, typography, colours, interactive elements, text, visual errors, accessibility, consistency | `{analysis, model}` |
| `compare_screenshots` | what changed between two screenshots? — before/after visual diff (layout / text / style / new / removed / regressions) | `{diff, model}` |

Image input is accepted three ways (pick one): `path` (a file under
`CERASE_TOOL_WORKSPACE_ROOT`), `image_url`, or `image_base64`.
`compare_screenshots` takes the two-image variants (`path1`/`image1_url`/
`image1_base64` and `path2`/…).

Async by design: the tools are ~100% LLM-wait, so concurrent requests run
on parallel I/O lanes inside the single runner container (no per-modality
queue). ffmpeg (audio normalisation) runs as an async subprocess.

## Long audio: the chunker

`transcribe` cuts anything longer than a chunk (`chunker.py`), transcribes the
pieces concurrently and re-assembles the text. Each piece after the first
repeats the previous one's last seconds so a word cannot be lost on a cut, and
the repeated words are located and dropped when the pieces are joined. A caller
that knows where the speakers change can hand over a speaker timeline: the cuts
move onto the turn changes, where nothing needs repeating.

Every piece — including a recording short enough to need no cut — gets a second
of silence in front of it. Audio that starts on a word comes back with that
first sentence missing.

| Knob | Default | What it decides |
|---|---|---|
| `CERASE_TRANSCRIBE_CHUNK_SECONDS` | 120 | how long a piece is, and how soon the first text arrives |
| `CERASE_TRANSCRIBE_CHUNK_OVERLAP_SECONDS` | 6 | how much audio each blind cut repeats |
| `CERASE_TRANSCRIBE_CONCURRENCY` | 4 | pieces of one recording in flight at once |
| `CERASE_TRANSCRIBE_LEAD_SILENCE_SECONDS` | 1 | silence in front of every piece |

## The same code as an HTTP endpoint

`transcription_api.py` serves `POST /v1/audio/transcriptions` — the interface
meeting drivers already speak — over the same chunker. The compose service
`cerase-transcription` runs this image on that entrypoint:

```sh
python -m uvicorn --app-dir /app --factory transcription_api:create_app --host 0.0.0.0 --port 8080
```

It takes the OpenAI fields (`file`, `model`, `language`, `response_format`,
`stream`) plus two of its own: `agent_id` (or the `X-Cerase-Agent-Id` header),
which every model call is billed to and which is required, and
`speaker_timeline`, a JSON array of `{start, end, speaker}`. Callers present
`CERASE_INTERNAL_SECRET` as a bearer; an unset secret refuses every request.
With `stream=true` each piece leaves as a `transcript.text.delta` event as soon
as it lands, and the run ends with `transcript.text.done`.

Env: `LITELLM_BASE_URL`, `LITELLM_MASTER_KEY` (scoped service key),
`CERASE_MULTIMODAL_ALIAS` (default `multimodal`),
`CERASE_TOOL_WORKSPACE_ROOT` (path-traversal guard root),
`CERASE_INTERNAL_SECRET` (the HTTP endpoint's bearer).
