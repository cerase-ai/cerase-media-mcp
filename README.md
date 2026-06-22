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

Env: `LITELLM_BASE_URL`, `LITELLM_MASTER_KEY` (scoped service key),
`CERASE_MULTIMODAL_ALIAS` (default `multimodal`),
`CERASE_TOOL_WORKSPACE_ROOT` (path-traversal guard root).
