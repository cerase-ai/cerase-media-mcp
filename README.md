# cerase-media MCP

First-party multimodal understanding (M-MEDIA-1 = the merge of the former
`cerase-ocr` + `cerase-transcriber`): three **async** tools over the
`multimodal` tool-model alias through cerase-litellm, billed per-agent.

| Tool | Question it answers | Returns |
|---|---|---|
| `ocr` | what is WRITTEN in this image? | `{text, model}` |
| `describe_image` | what does this image SHOW? | `{description, model}` |
| `transcribe` | what does this audio say? | `{text, model}` |

Async by design: the tools are ~100% LLM-wait, so concurrent requests run
on parallel I/O lanes inside the single runner container (no per-modality
queue). ffmpeg (audio normalisation) runs as an async subprocess.

Env: `LITELLM_BASE_URL`, `LITELLM_MASTER_KEY` (scoped service key),
`CERASE_MULTIMODAL_ALIAS` (default `multimodal`),
`CERASE_TOOL_WORKSPACE_ROOT` (path-traversal guard root).
