#!/usr/bin/env python3
"""Cerase Media MCP — first-party multimodal understanding via cerase-litellm.

M-MEDIA-1: the merge of cerase-ocr + cerase-transcriber (operator
2026-06-12) — one container, three ASYNC tools, so concurrent requests
ride parallel I/O lanes instead of one queue (the tools are ~100%%
LLM-wait). Uses the `multimodal` tool-model alias through cerase-litellm
and injects the calling Agent's id into LiteLLM metadata for per-agent
billing (×1 multimodal).

Tools:
  - ocr(agent_id, path?, image_url?, image_base64?, prompt?)
      → {text, model} — WHAT IS WRITTEN (verbatim transcription).
  - describe_image(agent_id, path?, image_url?, image_base64?, prompt?)
      → {description, model} — WHAT IS VISIBLE (scene description).
  - analyze_ui(agent_id, path?, image_url?, image_base64?)
      → {analysis, model} — structured UX/UI audit report.
  - compare_screenshots(agent_id, path1?, image1_url?, image1_base64?,
      path2?, image2_url?, image2_base64?)
      → {diff, model} — visual diff between two screenshots.
  - transcribe(agent_id, path?, audio_url?, audio_base64?, language?)
      → {text, model} — audio → text (ffmpeg-normalised to mono 16k mp3).

Exactly one source argument must be supplied per call. `agent_id` is
bound by the gateway (same pattern as cerase-memory's user_id).

Env vars:
  - LITELLM_BASE_URL, LITELLM_MASTER_KEY
  - CERASE_MULTIMODAL_ALIAS (default `multimodal`)
"""
from __future__ import annotations

import asyncio
import base64
import ipaddress
import mimetypes
import os
import socket
import tempfile
from typing import Any
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("cerase-media")

_MULTIMODAL_ALIAS = os.environ.get("CERASE_MULTIMODAL_ALIAS", "multimodal")

_OCR_PROMPT = (
    "Transcribe ALL text visible in this image exactly, preserving "
    "reading order and line breaks. Output only the transcribed text, "
    "with no commentary."
)

_DESCRIBE_PROMPT = (
    "Describe what is visible in this image: the overall scene, objects, "
    "people, actions, setting, colors and layout, plus a short summary of "
    "any readable text (do not transcribe it verbatim). Be factual and "
    "concrete; never guess beyond what is visible."
)

_ANALYZE_UI_PROMPT = (
    "You are a UX/UI audit expert. Analyze this screenshot of a user "
    "interface and return a structured report covering these dimensions:\n"
    "1. **Layout** — overall page structure, grid, alignment, whitespace.\n"
    "2. **Typography** — font sizes, weights, readability, hierarchy.\n"
    "3. **Colors** — palette, contrast, colour-blind accessibility.\n"
    "4. **Interactive elements** — buttons, links, inputs, their states "
    "(hover/active/disabled) if visible.\n"
    "5. **Text content** — all readable text, labels, headings, button "
    "copy, error messages, empty states.\n"
    "6. **Visual errors** — broken layouts, misaligned elements, "
    "overflow, truncated text, missing content areas.\n"
    "7. **Accessibility** — contrast violations, missing focus indicators, "
    "small touch targets, missing alt text indicators.\n"
    "8. **Consistency** — repeated patterns, style deviations.\n"
    "Be concrete: reference specific elements by position or label. "
    "Output in Markdown with clear section headings."
)

_COMPARE_SCREENSHOTS_PROMPT = (
    "You are comparing two screenshots of a user interface — "
    "Image 1 is the BEFORE (baseline), Image 2 is the AFTER (changed). "
    "List every visual difference you can detect, grouped as:\n"
    "1. **Layout changes** — moved elements, resized areas, new/deleted sections.\n"
    "2. **Text changes** — added, removed, or modified text.\n"
    "3. **Color/style changes** — background, border, font changes.\n"
    "4. **New elements** — buttons, inputs, images that appeared.\n"
    "5. **Removed elements** — elements present in Image 1 but absent in Image 2.\n"
    "6. **Regressions** — broken layouts, overflow, misalignment introduced.\n"
    "Be precise: describe the location of each change. "
    "If the two images look identical, state 'No visual differences detected.' "
    "Output in Markdown with clear section headings."
)


def _safe_local_path(path: str) -> str:
    """Path-traversal guard — refuse a `path` that escapes the shared
    workspace root (the agent supplies it)."""
    root = os.path.realpath(os.environ.get("CERASE_TOOL_WORKSPACE_ROOT", "/workspace"))
    resolved = os.path.realpath(path)
    if resolved != root and not resolved.startswith(root + os.sep):
        raise ValueError("path escapes the workspace root")
    return resolved


# M-SEC-SAFEFETCH-1 — cap on remote audio downloads (bytes).
_MAX_FETCH_BYTES = int(os.environ.get("CERASE_FETCH_MAX_BYTES", 50 * 1024 * 1024))


def _validate_fetch_url(url: str) -> str:
    """M-SEC-SAFEFETCH-1 — SSRF/LFI guard for a caller-supplied fetch URL.

    Only http(s) URLs whose host resolves to a public address may be
    fetched server-side: file:// / ftp:// / any other scheme is refused,
    as is any host that is — or resolves to — a loopback, link-local,
    private (RFC1918), reserved or otherwise non-public address (kills
    the cloud-metadata classic 169.254.169.254 and pivots into the
    compose-internal network). `CERASE_FETCH_ALLOWED_HOSTS` (comma-
    separated, exact hostnames, case-insensitive) optionally pins the
    reachable hosts. Fail-closed: unparseable or unresolvable → raise.
    Mirrors the PHP contract in control-plane `App\\Support\\SafeHttp`.
    """
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").strip().lower()
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"unparseable URL — refusing to fetch: {exc}") from exc
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"URL scheme {parsed.scheme!r} refused — only http/https may be fetched"
        )
    if not host:
        raise ValueError("URL has no host — refusing to fetch")

    allowlist = {
        h.strip().lower()
        for h in os.environ.get("CERASE_FETCH_ALLOWED_HOSTS", "").split(",")
        if h.strip()
    }
    if allowlist and host not in allowlist:
        raise ValueError(f"host {host!r} is not on the fetch allowlist")

    # Name-based fast fail — resolver-independent.
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("refusing to fetch localhost")

    try:
        infos = socket.getaddrinfo(
            host, port or (443 if parsed.scheme == "https" else 80),
            proto=socket.IPPROTO_TCP,
        )
    except (socket.gaierror, OSError) as exc:
        raise ValueError(
            f"could not resolve host {host!r} — refusing to fetch (fail-closed)"
        ) from exc
    if not infos:
        raise ValueError(
            f"could not resolve host {host!r} — refusing to fetch (fail-closed)"
        )
    for info in infos:
        addr = ipaddress.ip_address(info[4][0].split("%")[0])
        if (
            str(addr) == "169.254.169.254"  # cloud metadata — named explicitly
            or addr.is_loopback
            or addr.is_link_local
            or addr.is_private
            or addr.is_reserved
            or addr.is_multicast
            or addr.is_unspecified
            or not addr.is_global  # CGNAT, TEST-NETs, anything else non-public
        ):
            raise ValueError(
                f"host {host!r} points at a private/reserved address ({addr}) — "
                "server-side fetch refused"
            )
    return url


def _client():
    from openai import AsyncOpenAI

    base = os.environ.get("LITELLM_BASE_URL", "http://cerase-litellm:4000").rstrip("/")
    return AsyncOpenAI(
        api_key=os.environ.get("LITELLM_MASTER_KEY", ""),
        base_url=base + "/v1",
    )


def _one_source(*sources: str | None) -> None:
    if len([s for s in sources if s]) != 1:
        raise ValueError("supply exactly one source argument")


async def _load_workspace_bytes(agent_id: str, path: str, binding: str = "") -> bytes:
    """M-UPLOAD-2 — read an uploaded workspace file's CONTENT.

    This is a SHARED runner that mounts no agent work volume, so a `path`
    cannot be `open()`-ed locally in production. Try a local mount first
    (dev/test where CERASE_TOOL_WORKSPACE_ROOT IS the agent's workspace), then
    fall back to the control-plane internal API, which owns workspace access
    (docker exec) and serves the file scoped to (agent_id, path). The image
    `*_url` path can't be used here — the EXTERNAL vision model can't reach an
    internal URL — so we always deliver bytes.
    """
    try:
        local = _safe_local_path(path)
        if os.path.isfile(local):
            with open(local, "rb") as f:
                return f.read()
    except ValueError:
        pass  # not a safe local path → let the control-plane re-guard + serve

    cp = os.environ.get("CERASE_CONTROL_PLANE_URL", "").rstrip("/")
    secret = os.environ.get("CERASE_INTERNAL_SECRET", "")
    if not cp or not secret:
        raise ValueError(
            "workspace `path` given but no local file and no control-plane "
            "configured (CERASE_CONTROL_PLANE_URL / CERASE_INTERNAL_SECRET)"
        )
    import httpx

    # M-SEC-TOKEN-BINDING-1: the control-plane broker requires the calling
    # agent's binding (gateway-injected tool arg) besides the shared bearer.
    headers = {"Authorization": f"Bearer {secret}"}
    if binding:
        headers["X-Cerase-Agent-Binding"] = binding
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{cp}/api/internal/workspace-file/{agent_id}",
            params={"path": path},
            headers=headers,
        )
        resp.raise_for_status()
        return resp.content


async def _image_data_url(
    agent_id: str, path: str | None, image_url: str | None, image_base64: str | None,
    binding: str = "",
) -> str:
    """Resolve the single image source into a URL the vision model can
    consume (workspace paths become data URLs)."""
    _one_source(path, image_url, image_base64)
    if path:
        data = await _load_workspace_bytes(agent_id, path, binding)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    return image_url or image_base64  # type: ignore[return-value]


async def _load_audio_bytes(
    agent_id: str, path: str | None, audio_url: str | None, audio_base64: str | None,
    binding: str = "",
) -> bytes:
    _one_source(path, audio_url, audio_base64)
    if path:
        return await _load_workspace_bytes(agent_id, path, binding)
    if audio_url:
        # M-SEC-SAFEFETCH-1: only public http(s) targets — never file://
        # nor loopback/private/metadata addresses (SSRF); size-bounded
        # stream (httpx does not follow redirects by default — keep that).
        _validate_fetch_url(audio_url)
        import httpx

        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream("GET", audio_url) as resp:
                resp.raise_for_status()
                total = 0
                chunks: list[bytes] = []
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > _MAX_FETCH_BYTES:
                        raise ValueError(
                            f"audio download exceeds the {_MAX_FETCH_BYTES}-byte limit"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
    payload = audio_base64 or ""
    if "," in payload and payload.strip().startswith("data:"):
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


async def _normalise_to_mp3(raw: bytes) -> bytes:
    """Transcode arbitrary audio to mono 16k mp3 via ffmpeg (async
    subprocess — a long transcode never blocks the other lanes)."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "in")
        dst = os.path.join(d, "out.mp3")
        with open(src, "wb") as f:
            f.write(raw)
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", dst,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-400:]}")
        with open(dst, "rb") as f:
            return f.read()


async def _multimodal(agent_id: str, content: list[dict[str, Any]]) -> str:
    """One multimodal call, billed to the calling agent."""
    resp = await _client().chat.completions.create(
        model=_MULTIMODAL_ALIAS,
        messages=[{"role": "user", "content": content}],
        extra_body={"metadata": {"cerase_agent_id": agent_id}},
    )
    return (resp.choices[0].message.content if resp.choices else "") or ""


@mcp.tool()
async def ocr(
    agent_id: str,
    path: str | None = None,
    image_url: str | None = None,
    image_base64: str | None = None,
    prompt: str | None = None,
    agent_binding: str = "",
) -> dict[str, Any]:
    """Extract the TEXT written in an image via a vision LLM.

    Use when the user wants what is WRITTEN in an uploaded scan / photo /
    screenshot ("cosa c'è scritto in questa immagine?"). To know what the
    picture SHOWS instead, use `describe_image`.

    Args:
        agent_id: Cerase Agent PK — bound by the gateway. Required.
        path: workspace file path (the form the attachment-receiver
            skill uses). Use this OR image_url OR image_base64.
        image_url: http(s) URL of the image.
        image_base64: a `data:image/...;base64,...` data URL.
        prompt: optional instruction override (default = full
            transcription).
        agent_binding: injected by the platform (M-SEC-TOKEN-BINDING-1
            second factor for the workspace-file broker) — do not set it.

    Returns:
        dict with `text` (the transcription) and `model`.
    """
    if not agent_id:
        raise ValueError("agent_id is required (cannot be empty)")
    url = await _image_data_url(agent_id, path, image_url, image_base64, agent_binding)
    text = await _multimodal(agent_id, [
        {"type": "text", "text": prompt or _OCR_PROMPT},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    return {"text": text, "model": _MULTIMODAL_ALIAS}


@mcp.tool()
async def describe_image(
    agent_id: str,
    path: str | None = None,
    image_url: str | None = None,
    image_base64: str | None = None,
    prompt: str | None = None,
    agent_binding: str = "",
) -> dict[str, Any]:
    """Describe what is VISIBLE in an image via a vision LLM.

    Use when the user wants to know what a picture SHOWS — scene, objects,
    people, context ("cosa si vede / cosa è raffigurato in questa foto?").
    To extract the written text verbatim, use `ocr` instead.

    Args:
        agent_id: Cerase Agent PK — bound by the gateway. Required.
        path: workspace file path (the form the attachment-receiver
            skill uses). Use this OR image_url OR image_base64.
        image_url: http(s) URL of the image.
        image_base64: a `data:image/...;base64,...` data URL.
        prompt: optional specific question or instruction — pass the
            user's own question in the user's language to get the answer
            in that language.
        agent_binding: injected by the platform (M-SEC-TOKEN-BINDING-1
            second factor for the workspace-file broker) — do not set it.

    Returns:
        dict with `description` and `model`.
    """
    if not agent_id:
        raise ValueError("agent_id is required (cannot be empty)")
    url = await _image_data_url(agent_id, path, image_url, image_base64, agent_binding)
    description = await _multimodal(agent_id, [
        {"type": "text", "text": prompt or _DESCRIBE_PROMPT},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    return {"description": description, "model": _MULTIMODAL_ALIAS}


@mcp.tool()
async def analyze_ui(
    agent_id: str,
    path: str | None = None,
    image_url: str | None = None,
    image_base64: str | None = None,
    agent_binding: str = "",
) -> dict[str, Any]:
    """Analyze a UI screenshot and return a structured UX/UI audit.

    Returns a detailed report covering layout, typography, colors,
    interactive elements, text content, visual errors, accessibility,
    and consistency.

    Args:
        agent_id: Cerase Agent PK — bound by the gateway. Required.
        path: workspace file path (the form the attachment-receiver
            skill uses). Use this OR image_url OR image_base64.
        image_url: http(s) URL of the screenshot.
        image_base64: a `data:image/...;base64,...` data URL.
        agent_binding: injected by the platform (M-SEC-TOKEN-BINDING-1
            second factor for the workspace-file broker) — do not set it.

    Returns:
        dict with `analysis` (Markdown report) and `model`.
    """
    if not agent_id:
        raise ValueError("agent_id is required (cannot be empty)")
    url = await _image_data_url(agent_id, path, image_url, image_base64, agent_binding)
    analysis = await _multimodal(agent_id, [
        {"type": "text", "text": _ANALYZE_UI_PROMPT},
        {"type": "image_url", "image_url": {"url": url}},
    ])
    return {"analysis": analysis, "model": _MULTIMODAL_ALIAS}


@mcp.tool()
async def compare_screenshots(
    agent_id: str,
    path1: str | None = None,
    image1_url: str | None = None,
    image1_base64: str | None = None,
    path2: str | None = None,
    image2_url: str | None = None,
    image2_base64: str | None = None,
    agent_binding: str = "",
) -> dict[str, Any]:
    """Compare two UI screenshots and report visual differences.

    Image 1 is the BEFORE (baseline), Image 2 is the AFTER (changed).
    Returns a structured diff covering layout, text, colors, new/removed
    elements, and regressions.

    Args:
        agent_id: Cerase Agent PK — bound by the gateway. Required.
        path1: workspace file path for the baseline screenshot.
            Use this OR image1_url OR image1_base64.
        image1_url: http(s) URL of the baseline screenshot.
        image1_base64: data-URL of the baseline screenshot.
        path2: workspace file path for the changed screenshot.
            Use this OR image2_url OR image2_base64.
        image2_url: http(s) URL of the changed screenshot.
        image2_base64: data-URL of the changed screenshot.
        agent_binding: injected by the platform (M-SEC-TOKEN-BINDING-1
            second factor for the workspace-file broker) — do not set it.

    Returns:
        dict with `diff` (Markdown report) and `model`.
    """
    if not agent_id:
        raise ValueError("agent_id is required (cannot be empty)")
    try:
        _one_source(path1, image1_url, image1_base64)
    except ValueError:
        raise ValueError("supply exactly one source for image1 (path1, image1_url, or image1_base64)")
    try:
        _one_source(path2, image2_url, image2_base64)
    except ValueError:
        raise ValueError("supply exactly one source for image2 (path2, image2_url, or image2_base64)")
    url1 = await _image_data_url(agent_id, path1, image1_url, image1_base64, agent_binding)
    url2 = await _image_data_url(agent_id, path2, image2_url, image2_base64, agent_binding)
    diff_text = await _multimodal(agent_id, [
        {"type": "text", "text": _COMPARE_SCREENSHOTS_PROMPT},
        {"type": "image_url", "image_url": {"url": url1}},
        {"type": "image_url", "image_url": {"url": url2}},
    ])
    return {"diff": diff_text, "model": _MULTIMODAL_ALIAS}


@mcp.tool()
async def transcribe(
    agent_id: str,
    path: str | None = None,
    audio_url: str | None = None,
    audio_base64: str | None = None,
    language: str | None = None,
    agent_binding: str = "",
) -> dict[str, Any]:
    """Transcribe an audio file to text via a multimodal LLM.

    Use when the user sends a voice note / audio recording and wants it
    in text, or asks "cosa dice questo audio?".

    Args:
        agent_id: Cerase Agent PK — bound by the gateway. Required.
        path: workspace file path (the form the attachment-receiver
            skill uses). Use this OR audio_url OR audio_base64.
        audio_url: http(s) URL of the audio — public remote hosts only
            (local files must use `path`, not a file:// URL).
        audio_base64: a base64 / data-URL audio payload.
        language: optional ISO hint (e.g. "it") to bias the model.
        agent_binding: injected by the platform (M-SEC-TOKEN-BINDING-1
            second factor for the workspace-file broker) — do not set it.

    Returns:
        dict with `text` (the transcription) and `model`.
    """
    if not agent_id:
        raise ValueError("agent_id is required (cannot be empty)")
    mp3 = await _normalise_to_mp3(
        await _load_audio_bytes(agent_id, path, audio_url, audio_base64, agent_binding)
    )
    b64 = base64.b64encode(mp3).decode("ascii")
    hint = f" The audio is in {language}." if language else ""

    text = await _multimodal(agent_id, [
        {
            "type": "text",
            "text": "Transcribe this audio verbatim. Output only the "
            "transcription, no commentary." + hint,
        },
        {"type": "input_audio", "input_audio": {"data": b64, "format": "mp3"}},
    ])
    return {"text": text, "model": _MULTIMODAL_ALIAS}


if __name__ == "__main__":
    mcp.run()
