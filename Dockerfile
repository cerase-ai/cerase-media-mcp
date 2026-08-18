# Cerase Media MCP — multimodal understanding (image text/description +
# audio transcription) via the `multimodal` alias through cerase-litellm.
# M-MEDIA-1: the merge of cerase-ocr + cerase-transcriber, ASYNC tools so
# concurrent requests ride parallel I/O lanes in one container.
#
# Exposes 3 tools: ocr, describe_image, transcribe. FastMCP stdio bridged
# by mcp-proxy — same shape as the other cerase-* MCP images.
FROM python:3.13.9-slim@sha256:326df678c20c78d465db501563f3492d17c42a4afe33a1f2bf5406a1d56b0e86

# ffmpeg transcodes arbitrary uploaded audio to a compact mono 16k mp3
# before it goes upstream.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.lock \
    && rm /tmp/requirements.txt /tmp/requirements.lock

COPY server.py chunker.py transcription_api.py /app/

# The liveness probe below is a real MCP client, not a socket connect: with the
# stdio child gone mcp-proxy keeps the listener open and answers the handshake
# from what it recorded at its own startup, so the connect went on passing for a
# container with no server left in it. The script's header carries the detail.
COPY scripts/healthcheck.py /app/healthcheck.py

RUN groupadd -r appuser \
 && useradd -r -g appuser -u 1000 -m -d /home/appuser -s /usr/sbin/nologin appuser \
 && chown -R appuser:appuser /app
USER appuser
WORKDIR /home/appuser

EXPOSE 3000
# The same code serves an OpenAI-compatible transcription endpoint when the
# entrypoint is replaced with the uvicorn one the compose service uses. The
# port is declared here so the two surfaces stay documented together.
EXPOSE 8080

# M-CI-3: image-level liveness — runtime-spawned MCP containers have no
# compose healthcheck, this is the only signal `docker ps`/doctor sees.
#
# It completes the handshake and lists this server's tools, which is the first
# request that has to reach the stdio child. The script keeps itself inside the
# timeout by budget, so the timeout below is its ceiling and not its schedule.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 /app/healthcheck.py || exit 1

ENTRYPOINT ["sh", "-c", "exec mcp-proxy --port 3000 --host 0.0.0.0 --pass-environment -- python /app/server.py"]
