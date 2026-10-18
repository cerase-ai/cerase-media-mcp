"""Find where somebody speaks in a recording.

Transcribing silence is paid for and buys nothing. Worse, a model handed a
recording in which nobody spoke tends to answer with sentences nobody said, so
a stretch with no voice in it has to be found before the model hears it. The
speech stretches also decide whether a recording is a meeting at all.

The detector is Silero VAD, the same model file the meeting bot runs to decide
a call is over, fed the same way: 512 samples at 16 kHz, each window prefixed
with the previous window's last 64 samples, and the recurrent state carried
from one window to the next. numpy and onnxruntime are imported inside the functions
that use them, so the pure planning rules below import on a python without
either.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Iterable

SAMPLE_RATE = 16000

# The model was trained on this window: 512 samples, 32 milliseconds.
WINDOW_SAMPLES = 512
WINDOW_SECONDS = WINDOW_SAMPLES / SAMPLE_RATE

# Each window is prefixed with the previous window's last 64 samples, which is
# the context the model expects.
CONTEXT_SAMPLES = 64

# A window this confident starts a run.
SPEECH_THRESHOLD = 0.5

# The model authors' own end threshold, below the start threshold so a run is
# not broken by a single uncertain window inside a sentence.
SILENCE_THRESHOLD = 0.35

# Eight windows are 250 milliseconds: shorter than that is a click or a breath,
# not somebody speaking.
SPEECH_WINDOWS = 8

# A little audio either side keeps the first and last syllable of a stretch:
# the detector's run can begin after a word has already started.
SPEECH_PAD_SECONDS = float(
    os.environ.get("CERASE_TRANSCRIBE_SPEECH_PAD_SECONDS", 1.0)
)

# Two stretches separated by less than this are planned as one. A pause of a
# few seconds is not worth leaving a hole in the plan and paying a whole model
# call to skip it.
SKIPPED_SILENCE_SECONDS = float(
    os.environ.get("CERASE_TRANSCRIBE_SKIPPED_SILENCE_SECONDS", 10.0)
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models", "silero_vad.onnx"
)

# How much ffmpeg output is read at a time. The stream is consumed window by
# window, so the recording is never held in memory: it can last ten hours.
_READ_BYTES = 65536

_SESSION = None


def speech_runs(
    probabilities: Iterable[float],
) -> list[tuple[float, float]]:
    """Where speech runs, in seconds, from one probability per window.

    A run starts at the first window above `SPEECH_THRESHOLD` and continues
    for as long as windows stay above `SILENCE_THRESHOLD`; the two thresholds
    are what keep a soft word inside a sentence from splitting it. Windows
    arriving after the run would have ended are the start of the next run.

    Takes any iterable, including a generator, because the probabilities are
    produced as the audio is decoded rather than collected first.
    """
    runs: list[tuple[float, float]] = []
    start: int | None = None
    index = 0
    for probability in probabilities:
        if start is None:
            if probability >= SPEECH_THRESHOLD:
                start = index
        elif probability < SILENCE_THRESHOLD:
            if index - start >= SPEECH_WINDOWS:
                runs.append(
                    (start * WINDOW_SECONDS, index * WINDOW_SECONDS)
                )
            start = None
        index += 1
    if start is not None and index - start >= SPEECH_WINDOWS:
        runs.append((start * WINDOW_SECONDS, index * WINDOW_SECONDS))
    return runs


def stretches(
    runs: Iterable[tuple[float, float]], duration: float
) -> list[tuple[float, float]]:
    """The parts of a recording worth transcribing.

    Each run is padded, clipped to the recording, and runs closer together than
    `SKIPPED_SILENCE_SECONDS` are merged, so a short pause inside a sentence
    does not become a gap in the transcript.
    """
    duration = float(duration)
    if duration <= 0:
        return []
    padded: list[tuple[float, float]] = []
    for run_start, run_end in runs:
        start = max(0.0, float(run_start) - SPEECH_PAD_SECONDS)
        end = min(duration, float(run_end) + SPEECH_PAD_SECONDS)
        if end > start:
            padded.append((start, end))
    padded.sort()
    merged: list[tuple[float, float]] = []
    for start, end in padded:
        if merged and start - merged[-1][1] <= SKIPPED_SILENCE_SECONDS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def load():
    """The shared inference session, loaded once.

    One intra-op and one inter-op thread: the detector runs a window at a time
    as the audio arrives, and a recording of ten hours must not claim the
    machine's cores from the model calls it exists to avoid.
    """
    global _SESSION
    if _SESSION is None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        _SESSION = onnxruntime.InferenceSession(
            MODEL_PATH,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    return _SESSION


def speech_runs_of(
    path: str, audio_filter: str | None = None
) -> list[tuple[float, float]]:
    """Decode `path` to mono 16 kHz and return where somebody speaks.

    ffmpeg streams the decoded samples, and every whole window is run through
    the model in order with the state the previous window returned. The
    recording is read in blocks, so its length does not decide how much memory
    this needs. A filter is applied only when the caller names one: a caller
    that has already levelled the audio must hand the detector exactly what the
    transcription model will hear.
    """
    args = ["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-map", "0:a:0?"]
    if audio_filter:
        args += ["-af", audio_filter]
    args += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]

    # stderr goes to a file, not a pipe: stdout is read to the end before
    # stderr is looked at, and a pipe nobody drains would block ffmpeg once
    # it filled with errors.
    with tempfile.TemporaryFile() as stderr_file:
        proc = subprocess.Popen(  # noqa: S603 - an argv list, no shell
            args,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
        )
        try:
            runs = speech_runs(_window_probabilities(proc.stdout))
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            if proc.stdout is not None:
                proc.stdout.close()

        proc.wait()
        if proc.returncode != 0:
            stderr_file.seek(0, os.SEEK_END)
            stderr_file.seek(max(0, stderr_file.tell() - 400))
            stderr = stderr_file.read()
            raise RuntimeError(
                f"ffmpeg failed: {stderr.decode(errors='replace')}"
            )
    return runs


def _window_probabilities(stream) -> Iterable[float]:
    """Yield one speech probability per whole window read from `stream`.

    The input the model receives is the previous window's context followed by
    the current window's samples, and the state it returns becomes the next
    window's. A partial window at the end of the stream is dropped.
    """
    import numpy as np

    session = load()
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
    sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
    window_bytes = WINDOW_SAMPLES * 4
    pending = b""

    while True:
        block = stream.read(_READ_BYTES)
        if not block:
            break
        pending += block
        whole = len(pending) // window_bytes
        if whole == 0:
            continue
        usable = whole * window_bytes
        samples = np.frombuffer(pending[:usable], dtype="<f4")
        for offset in range(0, whole * WINDOW_SAMPLES, WINDOW_SAMPLES):
            window = samples[offset:offset + WINDOW_SAMPLES]
            model_input = np.concatenate((context, window))[None, :]
            output, state = session.run(
                ["output", "stateN"],
                {
                    "input": model_input,
                    "state": state,
                    "sr": sample_rate,
                },
            )
            yield float(output.reshape(-1)[0])
            context = window[-CONTEXT_SAMPLES:].copy()
        pending = pending[usable:]
