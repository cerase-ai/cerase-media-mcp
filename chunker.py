"""Chunk planning and transcript re-stitching for long audio.

One hour of speech in one payload does not work: the model returns empty or
runs away, the gateway bounds the call, and nobody waits seven minutes for a
transcript. The fix is to cut the audio into pieces, transcribe them
independently and put the text back together.

This module is the part of that with no I/O in it -- where the cuts go, and
how two neighbouring transcripts are joined. It is pure on purpose: the same
planner and the same stitcher serve the MCP tool a voice note arrives on and
the OpenAI-compatible HTTP endpoint a meeting driver posts to, and a pure
function is the only kind a test can exercise without a model, a gateway and
an agent.

Two ways to cut:

  - Blind, when nothing is known about the audio. Cuts land on a clock and a
    word will sooner or later straddle one, so each piece after the first
    starts a few seconds BEFORE its predecessor ended. The straddled word is
    then whole in one of the two, and the stitcher drops the repetition.
  - On a speaker timeline, when the caller has one. A turn change is a
    boundary no word crosses, so a cut placed there needs no overlap at all
    and costs no duplicated audio.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

# How long a piece should be. Below this the model handles a payload with no
# trouble; above it the failures the chunker exists for start. The value is
# also what decides how soon the first piece of text comes back, so it is a
# latency knob as much as a correctness one.
CHUNK_TARGET_SECONDS = float(
    os.environ.get("CERASE_TRANSCRIBE_CHUNK_SECONDS", 120)
)

# How much of the previous piece each blind cut repeats. It has to be longer
# than the longest word anyone says plus the pause around it, and every second
# of it is audio paid for twice, so it is small.
CHUNK_OVERLAP_SECONDS = float(
    os.environ.get("CERASE_TRANSCRIBE_CHUNK_OVERLAP_SECONDS", 6)
)

# A tail shorter than this is folded into the piece before it. A two-second
# final chunk costs a whole model call to transcribe a syllable, and its
# transcript is the one most likely to be junk.
MIN_TAIL_SECONDS = 20.0

# How far a cut may travel to land on a turn change, as a fraction of the
# target. Wider than this and the pieces stop resembling the size that was
# measured; narrower and most boundaries never find a turn to sit on.
TURN_SNAP_FRACTION = 0.35

# The stitcher will not believe an overlap shorter than this many words. Two
# words repeat by coincidence all the time -- "e poi", "and the" -- and cutting
# on a coincidence deletes real speech.
MIN_OVERLAP_WORDS = 3

# Speech runs about three words a second; five is generous. The stitcher only
# looks this far into a piece for the repetition, so a transcript that repeats
# a phrase minutes later cannot be mistaken for the seam.
WORDS_PER_SECOND = 5.0

# Slack added to that window, so a boundary that fell in a pause and produced
# a slightly longer overlap is still found.
OVERLAP_WINDOW_SLACK_WORDS = 10

# How much of the two candidate word runs must agree for the seam to be
# accepted. Exact equality is too strict: the same audio transcribed twice
# comes back with a comma moved or a filler dropped.
OVERLAP_AGREEMENT = 0.8

_WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class Chunk:
    """One piece of audio to transcribe.

    `lead_in` is how many seconds at the head of the piece are a repeat of the
    piece before it. Zero means the cut sits on a turn change, and the
    stitcher must not go looking for a repetition that is not there.
    """

    index: int
    start: float
    end: float
    lead_in: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class Turn:
    """One speaker's uninterrupted stretch, as a diarising caller reports it."""

    start: float
    end: float
    speaker: str = ""


def turn_boundaries(turns: Sequence[Turn]) -> list[float]:
    """The instants where the speaker changes.

    The start of the first turn is not one: it is the start of the audio, not
    a change. Consecutive turns by the same speaker do not produce one either,
    which is what makes a timeline of many short turns by one person behave
    like no timeline at all rather than like a very fine grid.
    """
    out: list[float] = []
    previous: str | None = None
    for turn in turns:
        if previous is not None and turn.speaker != previous:
            out.append(float(turn.start))
        previous = turn.speaker
    return sorted(set(out))


def plan_chunks(
    duration_seconds: float,
    target_seconds: float | None = None,
    overlap_seconds: float | None = None,
    turns: Sequence[Turn] | None = None,
) -> list[Chunk]:
    """Where to cut audio of this length.

    Audio that already fits comes back as a single chunk with no overlap, so a
    voice note costs exactly what it cost before there was a chunker.
    """
    target = float(target_seconds if target_seconds is not None else CHUNK_TARGET_SECONDS)
    overlap = float(
        overlap_seconds if overlap_seconds is not None else CHUNK_OVERLAP_SECONDS
    )
    duration = float(duration_seconds)
    if duration <= 0:
        return []
    if target <= 0:
        raise ValueError("target_seconds must be positive")
    if overlap < 0:
        raise ValueError("overlap_seconds cannot be negative")
    if overlap >= target:
        raise ValueError("overlap_seconds must be shorter than target_seconds")
    if duration <= target:
        return [Chunk(index=0, start=0.0, end=duration, lead_in=0.0)]

    changes = turn_boundaries(turns or [])
    snap_window = target * TURN_SNAP_FRACTION

    cuts: list[tuple[float, bool]] = []  # instant, and whether it is a turn change
    position = target
    while position < duration - MIN_TAIL_SECONDS:
        snapped = _nearest_change(changes, position, snap_window)
        previous = cuts[-1][0] if cuts else 0.0
        # A turn change is only worth cutting on if it leaves a piece worth
        # transcribing on either side of it.
        if (
            snapped is not None
            and snapped > previous + max(overlap, MIN_TAIL_SECONDS)
            and snapped < duration - MIN_TAIL_SECONDS
        ):
            cuts.append((snapped, True))
            position = snapped + target
        else:
            cuts.append((position, False))
            position += target
    cuts.append((duration, False))

    chunks: list[Chunk] = []
    start_of_next = 0.0
    for index, (cut, _) in enumerate(cuts):
        previous_was_change = cuts[index - 1][1] if index > 0 else False
        lead_in = 0.0 if index == 0 or previous_was_change else overlap
        chunks.append(
            Chunk(
                index=index,
                start=max(0.0, start_of_next - lead_in),
                end=cut,
                lead_in=min(lead_in, start_of_next),
            )
        )
        start_of_next = cut
    return chunks


def _nearest_change(changes: Sequence[float], target: float, window: float) -> float | None:
    """The turn change closest to `target`, or None if none is near enough."""
    best: float | None = None
    best_distance = window
    for change in changes:
        distance = abs(change - target)
        if distance <= best_distance:
            best, best_distance = change, distance
    return best


class Stitcher:
    """Assembles per-chunk transcripts into one, removing the repeated audio.

    Incremental rather than a single pass over a finished list, because the
    caller streams: a chunk's text has to be emitted the moment it is ready,
    and what it adds to the transcript is only knowable once the seam with the
    previous chunk has been resolved.

    A chunk with no lead-in is appended as it stands -- its cut sits on a turn
    change and no audio was repeated. A chunk with one had its first seconds
    already transcribed as part of its predecessor, so the repeated words are
    located and dropped rather than trusted to be absent: the model is not
    told about the overlap and could not honour such an instruction anyway.
    """

    def __init__(self) -> None:
        self.text = ""

    def add(self, chunk: Chunk, text: str) -> str:
        """Fold one chunk's transcript in; return what it appended."""
        piece = (text or "").strip()
        if not piece:
            return ""
        if not self.text:
            self.text = piece
            return piece
        if chunk.lead_in <= 0:
            self.text = self.text + "\n" + piece
            return "\n" + piece
        kept = _drop_repeated_head(self.text, piece, chunk.lead_in)
        if not kept:
            return ""
        self.text = self.text + " " + kept
        return " " + kept


def stitch(chunks: Sequence[Chunk], texts: Sequence[str]) -> str:
    """Join per-chunk transcripts back into one."""
    if len(chunks) != len(texts):
        raise ValueError("a transcript is required for every chunk")
    stitcher = Stitcher()
    for chunk, text in zip(chunks, texts):
        stitcher.add(chunk, text)
    return stitcher.text


def _drop_repeated_head(previous: str, piece: str, lead_in: float) -> str:
    """`piece` without the words its predecessor already ends with."""
    window = int(lead_in * WORDS_PER_SECOND) + OVERLAP_WINDOW_SLACK_WORDS
    tail = _words(previous)[-window:]
    head = list(_WORD_RE.finditer(piece))[:window]
    if len(tail) < MIN_OVERLAP_WORDS or len(head) < MIN_OVERLAP_WORDS:
        return piece
    head_words = [_normalise(m.group(0)) for m in head]

    # The repeated words are looked for at every offset into the head, not
    # only at its first word. A piece cut mid-sentence often opens with a
    # few words the model invented before it found the thread -- measured
    # against real audio, that is where the seams that fail, fail -- and an
    # anchored search gives up on the whole overlap because of them.
    for size in range(min(len(tail), len(head_words)), MIN_OVERLAP_WORDS - 1, -1):
        left = tail[-size:]
        for offset in range(0, len(head_words) - size + 1):
            right = head_words[offset:offset + size]
            agreed = sum(1 for a, b in zip(left, right) if a == b)
            if agreed / size >= OVERLAP_AGREEMENT:
                return piece[head[offset + size - 1].end():].lstrip(" ,.;:-—")
    return piece


def _words(text: str) -> list[str]:
    return [_normalise(m.group(0)) for m in _WORD_RE.finditer(text)]


def _normalise(word: str) -> str:
    return word.lower().replace("’", "'")
