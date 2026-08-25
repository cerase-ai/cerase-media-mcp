"""Where long audio is cut, and how the pieces are put back together.

Runs on the host python3 (the unit tier's `unittest discover`): the chunker
takes no dependency on the model, on ffmpeg or on the MCP stack, which is the
reason it is a separate module from the server at all.

The reconstruction cases are the chunk-size measurement kept rather than done
once. They cut a long passage of NON-REPETITIVE prose at a range of sizes,
hand each piece to a transcriber that is honest about what it was given, and
require the assembled text back word for word. A repeated phrase would make
this pass for the wrong reason -- the stitcher would find its seam anywhere --
so the passage says something different in every sentence.
"""
from __future__ import annotations

import math
import os
import re
import unittest

import chunker
from chunker import Chunk, Turn

_PASSAGE = open(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tests", "live", "fixtures", "chunker-speech.txt",
    ),
    encoding="utf-8",
).read().split()

# Words per second the fake recording is spoken at. Ordinary speech is around
# three. The value is deliberately not a round one: at a rate that divides the
# chunk size evenly every blind cut would land between two words, and the
# boundary problem the overlap exists for would never occur in the fixture.
_SPOKEN_RATE = 2.87
_SPOKEN_DURATION = len(_PASSAGE) / _SPOKEN_RATE


def _spoken(start: float, end: float) -> str:
    """What a perfect transcriber returns for the audio between two instants.

    Whole words only, which is what makes the boundary problem real: a cut
    inside a word gives one side a word the other side also claims.
    """
    first = max(0, math.floor(start * _SPOKEN_RATE + 1e-6))
    last = min(len(_PASSAGE), math.ceil(end * _SPOKEN_RATE - 1e-6))
    return " ".join(_PASSAGE[first:last])


class PlanCoversTheAudioTest(unittest.TestCase):
    def test_audio_that_fits_is_one_chunk_with_no_overlap(self):
        chunks = chunker.plan_chunks(45.0, target_seconds=120.0)
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].start, chunks[0].end), (0.0, 45.0))
        self.assertEqual(chunks[0].lead_in, 0.0)

    def test_silence_is_no_chunks(self):
        self.assertEqual(chunker.plan_chunks(0.0), [])

    def test_every_instant_of_the_audio_is_inside_some_chunk(self):
        chunks = chunker.plan_chunks(3600.0, target_seconds=120.0, overlap_seconds=6.0)
        self.assertEqual(chunks[0].start, 0.0)
        self.assertEqual(chunks[-1].end, 3600.0)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertLessEqual(later.start, earlier.end)

    def test_each_later_chunk_repeats_exactly_the_overlap(self):
        chunks = chunker.plan_chunks(900.0, target_seconds=120.0, overlap_seconds=6.0)
        self.assertGreater(len(chunks), 2)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(earlier.end - later.start, 6.0, places=6)
            self.assertAlmostEqual(later.lead_in, 6.0, places=6)

    def test_no_chunk_is_a_sliver(self):
        # A cut landing a few seconds before the end would spend a whole model
        # call on a syllable and return the least reliable text of the run.
        for duration in (121.0, 125.0, 139.0, 241.0, 3607.0):
            for chunk in chunker.plan_chunks(duration, target_seconds=120.0):
                self.assertGreater(chunk.duration, chunker.MIN_TAIL_SECONDS)

    def test_an_overlap_as_long_as_the_chunk_is_refused(self):
        with self.assertRaises(ValueError):
            chunker.plan_chunks(600.0, target_seconds=60.0, overlap_seconds=60.0)


class SpeakerTimelineTest(unittest.TestCase):
    def test_a_cut_moves_onto_a_turn_change_and_needs_no_overlap(self):
        turns = [Turn(0, 131, "a"), Turn(131, 400, "b")]
        chunks = chunker.plan_chunks(400.0, target_seconds=120.0, turns=turns)
        self.assertEqual(chunks[0].end, 131.0)
        self.assertEqual(chunks[1].start, 131.0)
        self.assertEqual(chunks[1].lead_in, 0.0)

    def test_a_turn_change_too_far_away_is_not_used(self):
        turns = [Turn(0, 300, "a"), Turn(300, 400, "b")]
        chunks = chunker.plan_chunks(400.0, target_seconds=120.0, turns=turns)
        self.assertEqual(chunks[0].end, 120.0)
        self.assertEqual(chunks[1].lead_in, chunker.CHUNK_OVERLAP_SECONDS)

    def test_the_same_speaker_twice_running_is_not_a_change(self):
        turns = [Turn(0, 60, "a"), Turn(60, 130, "a"), Turn(130, 400, "b")]
        self.assertEqual(chunker.turn_boundaries(turns), [130.0])

    def test_the_start_of_the_first_turn_is_not_a_change(self):
        self.assertEqual(chunker.turn_boundaries([Turn(0, 60, "a")]), [])


class StitchTest(unittest.TestCase):
    def test_the_repeated_words_at_a_seam_are_dropped_once(self):
        chunks = [Chunk(0, 0, 120), Chunk(1, 114, 240, lead_in=6.0)]
        joined = chunker.stitch(chunks, [
            "the surveyor counted the shipping containers twice on the northern quay",
            "on the northern quay before the tide turned",
        ])
        self.assertEqual(
            joined,
            "the surveyor counted the shipping containers twice on the northern quay"
            " before the tide turned",
        )

    def test_a_seam_survives_the_model_hearing_it_slightly_differently(self):
        # The same audio transcribed twice does not come back identical, so an
        # exact match on the overlap would fail exactly where it is needed.
        chunks = [Chunk(0, 0, 120), Chunk(1, 114, 240, lead_in=6.0)]
        joined = chunker.stitch(chunks, [
            "a ledger from the previous winter listed nine deliveries",
            "a ledger from the previous winter listed nine deliverys that nobody signed",
        ])
        self.assertEqual(
            joined,
            "a ledger from the previous winter listed nine deliveries that nobody signed",
        )

    def test_a_seam_survives_a_piece_that_opens_with_words_nobody_said(self):
        # Measured against real audio: a piece cut mid-sentence sometimes opens
        # with a few invented words before the model finds the thread. Anchored
        # at the first word, the search gives up and the overlap ships twice.
        chunks = [Chunk(0, 0, 120), Chunk(1, 114, 240, lead_in=6.0)]
        joined = chunker.stitch(chunks, [
            "the night watchman wrote down the number of every lorry that passed the gate",
            "in turn around man with lorry that passed the gate a crate of instruments arrived",
        ])
        self.assertEqual(
            joined,
            "the night watchman wrote down the number of every lorry that passed the gate"
            " a crate of instruments arrived",
        )

    def test_a_coincidental_two_word_repeat_is_not_treated_as_the_seam(self):
        chunks = [Chunk(0, 0, 120), Chunk(1, 114, 240, lead_in=6.0)]
        joined = chunker.stitch(chunks, [
            "the crew moved the timber under the long shed and the",
            "and the road inspector preferred to walk the culverts himself",
        ])
        self.assertIn("long shed and the and the road inspector", joined)

    def test_a_turn_cut_seam_joins_without_looking_for_a_repeat(self):
        chunks = [Chunk(0, 0, 131), Chunk(1, 131, 400, lead_in=0.0)]
        joined = chunker.stitch(chunks, ["first speaker said this", "first speaker said this"])
        self.assertEqual(joined, "first speaker said this\nfirst speaker said this")

    def test_an_empty_piece_does_not_break_the_transcript(self):
        chunks = [Chunk(0, 0, 120), Chunk(1, 114, 240, lead_in=6.0), Chunk(2, 234, 360, 6.0)]
        joined = chunker.stitch(chunks, ["first part here", "", "second part here"])
        self.assertIn("first part here", joined)
        self.assertIn("second part here", joined)

    def test_a_transcript_per_chunk_is_required(self):
        with self.assertRaises(ValueError):
            chunker.stitch([Chunk(0, 0, 120)], ["one", "two"])


class ReconstructsRealProseTest(unittest.TestCase):
    """The chunk size measurement, expressed as a property.

    Cutting the passage and putting it back must return the passage. It is run
    across a range of sizes rather than at the shipped one alone, so the value
    can be changed on evidence about the model without anyone wondering whether
    the assembly still holds at the new number.
    """

    def _reconstruct(self, target: float, overlap: float, turns=None) -> str:
        chunks = chunker.plan_chunks(
            _SPOKEN_DURATION, target_seconds=target, overlap_seconds=overlap, turns=turns
        )
        texts = [_spoken(c.start, c.end) for c in chunks]
        return re.sub(r"\s+", " ", chunker.stitch(chunks, texts)).strip()

    def test_the_shipped_chunk_size_reconstructs_the_passage_exactly(self):
        self.assertEqual(
            self._reconstruct(chunker.CHUNK_TARGET_SECONDS, chunker.CHUNK_OVERLAP_SECONDS),
            " ".join(_PASSAGE),
        )

    def test_it_reconstructs_at_every_chunk_size_worth_considering(self):
        for target in (30.0, 60.0, 90.0, 120.0, 180.0, 240.0):
            with self.subTest(target=target):
                self.assertEqual(
                    self._reconstruct(target, chunker.CHUNK_OVERLAP_SECONDS),
                    " ".join(_PASSAGE),
                )

    def test_it_reconstructs_when_the_cuts_follow_a_speaker_timeline(self):
        # A diariser reports turn changes BETWEEN words, never inside one, so
        # the timeline is built from word positions rather than from the clock.
        every = 71
        turns = [
            Turn(word / _SPOKEN_RATE, (word + every) / _SPOKEN_RATE,
                 "a" if index % 2 else "b")
            for index, word in enumerate(range(0, len(_PASSAGE), every))
        ]
        reconstructed = self._reconstruct(120.0, 6.0, turns).replace("\n", " ")
        self.assertEqual(re.sub(r"\s+", " ", reconstructed), " ".join(_PASSAGE))

    def test_the_passage_is_not_repetitive_enough_to_flatter_the_stitcher(self):
        # Guards the fixture, not the code: a passage with a repeated phrase
        # would let the stitcher find a seam in the wrong place and still pass
        # every case above.
        lowered = [w.lower().strip(".,") for w in _PASSAGE]
        five_grams = [tuple(lowered[i:i + 5]) for i in range(len(lowered) - 4)]
        self.assertEqual(len(five_grams), len(set(five_grams)))
        self.assertGreater(_SPOKEN_DURATION, 120.0)


if __name__ == "__main__":
    unittest.main()
