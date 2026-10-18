"""The voice detector's pure window and stretch rules.

The probabilities are fed in directly: no model, no ffmpeg, no numpy. The
tests pin the arithmetic that turns a per-window score into the stretches of
audio worth sending to a transcription model.
"""
from __future__ import annotations

import unittest

import vad


class SpeechRunsTest(unittest.TestCase):
    def test_eight_windows_at_speech_level_make_one_run(self):
        runs = vad.speech_runs([0.9] * 8)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0][0], 0.0)
        self.assertAlmostEqual(runs[0][1], 8 * vad.WINDOW_SECONDS)

    def test_seven_windows_are_too_short_to_be_a_run(self):
        self.assertEqual(vad.speech_runs([0.9] * 7), [])

    def test_a_run_continues_through_windows_above_the_silence_floor(self):
        probabilities = [0.9] * 8 + [0.4] * 3 + [0.2]
        runs = vad.speech_runs(probabilities)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0][0], 0.0)
        self.assertAlmostEqual(runs[0][1], 11 * vad.WINDOW_SECONDS)

    def test_a_window_below_the_silence_floor_ends_the_run(self):
        probabilities = [0.9] * 8 + [0.34] + [0.9] * 8
        runs = vad.speech_runs(probabilities)
        self.assertEqual(len(runs), 2)
        self.assertAlmostEqual(runs[0][0], 0.0)
        self.assertAlmostEqual(runs[0][1], 8 * vad.WINDOW_SECONDS)
        self.assertAlmostEqual(runs[1][0], 9 * vad.WINDOW_SECONDS)
        self.assertAlmostEqual(runs[1][1], 17 * vad.WINDOW_SECONDS)

    def test_isolated_spikes_make_no_run(self):
        self.assertEqual(vad.speech_runs([0.9, 0.0] * 20), [])

    def test_a_run_that_reaches_the_last_window_ends_at_the_audio_end(self):
        probabilities = [0.1] * 3 + [0.9] * 8
        runs = vad.speech_runs(probabilities)
        self.assertEqual(len(runs), 1)
        self.assertAlmostEqual(runs[0][0], 3 * vad.WINDOW_SECONDS)
        self.assertAlmostEqual(runs[0][1], len(probabilities) * vad.WINDOW_SECONDS)


class StretchesTest(unittest.TestCase):
    def _assert_stretches(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for got, want in zip(actual, expected):
            self.assertAlmostEqual(got[0], want[0])
            self.assertAlmostEqual(got[1], want[1])

    def test_a_run_near_the_start_is_padded_and_clipped_to_zero(self):
        self._assert_stretches(vad.stretches([(0.2, 5.0)], 245.0), [(0.0, 6.0)])

    def test_runs_five_seconds_apart_are_merged(self):
        self._assert_stretches(
            vad.stretches([(10.0, 20.0), (25.0, 30.0)], 100.0),
            [(9.0, 31.0)],
        )

    def test_runs_fifteen_seconds_apart_stay_two_padded_stretches(self):
        self._assert_stretches(
            vad.stretches([(10.0, 20.0), (35.0, 40.0)], 100.0),
            [(9.0, 21.0), (34.0, 41.0)],
        )

    def test_no_runs_are_no_stretches(self):
        self.assertEqual(vad.stretches([], 245.0), [])

    def test_a_run_at_the_end_is_clipped_to_the_duration(self):
        self._assert_stretches(vad.stretches([(45.0, 50.0)], 50.0), [(44.0, 50.0)])


if __name__ == "__main__":
    unittest.main()
