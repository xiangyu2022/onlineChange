"""Regression tests for binary detection with Bernoulli observations."""
import math
import unittest

import numpy as np

from onlineChange import stat


class BinaryDetectionTests(unittest.TestCase):
    def assert_detection(self, expected_stop, **kwargs):
        observation_times = []

        def observe_one(time):
            # Bound the fixture so a stopping-rule regression cannot hang.
            self.assertLess(time, expected_stop)
            observation_times.append(time)
            return 1

        # Bernoulli probabilities change from 1/5 to 4/5. At x=1,
        # the log likelihood ratio is log(4) and Hyvarinen scores differ by .6.
        stop = stat.detect_bin_stat(
            GEN=observe_one,
            alpha=0.5,
            ULP0=lambda x: -math.log(4) * np.sum(x),
            ULP1=lambda x: math.log(4) * np.sum(x),
            **kwargs
        )
        self.assertEqual(stop, expected_stop)
        self.assertEqual(observation_times, list(range(1, expected_stop)))

    def test_default_hyvarinen_score(self):
        self.assert_detection(4)

    def test_hyvarinen_score_names(self):
        for score in ("hyvarinen", "HyV"):
            with self.subTest(score=score):
                self.assert_detection(4, score=score)

    def test_logarithmic_score_names(self):
        for score in ("logarithmic", "LoG"):
            with self.subTest(score=score):
                self.assert_detection(
                    3,
                    score=score,
                    par0=math.log(1.25),
                    par1=math.log(5),
                )

    def test_logarithmic_computes_normalizers(self):
        self.assert_detection(3, score="LoG", lenx=1)


if __name__ == "__main__":
    unittest.main()
