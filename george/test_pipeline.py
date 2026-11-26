"""Meaningful checks for split isolation, multi-label scoring and cold starts."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from genres import GENRE_IDS, EXCLUDED_IDS
from model import GenreRecommender
from pipeline import evaluate, temporal_split, paired_bootstrap


class RecommenderTests(unittest.TestCase):
    def setUp(self):
        self.gg = pd.DataFrame({'appid': [1, 2, 3, 3], 'genre': ['Action', 'RPG', 'Action', 'RPG']})
        self.train = pd.DataFrame({'author_steamid': ['a', 'a', 'b'], 'appid': [1, 3, 2]})

    def test_global_mode_ties_are_deterministic_and_cold_start_works(self):
        model = GenreRecommender(alpha=0).fit(self.train, self.gg)
        result = model.predict(['a', 'b', 'unknown'])
        self.assertEqual(result.recommended_genre.tolist(), ['Action', 'RPG', 'Action'])
        self.assertEqual(result.history_count.tolist(), [2, 1, 0])
        self.assertTrue(np.isfinite(result.score).all())

    def test_smoothing_calculation_and_baseline(self):
        model = GenreRecommender(alpha=3).fit(self.train, self.gg)
        self.assertAlmostEqual(model.predict(['b']).score.iloc[0], (1 + 3 * 2 / 3) / 4)
        baseline = GenreRecommender(global_only=True).fit(self.train, self.gg)
        self.assertEqual(baseline.predict(['a', 'b']).recommended_genre.tolist(), ['Action', 'Action'])

    def test_chronological_split_has_no_time_or_pair_overlap(self):
        r = pd.DataFrame({'event_time': [1, 2, 3, 4, 5, 6, 7, 7, 8, 9],
                          'recommendationid': list(map(str, range(10)))})
        split, _, _ = temporal_split(r)
        groups = [split[split.split.eq(s)] for s in ['train', 'validation', 'test']]
        self.assertLess(groups[0].event_time.max(), groups[1].event_time.min())
        self.assertLess(groups[1].event_time.max(), groups[2].event_time.min())
        self.assertEqual(split.groupby('event_time').split.nunique().max(), 1)
        self.assertEqual(sum(len(g) for g in groups), len(r))

    def test_multilabel_hit_user_weighting_and_no_holdout_updates(self):
        model = GenreRecommender(alpha=0).fit(self.train, self.gg)
        before = model.counts.copy()
        heldout = pd.DataFrame({'author_steamid': ['a', 'a', 'b'], 'appid': [1, 3, 1]})
        metrics, _, _ = evaluate(model, heldout, self.gg)
        self.assertEqual(metrics['all_user_hit_at_1'], .5)
        self.assertAlmostEqual(metrics['review_hit_at_1'], 2 / 3)
        pd.testing.assert_frame_equal(before, model.counts)

    def test_saved_model_reproduces_predictions(self):
        model = GenreRecommender(alpha=5).fit(self.train, self.gg)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.npz'
            model.save(path)
            pd.testing.assert_frame_equal(model.predict(['a', 'b', 'new']),
                                          GenreRecommender.load(path).predict(['a', 'b', 'new']))

    def test_mapping_has_no_overlapping_ids(self):
        ids = [i for groups in [GENRE_IDS, EXCLUDED_IDS] for values in groups.values() for i in values]
        self.assertEqual(len(ids), len(set(ids)))

    def test_bootstrap_pairing(self):
        a = pd.DataFrame({'hit': [0., 1.]}, index=['a', 'b'])
        b = pd.DataFrame({'hit': [1., 0.]}, index=['b', 'a'])
        result = paired_bootstrap(a, b)
        self.assertEqual(result['mean_lift'], 0)
        self.assertEqual(result['ci_95_high'], 0)


if __name__ == '__main__':
    unittest.main()
