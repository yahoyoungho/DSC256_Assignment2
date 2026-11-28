"""Leakage, target semantics, omissions, ranking and learned-model regression tests."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from features import Events, History, previous_sets, snapshot_examples
from genres import ID_TO_GENRE
from metrics import evaluate_scores, ranking, rank_scores
from models import Baseline, Learned, SnapshotRecommender
from pipeline import clean
from diagnostics import replay_features


class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.e = Events(np.array([0, 0, 0, 1, 1]), np.array([10, 20, 20, 5, 15]),
                        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float32))
        self.h = History(self.e, 3, 21)

    def test_previous_is_strictly_earlier_and_ties_do_not_leak(self):
        p = previous_sets(self.e)
        np.testing.assert_array_equal(p[1], [1, 0, 0])
        np.testing.assert_array_equal(p[2], [1, 0, 0])
        np.testing.assert_array_equal(self.h.last[0], [0, 1, 1])

    def test_rejects_future_history(self):
        with self.assertRaises(ValueError):
            History(self.e, 3, 20)

    def test_fractional_history_conserves_one_vote_per_game(self):
        e = Events(np.array([0, 0]), np.array([1, 2]), np.array([[1, 1, 0], [0, 0, 1]], dtype=np.float32))
        h = History(e, 1, 3)
        np.testing.assert_allclose(h.profile(fractional=True)[0], [.25, .25, .5])
        np.testing.assert_allclose(h.profile()[0], [.5, .5, .5])

    def test_half_life_has_time_units(self):
        e = Events(np.array([0, 0]), np.array([1, 86401]), np.array([[1, 0], [0, 1]], dtype=np.float32))
        np.testing.assert_allclose(History(e, 1, 86402).profile(1)[0], [1 / 3, 2 / 3], rtol=1e-6)

    def test_old_history_does_not_numerically_turn_into_global_fallback(self):
        e = Events(np.array([0, 1, 1]), np.array([1, 2, 3]), np.array([[1, 0], [0, 1], [0, 1]], dtype=np.float32))
        old = History(e, 2, 100 * 365 * 86400)
        np.testing.assert_allclose(old.profile(30)[0], [1, 0])

    def test_no_mutation_during_baseline_prediction_and_cold_fallback(self):
        counts = self.h.counts.copy()
        for kind in ['user', 'last', 'markov1', 'recency']:
            m = Baseline(kind, half_life=30).fit(self.h)
            np.testing.assert_allclose(m.scores(self.h, [2])[0], self.h.prior)
        np.testing.assert_array_equal(counts, self.h.counts)

    def test_random_is_batch_invariant(self):
        m = Baseline('random').fit(self.h)
        np.testing.assert_array_equal(m.scores(self.h, [0, 2])[1], m.scores(self.h, [2])[0])

    def test_multilabel_metrics_and_random_floor(self):
        e = Events(np.array([0, 0, 1]), np.array([22, 23, 24]),
                   np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32))
        scores = np.array([[3, 2, 1], [3, 2, 1], [3, 2, 1]], dtype=float)
        metrics, _, cohorts, _, _ = evaluate_scores(e, scores, self.h.n, self.h.prior)
        self.assertEqual(metrics['user_hit_at_1'], .5)
        self.assertAlmostEqual(metrics['review_hit_at_1'], 2 / 3)
        self.assertAlmostEqual(metrics['uniform_random_expected_hit'], (0.5 + 1 / 3) / 2)
        self.assertLessEqual(metrics['user_ndcg_at_3'], 1)

    def test_rank_mixing_ignores_score_scale(self):
        s = np.array([[.2, .5, .1]])
        np.testing.assert_array_equal(rank_scores(s, self.h.prior), rank_scores(100 * s + 10, self.h.prior))

    def test_snapshot_roundtrip_unknown_users(self):
        m = SnapshotRecommender(['001', '002'], ['A', 'B', 'C'],
                                np.array([[0, 1, 0], [1, 0, 0]]), np.array([2, 1]), np.array([.6, .3, .1]))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'model.npz'; m.save(p)
            restored = SnapshotRecommender.load(p)
            self.assertEqual(restored.predict(['001', 'new']).recommended_genre.tolist(), ['B', 'A'])

    def test_target_labels_do_not_enter_snapshot_features(self):
        e = Events(np.tile(np.arange(3), 10), np.arange(1, 31), np.tile(np.eye(3, dtype=np.float32), (10, 1)))
        x, y, u = snapshot_examples(e, 3)
        changed = Events(e.u.copy(), e.t.copy(), e.y.copy())
        changed.y[-1] = [1, 1, 1]
        xx, yy, uu = snapshot_examples(changed, 3)
        np.testing.assert_array_equal(x, xx)
        self.assertFalse(np.array_equal(y, yy))

    def test_all_learned_families_train_and_reload(self):
        e = Events(np.tile(np.arange(3), 10), np.arange(1, 31), np.tile(np.eye(3, dtype=np.float32), (10, 1)))
        h = History(e, 4, 31)
        for kind in ['logistic', 'mf', 'fmc', 'fpmc']:
            model = Learned(kind, dim=2, max_epochs=2).fit(h)
            scores = model.scores(h, [0, 1, 2, 3])
            self.assertTrue(np.isfinite(scores).all())
            self.assertLessEqual(model.curve[-1]['training_loss'], model.curve[0]['training_loss'] + .01)
            with tempfile.TemporaryDirectory() as d:
                p = Path(d) / 'params.npz'; model.save_parameters(p)
                restored = Learned(kind, dim=2).load_parameters(p, h.prior)
                np.testing.assert_allclose(scores, restored.scores(h, [0, 1, 2, 3]))

    def test_replay_reveals_only_strictly_earlier_events(self):
        target = np.array([False, True, True, False, True])
        f = replay_features(self.e, target, 90)
        # Both user-0 events at t=20 have exactly one earlier review.
        np.testing.assert_array_equal(f['n'], [1, 1, 1])
        np.testing.assert_array_equal(f['counts'][0], [1, 0, 0])
        np.testing.assert_array_equal(f['counts'][1], [1, 0, 0])
        changed = Events(self.e.u, self.e.t, self.e.y.copy())
        changed.y[1] = [1, 1, 1]
        np.testing.assert_array_equal(f['counts'], replay_features(changed, target, 90)['counts'])


class CleanupTests(unittest.TestCase):
    def test_indie_is_removed_as_label_not_as_entire_game(self):
        self.assertEqual(len(set(ID_TO_GENRE.values())), 9)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); out = root / 'out'; out.mkdir()
            pd.DataFrame({'appid': [1, 2, 3], 'type': ['game', 'game', 'game']}).to_csv(root / 'applications.csv', index=False)
            pd.DataFrame({'id': [31, 122], 'name': ['Indie', 'Action']}).to_csv(root / 'genres.csv', index=False)
            pd.DataFrame({'appid': [1, 1, 2, 3], 'genre_id': [31, 122, 31, 122]}).to_csv(root / 'application_genres.csv', index=False)
            pd.DataFrame({'recommendationid': ['1', '2', '3'], 'appid': [1, 2, 3], 'author_steamid': ['001', '002', '003'],
                          'timestamp_created': [1600000000, 1600000000, 1900000000],
                          'timestamp_updated': [1600000000, 1600000000, 1900000000], 'voted_up': ['True'] * 3,
                          'updated_at': ['2025-09-08 00:00:00+00:00'] * 3}).to_csv(root / 'reviews.csv', index=False)
            r, gg, users, audit = clean(root, out)
            self.assertEqual(r.appid.tolist(), [1])
            self.assertEqual(users, ['001', '002', '003'])
            self.assertEqual(audit['invalid_or_inconsistent_dates_removed'], 1)
            self.assertEqual(audit['indie_tagged_games_retained'], 1)
            self.assertEqual(audit['indie_tagged_games_without_gameplay_genre'], 1)


if __name__ == '__main__':
    unittest.main()
