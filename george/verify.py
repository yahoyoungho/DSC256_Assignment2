"""Verify full-data split integrity, saved inference and reported metrics."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from features import encode, History
from metrics import evaluate_scores
from models import SnapshotRecommender

HERE = Path(__file__).resolve().parent


def verify(output=HERE / 'outputs_v2'):
    s = json.loads((output / 'summary.json').read_text())
    r = pd.read_csv(output / 'split_interactions.csv.gz', dtype={'author_steamid': 'string', 'recommendationid': 'string'})
    gg = pd.read_csv(output / 'game_genres.csv.gz')
    recs = pd.read_csv(output / 'user_recommendations.csv.gz', dtype={'author_steamid': 'string'})
    assert len(s['genres']) == 9 and 'Indie' not in s['genres']
    assert set(gg.genre) == set(s['genres']) and not gg.duplicated().any()
    assert not r.recommendationid.duplicated().any()
    assert not r.duplicated(['author_steamid', 'appid']).any()
    assert r.appid.isin(gg.appid).all()
    parts = [r[r.split.eq(k)] for k in ['train', 'validation', 'test']]
    assert parts[0].event_time.max() < parts[1].event_time.min()
    assert parts[1].event_time.max() < parts[2].event_time.min()
    assert r.groupby('event_time').split.nunique().max() == 1
    audit = s['audit']
    removals = ['invalid_required_fields_removed', 'invalid_or_inconsistent_dates_removed',
                'duplicate_review_ids_removed', 'duplicate_user_game_pairs_removed',
                'unmatched_application_reviews_removed', 'non_game_reviews_removed',
                'missing_usable_genre_reviews_removed', 'negative_reviews_excluded']
    assert audit['raw_reviews'] - sum(audit[k] for k in removals) == len(r)
    assert len(recs) == s['recommendation_users'] == audit['raw_users']
    assert recs.author_steamid.is_unique and recs.recommended_genre.isin(s['genres']).all()
    evaluated = SnapshotRecommender.load(output / 'evaluation_model.npz')
    deployed = SnapshotRecommender.load(output / 'recommendation_model.npz')
    assert evaluated.users.equals(deployed.users)
    before = r[r.split.ne('test')].groupby('author_steamid').size().reindex(evaluated.users, fill_value=0).to_numpy()
    full = r.groupby('author_steamid').size().reindex(deployed.users, fill_value=0).to_numpy()
    np.testing.assert_array_equal(before, evaluated.counts)
    np.testing.assert_array_equal(full, deployed.counts)
    saved = pd.read_csv(output / 'test_predictions.csv.gz', dtype={'author_steamid': 'string', 'recommendationid': 'string'})
    restored = evaluated.predict(saved.author_steamid.tolist())
    np.testing.assert_array_equal(restored.recommended_genre, saved.recommended_genre)
    predicted = deployed.predict(recs.author_steamid.tolist())
    np.testing.assert_array_equal(predicted.recommended_genre, recs.recommended_genre)
    e = encode(parts[2], evaluated.users, gg, s['genres'])
    indices = evaluated.users.get_indexer(parts[2].author_steamid)
    metrics = evaluate_scores(e, evaluated.scores[indices], evaluated.counts, evaluated.prior)[0]
    reported = pd.read_csv(output / 'test_metrics.csv').set_index('model').loc[s['selected_model']]
    for metric in ['user_hit_at_1', 'review_hit_at_1', 'user_ndcg_at_3', 'genre_coverage']:
        assert np.isclose(metrics[metric], reported[metric]), metric
    global_prior = encode(r[r.split.ne('test')], evaluated.users, gg, s['genres']).y.mean(axis=0)
    np.testing.assert_allclose(evaluated.prior, global_prior)
    assert evaluated.predict(['never_seen']).recommended_genre.iloc[0] == s['evaluation_global_mode']
    current_hash = hashlib.sha256(b''.join((HERE / f).read_bytes() for f in
        ['genres.py', 'pipeline.py', 'features.py', 'metrics.py', 'models.py', 'experiments.py'])).hexdigest()
    assert current_hash == s['source_code_sha256'], 'Results are stale after source changes; rerun experiments'
    replay = pd.read_csv(output / 'replay_diagnostics.csv')
    frozen = replay[replay.protocol.eq('frozen')].set_index('model')
    original = pd.read_csv(output / 'test_metrics.csv').set_index('model')
    np.testing.assert_allclose(frozen.loc[original.index].user_hit_at_1, original.user_hit_at_1)
    result = {'status': 'passed', 'checks': ['nine-genre vocabulary and unique catalog links',
              'unique review IDs and user-game pairs', 'strict chronological splits and timestamp ties',
              'cleanup row conservation', 'one valid recommendation per original user',
              'evaluation and deployment histories fit only their respective partitions',
              'saved-model predictions reproduce all exported recommendations',
              'selected model metrics recomputed from saved scores', 'training-only global prior',
              'unknown-user fallback', 'source hash matches executed results',
              'diagnostic frozen scores reproduce main experiment'],
              'users': len(recs), 'test_reviews': len(saved)}
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    verify()
