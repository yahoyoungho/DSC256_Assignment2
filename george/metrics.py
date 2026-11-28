"""Multi-label ranking evaluation, with users as the primary averaging unit."""
import numpy as np
import pandas as pd


def ranking(scores, prior):
    # Stable sorting resolves exact ties by training popularity, then genre order.
    tie = np.argsort(-prior, kind='stable')
    return tie[np.argsort(-scores[:, tie], axis=1, kind='stable')]


def rank_scores(scores, prior):
    order = ranking(scores, prior)
    out = np.empty_like(scores, dtype=np.float32)
    out[np.arange(len(scores))[:, None], order] = np.arange(scores.shape[1] - 1, -1, -1) / (scores.shape[1] - 1)
    return out


def evaluate_scores(e, scores, n_history, prior):
    if scores.shape != e.y.shape or not np.isfinite(scores).all():
        raise ValueError('Scores must be finite and match the event label matrix')
    order = ranking(scores, prior)
    prediction = order[:, 0]
    hits = e.y[np.arange(len(e)), prediction]
    discount = 1 / np.log2(np.arange(2, 5))
    dcg = (np.take_along_axis(e.y, order[:, :3], axis=1) * discount).sum(axis=1)
    idcg = (np.arange(3)[None, :] < e.y.sum(axis=1)[:, None]) @ discount
    frame = pd.DataFrame({'u': e.u, 'hit': hits, 'ndcg3': dcg / idcg, 'history': n_history[e.u]})
    per_user = frame.groupby('u').agg(hit=('hit', 'mean'), ndcg3=('ndcg3', 'mean'), history=('history', 'first'))
    results = {'users': len(per_user), 'reviews': len(e), 'user_hit_at_1': float(per_user.hit.mean()),
               'review_hit_at_1': float(hits.mean()), 'user_ndcg_at_3': float(per_user.ndcg3.mean()),
               'genre_coverage': int(np.unique(prediction).size),
               'uniform_random_expected_hit': float(pd.Series(e.y.sum(axis=1) / e.y.shape[1]).groupby(e.u).mean().mean())}
    cohorts = []
    h = per_user.history
    masks = {'cold': h == 0, 'warm': h > 0, '1-2': h.between(1, 2), '3-4': h.between(3, 4), '5+': h >= 5}
    for label, mask in masks.items():
        part = per_user[mask]
        cohorts.append({'cohort': label, 'users': len(part), 'user_hit_at_1': float(part.hit.mean()) if len(part) else None,
                        'user_ndcg_at_3': float(part.ndcg3.mean()) if len(part) else None})
    per_genre = []
    for g in range(e.y.shape[1]):
        target = e.y[:, g].astype(bool)
        chosen = prediction == g
        per_genre.append({'genre_index': g, 'target_reviews': int(target.sum()), 'predicted_reviews': int(chosen.sum()),
                          'genre_recall_at_1': float(chosen[target].mean()) if target.any() else None,
                          'genre_precision_at_1': float(target[chosen].mean()) if chosen.any() else None,
                          'any_genre_hit_on_this_target_cohort': float(hits[target].mean()) if target.any() else None})
    return results, per_user, pd.DataFrame(cohorts), pd.DataFrame(per_genre), prediction


def bootstrap_lift(baseline, selected, seed=2026, samples=1000):
    delta = selected.hit.subtract(baseline.hit).dropna().to_numpy()
    values, counts = np.unique(delta, return_counts=True)
    draws = np.random.default_rng(seed).multinomial(len(delta), counts / len(delta), size=samples)
    means = draws @ values / len(delta)
    return {'lift': float(delta.mean()), 'ci_low': float(np.quantile(means, .025)),
            'ci_high': float(np.quantile(means, .975)), 'samples': samples, 'seed': seed}
