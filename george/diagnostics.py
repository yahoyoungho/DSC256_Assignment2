"""Additional fixed-configuration diagnostics; never used to select the winner.

Sequential replay reveals an event only after prediction, handles simultaneous
events as baskets, and keeps all learned parameters/global priors frozen.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from features import History, encode, previous_sets, DAY
from models import Baseline, Learned, RankEnsemble
from metrics import evaluate_scores, bootstrap_lift, rank_scores
from experiments import refit, write_json

HERE = Path(__file__).resolve().parent


def replay_features(events, target_mask, half_life):
    """All vectors describe strictly earlier timestamps for the same user.

    Exponential weights use a fixed epoch; the common time scale cancels in
    the normalized profile. Shifting cumulative sums avoids subtractive loss.
    """
    k = events.y.shape[1]
    frame = pd.DataFrame({'u': events.u, 't': events.t, 'n': np.ones(len(events))})
    origin = 946684800  # 2000-01-01 UTC, fixed and independent of target data.
    w = np.exp2((events.t - origin) / (half_life * DAY))
    frame['weight'] = w
    for j in range(k):
        frame[f'g{j}'] = events.y[:, j]
        frame[f'w{j}'] = events.y[:, j] * w
        frame[f'f{j}'] = events.y[:, j] / events.y.sum(axis=1)
    grouped = frame.groupby(['u', 't'], sort=True).sum()
    before = grouped.groupby(level=0).cumsum().groupby(level=0).shift().fillna(0)
    target_index = pd.MultiIndex.from_arrays([events.u[target_mask], events.t[target_mask]])
    before = before.reindex(target_index)
    n = before.n.to_numpy()
    counts = before[[f'g{j}' for j in range(k)]].to_numpy()
    fractional = before[[f'f{j}' for j in range(k)]].to_numpy()
    weights = before.weight.to_numpy()
    weighted = before[[f'w{j}' for j in range(k)]].to_numpy()
    times = pd.Series(grouped.index.get_level_values('t').to_numpy(), index=grouped.index)
    prior_time = times.groupby(level=0).shift().fillna(0).reindex(target_index).to_numpy()
    return {'n': n, 'counts': counts, 'fractional': fractional,
            'weighted': weighted, 'weights': weights, 'last_time': prior_time,
            'last': previous_sets(events)[target_mask]}


def replay_scores(model, h, target, f, recency_f):
    n = f['n']
    profile = np.divide(f['counts'], n[:, None], out=np.broadcast_to(h.prior, f['counts'].shape).copy(), where=n[:, None] > 0)
    decayed90 = np.divide(f['weighted'], f['weights'][:, None], out=np.broadcast_to(h.prior, f['counts'].shape).copy(), where=f['weights'][:, None] > 0)
    if isinstance(model, RankEnsemble):
        return sum(w * rank_scores(replay_scores(m, h, target, f, recency_f), h.prior)
                   for m, w in zip(model.members, model.weights))
    if isinstance(model, Baseline):
        if model.kind in ['global', 'recent', 'random']:
            return model.scores(h, target.u)
        if model.kind == 'user':
            scores = profile.copy()
        elif model.kind == 'fractional':
            scores = np.divide(f['fractional'], n[:, None], out=np.broadcast_to(h.prior, f['counts'].shape).copy(), where=n[:, None] > 0)
        elif model.kind == 'recency':
            scores = np.divide(recency_f['weighted'], recency_f['weights'][:, None], out=np.broadcast_to(h.prior, f['counts'].shape).copy(), where=recency_f['weights'][:, None] > 0)
        elif model.kind == 'last':
            scores = f['last'].copy()
        elif model.kind == 'markov1':
            scores = f['last'] @ model.transition / np.maximum(f['last'].sum(axis=1, keepdims=True), 1)
        else:
            raise ValueError(model.kind)
        scores[n == 0] = h.prior
        return scores
    model.net.eval()
    with torch.no_grad():
        if model.kind == 'logistic':
            age = np.minimum((target.t - f['last_time']) / DAY, 3650)
            age[n == 0] = 3650
            x = np.column_stack([profile, decayed90, f['last'],
                                 np.broadcast_to(h.popularity(90), (len(target), h.k)),
                                 np.log1p(n) / 10, np.log1p(age) / 10]).astype(np.float32)
            scores = model.net(torch.from_numpy(x)).sigmoid().numpy()
            scores[n == 0] = h.prior
        else:
            uid = torch.from_numpy(model.user_map[target.u].astype(np.int64))
            scores = model.net(uid, torch.from_numpy(f['last'])).numpy()
            if model.kind != 'fmc':
                absent = model.user_map[target.u] == 0
                p = np.clip(profile[absent], .001, .999)
                scores[absent] = np.log(p / (1 - p))
            p = np.clip(h.prior, .001, .999)
            scores[n == 0] = np.log(p / (1 - p))
    return scores


def run(output):
    summary = json.loads((output / 'summary.json').read_text())
    specs = json.loads((output / 'selected_models.json').read_text())['specs']
    r = pd.read_csv(output / 'split_interactions.csv.gz', dtype={'author_steamid': 'string', 'recommendationid': 'string'})
    recs = pd.read_csv(output / 'user_recommendations.csv.gz', dtype={'author_steamid': 'string'})
    users = pd.Index(recs.author_steamid, dtype='string')
    gg = pd.read_csv(output / 'game_genres.csv.gz')
    e = encode(r, users, gg, summary['genres'])
    mask = r.split.eq('test').to_numpy()
    target = e.take(mask)
    cutoff = int(pd.Timestamp(summary['test_start_utc']).timestamp())
    h = History(e.take(e.t < cutoff), len(users), cutoff)
    tag = summary['source_code_sha256'] + summary['clean_event_sha256'] + ':trainval'
    f = replay_features(e, mask, 90)
    half = specs['recency_profile']['config']['half_life']
    rf = f if half == 90 else replay_features(e, mask, half)
    rows = []
    user_metrics = {}
    for name, spec in specs.items():
        model = refit(spec, h, output / 'checkpoints', tag)
        frozen = evaluate_scores(target, model.scores(h, target.u), h.n, h.prior)
        replay = evaluate_scores(target, replay_scores(model, h, target, f, rf), h.n, h.prior)
        user_metrics[name] = frozen[1]
        for protocol, result in [('frozen', frozen), ('sequential_replay', replay)]:
            rows.append({'model': name, 'protocol': protocol, **result[0]})
    pd.DataFrame(rows).to_csv(output / 'replay_diagnostics.csv', index=False)
    winner = summary['selected_model']
    comparisons = {name: bootstrap_lift(user_metrics[name], user_metrics[winner])
                   for name in ['global_mode', 'user_mode', 'recency_profile', 'mf']}
    write_json(output / 'paired_comparisons.json', comparisons)
    # Honest measurement of actual served artifact size, separate from search cache.
    write_json(output / 'artifact_sizes.json', {
        'evaluation_model_bytes': (output / 'evaluation_model.npz').stat().st_size,
        'recommendation_model_bytes': (output / 'recommendation_model.npz').stat().st_size,
        'checkpoint_cache_bytes': sum(p.stat().st_size for p in (output / 'checkpoints').glob('*.npz')),
        'notes': 'Replay uses fixed weights and priors, updates only per-user history, and is not used for selection. Cohorts remain defined at the original cutoff.'})
    print('Wrote replay diagnostics, paired comparisons and artifact sizes.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=HERE / 'outputs_v2')
    run(p.parse_args().output_dir)
