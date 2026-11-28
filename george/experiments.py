"""Reproducible nine-genre experiment suite. All tuning precedes test scoring."""
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from features import History, encode, snapshot_examples, previous_sets
from metrics import evaluate_scores, bootstrap_lift, rank_scores
from models import Baseline, Learned, RankEnsemble, SnapshotRecommender
from pipeline import clean, temporal_split

HERE = Path(__file__).resolve().parent


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def fit_cached(config, h, cache_dir, tag, validation=None, epochs=None, training_cache=None):
    key = hashlib.sha256(json.dumps([config, tag, epochs], sort_keys=True).encode()).hexdigest()[:20]
    params, info = cache_dir / f'{key}.npz', cache_dir / f'{key}.json'
    model = Learned(**config)
    if params.exists() and info.exists():
        record = json.loads(info.read_text())
        model.load_parameters(params, h.prior)
        model.best_epoch, model.curve = record['best_epoch'], record['curve']
        return model, record
    started = time.perf_counter()
    model.fit(h, validation=validation, epochs=epochs, training_cache=training_cache)
    record = {'config': config, 'best_epoch': model.best_epoch, 'curve': model.curve,
              'training_seconds': time.perf_counter() - started,
              'training_examples': model.training_examples, 'parameter_bytes': 0}
    model.save_parameters(params)
    record['parameter_bytes'] = params.stat().st_size
    write_json(info, record)
    return model, record


def refit(spec, h, cache_dir, tag):
    if spec['family'] == 'ensemble':
        return RankEnsemble([refit(s, h, cache_dir, tag) for s in spec['members']], spec['weights'])
    if spec['family'] == 'baseline':
        return Baseline(**spec['config']).fit(h)
    members = [fit_cached(config, h, cache_dir, tag, epochs=epoch)[0]
               for config, epoch in zip(spec['configs'], spec['epochs'])]
    return RankEnsemble(members, [1 / len(members)] * len(members))


def model_metrics(model, h, events):
    started = time.perf_counter()
    scores = model.scores(h, events.u)
    result = evaluate_scores(events, scores, h.n, h.prior)
    result[0]['scoring_seconds'] = time.perf_counter() - started
    return scores, result


def run(data_dir, output, config_path=HERE / 'experiment_config.json'):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / 'checkpoints'
    cache_dir.mkdir(exist_ok=True)
    config = json.loads(config_path.read_text())
    print('Cleaning nine-genre data...', flush=True)
    r, gg, all_users, audit = clean(data_dir, output)
    r, first, second = temporal_split(r)
    r.to_csv(output / 'split_interactions.csv.gz', index=False)
    users = pd.Index(all_users, dtype='string')
    genres = sorted(gg.genre.unique())
    e = encode(r, users, gg, genres)
    train, val, test = [e.take(r.split.eq(s).to_numpy()) for s in ['train', 'validation', 'test']]
    h = History(train, len(users), first)
    print(f'Train {len(train):,}; validation {len(val):,}; test {len(test):,}; genres {len(genres)}', flush=True)
    source_hash = hashlib.sha256(b''.join((HERE / f).read_bytes() for f in
        ['genres.py', 'pipeline.py', 'features.py', 'metrics.py', 'models.py', 'experiments.py'])).hexdigest()
    data_hash = hashlib.sha256(e.u.tobytes() + e.t.tobytes() + e.y.tobytes()).hexdigest()
    tag = source_hash + data_hash
    pd.DataFrame({'genre': genres, 'positive_review_memberships': train.y.sum(axis=0),
                  'membership_rate': h.prior}).to_csv(output / 'train_genre_distribution.csv', index=False)
    pd.Series(pd.to_datetime(e.t, unit='s').year).value_counts().sort_index().rename_axis('year').rename('reviews').to_csv(output / 'reviews_by_year.csv')
    rows, specs, fitted, curves = [], {}, {}, []
    search_rows = []

    # Baseline settings are selected on validation with exactly the same users.
    choices = {
        'uniform_random': [('random', {})], 'global_mode': [('global', {})],
        'recent_popularity': [('recent', {'days': d}) for d in config['recent_windows_days']],
        'user_mode': [('user', {})], 'repeat_last': [('last', {})],
        'recency_profile': [('recency', {'half_life': d}) for d in config['half_lives_days']],
        'fractional_profile': [('fractional', {})],
        'markov1': [('markov1', {'alpha': a}) for a in config['markov_prior_strengths']],
    }
    for name, options in choices.items():
        best = None
        for kind, params in options:
            clock = time.perf_counter()
            m = Baseline(kind, **params).fit(h)
            _, evaluated = model_metrics(m, h, val)
            row = {'model': name, 'config': json.dumps(m.config(), sort_keys=True),
                   'training_seconds': time.perf_counter() - clock - evaluated[0]['scoring_seconds'], **evaluated[0]}
            search_rows.append(row)
            if best is None or row['user_hit_at_1'] > best[0]['user_hit_at_1']:
                best = row, m
        row, model = best
        rows.append(row); fitted[name] = model
        specs[name] = {'family': 'baseline', 'config': model.config()}
        print(f"Validation {name}: {row['user_hit_at_1']:.4%}", flush=True)

    print('Building historical snapshot features and strict-time transitions...', flush=True)
    logistic_cache = snapshot_examples(train, len(users))
    sequence_cache = previous_sets(train)
    seed_rows = []
    for family in ['logistic', 'mf', 'fmc', 'fpmc']:
        candidates = []
        if family == 'logistic':
            candidates = [dict(kind=family, dim=4, lr=config['learning_rate'], l2=l2,
                               seed=config['seeds'][0], max_epochs=config['max_epochs']) for l2 in config['logistic_l2']]
        else:
            candidates = [dict(kind=family, dim=d, lr=config['learning_rate'], l2=config['factor_l2'],
                               seed=config['seeds'][0], max_epochs=config['max_epochs']) for d in config['factor_dimensions']]
        best = None
        training_cache = logistic_cache if family == 'logistic' else sequence_cache

        def trial(c):
            m, record = fit_cached(c, h, cache_dir, tag + ':train', validation=val, training_cache=training_cache)
            _, evaluated = model_metrics(m, h, val)
            row = {'model': family, 'config': json.dumps(c, sort_keys=True), 'best_epoch': m.best_epoch,
                   'training_seconds': record['training_seconds'], 'parameter_bytes': record['parameter_bytes'], **evaluated[0]}
            search_rows.append(row)
            curves.extend({'model': family, 'config': json.dumps(c, sort_keys=True), **point} for point in m.curve)
            print(f"Validation {family} d={c['dim']} lr={c['lr']} l2={c['l2']} seed={c['seed']}: {row['user_hit_at_1']:.4%}, epoch {m.best_epoch}", flush=True)
            return row, m, c

        for c in candidates:
            attempt = trial(c)
            if best is None or attempt[0]['user_hit_at_1'] > best[0]['user_hit_at_1']:
                best = attempt
        if family != 'logistic':
            for refinement in config['factor_refinements']:
                attempt = trial({**best[2], **refinement})
                if attempt[0]['user_hit_at_1'] > best[0]['user_hit_at_1']:
                    best = attempt
        row, model, chosen = best
        members, configs, epochs = [model], [chosen], [model.best_epoch]
        seed_rows.append({'model': family, 'seed': chosen['seed'], **row})
        times = row['training_seconds']
        for seed in config['seeds'][1:]:
            seed_config = {**chosen, 'seed': seed}
            seed_row, seed_model, _ = trial(seed_config)
            members.append(seed_model); configs.append(seed_config); epochs.append(seed_model.best_epoch)
            times += seed_row['training_seconds']
            seed_rows.append({'model': family, 'seed': seed, **seed_row})
        combined = RankEnsemble(members, [1 / len(members)] * len(members))
        _, evaluation = model_metrics(combined, h, val)
        rows.append({'model': family, 'config': json.dumps(chosen, sort_keys=True),
                     'training_seconds': times, **evaluation[0]})
        fitted[family] = combined
        specs[family] = {'family': family, 'configs': configs, 'epochs': epochs}
        pd.DataFrame(search_rows).to_csv(output / 'validation_search.csv', index=False)
        pd.DataFrame(curves).to_csv(output / 'training_curves.csv', index=False)
    del logistic_cache, sequence_cache
    gc.collect()
    # Two strongest different families. Rank normalization is applied before
    # mixing, including for sigmoid outputs versus BPR's uncalibrated logits.
    eligible = sorted([row for row in rows if row['model'] not in ['uniform_random', 'fractional_profile']],
                      key=lambda row: -row['user_hit_at_1'])
    a, b = [row['model'] for row in eligible[:2]]
    best_mix = None
    for weight in config['ensemble_weights']:
        m = RankEnsemble([fitted[a], fitted[b]], [weight, 1 - weight])
        _, evaluation = model_metrics(m, h, val)
        row = {'model': 'ensemble', 'config': json.dumps({'members': [a, b], 'weights': [weight, 1 - weight]}),
               'training_seconds': 0., **evaluation[0]}
        search_rows.append(row)
        if best_mix is None or row['user_hit_at_1'] > best_mix[0]['user_hit_at_1']:
            best_mix = row, m, weight
    row, mix, weight = best_mix
    rows.append(row); fitted['ensemble'] = mix
    specs['ensemble'] = {'family': 'ensemble', 'members': [specs[a], specs[b]], 'weights': [weight, 1 - weight]}
    # Stable tie rule favors earlier simpler entries. No test data accessed for selection.
    winner = max(rows, key=lambda row: row['user_hit_at_1'])['model']
    write_json(output / 'selected_models.json', {'winner': winner, 'specs': specs})
    pd.DataFrame(rows).to_csv(output / 'validation_metrics.csv', index=False)
    pd.DataFrame(search_rows).to_csv(output / 'validation_search.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(output / 'seed_metrics.csv', index=False)
    print(f'Selected on validation: {winner}. Refitting locked configurations.', flush=True)
    del fitted
    gc.collect()

    # Earlier rolling-origin diagnostics use fixed settings: no test-based tuning.
    rolling = []
    for start, end in config['development_fold_quantiles']:
        times = np.sort(e.t)
        left, right = int(times[int(start * len(e))]), int(times[int(end * len(e))])
        fh = History(e.take(e.t < left), len(users), left)
        fe = e.take((e.t >= left) & (e.t < right))
        for name in dict.fromkeys(['global_mode', 'user_mode', 'recency_profile', 'markov1', winner]):
            model = refit(specs[name], fh, cache_dir, tag + f':fold:{left}')
            _, evaluated = model_metrics(model, fh, fe)
            rolling.append({'model': name, 'cutoff_utc': pd.to_datetime(left, unit='s', utc=True).isoformat(), **evaluated[0]})
        del fh
    pd.DataFrame(rolling).to_csv(output / 'development_folds.csv', index=False)

    fit_events = e.take(e.t < second)
    th = History(fit_events, len(users), second)
    test_rows, cohort_rows, genre_rows = [], [], []
    base_users = selected_users = None
    for name, spec in specs.items():
        clock = time.perf_counter()
        model = refit(spec, th, cache_dir, tag + ':trainval')
        training_seconds = time.perf_counter() - clock
        scores, evaluated = model_metrics(model, th, test)
        metrics, per_user, cohorts, per_genre, predictions = evaluated
        test_rows.append({'model': name, 'refit_seconds': training_seconds, **metrics})
        cohort_rows.append(cohorts.assign(model=name))
        per_genre['genre'] = [genres[i] for i in per_genre.genre_index]
        genre_rows.append(per_genre.assign(model=name))
        print(f"Test {name}: {metrics['user_hit_at_1']:.4%}", flush=True)
        if name == 'global_mode':
            base_users = per_user
        if name == winner:
            selected_users = per_user
            out = r.loc[r.split.eq('test')].copy()
            out['recommended_genre'] = np.asarray(genres)[predictions]
            out['hit'] = test.y[np.arange(len(test)), predictions]
            out['history_count'] = th.n[test.u]
            out.to_csv(output / 'test_predictions.csv.gz', index=False)
            snapshot = SnapshotRecommender(users, genres, model.scores(th, np.arange(len(users))), th.n, th.prior)
            snapshot.save(output / 'evaluation_model.npz')
        del model
    pd.DataFrame(test_rows).to_csv(output / 'test_metrics.csv', index=False)
    pd.concat(cohort_rows).to_csv(output / 'test_cohorts.csv', index=False)
    pd.concat(genre_rows).to_csv(output / 'test_per_genre.csv', index=False)

    # Review-edit sensitivity: same calendar boundaries, unedited records only.
    sensitivity = []
    unedited = ~r.edited.to_numpy(dtype=bool)
    uh = History(e.take((e.t < second) & unedited), len(users), second)
    ue = e.take((e.t >= second) & unedited)
    for name in ['global_mode', 'user_mode', 'repeat_last', 'markov1']:
        model = refit(specs[name], uh, cache_dir, tag + ':unedited')
        _, result = model_metrics(model, uh, ue)
        sensitivity.append({'model': name, **result[0]})
    pd.DataFrame(sensitivity).to_csv(output / 'unedited_sensitivity.csv', index=False)
    del uh, th, h
    gc.collect()

    # Refit winner only after evaluation, using selected epochs without test tuning.
    final = History(e, len(users), int(e.t.max()) + 1)
    deployment = refit(specs[winner], final, cache_dir, tag + ':all')
    scores = deployment.scores(final, np.arange(len(users)))
    snapshot = SnapshotRecommender(users, genres, scores, final.n, final.prior)
    snapshot.save(output / 'recommendation_model.npz')
    recs = snapshot.predict(users.tolist())
    recs['global_baseline_genre'] = genres[int(np.argmax(final.prior))]
    recs.to_csv(output / 'user_recommendations.csv.gz', index=False)
    summary = {'task_version': 'nine-gameplay-genres-v2', 'audit': audit, 'genres': genres,
               'selected_model': winner, 'selected_spec': specs[winner],
               'split_counts': r.split.value_counts().to_dict(),
               'validation_start_utc': pd.to_datetime(first, unit='s', utc=True).isoformat(),
               'test_start_utc': pd.to_datetime(second, unit='s', utc=True).isoformat(),
               'event_time_min_utc': pd.to_datetime(e.t.min(), unit='s', utc=True).isoformat(),
               'event_time_max_utc': pd.to_datetime(e.t.max(), unit='s', utc=True).isoformat(),
               'evaluation_global_mode': genres[int(np.argmax(fit_events.y.mean(axis=0)))],
               'test_lift': bootstrap_lift(base_users, selected_users, samples=config['bootstrap_samples']),
               'recommendation_users': len(recs), 'recommendation_cold_users': int((final.n == 0).sum()),
               'source_code_sha256': source_hash, 'clean_event_sha256': data_hash,
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__},
               'wall_seconds': time.perf_counter() - started, 'config': config,
               'evaluation_status': 'Exploratory temporal holdout: overlapping events were inspected in v1; not an untouched confirmatory test.',
               'development_fold_status': 'Earlier temporal diagnostics with settings selected on the later validation window; not independent nested cross-validation.'}
    write_json(output / 'summary.json', summary)
    print(json.dumps({'winner': winner, 'lift': summary['test_lift'], 'seconds': summary['wall_seconds']}, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE.parent / 'steam_dataset_2025_csv')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs_v2')
    parser.add_argument('--config', type=Path, default=HERE / 'experiment_config.json')
    args = parser.parse_args()
    run(args.data_dir, args.output_dir, args.config)
