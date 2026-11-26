"""Run cleanup, chronological model selection, holdout evaluation and refit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from genres import ID_TO_GENRE, ID_TO_EXCLUSION
from model import GenreRecommender

HERE = Path(__file__).resolve().parent


def clean(data_dir, output):
    audit = {}
    apps = pd.read_csv(data_dir / 'applications.csv', usecols=['appid', 'type'])
    if apps.appid.duplicated().any():
        raise ValueError('Application IDs must be unique')
    audit['application_types'] = apps.type.value_counts().to_dict()
    names = pd.read_csv(data_dir / 'genres.csv')
    unknown = set(names.id) - set(ID_TO_GENRE) - set(ID_TO_EXCLUSION)
    if unknown:
        raise ValueError(f'Unmapped genre IDs require review: {unknown}')
    names['canonical_genre'] = names.id.map(ID_TO_GENRE)
    names['exclusion_reason'] = names.id.map(ID_TO_EXCLUSION)
    names.to_csv(output / 'genre_mapping.csv', index=False)
    links = pd.read_csv(data_dir / 'application_genres.csv')
    audit['raw_genre_links'] = len(links)
    if not set(links.genre_id).issubset(set(names.id)):
        raise ValueError('Genre links contain unknown IDs')
    links['genre'] = links.genre_id.map(ID_TO_GENRE)
    games = set(apps.loc[apps.type.eq('game'), 'appid'])
    gg = links.loc[links.appid.isin(games) & links.genre.notna(), ['appid', 'genre']].drop_duplicates()
    audit['normalized_game_genre_links'] = len(gg)
    audit['games_with_usable_genres'] = int(gg.appid.nunique())
    gg.to_csv(output / 'game_genres.csv.gz', index=False)

    cols = ['recommendationid', 'appid', 'author_steamid', 'timestamp_created', 'timestamp_updated', 'voted_up']
    r = pd.read_csv(data_dir / 'reviews.csv', usecols=cols,
                    dtype={'author_steamid': 'string', 'recommendationid': 'string', 'voted_up': 'string'})
    audit['raw_reviews'] = len(r)
    # Preserve 64-bit Steam IDs as strings. Keep all identifiable users for output,
    # including users with negative reviews only or excluded application types.
    valid_user = r.author_steamid.str.fullmatch(r'\d+', na=False)
    all_users = sorted(r.loc[valid_user, 'author_steamid'].unique())
    audit['raw_users'] = len(all_users)
    counts = r.loc[valid_user].groupby('author_steamid').size()
    audit['raw_user_review_quantiles'] = {str(k): float(v) for k, v in counts.quantile([0, .5, .75, .9, .95, .99, 1]).items()}
    counts.value_counts().sort_index().rename_axis('review_count').rename('users').to_csv(output / 'user_history_distribution.csv')
    r['voted_up'] = r.voted_up.str.lower().map({'true': True, 'false': False})
    for c in ['appid', 'timestamp_created', 'timestamp_updated']:
        r[c] = pd.to_numeric(r[c], errors='coerce')
    valid = valid_user & r.recommendationid.str.fullmatch(r'\d+', na=False) & r.voted_up.notna()
    for c in ['appid', 'timestamp_created', 'timestamp_updated']:
        valid &= r[c].notna() & np.isfinite(r[c]) & r[c].gt(0) & r[c].mod(1).eq(0)
    audit['invalid_required_fields_removed'] = int((~valid).sum())
    r = r.loc[valid].copy()
    # The stored vote/text may be edited. Its earliest defensible availability
    # is the last update, not necessarily the original review creation time.
    r['event_time'] = r[['timestamp_created', 'timestamp_updated']].max(axis=1).astype('int64')
    audit['edited_reviews'] = int(r.timestamp_updated.gt(r.timestamp_created).sum())
    r = r.sort_values(['event_time', 'recommendationid'], kind='stable')
    before = len(r)
    r = r.drop_duplicates('recommendationid', keep='last')
    audit['duplicate_review_ids_removed'] = before - len(r)
    before = len(r)
    r = r.drop_duplicates(['author_steamid', 'appid'], keep='last')
    audit['duplicate_user_game_pairs_removed'] = before - len(r)
    audit['unmatched_application_reviews_removed'] = int((~r.appid.isin(apps.appid)).sum())
    r = r[r.appid.isin(apps.appid)]
    audit['non_game_reviews_removed'] = int((~r.appid.isin(games)).sum())
    r = r[r.appid.isin(games)]
    audit['missing_usable_genre_reviews_removed'] = int((~r.appid.isin(gg.appid)).sum())
    r = r[r.appid.isin(gg.appid)]
    audit['negative_reviews_excluded'] = int((~r.voted_up.astype(bool)).sum())
    r = r[r.voted_up.astype(bool)].copy()
    r['appid'] = r.appid.astype('int64')
    r = r[['recommendationid', 'author_steamid', 'appid', 'event_time']].reset_index(drop=True)
    audit['clean_positive_reviews'] = len(r)
    audit['clean_positive_users'] = int(r.author_steamid.nunique())
    audit['clean_reviewed_games'] = int(r.appid.nunique())
    return r, gg, all_users, audit


def temporal_split(r):
    """Global time boundaries, with timestamp ties kept in a single partition."""
    if len(r) < 3:
        raise ValueError('Need at least three interactions')
    times = np.sort(r.event_time.to_numpy())
    first, second = int(times[int(.70 * len(times))]), int(times[int(.85 * len(times))])
    result = r.copy()
    result['split'] = np.where(r.event_time < first, 'train',
                               np.where(r.event_time < second, 'validation', 'test'))
    if set(result.split) != {'train', 'validation', 'test'}:
        raise ValueError('Insufficient distinct timestamps for three chronological partitions')
    return result, first, second


def evaluate(model, holdout, gg):
    """One prediction per user; a hit means it is a genre of the liked game.

    Average hits per user first so prolific reviewers do not dominate selection.
    No holdout interactions update a profile during evaluation.
    """
    predictions = model.predict(holdout.author_steamid.drop_duplicates())
    scored = holdout.merge(predictions, on='author_steamid', validate='many_to_one')
    matches = gg.rename(columns={'genre': 'recommended_genre'}).assign(hit=1)
    scored = scored.merge(matches, on=['appid', 'recommended_genre'], how='left', validate='many_to_one')
    scored['hit'] = scored.hit.fillna(0).astype(int)
    user_scores = scored.groupby('author_steamid').agg(hit=('hit', 'mean'), history_count=('history_count', 'first'))
    result = {}
    for label, mask in [('all', np.ones(len(user_scores), dtype=bool)),
                        ('warm', user_scores.history_count.gt(0)),
                        ('cold', user_scores.history_count.eq(0))]:
        group = user_scores.loc[mask, 'hit']
        result[label + '_users'] = len(group)
        result[label + '_user_hit_at_1'] = float(group.mean()) if len(group) else None
    result['review_hit_at_1'] = float(scored.hit.mean())
    result['reviews'] = len(scored)
    result['prediction_genres'] = int(predictions.recommended_genre.nunique())
    return result, scored, user_scores


def paired_bootstrap(baseline, selected, seed=2026, samples=1000):
    """Paired user-level resampling preserves within-user review dependence."""
    delta = selected.hit.subtract(baseline.hit).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    # Resampling counts over distinct delta values is exactly equivalent to
    # resampling users, but avoids allocating samples x users matrices.
    values, counts = np.unique(delta, return_counts=True)
    means = rng.multinomial(len(delta), counts / len(delta), size=samples) @ values / len(delta)
    return {'mean_lift': float(delta.mean()), 'ci_95_low': float(np.quantile(means, .025)),
            'ci_95_high': float(np.quantile(means, .975)), 'samples': samples, 'seed': seed}


def main(data_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    print('Cleaning local Steam export...', flush=True)
    r, gg, all_users, audit = clean(data_dir, output)
    r, first, second = temporal_split(r)
    r.to_csv(output / 'split_interactions.csv.gz', index=False)
    train = r[r.split.eq('train')]
    val = r[r.split.eq('validation')]
    test = r[r.split.eq('test')]
    print(f'Positive review split: {len(train):,} / {len(val):,} / {len(test):,}', flush=True)
    # Fit counts once; the candidates vary only in smoothing, never their data.
    model = GenreRecommender().fit(train, gg)
    global_counts = model.global_counts.rename_axis('genre').rename('positive_review_votes').reset_index()
    global_counts['membership_rate'] = global_counts.positive_review_votes / len(train)
    global_counts.sort_values('positive_review_votes', ascending=False).to_csv(output / 'train_genre_distribution.csv', index=False)
    candidates = [('global_mode', 0, True), ('personal_mode', 0, False)]
    candidates += [(f'smoothed_alpha_{alpha}', alpha, False) for alpha in [1, 5, 20, 100]]
    rows = []
    for name, alpha, global_only in candidates:
        model.alpha, model.global_only = alpha, global_only
        metrics, _, _ = evaluate(model, val, gg)
        rows.append({'model': name, 'alpha': alpha, 'global_only': global_only, **metrics})
    # Exact ties favor the first, simpler candidate. Test is not used here.
    best = max(rows, key=lambda row: row['all_user_hit_at_1'])
    pd.DataFrame(rows).to_csv(output / 'validation_metrics.csv', index=False)
    print(f"Selected on validation: {best['model']}", flush=True)
    history = r[r.split.ne('test')]
    selected = GenreRecommender(best['alpha'], best['global_only']).fit(history, gg)
    baseline = GenreRecommender(global_only=True).fit(history, gg)
    test_rows = []
    baseline_metrics, _, baseline_users = evaluate(baseline, test, gg)
    test_rows.append({'model': 'global_mode', **baseline_metrics})
    selected_metrics, predictions, selected_users = evaluate(selected, test, gg)
    test_rows.append({'model': best['model'], **selected_metrics})
    pd.DataFrame(test_rows).to_csv(output / 'test_metrics.csv', index=False)
    predictions.to_csv(output / 'test_predictions.csv.gz', index=False)
    selected.save(output / 'evaluation_model.npz')
    summary = {'audit': audit, 'split_counts': r.split.value_counts().to_dict(),
               'validation_start_utc': pd.to_datetime(first, unit='s', utc=True).isoformat(),
               'test_start_utc': pd.to_datetime(second, unit='s', utc=True).isoformat(),
               'event_time_min_utc': pd.to_datetime(r.event_time.min(), unit='s', utc=True).isoformat(),
               'event_time_max_utc': pd.to_datetime(r.event_time.max(), unit='s', utc=True).isoformat(),
               'selected_model': best['model'], 'alpha': best['alpha'],
               'evaluation_global_mode': baseline.global_mode,
               'test_lift': paired_bootstrap(baseline_users, selected_users),
               'source_files': {p.name: {'bytes': p.stat().st_size} for p in data_dir.glob('*.csv')},
               'manifest_sha256': hashlib.sha256((data_dir / 'MANIFEST.json').read_bytes()).hexdigest(),
               'versions': {'numpy': np.__version__, 'pandas': pd.__version__}}
    # Only AFTER holdout evaluation, refit for present-day recommendations.
    deployment = GenreRecommender(best['alpha'], best['global_only']).fit(r, gg)
    deployment.save(output / 'recommendation_model.npz')
    recs = deployment.predict(all_users)
    recs['global_baseline_genre'] = deployment.global_mode
    recs.to_csv(output / 'user_recommendations.csv.gz', index=False)
    summary['recommendation_users'] = len(recs)
    summary['recommendation_cold_users'] = int(recs.history_count.eq(0).sum())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'selected': best['model'], 'test': test_rows, 'lift': summary['test_lift']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE.parent / 'steam_dataset_2025_csv')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs')
    args = parser.parse_args()
    main(args.data_dir, args.output_dir)
