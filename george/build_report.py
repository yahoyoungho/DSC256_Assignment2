"""Execute a small Jupyter-compatible analysis notebook and export standalone HTML.

Only standard Python, NumPy and pandas are needed. HTML markdown cells are intentional:
they render in Jupyter and in this lightweight exporter without another dependency.
Cell outputs below are captured from execution, never filled with invented results.
"""
from contextlib import redirect_stdout
import html
import io
import json
from pathlib import Path
import textwrap

HERE = Path(__file__).resolve().parent


def build():
    cells = []

    def md(source):
        cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': textwrap.dedent(source).strip()})

    def code(source):
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': textwrap.dedent(source).strip(),
                      'execution_count': None, 'outputs': []})

    md('''
    <h1>Steam genre recommendation — George's first experiment</h1>
    <p>Meetup preparation: model choice, independent data cleanup and analysis,
    global-mode baseline, and a chronological holdout. This is an executed prototype
    report, not the completed group presentation or final related-work survey.</p>
    <h2>1. Predictive task and model decision</h2>
    <p>Given a user's past positive game reviews, recommend one Steam genre. A future
    positive review supplies a set of acceptable genre labels. Success means the
    recommendation belongs to that set. We do not observe a user's true favorite genre,
    and we are not predicting the genre of a supplied target game.</p>
    <p>Before modeling, inspection found 1,048,148 reviews and 701,884 authors; at least
    75% of authors have only one review. A content-based genre-frequency profile is a
    suitable initial family because it can work with a single prior interaction and
    supports a transparent global fallback. We compare its unsmoothed and smoothed
    variants rather than assuming a more complex model will win.</p>
    <table><tr><th>Model family</th><th>Decision before evaluation</th></tr>
    <tr><td>Global genre mode</td><td>Required baseline, cold-start fallback.</td></tr>
    <tr><td>Personal genre mode / smoothed profile</td><td>Primary candidates; small vocabulary, interpretable preferences.</td></tr>
    <tr><td>Collaborative filtering / factorization</td><td>Useful future benchmark; very sparse user histories are a challenge. Not empirically ruled out.</td></tr>
    <tr><td>Text or neural model</td><td>Not justified as a first step. Target-review text would leak information unavailable at recommendation time.</td></tr></table>
    <p>“Best” below means best validation performance among the six tested configurations,
    not best among all recommender systems.</p>
    ''')
    code('''
    from pathlib import Path
    import json, sys
    import numpy as np
    import pandas as pd
    ROOT = Path.cwd() if (Path.cwd() / 'pipeline.py').exists() else Path.cwd() / 'george'
    ROOT = ROOT.resolve()
    sys.path.insert(0, str(ROOT))
    from model import GenreRecommender
    from pipeline import main
    OUT = ROOT / 'outputs'
    # Reuse the pipeline results; if absent, reproduce them from the local CSVs.
    # To refresh after source/code changes: python3 george/pipeline.py
    if not (OUT / 'summary.json').exists():
        main(ROOT.parent / 'steam_dataset_2025_csv', OUT)
    summary = json.loads((OUT / 'summary.json').read_text())
    def table(frame):
        display_html(frame.to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}'))
    print('Recorded package versions:', summary['versions'])
    print('Selected model:', summary['selected_model'], '| alpha:', summary['alpha'])
    ''')
    md('''
    <h2>2. Data and cleanup</h2>
    <p>Sources: local <code>steam_dataset_2025_csv</code> applications, reviews,
    application_genres and genres tables. The manifest describes table sizes but does
    not establish the upstream source, sampling method or license. Confirm these with
    the team. Authors embedded in reviews are the available user data; there is no
    separate complete user library table.</p>
    <p>Steam IDs stay strings. Validate required identifiers, votes and times; deduplicate
    review IDs and user–game pairs, retaining the latest available state. Restrict to
    games with usable genre labels, then positive votes. Merge translated genre labels
    into ten canonical labels and deduplicate game–genre pairs. Exclude business models,
    release statuses, content descriptors, and software/video categories. Retain Indie
    as a Steam catalog genre; the team should discuss whether that target is useful.</p>
    <p>The following removal counts are sequential, so a negative DLC review is counted
    under non-game removal rather than again as a negative review.</p>
    ''')
    code('''
    audit = summary['audit']
    scalar_audit = [{'measure': k, 'count': v} for k, v in audit.items() if not isinstance(v, dict)]
    table(pd.DataFrame(scalar_audit))
    table(pd.DataFrame(list(audit['raw_user_review_quantiles'].items()), columns=['quantile', 'reviews_per_user']))
    print('Application types:', audit['application_types'])
    ''')
    code('''
    mapping = pd.read_csv(OUT / 'genre_mapping.csv')
    table(mapping.groupby('canonical_genre').size().rename('source_labels').reset_index())
    print('Excluded source labels:', int(mapping.canonical_genre.isna().sum()))
    genre_distribution = pd.read_csv(OUT / 'train_genre_distribution.csv')
    table(genre_distribution)
    ''')
    md('''
    <p>The training table counts positive interaction–genre memberships, not catalog
    titles. Multi-genre games count once toward each distinct genre, so membership
    rates can sum above 100%. Broad categories can dominate this target.</p>
    <h2>3. Model formulation and implementation</h2>
    <p>Let c(u,g) count the user's positive reviews whose games have genre g, n(u)
    count their positive reviews, and p(g) be the corresponding global membership rate.
    Score g by <code>(c(u,g) + alpha * p(g)) / (n(u) + alpha)</code> and select the
    largest score. An unseen user gets p(g). These are membership-rate estimates,
    not a probability distribution over mutually exclusive classes.</p>
    <p>Alpha=0 is personal mode. The global-only baseline always picks the most frequent
    training genre. Ties use global frequency, then alphabetical order. Validation
    maximizes user-averaged Hit@1 over the fixed grid; ties prefer the simpler candidate.
    Count aggregation costs O(number of interaction–genre memberships), with O(users ×
    genres) profile storage and scoring. This is small compared with learning one
    embedding per game and user, though it cannot discover preferences absent from
    the selected genre representation.</p>
    ''')
    code('''
    import inspect
    print(inspect.getsource(GenreRecommender.fit))
    print(inspect.getsource(GenreRecommender.predict))
    ''')
    md('''
    <h2>4. Holdout protocol and results</h2>
    <p>Sort by <code>max(timestamp_created, timestamp_updated)</code> and use global
    70/15/15 chronological partitions, keeping timestamp ties together. Last-update time
    is used because the stored review vote may have changed after creation. This tests
    later available positive review states, not strictly first-time purchases.</p>
    <p>Choose the model using training and validation only; refit its selected
    configuration on train + validation and evaluate the test period with frozen
    histories. A warm user has positive history in the fitted partition. Cold users are
    included rather than silently removed. Training genre metadata is a present-day
    snapshot, so historical taxonomy changes remain an unresolved limitation.</p>
    <p>Primary metric: average the binary genre hit across reviews within each user,
    then across users. This gives every evaluated user equal weight. Also show
    review-weighted hits and warm/cold cohorts. This is multi-label Hit@1, not
    single-label classification accuracy, recall of all interests, or causal evidence
    that a recommendation improves satisfaction.</p>
    ''')
    code('''
    table(pd.DataFrame(list(summary['split_counts'].items()), columns=['partition', 'positive_reviews']))
    for key in ['event_time_min_utc', 'validation_start_utc', 'test_start_utc', 'event_time_max_utc']:
        print(key, summary[key])
    validation = pd.read_csv(OUT / 'validation_metrics.csv')
    table(validation[['model', 'alpha', 'all_user_hit_at_1', 'warm_user_hit_at_1', 'cold_user_hit_at_1', 'prediction_genres']])
    ''')
    md('''
    <p>Personal mode and alpha=1 tied on validation; the stated tie rule selects personal
    mode. Larger prior strengths reduced validation performance. Thus the data supports
    the simpler member of the initial model family. The global fallback is still used
    for users with no history.</p>
    ''')
    code('''
    test = pd.read_csv(OUT / 'test_metrics.csv')
    table(test[['model', 'all_user_hit_at_1', 'warm_user_hit_at_1', 'cold_user_hit_at_1', 'review_hit_at_1', 'prediction_genres']])
    row = test.iloc[0]
    print(f"Test users: {int(row.all_users):,}; warm: {int(row.warm_users):,}; cold: {int(row.cold_users):,}")
    print('Global mode:', summary['evaluation_global_mode'])
    lift = summary['test_lift']
    print(f"Paired user-level lift: {100 * lift['mean_lift']:.3f} percentage points")
    print(f"95% bootstrap interval: [{100 * lift['ci_95_low']:.3f}, {100 * lift['ci_95_high']:.3f}] percentage points")
    print(f"Bootstrap: {lift['samples']} resamples, seed {lift['seed']}")
    ''')
    md('''
    <p>Most test users are cold, so the overall improvement is modest while warm-user
    improvement is larger. The interval resamples users with their paired model
    outcomes, preserving within-user dependence; it does not account for dependence
    between users reviewing the same game or uncertainty under a different time period.</p>
    <h3>Artifact verification and inference</h3>
    <p>Check chronological isolation, unique pairs, full recommendation coverage, and
    predictions reproduced from the saved evaluation model. The separate deployment
    model is refit on all data only after scoring and is never used for test metrics.</p>
    ''')
    code('''
    split = pd.read_csv(OUT / 'split_interactions.csv.gz', dtype={'author_steamid': 'string', 'recommendationid': 'string'})
    assert not split.duplicated(['author_steamid', 'appid']).any()
    assert not split.recommendationid.duplicated().any()
    train = split[split.split.eq('train')]
    val = split[split.split.eq('validation')]
    holdout = split[split.split.eq('test')]
    assert train.event_time.max() < val.event_time.min() <= val.event_time.max() < holdout.event_time.min()
    saved = pd.read_csv(OUT / 'test_predictions.csv.gz', dtype={'author_steamid': 'string', 'recommendationid': 'string'})
    model = GenreRecommender.load(OUT / 'evaluation_model.npz')
    reproduced = model.predict(saved.author_steamid.drop_duplicates())
    check = saved.merge(reproduced, on='author_steamid', suffixes=('_saved', '_loaded'))
    assert check.recommended_genre_saved.eq(check.recommended_genre_loaded).all()
    assert np.isclose(saved.groupby('author_steamid').hit.mean().mean(), test.iloc[1].all_user_hit_at_1)
    recs = pd.read_csv(OUT / 'user_recommendations.csv.gz', dtype={'author_steamid': 'string'})
    assert len(recs) == summary['audit']['raw_users']
    assert recs.author_steamid.is_unique and recs.recommended_genre.notna().all()
    print('All artifact checks passed.')
    print(f"Final recommendations: {len(recs):,}; no positive history: {int(recs.history_count.eq(0).sum()):,}")
    table(recs.recommended_genre.value_counts().rename_axis('genre').rename('recommended_users').reset_index())
    deployment = GenreRecommender.load(OUT / 'recommendation_model.npz')
    table(deployment.predict(['new_user']))
    ''')
    md('''
    <h2>5. Findings, limitations and discussion for the meetup</h2>
    <ul>
    <li>Use personal genre mode with global Indie fallback as the current selected model.
    It is best among the tested configurations; factorization and learned classifiers
    have not been compared.</li>
    <li>Discuss whether Indie belongs in the prediction vocabulary and whether to predict
    a single favorite genre or retain the current multi-label future-review proxy.</li>
    <li>Cold start is the main coverage limitation. Optional preference onboarding or
    additional historical user data could help; current lifetime ownership/playtime
    aggregates cannot safely be treated as past features.</li>
    <li>Positive-only reviews are an explicit feedback proxy. Missing or negative reviews
    do not establish dislike of every genre. The sample may be biased by which games
    and users were collected.</li>
    <li>Chronological evaluation has no future interactions in fitted profiles, but
    snapshot genres and unavailable edit histories prevent a fully historical replay.</li>
    <li>For later experiments, consider historical-text features, recency weights,
    collaborative filtering, and an Indie-excluded target. Reserve a fresh final
    holdout before further tuning against these observed test results.</li>
    </ul>
    <h3>Related-work starting point and assignment status</h3>
    <p><a href="https://developers.google.com/machine-learning/recommendation/content-based/basics">Google's content-based filtering overview</a>
    motivates combining item attributes with past preferences; its
    <a href="https://developers.google.com/machine-learning/recommendation/content-based/summary">advantages and limitations discussion</a>
    explains the dependence on item representation and existing interests. Our count
    model is a deliberately simple genre-level implementation. These sources do not
    provide a directly comparable result on this custom dataset/split.</p>
    <p>The assignment PDF asks for a readable notebook exported as HTML, predictive
    task, EDA, modeling, evaluation, related work, and a roughly 20-minute presentation.
    This report supports the first four sections for the meetup. The final group work
    still needs verified dataset provenance, a substantive research comparison,
    the recorded presentation and video URL, and subsequent peer grading. No video
    or submission has been fabricated.</p>
    ''')

    namespace = {'__name__': '__notebook__'}
    html_parts = []
    count = 0
    import os
    previous_cwd = Path.cwd()
    os.chdir(HERE)
    try:
        for i, cell in enumerate(cells):
            cell['id'] = f'cell-{i:03d}'
            if cell['cell_type'] == 'markdown':
                html_parts.append(cell['source'])
                continue
            count += 1
            cell['execution_count'] = count
            outputs = cell['outputs']
            stream = io.StringIO()

            def display_html(value):
                # Flush prior printed text so mixed displays retain execution order.
                if stream.tell():
                    outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue()})
                    stream.seek(0)
                    stream.truncate(0)
                outputs.append({'output_type': 'display_data', 'metadata': {},
                                'data': {'text/html': value, 'text/plain': 'Rendered data table'}})

            namespace['display_html'] = display_html
            with redirect_stdout(stream):
                exec(compile(cell['source'], f'analysis.ipynb cell {count}', 'exec'), namespace)
            if stream.getvalue():
                outputs.append({'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue()})
            html_parts.append(f'<details><summary>Code cell {count}</summary><pre>{html.escape(cell["source"])}</pre></details>')
            for output in outputs:
                html_parts.append('<pre>' + html.escape(output['text']) + '</pre>' if output['output_type'] == 'stream'
                                  else '<div class="table-wrap">' + output['data']['text/html'] + '</div>')
    finally:
        os.chdir(previous_cwd)
    # A fallback display helper makes the exported notebook runnable in normal Jupyter.
    first_code = next(c for c in cells if c['cell_type'] == 'code')
    first_code['source'] = ('try:\n    display_html\nexcept NameError:\n'
                            '    from IPython.display import HTML, display\n'
                            '    def display_html(value):\n        display(HTML(value))\n\n') + first_code['source']
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python', 'version': '3'}}, 'nbformat': 4, 'nbformat_minor': 5}
    (HERE / 'analysis.ipynb').write_text(json.dumps(notebook, indent=2) + '\n')
    style = '''body{max-width:1100px;margin:48px auto;padding:0 24px;color:#20232a;font:16px/1.6 system-ui,sans-serif}
    h1,h2,h3{line-height:1.2}h2{margin-top:48px}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}
    th,td{text-align:left;padding:9px;border-bottom:1px solid #ddd}th{background:#f2f4f6}pre{padding:16px;background:#f4f5f6;overflow:auto;font-size:13px}
    .table-wrap{overflow:auto}details{margin-top:20px}summary{cursor:pointer;color:#245684}a{color:#245684}code{font-size:.9em}'''
    document = '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    document += '<title>George — Steam genre recommendation</title><style>' + style + '</style></head><body>'
    document += '\n'.join(html_parts) + '</body></html>\n'
    (HERE / 'workbook.html').write_text(document)
    print(f'Executed {count} code cells; wrote analysis.ipynb and workbook.html')


if __name__ == '__main__':
    build()
