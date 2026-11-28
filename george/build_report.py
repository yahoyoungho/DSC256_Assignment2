"""Execute analysis cells in-process, validate notebook, export with nbconvert."""
import base64
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import textwrap
import nbformat
from nbconvert import HTMLExporter

HERE = Path(__file__).resolve().parent


def build():
    notebook = nbformat.v4.new_notebook()
    notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    cells = notebook.cells

    def md(s):
        cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(s).strip()))

    def code(s):
        cells.append(nbformat.v4.new_code_cell(textwrap.dedent(s).strip()))

    md('''
    # Steam genre recommendation: nine-genre experiments

    George's independent implementation following the November 25 meeting.
    **Question:** Do recency, latent preferences or transitions improve recommendations
    beyond a user's long-term genre mode?

    Given positive review history before a cutoff, recommend **one** genre. A hit means
    it belongs to a later positively reviewed game's genre set. This is a review-derived
    preference proxy, not a directly observed favorite genre or causal satisfaction measure.

    **Exploratory evaluation:** Overlapping test events were inspected in the original
    ten-genre prototype. Changing the genre vocabulary does not make them an untouched
    test. This run selects settings, checkpoints and ensemble weights on validation only.
    ''')
    code('''
    import json, sys, os
    from pathlib import Path
    import numpy as np
    import pandas as pd
    ROOT = Path.cwd() if (Path.cwd() / 'experiments.py').exists() else Path.cwd() / 'george'
    ROOT = ROOT.resolve(); sys.path.insert(0, str(ROOT))
    OUT = ROOT / 'outputs_v2'
    summary = json.loads((OUT / 'summary.json').read_text())
    try:
        display_html
    except NameError:
        from IPython.display import HTML, Image, display
        def display_html(value): display(HTML(value))
        def display_png(path): display(Image(filename=str(path)))
    def table(frame):
        display_html(frame.to_html(index=False, border=0, float_format=lambda x: f'{x:.6f}'))
    print('Versions:', summary['versions'])
    print('Selected family:', summary['selected_model'])
    print(summary['evaluation_status'])
    ''')
    md('''
    ## 1. Model decision and data cleanup

    At least 75% of the 701,884 raw authors have only one review. Start with history
    profiles and a shared logistic model, then test latent/sequential models empirically.
    No architecture is assumed best in advance.

    Inputs are the local reviews, applications, application_genres and genres CSVs.
    The manifest does not verify upstream sampling, source URL or license; these remain
    team questions. Authors are the user data, not complete user game libraries.

    Preserve IDs as strings. Validate dates against a broad year-2000 lower bound,
    creation/update order, and the row's extraction timestamp. Deduplicate review IDs
    and user–game pairs, keeping the latest state. Keep games, usable genres, and positive
    votes. Negative reviews do not imply dislike of every genre and are excluded from
    preference counts. Removal counts below are sequential.

    Normalize translated genre labels into **Action, Adventure, Casual, Massively
    Multiplayer, RPG, Racing, Simulation, Sports and Strategy**. Remove Indie,
    release/business-model labels, content descriptors and software/video labels.
    **Indie + Action stays as Action.** Drop a game only if no approved genre remains.
    Unknown source IDs fail validation; approved rare genres are retained.

    All eligible target labels count as correct; no arbitrary primary-genre priority.
    The earlier 58.13% result allowed Indie and is not directly comparable.
    ''')
    code('''
    audit = summary['audit']
    table(pd.DataFrame([{'measure': k, 'value': v} for k, v in audit.items() if not isinstance(v, (list, dict))]))
    mapping = pd.read_csv(OUT / 'genre_mapping.csv')
    table(mapping.groupby('canonical_genre').size().rename('source_label_count').reset_index())
    print('Excluded source names:', int(mapping.canonical_genre.isna().sum()))
    print('Full mapping: genre_mapping.csv; full-catalog label audit: genre_omission_audit.csv')
    ''')
    code('''
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG = OUT / 'figures'; FIG.mkdir(exist_ok=True)
    dist = pd.read_csv(OUT / 'train_genre_distribution.csv').sort_values('membership_rate')
    years = pd.read_csv(OUT / 'reviews_by_year.csv')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout='constrained')
    axes[0].barh(dist.genre, 100 * dist.membership_rate, color='#246b82')
    axes[0].set(xlabel='Positive training reviews with genre (%)', title='Training genre memberships')
    axes[1].bar(years.year, years.reviews, color='#477a58')
    axes[1].set(xlabel='Available-state year', ylabel='Positive reviews', title='Strong concentration in 2025')
    fig.savefig(FIG / 'data_overview.png', dpi=160); plt.close(fig)
    display_png(FIG / 'data_overview.png')
    ''')
    md('''
    Multi-genre reviews contribute one count to each genre, so rates need not sum to one.
    The fractional-profile ablation distributes one total vote per game while retaining
    the same multi-hot targets. Inverse-frequency class weighting is not the default:
    it changes the objective and may trade overall Hit@1 for rare-genre coverage.

    ## 2. Temporal protocol and historical features

    Global chronological partitions approximate 70/15/15, with timestamp ties together.
    Event time is max(creation, update), since only the final vote state is available.
    This predicts later available review states, not play order. Current genre metadata
    and missing edit histories prevent a fully historical replay.

    Every main model uses frozen histories. Fit on training; tune on validation; refit
    on train + validation using selected epoch counts; score test. Simultaneous reviews
    form a genre basket, and transitions always originate at a strictly earlier timestamp.

    Logistic training creates three disjoint target windows inside fitting data, beginning
    at its 40%, 60% and 80% timestamps. Features freeze at each window's start. A target
    never enters its own profile. Inputs are long-term genre rates, 90-day-half-life
    rates, last genre set, recent global rates, log history length and log review age.
    Fixed feature scaling does not require fitting transforms on test data.
    ''')
    code('''
    table(pd.DataFrame(list(summary['split_counts'].items()), columns=['partition', 'reviews']))
    for key in ['event_time_min_utc', 'validation_start_utc', 'test_start_utc', 'event_time_max_utc']:
        print(key, summary[key])
    print('Test global mode:', summary['evaluation_global_mode'])
    ''')
    md('''
    ## 3. Models, implementation and optimization

    | Family | Implementation |
    |---|---|
    | Uniform random | Stable seeded per-user random ranks. |
    | Global mode | Most frequent training interaction–genre membership. |
    | Recent popularity | Mode in 30/90/180 days before the cutoff. |
    | User mode | Most frequent genre in positive user history. |
    | Repeat last | Last basket's genre indicators; training-frequency tie break. |
    | Recency profile | Weights 2^(-age_days / half_life), half-life 30/90/180 days. |
    | Fractional profile | One total vote per reviewed game. |
    | Markov-1 | Source-basket transitions smoothed toward global rates; alpha 0.1/1/10. |
    | Logistic | PyTorch linear layer with nine outputs, binary cross-entropy with logits. |
    | MF-BPR | User and genre factors plus genre bias; no sequence term. |
    | FMC-BPR | Factorized previous-to-next genre transitions, no user factor. |
    | FPMC-BPR | Preference and transition factors together. |
    | Ensemble | Validation-selected weighted ranks from two leading families. |

    BPR is an objective, not an architecture or literal SVD. A nine-column genre matrix
    has rank at most nine; dimensions 2/4/8 are tested. The loss averages every pair of
    present versus absent target genres. An absent genre is not on this event; it is
    not assumed to be a lifelong dislike. Previously liked genres remain eligible.

    MF/FPMC estimate user factors only with at least two fitting reviews; other users
    receive profile fallback. Nobody is removed from evaluation. FMC shares transition
    factors. All learned models use the global prior for users with no positive history.
    Factor results therefore include explicit hybrid fallback behavior.

    Training uses Adam, user-balanced examples, explicit L2, batches up to 8192, a
    12-epoch cap and validation patience 3. Staged searches test dimensions, learning
    rates and L2 rather than a full Cartesian product. Shortlisted settings run with
    seeds 2026/2027/2028; their ranks are averaged. Each seed's epoch uses validation.
    Several fits hit the cap; this is a bounded first search, not proof of convergence.

    Rank averaging handles incomparable probability/logit scales but discards score
    magnitude. Markov-2, game-level Jaccard and deep networks remain future extensions.
    ''')
    code('''
    validation = pd.read_csv(OUT / 'validation_metrics.csv')
    test = pd.read_csv(OUT / 'test_metrics.csv')
    table(validation[['model', 'user_hit_at_1', 'user_ndcg_at_3', 'genre_coverage', 'config']])
    print('Selected specification:'); print(json.dumps(summary['selected_spec'], indent=2))
    seeds = pd.read_csv(OUT / 'seed_metrics.csv')
    table(seeds.groupby('model').agg(validation_mean=('user_hit_at_1', 'mean'), validation_std=('user_hit_at_1', 'std'),
                                    min_epoch=('best_epoch', 'min'), max_epoch=('best_epoch', 'max')).reset_index())
    ''')
    md('''
    ## 4. Evaluation and results

    **Primary:** binary Hit@1 averaged within users, then across users. Secondary:
    review-weighted Hit@1, user-averaged NDCG@3, genre coverage, per-genre support and
    precision/recall, and history cohorts. NDCG@3 evaluates an optional ranking; the
    product output remains one genre. Uniform random expected Hit@1 is target-set-size/9.

    Paired bootstrap intervals resample users, preserving within-user dependence.
    They do not account for shared games, model-selection uncertainty or time shift.
    ''')
    code('''
    table(test[['model', 'user_hit_at_1', 'review_hit_at_1', 'user_ndcg_at_3', 'genre_coverage', 'refit_seconds', 'scoring_seconds']])
    comparisons = json.loads((OUT / 'paired_comparisons.json').read_text())
    table(pd.DataFrame([{'baseline': k, **v} for k, v in comparisons.items()]))
    print('Timings include cache effects and exclude shared data/feature preparation.')
    print('Pipeline wall seconds:', round(summary['wall_seconds'], 2))
    print('Artifact sizes:', json.loads((OUT / 'artifact_sizes.json').read_text()))
    fig, ax = plt.subplots(figsize=(10, 5.8), layout='constrained')
    ordered = test.sort_values('user_hit_at_1')
    ax.barh(ordered.model, ordered.user_hit_at_1 * 100,
            color=['#246b82' if n == summary['selected_model'] else '#8c9aa3' for n in ordered.model])
    ax.set(xlabel='User-averaged Hit@1 (%)', title='Nine-genre exploratory temporal holdout', xlim=(0, 60))
    for i, v in enumerate(ordered.user_hit_at_1): ax.text(v * 100 + .4, i, f'{v:.2%}', va='center', fontsize=9)
    fig.savefig(FIG / 'model_comparison.png', dpi=160); plt.close(fig)
    display_png(FIG / 'model_comparison.png')
    ''')
    md('''
    The selected ensemble improves over global popularity, but its difference from
    user mode is tiny. Inspect that paired interval before claiming an ensemble win.
    Plain user mode is a compelling operational choice; exported predictions follow
    the validation-selected model. Added model complexity does not guarantee a gain.
    ''')
    code('''
    cohorts = pd.read_csv(OUT / 'test_cohorts.csv')
    selected = cohorts[cohorts.model.eq(summary['selected_model'])]
    table(selected)
    per_genre = pd.read_csv(OUT / 'test_per_genre.csv')
    table(per_genre[per_genre.model.eq(summary['selected_model'])].drop(columns=['genre_index', 'model']))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout='constrained')
    groups = ['cold', '1-2', '3-4', '5+']
    for model, color in [('global_mode', '#8c9aa3'), ('user_mode', '#477a58'), (summary['selected_model'], '#246b82')]:
        part = cohorts[cohorts.model.eq(model)].set_index('cohort').loc[groups]
        axes[0].plot(groups, part.user_hit_at_1 * 100, marker='o', label=model, color=color)
    axes[0].set(xlabel='Reviews available at cutoff', ylabel='User Hit@1 (%)', title='History matters'); axes[0].legend()
    axes[1].bar(groups, selected.set_index('cohort').loc[groups].users, color='#246b82')
    axes[1].set(xlabel='History cohort', ylabel='Test users', title='Most users have no positive history')
    fig.savefig(FIG / 'history_cohorts.png', dpi=160); plt.close(fig)
    display_png(FIG / 'history_cohorts.png')
    ''')
    md('''
    ## 5. Robustness and alternative protocol

    Minimum history is a cohort, not an exclusion of the full population. Earlier
    50–60% and 60–70% time windows are evaluated with fixed selected settings: these
    are retrospective diagnostics, not independent nested cross-validation. Unedited
    review sensitivity uses the same calendar boundaries and fixed baseline settings.

    Sequential replay separately reveals a review only after prediction and updates
    per-user history. Simultaneous reviews are revealed together. Learned parameters,
    global priors and transition matrices stay frozen; cohorts refer to history at
    the original cutoff. No replay result selects the winner.
    ''')
    code('''
    table(pd.read_csv(OUT / 'development_folds.csv')[['model', 'cutoff_utc', 'users', 'user_hit_at_1']])
    table(pd.read_csv(OUT / 'unedited_sensitivity.csv')[['model', 'users', 'reviews', 'user_hit_at_1']])
    replay = pd.read_csv(OUT / 'replay_diagnostics.csv')
    table(replay.pivot(index='model', columns='protocol', values='user_hit_at_1').reset_index())
    ''')
    md('''
    ## 6. Reproduction and inference

    Run `experiments.py`, `diagnostics.py`, `verify.py`, then `build_report.py` with the
    local virtual environment. `experiment_config.json`, `requirements-lock.txt`, code/data
    hashes, checkpoint metadata and seed results record the run. The evaluation model
    fits train + validation; the deployment model refits all positive data only after
    evaluation. Every original identifiable user receives a genre.

    Numeric NPZ snapshots need no pickle. Unknown IDs get global fallback. Incorporating
    new history requires rebuilding the batch snapshot; this is not an online service.
    Model scores are ranking values, not calibrated enjoyment probabilities.
    ''')
    code('''
    from models import SnapshotRecommender
    model = SnapshotRecommender.load(OUT / 'recommendation_model.npz')
    table(model.predict(['new_user']))
    print(f"Users served: {summary['recommendation_users']:,}; no positive history: {summary['recommendation_cold_users']:,}")
    print(json.dumps(json.loads((OUT / 'verification.json').read_text()), indent=2))
    ''')
    md('''
    ## 7. Tradeoffs, related work and assignment status

    - Gameplay genres are more interpretable than Indie, but exclude games lacking
      any approved label and still simplify diverse interests into broad categories.
    - Review sampling and positive-only feedback can bias results; there is no full
      ownership/play history. Lifetime playtime/ownership totals and target review
      text are not predictive inputs because of availability/leakage concerns.
    - Update time is conservative for final-vote availability, but is not play order.
      Snapshot genres and missing edit histories limit historical validity.
    - Shared logistic coefficients support sparse histories. User factors require
      fallback behavior, and the bounded search does not establish universal rankings.
    - An ensemble adds storage/inference cost for little improvement over user mode.
      Rank averaging sacrifices score magnitude to avoid mixing incompatible scales.
    - A genuinely untouched final test needs new data or an uninspected period.

    [BPR (Rendle et al., 2009)](https://proceedings.mlr.press/r7/rendle09a.html)
    motivates pairwise ranking. Our nine-genre adaptation enumerates all event-specific
    positive/negative pairs rather than sampling a large item space.
    [FPMC (Rendle et al., 2010)](https://archives.iw3c2.org/www2010/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf)
    combines preference and transition factors. Our genre baskets, sparse-user fallback
    and protocol differ from the original benchmark; published scores are not comparable.
    [PyTorch BCEWithLogitsLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
    supplies the logistic model's multi-label training loss.

    The assignment PDF requests task definition, EDA, modeling, evaluation, related
    work, an HTML notebook, a roughly 20-minute recorded presentation and peer grading.
    This report supports the technical sections. The team still needs verified data
    provenance, integrated group findings, the recording and real video URL, and peer
    grading. No video or final submission has been fabricated.
    ''')
    namespace = {'__name__': '__notebook__'}
    previous_cwd = Path.cwd(); os.chdir(HERE)
    try:
        count = 0
        for cell in cells:
            if cell.cell_type != 'code': continue
            count += 1; cell.execution_count = count
            stream = io.StringIO()
            def flush():
                if stream.tell():
                    cell.outputs.append(nbformat.v4.new_output('stream', name='stdout', text=stream.getvalue()))
                    stream.seek(0); stream.truncate(0)
            def display_html(value):
                flush(); cell.outputs.append(nbformat.v4.new_output('display_data', data={'text/html': value, 'text/plain': 'Data table'}))
            def display_png(path):
                flush(); cell.outputs.append(nbformat.v4.new_output('display_data', data={'image/png': base64.b64encode(Path(path).read_bytes()).decode(), 'text/plain': 'Figure'}))
            namespace.update(display_html=display_html, display_png=display_png)
            with redirect_stdout(stream):
                exec(compile(cell.source, f'analysis.ipynb cell {count}', 'exec'), namespace)
            flush()
    finally:
        os.chdir(previous_cwd)
    nbformat.validate(notebook)
    nbformat.write(notebook, HERE / 'analysis.ipynb')
    body, _ = HTMLExporter(template_name='lab').from_notebook_node(notebook)
    (HERE / 'workbook.html').write_text(body)
    assert body.startswith('<!DOCTYPE html>')
    print(f'Executed {count} cells; validated notebook and exported workbook.html.')


if __name__ == '__main__':
    build()
