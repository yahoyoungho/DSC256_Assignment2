# George — Steam genre recommendation experiments

Current task: recommend **one of nine gameplay genres** from a user's available
positive-review history. Multiple genres on a future positively reviewed game are
valid targets. There is no arbitrary primary-genre priority list.

Start with **[workbook.html](workbook.html)** for the executed analysis, figures,
comparison tables and limitations, or open **[analysis.ipynb](analysis.ipynb)**.
Current results live in `outputs_v2/`. The old `outputs/` directory preserves the
historical ten-genre prototype and is **not** used by the current experiment.

## Findings from this run

| Model | User-averaged test Hit@1 |
|---|---:|
| Global mode (Action) | 47.9293% |
| Recent popularity | 48.4412% |
| User mode | 49.7292% |
| Recency profile | 49.6537% |
| Logistic | 49.5064% |
| MF-BPR | 49.6942% |
| FMC-BPR | 49.3803% |
| FPMC-BPR | 49.6429% |
| Validation-selected ensemble | 49.7303% |

Validation selected a **75% user-mode / 25% MF rank ensemble**. The MF component
combines seeds 2026/2027/2028, eight factors, learning rate 0.003, L2 0.0001 and
validation-selected epochs 11/12/12. Other baselines and full metrics are in the report.

The selected ensemble improves over global mode by **1.801 percentage points**
(paired user-bootstrap 95% interval: **1.665–1.941 points**). Its gain over user mode
is only **0.0011 points**, with an interval spanning zero (**−0.0026 to 0.0052 points**).
There is no convincing evidence that its extra complexity is useful relative to
plain user mode. Saved recommendations follow the validation winner; user mode is
the practical simpler alternative. Do not select a different model by inspecting test.

**These are exploratory temporal holdout results.** Overlapping test events were
inspected in v1. Changing labels does not create an untouched confirmatory test.
The prior 58.13% result included Indie and is not directly comparable.

## Reproduce

Python 3.10+; this run used Python 3.12.10 on CPU. A local virtual environment has
already been installed in `george/.venv`.

```sh
# From the repository root, on a fresh setup:
python3 -m venv george/.venv
george/.venv/bin/python -m pip install -r george/requirements.txt

# Tests → cleanup/train/select/refit → diagnostics → artifact checks → notebook/HTML:
george/.venv/bin/python george/run_all.py
```

`requirements-lock.txt` records exact packages from the executed environment;
`requirements.txt` contains portable supported ranges. No GPU is required. Once
dependencies are installed, training and reporting need no network. Original CSVs
remain read-only and all work is confined to `george`.

Individual steps:

```sh
george/.venv/bin/python george/experiments.py
george/.venv/bin/python george/diagnostics.py
george/.venv/bin/python george/verify.py
george/.venv/bin/python george/build_report.py
george/.venv/bin/python -m unittest discover -s george -p 'test_*.py' -v
```

The experiment CLI accepts `--data-dir`, `--output-dir` and `--config`.
Diagnostics accepts `--output-dir`; report and full verification use the default
`outputs_v2/`. `pipeline.py` also runs the default current experiment for compatibility.
Model checkpoints are cached by code/data/config/partition hash. Repeated runs may
reuse fits; timings are local observations and include cache effects, not benchmarks.

## Data contract and genre omissions

- Approved genres: Action, Adventure, Casual, Massively Multiplayer, RPG, Racing,
  Simulation, Sports, Strategy. All nine remain, including less frequent genres.
- `genres.py` explicitly maps translated source IDs to canonical genres. Unknown
  IDs fail instead of being silently discarded. Text translation is unnecessary
  because review text is not used.
- Remove Indie, Early Access, Free to Play and other non-gameplay labels. **An
  Indie + Action game remains as Action**. A game with no surviving label is excluded
  from training/evaluation, not interpreted as a negative preference.
- 96,731 Indie-tagged games retain gameplay labels; 4,476 Indie-tagged games have
  no approved label. The latter count includes any other excluded labels they have.
- Validate string IDs, votes and timestamps; reject dates before 2000, updates before
  creation, or review updates after the row's extraction timestamp. Deduplicate review
  IDs and user–game pairs, keeping the latest available state. Keep `type == game`,
  mapped labels and positive votes. The audit records sequential removal counts.
- Final modeling data: **631,859 positive reviews, 459,048 users and 70,808 games**.
  Original identifiable users still receive recommendations: **701,884 total**,
  including **242,836** with no retained positive history.
- No separate user table or full library is present. Dataset provenance, sampling
  and licensing are not established by the supplied manifest and require confirmation.

## Evaluation contract

1. Global chronological train/validation/test split: **442,301 / 94,779 / 94,779**
   records. Timestamp ties remain together; user–game pairs never cross splits.
2. Event time is `max(timestamp_created, timestamp_updated)`, reflecting when the
   stored final vote could be available. Validation starts **2025-08-14 07:50:18 UTC**;
   test starts **2025-08-24 14:08:49 UTC**. Metadata is still a current snapshot.
3. Main evaluation freezes all histories at each cutoff. Validation selects settings,
   checkpoints, seed-model combinations and ensemble weights. Test is not used for
   any of these decisions. Refit train + validation with fixed selected epoch counts.
4. Primary metric: average genre Hit@1 within each user, then across users. Also report
   review-weighted Hit@1, user-averaged NDCG@3, per-genre support/precision/recall,
   genre coverage, and cold/1–2/3–4/5+ history cohorts. Cohorts use fitting history only.
5. A target game's entire approved genre set is relevant. Uniform random expected
   Hit@1 is its label count divided by nine, not automatically 1/9.
6. Paired bootstrap resamples users. It does not capture game-level dependence,
   model-selection uncertainty or temporal distribution shift.
7. Only after evaluation, refit the selected model on all retained positives for the
   exported full-population recommendations. Never score holdout with this final model.

## Models and important implementation choices

`experiment_config.json` records the staged search. Thirteen reported families/
ablations include random, global mode, recent popularity, user mode, repeat-last,
recency profile, fractional profile, Markov-1, logistic, MF, FMC, FPMC and ensemble.

**Profiles:** each positive game contributes to every distinct genre. Fractional
weighting is a separate ablation. Recency is measured in days with half-lives 30/90/180,
not an undefined per-review decay. Recent popularity uses 30/90/180-day windows.
All tie breaks use training popularity then alphabetical genre order.

**Logistic:** nine-output PyTorch linear model with `BCEWithLogitsLoss`. Historical
features include normalized long-term and decayed profiles, last genre set, recent
global rates, log history length and log time since the last review. Three internal
historical windows produce training examples with frozen prefix features. Target
reviews never enter their own features. Feature scaling is fixed; no test-fitted scaler.

**Markov-1:** fractional source-basket transitions and global-prior smoothing with
alpha 0.1/1/10. Simultaneous reviews form a basket; no fabricated within-timestamp order.

**MF/FMC/FPMC:** nine-genre adaptations, using dimensions 2/4/8. MF captures taste;
FMC captures transitions; FPMC combines both. BPR is a pairwise objective, not an
architecture or literal SVD. With nine genres, enumerate all present/absent event
genre pairs instead of sampling. A genre absent from this game is not treated as a
lifelong dislike, and previously liked genres remain recommendation candidates.

User factors require at least two fitting reviews. MF/FPMC use a profile fallback
otherwise; no users are removed from evaluation. FMC has shared transitions. All
learned models use global fallback when no positive history exists. These are explicit
hybrid fallbacks, so do not describe the results as pure factor predictions for everyone.

**Optimization:** Adam; explicit L2; per-user-balanced training examples; batch size
up to 8192; 12-epoch cap; patience 3 on validation Hit@1. Search seeds start at 2026,
then the best setting in each learned family is repeated at 2027 and 2028. Each seed
has its own validation-selected epoch; average its genre ranks for the family score.
Seed variation is descriptive, not an independent significance test. Several fits
reach the cap, so the search does not establish convergence or a universal best model.

**Ensemble:** take the two strongest eligible validation families and tune weights
0.25/0.5/0.75. Blend normalized ranks, not incompatible BPR logits and probabilities.
This sacrifices score magnitude and adds inference/storage cost.

## Diagnostics and tradeoffs

- **Normalization ablation:** full versus fractional genre votes under identical targets.
- **History cohorts:** evaluate 3+ and 5+ history through cohort tables instead of
  dropping the large majority of users. At test cutoff, 69,788 of 81,944 users are cold.
- **Earlier time windows:** fixed selected settings evaluated on 50–60% and 60–70%
  chronological windows. Settings came from later validation, so these are retrospective
  diagnostics, not independently tuned nested cross-validation.
- **Review-edit sensitivity:** baseline comparison on unedited reviews, same calendar
  cutoffs. Update order is not play order; this tests one source of distortion.
- **Sequential replay:** all families predict before revealing an event and update
  user history afterward. Same-time reviews are revealed together. Weights, global
  popularity and transitions stay fixed. This separate protocol does not select models;
  cohort status refers to the original cutoff. Selected-ensemble replay Hit@1 is 50.25%.
- **Genre-specific scores:** distinguish “recommended this genre correctly” from
  “hit any valid genre on games carrying this genre.” Rare genres have limited support.
- **Costs:** report runtime, parameter and snapshot sizes. Shared preprocessing and
  cached fits make per-model times unsuitable as strict algorithmic benchmarks.
- **Deferred:** Markov-2, game-level Jaccard, deep networks, inverse-frequency weighting
  and other target definitions. These were optional future extensions, not silently
  substituted for the implemented core comparisons.

The data cannot establish users' true favorite genres. Broad labels, review selection,
positive-only evidence, current metadata and incomplete user histories all constrain
the interpretation. Target review text and current lifetime ownership/playtime totals
are excluded as predictive features. New data or an uninspected period is needed for
a genuinely untouched final test.

## Files and inference

| File | Role |
|---|---|
| `genres.py`, `pipeline.py` | Canonical labels, auditable cleanup, temporal splitting |
| `features.py` | Frozen histories, strict-time baskets and snapshot examples |
| `models.py` | Baselines, PyTorch learners, rank ensembles and saved inference |
| `metrics.py` | Multi-label ranking metrics, cohorts and paired bootstrap |
| `experiments.py`, `experiment_config.json` | Search, validation selection, refits and predictions |
| `diagnostics.py` | Sequential replay and additional paired comparisons |
| `verify.py`, `test_*.py` | Full-artifact verification and 21 unit/regression tests |
| `build_report.py` | Executed notebook, figures and standard nbconvert HTML export |
| `run_all.py` | One-command reproduction |
| `model.py` | Original simple baseline retained for compatibility/tests |

`outputs_v2/` contains the mapping/omission audit, summary, all search and seed results,
training curves, test metrics/cohorts/per-genre results, diagnostics, plots and verification.
Large generated `.gz` and `.npz` artifacts are local and Git-ignored:

- `split_interactions.csv.gz` and `game_genres.csv.gz` — auditable cleaned inputs.
- `test_predictions.csv.gz` — selected-model per-event holdout predictions and hits.
- `user_recommendations.csv.gz` — one genre per identifiable original user.
- `evaluation_model.npz` — score snapshot fitted on train + validation.
- `recommendation_model.npz` — full-data deployment score snapshot.
- `checkpoints/*.npz` — actual learned parameter arrays; matching JSON records configs,
  epochs, curves and training times. Cached files may include previous code versions.

```python
import sys
sys.path.insert(0, 'george')
from models import SnapshotRecommender
model = SnapshotRecommender.load('george/outputs_v2/recommendation_model.npz')
print(model.predict(['76561198085405844', 'new_user']))
```

NPZ loading uses `allow_pickle=False`. Unknown IDs receive global fallback. A snapshot
serves known fitted histories; rerun the pipeline to incorporate new interactions.
Returned scores are ranking values, **not calibrated probabilities of enjoyment**.

## Sources and assignment status

Read against the local `158 _ 258 2026 Assignment 2.pdf`. The report supports task
definition, EDA, modeling, evaluation and related-work discussion. Final group synthesis,
verified data provenance, the recorded presentation and actual video URL, and later
peer grading remain outside this implementation.

- [BPR, Rendle et al. (2009)](https://proceedings.mlr.press/r7/rendle09a.html)
- [FPMC, Rendle et al. (2010)](https://archives.iw3c2.org/www2010/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf)
- [PyTorch multi-label loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

These motivate methods, not a directly comparable score for this custom Steam task.
