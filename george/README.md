# George — Steam genre recommendation prototype

Independent preparation for the next meetup: data audit/cleanup, global-mode baseline,
model selection, chronological holdout evaluation, and one recommendation per user.
Read `workbook.html` for the executed analysis or open `analysis.ipynb` in Jupyter.
All code and generated files stay in this folder; source CSVs are read-only.

## Model decision, made before implementation

The raw export has 1,048,148 reviews from 701,884 users; 75% of users have no more
than one review. Only 81,427 users have two or more positive reviews before cleaning.
The appropriate first model is a **content-based genre profile with shrinkage toward
global popularity**. It uses the genres of positively reviewed games as user features.
It is cheap, explainable, and can use one past interaction; unknown users get a global
fallback. This is a reasoned starting choice, not proof of universal optimality.

| Approach | Decision |
| --- | --- |
| Global mode | Required baseline; also a candidate in model selection. |
| Personal genre mode | Simple personalized comparison; fragile with little history. |
| Smoothed genre profile | Preferred initial family; tune prior strength on validation. |
| Matrix factorization / user-neighbor CF | Future comparison; sparse user histories and cold users limit useful learned user factors or neighbors. Not empirically ruled out. |
| Review text classifier | Does not directly solve this task if it reads the target review, which is unavailable when recommending. Historical text is a possible later feature. |
| Neural recommender | No established benefit here to justify initial complexity. Not benchmarked. |

For user u and genre g, `score = (count(u,g) + alpha * global_rate(g)) / (n(u) + alpha)`.
Counts use distinct positive user–game interactions. Each game contributes one vote to
each distinct canonical genre. These are smoothed multi-label membership rates, not a
probability distribution summing to one, nor calibrated probabilities of enjoyment.
Predict the largest score. Break ties by global frequency then alphabetically.
The global mode is the most frequent genre across positive **training interactions**,
not the most common genre in the game catalog and not a mode computed from the holdout.

## Run

Python 3.10+ with NumPy and pandas (already available in the development environment):

```sh
python3 -m pip install -r george/requirements.txt
python3 george/pipeline.py
python3 -m unittest discover -s george -p 'test_*.py' -v
python3 george/build_report.py
```

Paths resolve relative to the script, so running from other directories also works.
Override inputs/output with `--data-dir PATH --output-dir PATH` on `pipeline.py`.
The report builder reads the default `george/outputs` directory. No GPU, network,
scikit-learn, Jupyter installation, or original ZIP extraction is needed to run the analysis.
The report builder executes notebook cells and exports their actual outputs using a
small standard-library exporter; the resulting notebook is also Jupyter-compatible.

## Evaluation contract

- Task: recommend one genre that appears among the genres of a user's future positive
  game review. This is a proxy for preference, not a directly observed favorite genre.
- Global chronological train/validation/test partitions target 70%/15%/15% of cleaned
  positive interactions; tied timestamps stay together. No user–game pair crosses splits.
- Event time is `max(timestamp_created, timestamp_updated)` because the export contains
  final review states, not edit histories. This predicts later available positive review
  states, not necessarily first purchases or first-time reviews.
- Compare global mode, personal mode, and alpha in `{1, 5, 20, 100}` on validation.
  Select highest user-averaged Hit@1; exact ties favor the earlier, simpler candidate.
- Refit the selected configuration and baseline on train + validation, then evaluate
  once on test with frozen profiles. Warm and cold refer to positive history in that fit.
- Hit@1 is 1 if the single recommendation belongs to the target game's genre set.
  Average within each user, then across users, so prolific reviewers do not dominate.
  Also report review-weighted Hit@1, warm/cold scores, and predicted genre coverage.
- A seeded paired bootstrap over users gives a descriptive 95% interval for test lift.
  It does not account for game-level dependence or temporal distribution shift.
- Only after evaluation, refit on all cleaned positive reviews and recommend to every
  identifiable user in the raw reviews, including negative-only users via fallback.
  Do not use that final refit model to reproduce holdout scores.

## Outputs

Small audit/results files are under `outputs/`: `summary.json`, validation/test metrics,
genre mapping, training genre frequencies, and user-history distribution. Larger,
reproducible artifacts are locally present but Git-ignored:

- `split_interactions.csv.gz`: cleaned positive records and explicit split membership.
- `game_genres.csv.gz`: normalized, unique game–genre pairs.
- `test_predictions.csv.gz`: auditable per-review test predictions and hits.
- `user_recommendations.csv.gz`: one row per user, selected genre, history count, score,
  and the full-data global-baseline genre. Keep Steam IDs as strings when reading.
- `evaluation_model.npz`: selected model fit on train + validation.
- `recommendation_model.npz`: final full-data model for inference.

```python
import sys
sys.path.insert(0, 'george')
from model import GenreRecommender
model = GenreRecommender.load('george/outputs/recommendation_model.npz')
print(model.predict(['76561198085405844', 'new_user']))
```

## Limitations and meetup decisions

There is no separate user table or complete ownership/play log: users are review authors.
Current lifetime playtime/ownership counts, review text, and target game features are
not used as predictive inputs. Game genres are used as historical profile attributes
and evaluation labels. Metadata is a current snapshot, so historical taxonomy changes
cannot be reconstructed. Keep this limitation even with a chronological split.

The mapping normalizes translated genres; status, business-model, software/video and
content descriptors are excluded. `Indie` is deliberately retained as a Steam catalog
genre, although it is not a gameplay mechanic. Review the mapping and consider an
Indie-excluded sensitivity analysis as a separate future experiment. Multi-genre games
make Hit@1 easier than single-label classification. Negative reviews are not proof of
disliking every genre on a game, so they are excluded from preference counts.

The local export contains a manifest but does not document its upstream sampling or
collection method. Do not call it a representative sample of all Steam users. Its
directory says 2025 while manifest file metadata is later; use measured event dates.
Confirm the original dataset URL, licensing, sampling and collection protocol with the
team. Further tuning after seeing these test results requires a fresh final holdout.

## Assignment alignment and sources

Read against `158 _ 258 2026 Assignment 2.pdf`: predictive task/validity, EDA/processing,
model formulation, and baseline evaluation are covered for the meetup. The final
assignment also requires substantive related work, a roughly 20-minute recorded
presentation and `video_url.txt`, and later peer grading. Those are not completed by
this prototype. The exported `workbook.html` is an initial notebook report, not a claim
that the full group submission is finished.

Method background: [Google's content-based filtering explanation](https://developers.google.com/machine-learning/recommendation/content-based/basics)
and [advantages/limitations](https://developers.google.com/machine-learning/recommendation/content-based/summary).
These motivate using item attributes and historical preferences, but do not establish
our model as best. No published benchmark score is directly comparable to this custom
genre vocabulary, snapshot, and temporal protocol.
