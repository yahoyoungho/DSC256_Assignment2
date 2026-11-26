"""Next-genre prototypes on Steam reviews (Jacob).

Run steam_eda.ipynb first: it reads the raw Kaggle CSVs and writes clean/clean_reviews.csv.gz.
Run: python prototype_models.py

Setup per user (needs >= 3 reviews, ordered by date):
  history = all reviews before the held-out one, target = genre of the held-out review.
  Validation stage: target = 2nd to last review, models fit on everything before it.
  Test stage:       target = last review,        models fit on everything before it.
Fitting only on reviews that come before the target keeps the test review unseen.

Metrics: Acc@k / HitRate@k on the target's primary genre, and any-genre hit@k,
which counts a hit if any top-k genre is one of the held-out game's listed genres.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

# cleaned data written by the notebook (looks next to this file first, then in the working directory)
CLEAN_CANDIDATES = [Path(__file__).parent / "clean", Path("clean"), Path("Jacob/clean")]
MIN_EVENTS = 3
KS = (1, 3, 5)
DECAY = 0.7        # recency weight for the decayed user profile
MARKOV_ALPHA = 0.5  # add-alpha smoothing for transition counts


# ---------- data ----------
def load_sequences():
    path = next((d / "clean_reviews.csv.gz" for d in CLEAN_CANDIDATES if (d / "clean_reviews.csv.gz").exists()), None)
    if path is None:
        raise SystemExit("clean_reviews.csv.gz not found. Run every cell of steam_eda.ipynb first (it saves the cleaned data).")
    rev = pd.read_csv(path, usecols=["user_id", "date", "seq_idx", "primary_genre", "genres"], parse_dates=["date"])
    rev["genres"] = rev["genres"].str.split("|")
    rev = rev.sort_values(["user_id", "date", "seq_idx"], kind="stable")
    vocab = sorted({g for gs in rev["genres"] for g in gs})
    gidx = {g: i for i, g in enumerate(vocab)}
    seqs = {}
    for uid, grp in rev.groupby("user_id", sort=False):
        if len(grp) < MIN_EVENTS:
            continue
        primary = np.array([gidx[g] for g in grp["primary_genre"]])
        allg = [frozenset(gidx[g] for g in gs) for gs in grp["genres"]]
        seqs[uid] = (primary, allg)
    return vocab, seqs


# ---------- models: every model returns a probability vector over genres ----------
class GlobalMode:
    def fit(self, hists, n_genres):
        self.p = np.bincount(np.concatenate(hists), minlength=n_genres).astype(float)
        self.p /= self.p.sum()
        return self

    def predict(self, hist):
        return self.p


class UserMode:
    """User's own genre counts; global popularity breaks ties."""
    def fit(self, hists, n_genres):
        self.n, self.prior = n_genres, GlobalMode().fit(hists, n_genres).p
        return self

    def predict(self, hist):
        c = np.bincount(hist, minlength=self.n).astype(float)
        return c / c.sum() + 1e-6 * self.prior


class RepeatLast:
    def fit(self, hists, n_genres):
        self.n, self.prior = n_genres, GlobalMode().fit(hists, n_genres).p
        return self

    def predict(self, hist):
        p = np.zeros(self.n)
        p[hist[-1]] = 1.0
        return p + 1e-6 * self.prior


class DecayUser:
    """User's genre counts, newer reviews weighted more (DECAY ** age)."""
    def fit(self, hists, n_genres):
        self.n, self.prior = n_genres, GlobalMode().fit(hists, n_genres).p
        return self

    def predict(self, hist):
        w = DECAY ** np.arange(len(hist) - 1, -1, -1)
        p = np.bincount(hist, weights=w, minlength=self.n)
        return p / p.sum() + 1e-6 * self.prior


class Markov1:
    """P(next genre | previous genre) from consecutive pairs across all users."""
    def fit(self, hists, n_genres):
        T = np.full((n_genres, n_genres), MARKOV_ALPHA)
        for h in hists:
            np.add.at(T, (h[:-1], h[1:]), 1.0)
        self.T = T / T.sum(axis=1, keepdims=True)
        self.prior = GlobalMode().fit(hists, n_genres).p
        return self

    def predict(self, hist):
        return self.T[hist[-1]]


class Blend:
    """w_user * decayed user profile + w_markov * transition row + (1 - both) * global."""
    def __init__(self, w_user, w_markov):
        self.wu, self.wm = w_user, w_markov

    def fit(self, hists, n_genres):
        self.user = DecayUser().fit(hists, n_genres)
        self.markov = Markov1().fit(hists, n_genres)
        self.glob = self.markov.prior
        return self

    def predict(self, hist):
        wg = max(0.0, 1.0 - self.wu - self.wm)
        return self.wu * self.user.predict(hist) + self.wm * self.markov.predict(hist) + wg * self.glob


# ---------- evaluation ----------
def evaluate(model, seqs, t, n_genres):
    """t = -2 (validation) or -1 (test). Model must already be fit on events before t."""
    hits = {f"Acc@{k}": [] for k in KS}
    any_hits = {f"AnyGenre@{k}": [] for k in KS}
    for primary, allg in seqs.values():
        hist, target, target_genres = primary[:t], primary[t], allg[t]
        order = np.argsort(-model.predict(hist), kind="stable")
        for k in KS:
            top = order[:k]
            hits[f"Acc@{k}"].append(target in top)
            any_hits[f"AnyGenre@{k}"].append(any(g in target_genres for g in top))
    return {m: float(np.mean(v)) for m, v in {**hits, **any_hits}.items()}


def fit_stage(model, seqs, t, n_genres):
    """Fit on reviews strictly before index t for every user."""
    return model.fit([p[:t] for p, _ in seqs.values()], n_genres)


def main():
    vocab, seqs = load_sequences()
    n = len(vocab)
    print(f"{len(seqs):,} users with >= {MIN_EVENTS} reviews, {n} genres")

    # tune blend weights on validation (target = 2nd to last review)
    grid = [(wu, wm) for wu, wm in itertools.product([0, 0.25, 0.5, 0.75, 1.0], repeat=2) if wu + wm <= 1.0]
    best, best_acc = None, -1.0
    for wu, wm in grid:
        m = fit_stage(Blend(wu, wm), seqs, -2, n)
        acc = evaluate(m, seqs, -2, n)["Acc@1"]
        if acc > best_acc:
            best, best_acc = (wu, wm), acc
    print(f"best blend on validation: w_user={best[0]}, w_markov={best[1]} (Acc@1 {best_acc:.3f})")

    models = {
        "global mode": GlobalMode(),
        "repeat last genre": RepeatLast(),
        "user mode": UserMode(),
        "user decayed profile": DecayUser(),
        "markov-1": Markov1(),
        f"blend {best}": Blend(*best),
    }
    rows = {}
    for name, model in models.items():
        fit_stage(model, seqs, -1, n)
        rows[name] = evaluate(model, seqs, -1, n)
    print("\nTest (target = each user's last review)")
    print(pd.DataFrame(rows).T.round(3).to_string())


if __name__ == "__main__":
    main()
