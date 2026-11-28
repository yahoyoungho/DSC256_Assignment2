"""Historical feature construction. Targets never enter their own profiles."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

DAY = 86400


@dataclass
class Events:
    u: np.ndarray
    t: np.ndarray
    y: np.ndarray

    def take(self, mask):
        return Events(self.u[mask], self.t[mask], self.y[mask])

    def __len__(self):
        return len(self.u)


def encode(frame, users, game_genres, genres):
    catalog = pd.crosstab(game_genres.appid, game_genres.genre).reindex(columns=genres, fill_value=0)
    y = catalog.reindex(frame.appid).to_numpy(dtype=np.float32)
    if np.isnan(y).any() or (y.sum(axis=1) == 0).any():
        raise ValueError('Every target must have at least one canonical genre')
    u = users.get_indexer(frame.author_steamid)
    if (u < 0).any():
        raise ValueError('Unknown encoded user')
    return Events(u.astype(np.int32), frame.event_time.to_numpy(dtype=np.int64), y)


def previous_sets(e):
    """Previous strictly earlier timestamp's union of labels, never tied rows.

    Treat same-time reviews as a basket. No artificial genre order or arbitrary
    ordering of simultaneous events is introduced.
    """
    cols = [f'g{i}' for i in range(e.y.shape[1])]
    f = pd.DataFrame(e.y, columns=cols)
    f['u'], f['t'] = e.u, e.t
    grouped = f.groupby(['u', 't'], sort=True)[cols].max()
    prior = grouped.groupby(level=0).shift().fillna(0)
    return prior.reindex(pd.MultiIndex.from_arrays([e.u, e.t])).to_numpy(dtype=np.float32)


class History:
    def __init__(self, events, n_users, cutoff):
        if len(events) == 0 or events.t.max() >= cutoff:
            raise ValueError('History must be nonempty and strictly before cutoff')
        self.events, self.n_users, self.cutoff = events, n_users, int(cutoff)
        self.k = events.y.shape[1]
        self.n = np.bincount(events.u, minlength=n_users).astype(np.int32)
        self.counts = self.aggregate(events.y)
        self.prior = events.y.mean(axis=0)
        self.last_time = np.zeros(n_users, dtype=np.int64)
        np.maximum.at(self.last_time, events.u, events.t)
        self.last = np.zeros((n_users, self.k), dtype=np.float32)
        recent = events.t == self.last_time[events.u]
        np.maximum.at(self.last, events.u[recent], events.y[recent])
        self._decayed = {}

    def aggregate(self, values):
        return np.stack([np.bincount(self.events.u, weights=values[:, j], minlength=self.n_users)
                         for j in range(values.shape[1])], axis=1).astype(np.float32)

    def profile(self, half_life=None, fractional=False):
        key = (half_life, fractional)
        if key not in self._decayed:
            e = self.events
            # A common exponential factor cancels in a user's normalized profile.
            # Center on that user's latest review, keeping at least one weight=1
            # even when the whole history is many years before the cutoff.
            w = np.ones(len(e)) if half_life is None else np.exp2(-(self.last_time[e.u] - e.t) / (half_life * DAY))
            values = e.y / e.y.sum(axis=1, keepdims=True) if fractional else e.y
            counts = self.aggregate(values * w[:, None])
            denom = np.bincount(e.u, weights=w, minlength=self.n_users)
            result = np.divide(counts, denom[:, None], out=np.broadcast_to(self.prior, counts.shape).copy(), where=denom[:, None] > 0)
            self._decayed[key] = result.astype(np.float32)
        return self._decayed[key]

    def popularity(self, days=None):
        y = self.events.y if days is None else self.events.y[self.events.t >= self.cutoff - days * DAY]
        return y.mean(axis=0) if len(y) else self.prior

    def features(self, users):
        age = np.minimum((self.cutoff - self.last_time[users]) / DAY, 3650)
        age[self.n[users] == 0] = 3650
        return np.column_stack([self.profile()[users], self.profile(90)[users], self.last[users],
                                np.broadcast_to(self.popularity(90), (len(users), self.k)),
                                np.log1p(self.n[users]) / 10, np.log1p(age) / 10]).astype(np.float32)

    def transitions(self, alpha=1):
        prev = previous_sets(self.events)
        mask = prev.sum(axis=1) > 0
        prev = prev[mask] / prev[mask].sum(axis=1, keepdims=True)
        counts = prev.T @ self.events.y[mask]
        totals = prev.sum(axis=0)
        return (counts + alpha * self.prior) / (totals[:, None] + alpha)


def snapshot_examples(e, n_users):
    """Three disjoint target windows wholly inside fitting data.

    Each window freezes features at its left boundary. This mirrors the main
    frozen-cutoff evaluation, instead of using the target's immediate predecessor.
    """
    times = np.sort(e.t)
    edges = [int(times[int(q * len(times))]) for q in [.4, .6, .8]] + [int(times[-1]) + 1]
    xs, ys, us = [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        target = (e.t >= left) & (e.t < right)
        if not target.any():
            continue
        h = History(e.take(e.t < left), n_users, left)
        xs.append(h.features(e.u[target])); ys.append(e.y[target]); us.append(e.u[target])
    if not xs:
        raise ValueError('Not enough distinct timestamps for snapshot training')
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(us)
