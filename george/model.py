"""Small content-based recommender with a global genre-frequency prior."""
import numpy as np
import pandas as pd


class GenreRecommender:
    """Estimate genre membership rates with alpha pseudo-interactions.

    score(u,g) = (positive_count(u,g) + alpha * global_rate(g)) /
                 (positive_review_count(u) + alpha)

    Each distinct genre on a liked game receives one vote. Rates need not sum
    to one: a game can have multiple genres. Alpha=0 gives personal mode;
    global_only=True gives the requested global-mode baseline. Ties use
    global frequency then alphabetic genre order. Unknown users use the prior.
    """

    def __init__(self, alpha=5.0, global_only=False):
        if alpha < 0 or not np.isfinite(alpha):
            raise ValueError('alpha must be finite and nonnegative')
        self.alpha = float(alpha)
        self.global_only = global_only

    def fit(self, interactions, game_genres):
        if interactions.empty:
            raise ValueError('Cannot fit without positive training reviews')
        self.genres = sorted(game_genres.genre.unique())
        expanded = interactions[['author_steamid', 'appid']].merge(game_genres, on='appid')
        self.global_counts = expanded.genre.value_counts().reindex(self.genres, fill_value=0)
        self.prior = self.global_counts.to_numpy(dtype=float) / len(interactions)
        self.counts = pd.crosstab(expanded.author_steamid, expanded.genre).reindex(columns=self.genres, fill_value=0)
        self.history = interactions.groupby('author_steamid').size()
        self.tie_order = np.argsort(-self.prior, kind='stable')
        self.global_mode = self.genres[self.tie_order[0]]
        return self

    def predict(self, users):
        users = pd.Index(users, dtype='string')
        n = self.history.reindex(users, fill_value=0).to_numpy(dtype=float)
        if self.global_only:
            scores = np.broadcast_to(self.prior, (len(users), len(self.genres)))
        else:
            counts = self.counts.reindex(users, fill_value=0).to_numpy(dtype=float)
            denom = n + self.alpha
            scores = np.divide(counts + self.alpha * self.prior, denom[:, None],
                               out=np.broadcast_to(self.prior, counts.shape).copy(),
                               where=denom[:, None] > 0)
        best = self.tie_order[np.argmax(scores[:, self.tie_order], axis=1)]
        return pd.DataFrame({'author_steamid': users,
                             'recommended_genre': np.asarray(self.genres)[best],
                             'history_count': n.astype(int),
                             'score': scores[np.arange(len(users)), best]})

    def save(self, path):
        """Portable numeric artifact; no pickle or executable deserialization."""
        np.savez_compressed(path, genres=np.asarray(self.genres), prior=self.prior,
                            users=self.counts.index.to_numpy(dtype=str),
                            counts=self.counts.to_numpy(dtype=np.int32),
                            history=self.history.reindex(self.counts.index).to_numpy(),
                            alpha=self.alpha, global_only=self.global_only)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(float(data['alpha']), bool(data['global_only']))
            model.genres = data['genres'].tolist()
            model.prior = data['prior'].copy()
            users = pd.Index(data['users'], dtype='string')
            model.counts = pd.DataFrame(data['counts'], index=users, columns=model.genres)
            model.history = pd.Series(data['history'], index=users)
        model.tie_order = np.argsort(-model.prior, kind='stable')
        model.global_mode = model.genres[model.tie_order[0]]
        return model
