"""Baselines, logistic regression, genre MF/FMC/FPMC, and rank ensembles.

The factor models adapt next-basket ranking to nine genre labels. They do not
claim to reproduce the original game-level FPMC benchmark.
"""
import copy
import hashlib
import time
import numpy as np
import torch
from torch import nn
from features import previous_sets, snapshot_examples
from metrics import evaluate_scores, rank_scores

torch.set_num_threads(4)


class Baseline:
    def __init__(self, kind, **params):
        self.kind, self.params = kind, params

    def fit(self, history):
        self.transition = history.transitions(self.params.get('alpha', 1)) if self.kind == 'markov1' else None
        return self

    def scores(self, h, users):
        users = np.asarray(users)
        if self.kind == 'global':
            result = np.broadcast_to(h.prior, (len(users), h.k)).copy()
        elif self.kind == 'recent':
            result = np.broadcast_to(h.popularity(self.params['days']), (len(users), h.k)).copy()
        elif self.kind in ('user', 'recency', 'fractional'):
            result = h.profile(self.params.get('half_life'), self.kind == 'fractional')[users].copy()
        elif self.kind == 'last':
            result = h.last[users].copy()
        elif self.kind == 'markov1':
            last = h.last[users]
            result = last @ self.transition / np.maximum(last.sum(axis=1, keepdims=True), 1)
        elif self.kind == 'random':
            # Stable per-user seeded pseudo-random scores, invariant to batch order.
            x = (users[:, None].astype(np.uint64) + 1) * np.uint64(0x9E3779B185EBCA87)
            x = x ^ (np.arange(h.k, dtype=np.uint64)[None, :] + np.uint64(2026))
            x ^= x >> np.uint64(30); x *= np.uint64(0xBF58476D1CE4E5B9)
            x ^= x >> np.uint64(27); x *= np.uint64(0x94D049BB133111EB)
            x ^= x >> np.uint64(31)
            return (x >> np.uint64(11)).astype(float) / 2**53
        else:
            raise ValueError(self.kind)
        # Recent popularity has its own cold fallback; all other personalized
        # baselines use the all-history prior.
        if self.kind not in ('recent', 'global'):
            result[h.n[users] == 0] = h.prior
        return result

    def config(self):
        return {'kind': self.kind, **self.params}


class Factor(nn.Module):
    def __init__(self, users, genres, dim, kind, prior):
        super().__init__()
        self.kind = kind
        self.user = nn.Embedding(users + 1, dim, padding_idx=0)
        self.item = nn.Parameter(torch.randn(genres, dim) * .05)
        self.previous = nn.Parameter(torch.randn(genres, dim) * .05)
        self.following = nn.Parameter(torch.randn(genres, dim) * .05)
        p = np.clip(prior, .001, .999)
        self.bias = nn.Parameter(torch.tensor(np.log(p / (1 - p)), dtype=torch.float32))
        nn.init.normal_(self.user.weight, std=.05)
        with torch.no_grad():
            self.user.weight[0].zero_()

    def forward(self, uid, previous):
        result = self.bias.expand(len(uid), -1)
        if self.kind != 'fmc':
            result = result + self.user(uid) @ self.item.T
        if self.kind != 'mf':
            p = previous / previous.sum(dim=1, keepdim=True).clamp_min(1)
            result = result + (p @ self.previous) @ self.following.T
        return result


class Learned:
    def __init__(self, kind, dim=4, lr=.003, l2=.0001, seed=2026, max_epochs=8):
        self.kind, self.dim, self.lr, self.l2 = kind, dim, lr, l2
        self.seed, self.max_epochs = seed, max_epochs
        self.curve = []

    def config(self):
        return dict(kind=self.kind, dim=self.dim, lr=self.lr, l2=self.l2,
                    seed=self.seed, max_epochs=self.max_epochs)

    def fit(self, h, validation=None, epochs=None, training_cache=None):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        e = h.events
        if self.kind == 'logistic':
            x, y, u = training_cache if training_cache is not None else snapshot_examples(e, h.n_users)
            self.net = nn.Linear(x.shape[1], h.k)
            with torch.no_grad():
                self.net.weight.zero_()
                p = np.clip(h.prior, .001, .999)
                self.net.bias.copy_(torch.tensor(np.log(p / (1 - p))))
            inputs = torch.from_numpy(x)
            previous = None
        else:
            # Sparse user factors are only estimated with >=2 fitting reviews.
            # No evaluation users are dropped; short histories get a profile fallback.
            active = np.flatnonzero(h.n >= 2)
            self.user_map = np.zeros(h.n_users, dtype=np.int32)
            self.user_map[active] = np.arange(1, len(active) + 1)
            self.net = Factor(len(active), h.k, self.dim, self.kind, h.prior)
            prev = previous_sets(e) if training_cache is None else training_cache
            mask = (h.n[e.u] >= 2) & (e.y.sum(axis=1) < h.k)
            if self.kind == 'fmc':
                mask &= prev.sum(axis=1) > 0
            u, y = e.u[mask], e.y[mask]
            inputs = torch.from_numpy(self.user_map[u].astype(np.int64))
            previous = torch.from_numpy(prev[mask])
        if not len(y):
            raise ValueError('No eligible learned-model training examples')
        self.training_examples = len(y)
        targets = torch.from_numpy(y.astype(np.float32))
        counts = np.bincount(u, minlength=h.n_users)
        w = 1 / counts[u]
        weights = torch.tensor(w / w.mean(), dtype=torch.float32)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        best, best_state, stale = -np.inf, None, 0
        for epoch in range(1, (epochs or self.max_epochs) + 1):
            self.net.train()
            loss_sum = 0
            for batch in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y) / 8192)))):
                idx = torch.from_numpy(batch)
                optimizer.zero_grad()
                prediction = self.net(inputs[idx]) if self.kind == 'logistic' else self.net(inputs[idx], previous[idx])
                target = targets[idx]
                if self.kind == 'logistic':
                    loss = nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction='none').mean(dim=1)
                    reg = self.net.weight.square().mean()
                else:
                    # Exact BPR-style average over ALL positive/negative genre pairs.
                    # A negative is absent from this event, not a lifelong dislike.
                    pair = target[:, :, None] * (1 - target[:, None, :])
                    difference = prediction[:, :, None] - prediction[:, None, :]
                    loss = (nn.functional.softplus(-difference) * pair).sum((1, 2)) / pair.sum((1, 2)).clamp_min(1)
                    reg = self.net.bias.square().mean()
                    if self.kind != 'fmc':
                        reg = reg + self.net.user(inputs[idx]).square().mean() + self.net.item.square().mean()
                    if self.kind != 'mf':
                        reg = reg + self.net.previous.square().mean() + self.net.following.square().mean()
                objective = (loss * weights[idx]).mean() + self.l2 * reg
                objective.backward()
                optimizer.step()
                loss_sum += float(objective.detach()) * len(batch)
            value = None
            if validation is not None:
                value = evaluate_scores(validation, self.scores(h, validation.u), h.n, h.prior)[0]['user_hit_at_1']
                if value > best + 1e-10:
                    best, self.best_epoch, stale = value, epoch, 0
                    best_state = copy.deepcopy(self.net.state_dict())
                else:
                    stale += 1
            else:
                self.best_epoch = epoch
            self.curve.append({'epoch': epoch, 'training_loss': loss_sum / len(y), 'validation_user_hit_at_1': value})
            if validation is not None and stale >= 3:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def scores(self, h, users):
        self.net.eval()
        users = np.asarray(users)
        parts = []
        with torch.no_grad():
            for ids in np.array_split(users, max(1, int(np.ceil(len(users) / 32768)))):
                if self.kind == 'logistic':
                    p = self.net(torch.from_numpy(h.features(ids))).sigmoid().numpy()
                else:
                    uid = torch.from_numpy(self.user_map[ids].astype(np.int64))
                    p = self.net(uid, torch.from_numpy(h.last[ids])).numpy()
                    # Explicitly preserve a reliable fallback when a user factor
                    # cannot be estimated. FMC needs no individual embedding.
                    if self.kind != 'fmc':
                        short = self.user_map[ids] == 0
                        p[short] = np.log(np.clip(h.profile()[ids[short]], .001, .999) /
                                          (1 - np.clip(h.profile()[ids[short]], .001, .999)))
                p[h.n[ids] == 0] = h.prior if self.kind == 'logistic' else np.log(np.clip(h.prior, .001, .999) / (1 - np.clip(h.prior, .001, .999)))
                parts.append(p)
        return np.concatenate(parts)

    def save_parameters(self, path):
        data = {key: value.detach().numpy() for key, value in self.net.state_dict().items()}
        if hasattr(self, 'user_map'):
            data['user_map'] = self.user_map
        np.savez_compressed(path, **data)

    def load_parameters(self, path, prior):
        with np.load(path, allow_pickle=False) as data:
            if self.kind == 'logistic':
                self.net = nn.Linear(data['weight'].shape[1], data['weight'].shape[0])
            else:
                self.user_map = data['user_map'].copy()
                self.net = Factor(data['user.weight'].shape[0] - 1, len(prior), self.dim, self.kind, prior)
            self.net.load_state_dict({k: torch.tensor(data[k]) for k in data.files if k != 'user_map'})
        return self


class RankEnsemble:
    """Borda-style ranks avoid averaging incompatible probabilities/BPR logits."""
    def __init__(self, members, weights):
        self.members, self.weights = members, weights

    def scores(self, h, users):
        return sum(w * rank_scores(m.scores(h, users), h.prior)
                   for m, w in zip(self.members, self.weights))


class SnapshotRecommender:
    """Portable fitted score snapshot; rebuild pipeline to incorporate new history."""
    def __init__(self, users, genres, scores, counts, prior):
        import pandas as pd
        self.users = pd.Index(users, dtype='string')
        self.genres = np.asarray(genres)
        self.scores, self.counts, self.prior = scores, counts, prior

    def predict(self, users):
        import pandas as pd
        from metrics import ranking
        ids = self.users.get_indexer(users)
        known = ids >= 0
        scores = np.broadcast_to(self.prior, (len(ids), len(self.genres))).copy()
        scores[known] = self.scores[ids[known]]
        counts = np.zeros(len(ids), dtype=int)
        counts[known] = self.counts[ids[known]]
        top = ranking(scores, self.prior)[:, 0]
        return pd.DataFrame({'author_steamid': users, 'recommended_genre': self.genres[top],
                             'history_count': counts, 'score': scores[np.arange(len(ids)), top]})

    def save(self, path):
        np.savez_compressed(path, users=self.users.to_numpy(dtype=str), genres=self.genres,
                            scores=self.scores.astype(np.float32), counts=self.counts, prior=self.prior)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            return cls(**{k: data[k] for k in ['users', 'genres', 'scores', 'counts', 'prior']})
