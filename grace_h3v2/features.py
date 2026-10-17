import copy
import re
import warnings
import numpy as np
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import folds, metrics

TOKEN_RE = re.compile(r'0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?|[A-Za-z_]\w*|'
                      r'->|::|<<|>>|<=|>=|==|!=|&&|\|\||\+\+|--|[^\s]')
NUMERIC_NAMES = ('log_bytes', 'log_lines', 'log_tokens', 'if', 'else', 'for', 'while',
                 'switch', 'case', 'return', 'goto', 'break', 'sizeof', 'null',
                 'star', 'arrow', 'bracket', 'compare', 'logical', 'arithmetic',
                 'allocation', 'release', 'copy', 'input', 'error', 'assert')


def tokens(text):
    return TOKEN_RE.findall(text)


def numeric(rows):
    result = []
    for row in rows:
        code = row['code']
        t = tokens(code)
        lower = [w.lower() for w in t]
        count = lambda values: sum(w in values for w in lower)
        a = [np.log1p(len(code.encode('utf-8'))), np.log1p(len(code.splitlines())),
             np.log1p(len(t))]
        a += [lower.count(w) for w in NUMERIC_NAMES[3:13]]
        a += [count({'null', 'nullptr'}), lower.count('*'), lower.count('->'),
              lower.count('['), count({'<', '>', '<=', '>=', '==', '!='}),
              count({'&&', '||', '!'}), count({'+', '-', '*', '/', '%'}),
              count({'malloc', 'calloc', 'realloc', 'alloca', 'new'}),
              count({'free', 'delete'}), count({'memcpy', 'memmove', 'strcpy', 'strncpy'}),
              count({'read', 'recv', 'scanf', 'gets'}), count({'errno', 'error', 'fail'}),
              lower.count('assert')]
        result.append(a)
    return np.asarray(result, dtype=np.float32)


class DenseMLP:
    """Group-held-out epoch selection followed by fresh full-fold refit."""
    def __init__(self, cfg, seed, group_key, hidden=128):
        self.cfg, self.seed, self.group_key, self.hidden = cfg, seed, group_key, hidden

    def _new(self):
        return MLPClassifier(hidden_layer_sizes=(self.hidden,), activation='relu',
                             solver='adam', alpha=self.cfg['alpha'],
                             batch_size=self.cfg['batch_size'],
                             learning_rate_init=self.cfg['learning_rate'],
                             max_iter=1, random_state=self.seed, shuffle=True)

    def fit(self, x, y, rows):
        a, b = folds(rows, y, 3, self.seed + 109, self.group_key)[0]
        scaler = StandardScaler().fit(x[a])
        model = self._new()
        xa, xb = scaler.transform(x[a]), scaler.transform(x[b])
        best, stale, epoch_count = -np.inf, 0, 1
        for epoch in range(self.cfg['epochs']):
            model.partial_fit(xa, y[a], classes=np.asarray([0, 1]))
            mcc = metrics(y[b], model.predict(xb))['mcc']
            if mcc > best + 1e-8:
                best, stale, epoch_count = mcc, 0, epoch + 1
            else:
                stale += 1
            if stale >= self.cfg['patience']:
                break
        self.scaler = StandardScaler().fit(x)
        self.model = self._new()
        full = self.scaler.transform(x)
        for _ in range(epoch_count):
            self.model.partial_fit(full, y, classes=np.asarray([0, 1]))
        self.epochs_selected = epoch_count
        return self

    def predict_proba(self, x):
        return self.model.predict_proba(self.scaler.transform(x))


class View:
    def __init__(self, name, cfg, seed):
        self.name, self.cfg, self.seed = name, cfg, seed

    def fit(self, rows, embedding, y):
        name = self.name
        if name == 'lexical':
            opts = self.cfg['lexical']
            self.word = TfidfVectorizer(tokenizer=tokens, token_pattern=None,
                                       lowercase=False, ngram_range=(1, 2),
                                       max_features=opts['token_features'], min_df=2,
                                       sublinear_tf=True, dtype=np.float32)
            self.char = TfidfVectorizer(analyzer='char', lowercase=False,
                                       ngram_range=(3, 5), max_features=opts['char_features'],
                                       min_df=2, sublinear_tf=True, dtype=np.float32)
            from scipy.sparse import hstack
            codes = [r['code'] for r in rows]
            x = hstack([self.word.fit_transform(codes), self.char.fit_transform(codes)],
                       format='csr')
            self.model = SGDClassifier(loss='log_loss', alpha=opts['alpha'],
                                       max_iter=opts['max_iter'], tol=1e-4,
                                       random_state=self.seed)
            self.model.fit(x, y)
        elif name == 'semantic_mlp':
            self.model = DenseMLP(self.cfg['mlp'], self.seed, self.cfg['group_key'])
            self.model.fit(np.asarray(embedding), y, rows)
        else:
            x = numeric(rows) if name == 'numeric' else np.asarray(embedding)
            weighted = 'balanced' if name == 'semantic_lr_balanced' else None
            self.model = make_pipeline(StandardScaler(),
                                       LogisticRegression(C=self.cfg['linear_C'],
                                                          class_weight=weighted,
                                                          max_iter=2000, random_state=self.seed))
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                self.model.fit(x, y)
        return self

    def score(self, rows, embedding):
        if self.name == 'lexical':
            from scipy.sparse import hstack
            codes = [r['code'] for r in rows]
            x = hstack([self.word.transform(codes), self.char.transform(codes)], format='csr')
        else:
            x = numeric(rows) if self.name == 'numeric' else np.asarray(embedding)
        p = self.model.predict_proba(x)[:, 1]
        if not np.isfinite(p).all():
            raise FloatingPointError('Non-finite classifier score')
        return p


VIEWS = ('semantic_lr', 'semantic_lr_balanced', 'semantic_mlp', 'lexical', 'numeric')
ARMS = ('semantic_lr', 'semantic_lr_balanced', 'semantic_mlp', 'lexical',
        'fusion_semantic_lexical', 'fusion_all')


def fusion_input(scores, arm):
    names = ('semantic_mlp', 'lexical') if arm == 'fusion_semantic_lexical' else (
        'semantic_lr', 'semantic_mlp', 'lexical', 'numeric')
    return np.column_stack([logit(np.clip(scores[:, VIEWS.index(n)], 1e-6, 1-1e-6))
                            for n in names])


def fusion_model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=2000))
