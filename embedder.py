"""Turns text into vectors. Default: a local sentence-transformers model (free, runs offline
after the first download). 'tfidf' is a fallback for testing on machines without the model."""
import os, pickle
import numpy as np
import config


class Embedder:
    def __init__(self, kind=None):
        self.kind = kind or os.environ.get("RAG_EMBEDDER", "st")
        self._model = None
        self._tfidf = None

    def _st(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(config.EMBED_MODEL)
        return self._model

    def fit_encode(self, docs):
        """Embed every product document; vectors are normalised so a dot product = cosine similarity."""
        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
            m = self._tfidf.fit_transform(docs).astype(np.float32).toarray()
        else:
            m = self._st().encode(docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        return _normalise(np.asarray(m, dtype=np.float32))

    def encode_query(self, text):
        if self.kind == "tfidf":
            v = self._tfidf.transform([text]).astype(np.float32).toarray()
        else:
            v = self._st().encode([config.QUERY_PREFIX + text], normalize_embeddings=True)
        return _normalise(np.asarray(v, dtype=np.float32))[0]

    def save(self, folder):
        with open(os.path.join(folder, "embedder.pkl"), "wb") as f:
            pickle.dump({"kind": self.kind, "tfidf": self._tfidf}, f)

    @classmethod
    def load(cls, folder):
        with open(os.path.join(folder, "embedder.pkl"), "rb") as f:
            d = pickle.load(f)
        e = cls(d["kind"])
        e._tfidf = d["tfidf"]
        return e


def _normalise(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n
