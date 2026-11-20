"""Precomputed verdicts, so the demo can run without a GPU.

`precompute.py` runs the real engine on every input the demo can produce from the test claims and
the counterfactual pairs (as loaded, with predicted / gold evidence removed, with any one sentence
removed) and stores the verdicts. `CachedVerifier` answers from that store and falls back to the live
engine (loaded lazily, CPU if there is no GPU) for anything else, e.g. a pasted claim.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import threading

from engine import Verdict

CACHE_FILE = Path(__file__).with_name("cache") / "verdicts.jsonl.gz"


def cache_key(model: str, claim: str, articles: list[dict]) -> str:
    """`articles` exactly as the app passes them to the engine (removed sentences are "")."""
    payload = [model, claim.strip(), [[int(a["doc_id"]), a["title"], list(a["sentences"])] for a in articles]]
    return hashlib.sha1(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def to_record(key: str, verdict: Verdict) -> dict:
    return {"key": key, "label": verdict.label, "probs": verdict.label_probabilities,
            "sentences": [[d, i, p] for (d, i), p in verdict.sentence_probabilities.items()],
            "seconds": verdict.seconds, "tokens": verdict.input_tokens}


def from_record(record: dict) -> Verdict:
    return Verdict(record["label"], record["probs"], {(d, i): p for d, i, p in record["sentences"]},
                   record["seconds"], record["tokens"])


def read_cache(path: Path) -> dict[str, dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    if not path.exists():
        return {}
    with opener(path, "rt") as handle:
        return {r["key"]: r for r in map(json.loads, handle)}


class CachedVerifier:
    """Same interface as D1Verifier (`model_names`, `verify`); `verify(...).cached` tells the source."""

    def __init__(self, cache: dict[str, dict], model_names: list[str], load_engine=None):
        self.cache = cache
        self.model_names = model_names
        self._load_engine = load_engine
        self._engine = None
        self._lock = threading.Lock()

    def start_loading(self):
        """Load the live engine in the background so cached answers are available immediately."""
        if self._load_engine:
            threading.Thread(target=self.engine, daemon=True).start()

    def engine(self):
        with self._lock:
            if self._engine is None and self._load_engine:
                self._engine = self._load_engine()
        return self._engine

    def verify(self, claim: str, articles: list[dict], model: str) -> Verdict:
        record = self.cache.get(cache_key(model, claim, articles))
        if record:
            verdict = from_record(record)
            verdict.cached = True
            return verdict
        engine = self.engine()
        if engine is None:
            raise LookupError("This input was not precomputed and no live model is available.")
        verdict = engine.verify(claim, articles, model)
        verdict.cached = False
        return verdict
