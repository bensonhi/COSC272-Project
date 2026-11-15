"""HealthVer (Sarrouti et al., 2021) as an out-of-distribution test: human-labelled COVID-19 claims
from web search, each paired with an evidence passage from a scientific abstract.

`--eval-set healthver` (any baseline) converts the 1,823 test pairs to Samples: one article per pair
with an empty title and the passage split into sentences; Supports / Refutes / Neutral map to
SUPPORT / CONTRADICT / NEI. HealthVer has no sentence-level rationales, so evidence metrics on this
set are meaningless; only the label metrics count. The CSV is downloaded once to scratch.
"""

import csv
from pathlib import Path
import re
import urllib.request

from common import SCRATCH
from LiquidAI.data import Article, Sample

URL = "https://raw.githubusercontent.com/sarrouti/HealthVer/master/data/healthver_{split}.csv"
LABEL = {"Supports": "SUPPORT", "Refutes": "CONTRADICT", "Neutral": "NEI"}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def healthver_samples(split: str = "test") -> list[Sample]:
    path = SCRATCH / "healthver" / f"healthver_{split}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(split=split), path)
    samples = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sentences = [s.strip() for s in SENTENCE_END.split(row["evidence"].strip()) if s.strip()]
            pair_id = int(row["id"])
            article = Article(pair_id, "", sentences, [0] * len(sentences), [True] * len(sentences))
            samples.append(Sample(pair_id, 0, "original", row["claim"].strip(), LABEL[row["label"]],
                                  [article], str(Path(path))))
    return samples
