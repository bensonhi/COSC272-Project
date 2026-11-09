"""Add extra evidence sentences to the train claims.

Claims with an empty evidence field are copied unchanged. For every abstract
that already has gold evidence, Grok reads the claim, the abstract, and the
gold sentences, then returns other sentence indices with the same verdict.
Gold sentences keep SUPPORT or CONTRADICT. New sentences are saved one index
at a time as SUPPORT_more or CONTRADICT_more.

The result is data/claims_train_more.jsonl. API responses are cached under
data/cache/train_evidence_more/, so a rerun continues after an interruption.

    export XAI_API_KEY="..."
    .venv/bin/python augment_train_evidence.py
    .venv/bin/python augment_train_evidence.py --limit 2
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "claims_train.jsonl"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
OUTPUT_PATH = DATA_DIR / "claims_train_more.jsonl"
PREVIEW_PATH = DATA_DIR / "claims_train_more.preview.jsonl"
CACHE_DIR = DATA_DIR / "cache" / "train_evidence_more"

MODEL = "grok-4.7"
PROMPT_VERSION = 1
MAX_ATTEMPTS = 5

SYSTEM_PROMPT = (
    "You select additional evidence sentences from a scientific abstract. "
    "Return only sentence indices that appear in the abstract."
)


class ExtraSentences(BaseModel):
    sentences: list[int] = Field(
        description=(
            "0-based indices of additional sentences that individually give "
            "the same verdict as the original evidence. Empty when none qualify."
        )
    )


def load_jsonl(path):
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def original_label_and_indices(annotations):
    labels = {annotation["label"] for annotation in annotations}
    if labels not in ({"SUPPORT"}, {"CONTRADICT"}):
        raise ValueError(f"expected one gold label, found {sorted(labels)}")
    indices = []
    for annotation in annotations:
        indices.extend(annotation["sentences"])
    return labels.pop(), set(indices)


def filter_new_indices(proposed, abstract_length, banned):
    accepted = []
    for index in proposed:
        if isinstance(index, bool) or not isinstance(index, int):
            continue
        if index < 0 or index >= abstract_length or index in banned or index in accepted:
            continue
        accepted.append(index)
    accepted.sort()
    return accepted


def merge_evidence(annotations, new_indices, label):
    merged = copy.deepcopy(annotations)
    extra_label = f"{label}_more"
    for index in new_indices:
        merged.append({"sentences": [index], "label": extra_label})
    return merged


def augment_claim(claim, corpus, fetch_extra):
    evidence = claim.get("evidence") or {}
    if not evidence:
        return copy.deepcopy(claim)

    record = copy.deepcopy(claim)
    augmented = {}
    for doc_key, annotations in evidence.items():
        label, banned = original_label_and_indices(annotations)
        document = corpus.get(int(doc_key))
        if document is None:
            raise KeyError(f"claim {claim['id']} cites missing document {doc_key}")
        abstract = document["abstract"]
        proposed = fetch_extra(
            claim=claim,
            doc_key=str(doc_key),
            abstract=abstract,
            annotations=annotations,
            label=label,
        )
        accepted = filter_new_indices(proposed, len(abstract), banned)
        augmented[doc_key] = merge_evidence(annotations, accepted, label)
    record["evidence"] = augmented
    return record


def render_prompt(claim, abstract, annotations, label):
    numbered = "\n".join(f"[{index}] {sentence.strip()}" for index, sentence in enumerate(abstract))
    gold_lines = []
    for annotation in annotations:
        for index in annotation["sentences"]:
            gold_lines.append(f"[{index}] {abstract[index].strip()}")
    verdict = "support" if label == "SUPPORT" else "contradict"
    return (
        f"Claim:\n{claim['claim']}\n\n"
        f"The original evidence {verdict}s this claim. "
        f"Find other sentences in the abstract that also {verdict} it on their own. "
        "Skip background, topic sentences, and method details that do not state the result. "
        "Do not return an index that is already original evidence.\n\n"
        f"Original evidence:\n" + "\n".join(gold_lines) + "\n\n"
        f"Abstract:\n{numbered}"
    )


def cache_path(claim_id, doc_key):
    return CACHE_DIR / f"{claim_id}_{doc_key}.json"


def read_cache(claim_id, doc_key):
    path = cache_path(claim_id, doc_key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if payload.get("prompt_version") != PROMPT_VERSION:
        return None
    return payload["sentences"]


def write_cache(claim_id, doc_key, sentences):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(claim_id, doc_key)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "prompt_version": PROMPT_VERSION,
                "claim_id": claim_id,
                "doc_id": doc_key,
                "sentences": sentences,
            },
            ensure_ascii=False,
        )
    )
    temporary.replace(path)


def call_model(client, prompt):
    from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

    retriable = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            completion = client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=ExtraSentences,
                reasoning_effort="low",
            )
            parsed = completion.choices[0].message.parsed
            if parsed is None:
                raise ValueError("model returned no parsed sentences")
            return list(parsed.sentences)
        except retriable as error:
            last_error = error
            time.sleep(2**attempt)
    raise last_error


def make_fetcher(client):
    def fetch_extra(claim, doc_key, abstract, annotations, label):
        cached = read_cache(claim["id"], doc_key)
        if cached is not None:
            return cached
        prompt = render_prompt(claim, abstract, annotations, label)
        sentences = call_model(client, prompt)
        write_cache(claim["id"], doc_key, sentences)
        return sentences

    return fetch_extra


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def augment_split(train_rows, corpus, fetch_extra, limit=None):
    selected = train_rows if limit is None else train_rows[:limit]
    augmented = []
    extra_sentence_count = 0
    for number, claim in enumerate(selected, start=1):
        record = augment_claim(claim, corpus, fetch_extra)
        augmented.append(record)
        added = 0
        for annotations in record["evidence"].values():
            added += sum(annotation["label"].endswith("_more") for annotation in annotations)
        extra_sentence_count += added
        print(f"{number}/{len(selected)} claim {claim['id']}: +{added} sentences", flush=True)
    return augmented, extra_sentence_count


def self_test():
    abstract = [f"sentence {index}" for index in range(14)]
    corpus = {33409100: {"doc_id": 33409100, "abstract": abstract}}
    claim = {
        "id": 12,
        "claim": "Example claim.",
        "evidence": {
            "33409100": [
                {"sentences": [8], "label": "SUPPORT"},
                {"sentences": [12], "label": "SUPPORT"},
            ]
        },
        "cited_doc_ids": [33409100],
    }
    empty = {"id": 0, "claim": "No evidence.", "evidence": {}, "cited_doc_ids": [1]}

    def fetch_extra(**_kwargs):
        return [12, 9, 9, 11, -1, 99, True]

    augmented = augment_claim(claim, corpus, fetch_extra)
    copied = augment_claim(empty, corpus, fetch_extra)
    assert copied == empty
    assert augmented["evidence"]["33409100"] == [
        {"sentences": [8], "label": "SUPPORT"},
        {"sentences": [12], "label": "SUPPORT"},
        {"sentences": [9], "label": "SUPPORT_more"},
        {"sentences": [11], "label": "SUPPORT_more"},
    ]
    contradict = copy.deepcopy(claim)
    contradict["evidence"]["33409100"][0]["label"] = "CONTRADICT"
    contradict["evidence"]["33409100"][1]["label"] = "CONTRADICT"
    augmented = augment_claim(contradict, corpus, fetch_extra)
    assert augmented["evidence"]["33409100"][-1]["label"] == "CONTRADICT_more"
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Process only the first N train claims.")
    parser.add_argument("--self-test", action="store_true", help="Check the merge rules without calling the API.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if not os.environ.get("XAI_API_KEY"):
        raise SystemExit("Set XAI_API_KEY before running this script.")

    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ["XAI_API_KEY"],
        base_url="https://api.x.ai/v1",
        timeout=120,
    )
    train_rows = load_jsonl(TRAIN_PATH)
    corpus = {row["doc_id"]: row for row in load_jsonl(CORPUS_PATH)}
    augmented, extra_sentence_count = augment_split(
        train_rows,
        corpus,
        make_fetcher(client),
        limit=args.limit,
    )
    output_path = OUTPUT_PATH if args.limit is None else PREVIEW_PATH
    write_jsonl(output_path, augmented)
    print(f"wrote {len(augmented)} claims to {output_path}")
    print(f"added {extra_sentence_count} sentences")


if __name__ == "__main__":
    main()
