"""Merge train_more and dev, split 7:3, and write one JSON file per claim.

Train claims are augmented. A gold rationale is one evidence group, so a
multi-sentence rationale is deleted together. Each SUPPORT_more or
CONTRADICT_more sentence is its own group. The original group keeps every
sentence. Other groups drop 1, 2, ... evidence groups, one random subset per
count, and every claim that has evidence also gets one group with all
evidence sentences removed. New groups also drop about 15% of the
non-evidence sentences in each abstract, never more than half, and never the
sentence immediately before or after a kept evidence sentence.

Test claims keep the original abstracts. The split is an unstratified shuffle
with seed 42. Outputs are data/train/{id}.json and data/test/{id}.json.

    .venv/bin/python build_claim_splits.py
"""

import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
TRAIN_PATH = DATA_DIR / "claims_train.jsonl"
MORE_PATH = DATA_DIR / "claims_train_more.jsonl"
DEV_PATH = DATA_DIR / "claims_dev.jsonl"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

SEED = 42
TRAIN_PARTS = 7
SPLIT_PARTS = 10


def load_jsonl(path):
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def nearest_fifteen_percent(count):
    """Nearest integer to 15% of count. An exact half rounds up."""
    if count <= 0:
        return 0
    quotient, remainder = divmod(count * 15, 100)
    if remainder >= 50:
        return quotient + 1
    return quotient


def non_evidence_deletion_count(count):
    """About 15% of the non-evidence sentences, and at most half of them."""
    return min(nearest_fifteen_percent(count), count // 2)


def claim_polarity(groups):
    if not groups:
        return "NEI"
    labels = set()
    for group in groups:
        label = group["label"]
        if label.endswith("_more"):
            label = label[: -len("_more")]
        labels.add(label)
    if labels == {"SUPPORT"}:
        return "SUPPORT"
    if labels == {"CONTRADICT"}:
        return "CONTRADICT"
    raise ValueError(f"mixed evidence labels: {sorted(labels)}")


def cited_doc_ids(row):
    docs = []
    for doc_id in row.get("cited_doc_ids") or []:
        if isinstance(doc_id, bool) or not isinstance(doc_id, int):
            raise ValueError(f"claim {row['id']} has a non-integer cited doc id")
        docs.append(doc_id)
    return docs


def groups_from_evidence(evidence):
    groups = []
    for doc_key, annotations in (evidence or {}).items():
        doc_id = int(doc_key)
        for annotation in annotations:
            sentences = tuple(annotation["sentences"])
            if not sentences:
                raise ValueError(f"empty rationale for doc {doc_id}")
            groups.append(
                {
                    "doc_id": doc_id,
                    "sentences": sentences,
                    "label": annotation["label"],
                }
            )
    return groups


def build_claim(row, groups, origin):
    docs = cited_doc_ids(row)
    for group in groups:
        if group["doc_id"] not in docs:
            docs.append(group["doc_id"])
    numbered = []
    for index, group in enumerate(groups):
        numbered.append({**group, "group_id": f"g{index}"})
    return {
        "id": row["id"],
        "claim": row["claim"],
        "origin": origin,
        "docs": docs,
        "groups": numbered,
        "label": claim_polarity(numbered),
    }


def load_claims(corpus):
    train_by_id = {row["id"]: row for row in load_jsonl(TRAIN_PATH)}
    claims = []
    for row in load_jsonl(MORE_PATH):
        source = train_by_id.get(row["id"])
        if source is None or source["claim"] != row["claim"]:
            raise ValueError(f"claim {row['id']} is missing from {TRAIN_PATH.name}")
        groups = groups_from_evidence(source.get("evidence"))
        for group in groups_from_evidence(row.get("evidence")):
            if group["label"].endswith("_more"):
                if len(group["sentences"]) != 1:
                    raise ValueError(f"claim {row['id']} has a multi-sentence _more group")
                groups.append(group)
        claims.append(build_claim(row, groups, "train_more"))
    for row in load_jsonl(DEV_PATH):
        claims.append(build_claim(row, groups_from_evidence(row.get("evidence")), "dev"))

    ids = [claim["id"] for claim in claims]
    if len(ids) != len(set(ids)):
        raise ValueError("claim ids overlap between train_more and dev")
    for claim in claims:
        seen = set()
        for group in claim["groups"]:
            abstract = corpus[group["doc_id"]]["abstract"]
            for index in group["sentences"]:
                key = (group["doc_id"], index)
                if key in seen:
                    raise ValueError(f"claim {claim['id']} repeats evidence sentence {key}")
                if index < 0 or index >= len(abstract):
                    raise ValueError(f"claim {claim['id']} sentence {key} is outside the abstract")
                seen.add(key)
    return claims


def evidence_by_doc(groups):
    by_doc = {}
    for group in groups:
        by_doc.setdefault(group["doc_id"], set()).update(group["sentences"])
    return by_doc


def choose_variants(claim, rng):
    group_ids = [group["group_id"] for group in claim["groups"]]
    variants = [("original", [])]
    count = len(group_ids)
    if count == 0:
        return variants
    for drop_count in range(1, count):
        variants.append((f"drop_{drop_count}_groups", rng.sample(group_ids, drop_count)))
    variants.append(("drop_all_evidence", list(group_ids)))
    return variants


def processed_articles(claim, dropped_ids, corpus, rng, delete_background):
    groups = claim["groups"]
    dropped = [group for group in groups if group["group_id"] in dropped_ids]
    kept = [group for group in groups if group["group_id"] not in dropped_ids]
    all_evidence = evidence_by_doc(groups)
    kept_evidence = evidence_by_doc(kept)
    dropped_evidence = evidence_by_doc(dropped)

    articles = []
    deleted = []
    new_index_of = {}
    for doc_id in claim["docs"]:
        abstract = corpus[doc_id]["abstract"]
        evidence = all_evidence.get(doc_id, set())
        kept_here = kept_evidence.get(doc_id, set())
        removed = set(dropped_evidence.get(doc_id, set()))
        if delete_background:
            protected = set()
            for index in kept_here:
                if index - 1 >= 0:
                    protected.add(index - 1)
                if index + 1 < len(abstract):
                    protected.add(index + 1)
            eligible = [
                index
                for index in range(len(abstract))
                if index not in evidence and index not in protected
            ]
            target = non_evidence_deletion_count(len(abstract) - len(evidence))
            choose = min(target, len(eligible))
            removed.update(rng.sample(eligible, choose))

        sentences = []
        new_index = 0
        for index, sentence in enumerate(abstract):
            if index in removed:
                kind = "evidence" if index in evidence else "non_evidence"
                deleted.append(
                    {
                        "doc_id": doc_id,
                        "original_index": index,
                        "sentence": sentence,
                        "kind": kind,
                    }
                )
                continue
            sentences.append(
                {
                    "new_index": new_index,
                    "original_index": index,
                    "sentence": sentence,
                }
            )
            new_index_of[(doc_id, index)] = new_index
            new_index += 1
        articles.append(
            {
                "doc_id": doc_id,
                "title": corpus[doc_id]["title"],
                "sentences": sentences,
            }
        )
    return articles, deleted, new_index_of, kept


def evidence_records(claim, new_index_of):
    records = []
    doc_order = {doc_id: position for position, doc_id in enumerate(claim["docs"])}
    for group in claim["groups"]:
        for index in group["sentences"]:
            key = (group["doc_id"], index)
            records.append(
                {
                    "doc_id": group["doc_id"],
                    "group_id": group["group_id"],
                    "evidence_label": group["label"],
                    "original_index": index,
                    "sentence": None,
                    "kept": key in new_index_of,
                    "new_index": new_index_of.get(key),
                }
            )
    records.sort(key=lambda item: (doc_order[item["doc_id"]], item["original_index"]))
    return records


def fill_evidence_text(records, corpus):
    for record in records:
        record["sentence"] = corpus[record["doc_id"]]["abstract"][record["original_index"]]
    return records


def build_group(claim, variant, dropped_ids, corpus, rng):
    delete_background = variant != "original"
    articles, deleted, new_index_of, kept_groups = processed_articles(
        claim, dropped_ids, corpus, rng, delete_background
    )
    evidence = fill_evidence_text(evidence_records(claim, new_index_of), corpus)
    label = claim_polarity(kept_groups)
    return {
        "variant": variant,
        "dropped_group_ids": list(dropped_ids),
        "claim": claim["claim"],
        "evidence": evidence,
        "articles": articles,
        "deleted_sentences": deleted,
        "label": label,
    }


def train_payload(claim, corpus, seed):
    rng = random.Random(f"{seed}-claim-{claim['id']}")
    groups = [
        build_group(claim, variant, dropped_ids, corpus, rng)
        for variant, dropped_ids in choose_variants(claim, rng)
    ]
    return {
        "id": claim["id"],
        "origin": claim["origin"],
        "n_groups": len(groups),
        "groups": groups,
    }


def test_payload(claim, corpus):
    evidence = []
    for group in claim["groups"]:
        abstract = corpus[group["doc_id"]]["abstract"]
        for index in group["sentences"]:
            evidence.append(
                {
                    "doc_id": group["doc_id"],
                    "group_id": group["group_id"],
                    "evidence_label": group["label"],
                    "original_index": index,
                    "sentence": abstract[index],
                }
            )
    articles = []
    for doc_id in claim["docs"]:
        article = corpus[doc_id]
        articles.append(
            {
                "doc_id": doc_id,
                "title": article["title"],
                "sentences": [
                    {"original_index": index, "sentence": sentence}
                    for index, sentence in enumerate(article["abstract"])
                ],
            }
        )
    return {
        "id": claim["id"],
        "origin": claim["origin"],
        "claim": claim["claim"],
        "evidence": evidence,
        "articles": articles,
        "label": claim["label"],
    }


def check_train_payload(payload, claim, corpus):
    if payload["n_groups"] != len(payload["groups"]):
        raise AssertionError(f"claim {claim['id']} n_groups does not match")
    expected = 1 if not claim["groups"] else len(claim["groups"]) + 1
    if payload["n_groups"] != expected:
        raise AssertionError(f"claim {claim['id']} produced {payload['n_groups']} groups, expected {expected}")

    evidence = evidence_by_doc(claim["groups"])
    for group in payload["groups"]:
        if group["claim"] != claim["claim"]:
            raise AssertionError(f"claim {claim['id']} text changed")
        deleted = {(item["doc_id"], item["original_index"]) for item in group["deleted_sentences"]}
        for article in group["articles"]:
            doc_id = article["doc_id"]
            abstract = corpus[doc_id]["abstract"]
            seen = [item["original_index"] for item in article["sentences"]]
            if seen != sorted(seen) or len(seen) != len(set(seen)):
                raise AssertionError(f"claim {claim['id']} article {doc_id} order changed")
            for item in article["sentences"]:
                if item["sentence"] != abstract[item["original_index"]]:
                    raise AssertionError(f"claim {claim['id']} sentence text changed")
                if (doc_id, item["original_index"]) in deleted:
                    raise AssertionError(f"claim {claim['id']} kept a deleted sentence")
            if [item["new_index"] for item in article["sentences"]] != list(range(len(seen))):
                raise AssertionError(f"claim {claim['id']} new indices are not contiguous")
            removed_here = {index for doc, index in deleted if doc == doc_id}
            if set(seen) | removed_here != set(range(len(abstract))):
                raise AssertionError(f"claim {claim['id']} doc {doc_id} lost a sentence")

            non_evidence = [index for index in range(len(abstract)) if index not in evidence.get(doc_id, set())]
            deleted_background = [
                index for index in non_evidence if (doc_id, index) in deleted
            ]
            if group["variant"] == "original":
                if deleted_background or removed_here:
                    raise AssertionError(f"claim {claim['id']} original group deleted a sentence")
            else:
                if len(deleted_background) > len(non_evidence) // 2:
                    raise AssertionError(f"claim {claim['id']} deleted more than half the background")
                kept_evidence = evidence.get(doc_id, set()) - removed_here
                protected = set()
                for index in kept_evidence:
                    if index - 1 >= 0:
                        protected.add(index - 1)
                    if index + 1 < len(abstract):
                        protected.add(index + 1)
                if protected & set(deleted_background):
                    raise AssertionError(f"claim {claim['id']} deleted a neighbor of kept evidence")
                eligible = [index for index in non_evidence if index not in protected]
                target = min(non_evidence_deletion_count(len(non_evidence)), len(eligible))
                if len(deleted_background) != target:
                    raise AssertionError(f"claim {claim['id']} background deletion count is {len(deleted_background)}, expected {target}")

        kept_keys = set()
        for record in group["evidence"]:
            key = (record["doc_id"], record["original_index"])
            if record["sentence"] != corpus[record["doc_id"]]["abstract"][record["original_index"]]:
                raise AssertionError(f"claim {claim['id']} evidence text changed")
            if record["kept"]:
                kept_keys.add(key)
                match = [
                    item
                    for article in group["articles"]
                    if article["doc_id"] == record["doc_id"]
                    for item in article["sentences"]
                    if item["original_index"] == record["original_index"]
                ]
                if len(match) != 1 or match[0]["new_index"] != record["new_index"]:
                    raise AssertionError(f"claim {claim['id']} new evidence index does not match the article")
            elif record["new_index"] is not None or key not in deleted:
                raise AssertionError(f"claim {claim['id']} dropped evidence is inconsistent")

        for source in claim["groups"]:
            keys = {(source["doc_id"], index) for index in source["sentences"]}
            fully_kept = keys <= kept_keys
            if fully_kept and source["group_id"] in group["dropped_group_ids"]:
                raise AssertionError(f"claim {claim['id']} dropped group {source['group_id']} is still complete")
            if not fully_kept and source["group_id"] not in group["dropped_group_ids"]:
                raise AssertionError(f"claim {claim['id']} kept group {source['group_id']} was damaged")
            if len(keys & kept_keys) not in (0, len(keys)):
                raise AssertionError(f"claim {claim['id']} split group {source['group_id']}")

        if group["label"] != claim_polarity(
            [source for source in claim["groups"] if source["group_id"] not in group["dropped_group_ids"]]
        ):
            raise AssertionError(f"claim {claim['id']} variant {group['variant']} has the wrong label")


def check_test_payload(payload, claim, corpus):
    if payload["claim"] != claim["claim"] or payload["label"] != claim["label"]:
        raise AssertionError(f"test claim {claim['id']} does not match its source")
    if "deleted_sentences" in payload or "groups" in payload:
        raise AssertionError(f"test claim {claim['id']} contains train-only fields")
    evidence = {
        (item["doc_id"], item["original_index"])
        for item in payload["evidence"]
    }
    expected = {
        (group["doc_id"], index)
        for group in claim["groups"]
        for index in group["sentences"]
    }
    if evidence != expected:
        raise AssertionError(f"test claim {claim['id']} evidence does not match")
    for article in payload["articles"]:
        abstract = corpus[article["doc_id"]]["abstract"]
        got = [item["sentence"] for item in article["sentences"]]
        if got != abstract:
            raise AssertionError(f"test claim {claim['id']} article was modified")


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main():
    corpus = {row["doc_id"]: row for row in load_jsonl(CORPUS_PATH)}
    claims = load_claims(corpus)
    ids = sorted(claim["id"] for claim in claims)
    random.Random(SEED).shuffle(ids)
    train_count = (len(ids) * TRAIN_PARTS) // SPLIT_PARTS
    train_ids = set(ids[:train_count])
    by_id = {claim["id"]: claim for claim in claims}

    reset_dir(TRAIN_DIR)
    reset_dir(TEST_DIR)

    new_groups = 0
    variant_counts = {}
    new_labels = {}
    for claim_id in ids:
        claim = by_id[claim_id]
        if claim_id in train_ids:
            payload = train_payload(claim, corpus, SEED)
            check_train_payload(payload, claim, corpus)
            write_json(TRAIN_DIR / f"{claim_id}.json", payload)
            new_groups += payload["n_groups"] - 1
            for group in payload["groups"]:
                variant_counts[group["variant"]] = variant_counts.get(group["variant"], 0) + 1
                if group["variant"] != "original":
                    new_labels[group["label"]] = new_labels.get(group["label"], 0) + 1
        else:
            payload = test_payload(claim, corpus)
            check_test_payload(payload, claim, corpus)
            write_json(TEST_DIR / f"{claim_id}.json", payload)

    train_files = list(TRAIN_DIR.glob("*.json"))
    test_files = list(TEST_DIR.glob("*.json"))
    print(f"train json files: {len(train_files)}")
    print(f"test json files: {len(test_files)}")
    print(f"new train groups: {new_groups}")
    print(f"variant counts: {variant_counts}")
    print(f"new group labels: {new_labels}")


if __name__ == "__main__":
    main()
