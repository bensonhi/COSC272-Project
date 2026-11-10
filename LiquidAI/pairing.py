"""Match original claims to their evidence-deleted training versions."""

from __future__ import annotations

from .data import Sample


def build_evidence_pairs(samples: list[Sample]) -> dict[int, int]:
    """Return sufficient-index -> insufficient-index, once per eligible claim.

    Other variants and original NEI claims still receive ordinary supervision.
    Missing counterparts (e.g. after --max-groups) are never invented.
    """
    claims: dict[int, dict[str, int]] = {}
    for index, sample in enumerate(samples):
        if sample.variant not in ("original", "drop_all_evidence"):
            continue
        variants = claims.setdefault(sample.claim_id, {})
        if sample.variant in variants:
            raise ValueError(f"Claim {sample.claim_id}: ambiguous duplicate {sample.variant} version")
        variants[sample.variant] = index

    pairs = {}
    for claim_id, variants in claims.items():
        if "original" not in variants or "drop_all_evidence" not in variants:
            continue
        sufficient_index, insufficient_index = variants["original"], variants["drop_all_evidence"]
        sufficient, insufficient = samples[sufficient_index], samples[insufficient_index]
        if sufficient.label not in ("SUPPORT", "CONTRADICT") or insufficient.label != "NEI":
            raise ValueError(f"Claim {claim_id}: expected original SUPPORT/CONTRADICT and deleted NEI")
        if sufficient.claim != insufficient.claim:
            raise ValueError(f"Claim {claim_id}: paired versions have different claim text")
        originals = {a.doc_id: a for a in sufficient.articles}
        deleted = {a.doc_id: a for a in insufficient.articles}
        if originals.keys() != deleted.keys():
            raise ValueError(f"Claim {claim_id}: paired versions have different articles")
        for doc_id, article in deleted.items():
            original = originals[doc_id]
            if article.title != original.title:
                raise ValueError(f"Claim {claim_id}: paired article titles differ")
            for i, available in enumerate(article.sentence_mask):
                if not available:
                    continue
                if i >= len(original.sentences) or article.sentences[i] != original.sentences[i]:
                    raise ValueError(f"Claim {claim_id}: deleted version adds or changes sentence {(doc_id, i)}")
                if original.evidence_targets[i] or article.evidence_targets[i]:
                    raise ValueError(f"Claim {claim_id}: drop_all_evidence retains annotated evidence")
        pairs[sufficient_index] = insufficient_index
    if not pairs:
        raise ValueError("--loss pair requires at least one complete original / drop_all_evidence pair; check training data or increase --max-groups")
    return pairs
