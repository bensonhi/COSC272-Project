"""Pair identity, deletion validation and missing counterparts."""

from copy import deepcopy
import unittest

from ..data import samples_from_record
from ..pairing import build_evidence_pairs
from .fixtures import fixture


def paired_samples(claim_id=10):
    record = fixture()
    record["id"] = claim_id
    original = record["groups"][0]
    original["variant"] = "original"
    deleted = deepcopy(original)
    deleted.update(variant="drop_all_evidence", label="NEI")
    for article in deleted["articles"]:
        article["sentences"] = [s for s in article["sentences"] if s["original_index"] != 2]
        deleted["deleted_sentences"].append({"doc_id": article["doc_id"], "original_index": 2})
    for evidence in deleted["evidence"]:
        evidence["kept"] = False
    record["groups"].append(deleted)
    return samples_from_record(record)


class PairingTests(unittest.TestCase):
    def test_pairs_use_claim_identity_and_named_versions_not_adjacent_rows(self):
        a, b = paired_samples(10), paired_samples(11)
        extra = deepcopy(a[0])
        extra.variant = "drop_1_groups"
        samples = [a[1], b[0], extra, a[0], b[1]]
        self.assertEqual(build_evidence_pairs(samples), {3: 0, 1: 4})

    def test_original_nei_and_incomplete_claims_are_not_paired(self):
        a, b = paired_samples(10), paired_samples(11)
        b[1].variant = "original"
        self.assertEqual(build_evidence_pairs(a + [b[1]]), {0: 1})
        with self.assertRaisesRegex(ValueError, "at least one complete"):
            build_evidence_pairs(a[:1])

    def test_rejects_duplicate_versions_and_wrong_claim_or_labels(self):
        samples = paired_samples()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_evidence_pairs(samples + [deepcopy(samples[0])])
        for field, value, message in (("claim", "Another claim", "different claim"),
                                      ("label", "SUPPORT", "deleted NEI")):
            changed = deepcopy(samples)
            setattr(changed[1], field, value)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, message):
                build_evidence_pairs(changed)

    def test_rejects_changed_context_and_retained_evidence(self):
        changed = paired_samples()
        changed[1].articles[0].sentences[0] = "Changed background"
        with self.assertRaisesRegex(ValueError, "adds or changes"):
            build_evidence_pairs(changed)
        changed = paired_samples()
        changed[1].articles[0].sentences[2] = changed[0].articles[0].sentences[2]
        changed[1].articles[0].sentence_mask[2] = True
        with self.assertRaisesRegex(ValueError, "retains annotated evidence"):
            build_evidence_pairs(changed)
