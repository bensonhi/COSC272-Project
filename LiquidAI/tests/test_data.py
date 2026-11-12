"""Data schemas, evidence targets and original-index alignment."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from .. import data
from .fixtures import fixture


class DatasetTests(unittest.TestCase):
    def test_original_indices_masks_and_multiple_articles(self):
        sample = data.samples_from_record(fixture())[0]
        first, second = sample.articles
        self.assertEqual(first.sentences, ["Background.", "", "Evidence.", ""])
        self.assertEqual(first.evidence_targets, [0, 0, 1, 0])
        self.assertEqual(first.sentence_mask, [True, False, True, False])
        self.assertEqual(second.evidence_targets, [0, 0, 1])
        self.assertEqual(second.sentence_mask, [True, False, True])
        self.assertEqual(sample.label, "SUPPORT")
        state = json.dumps(sample.state())
        self.assertNotIn("DELETED SECRET", state)
        self.assertNotIn("evidence_label", state)
        self.assertNotIn("new_index", state)
        self.assertNotIn("SUPPORT", state)
        self.assertEqual(sample.state()["articles"][0]["sentences"]["2"], "Evidence.")

    def test_nei_with_all_evidence_removed(self):
        record = fixture()
        group = record["groups"][0]
        group["label"] = "NEI"
        for article in group["articles"]:
            article["sentences"] = [s for s in article["sentences"] if s["original_index"] != 2]
        for evidence in group["evidence"]:
            evidence["kept"] = False
        sample = data.samples_from_record(record)[0]
        self.assertEqual(sum(sum(a.evidence_targets) for a in sample.articles), 0)
        self.assertEqual(sample.articles[0].sentence_mask, [True, False, False, False])

    def test_original_variant_recovers_trailing_slots_without_leaking_text(self):
        record = fixture()
        variant = deepcopy(record["groups"][0])
        original = record["groups"][0]
        original["articles"][0]["sentences"].append({"original_index": 7, "sentence": "ONLY IN ORIGINAL"})
        record["groups"].append(variant)
        samples = data.samples_from_record(record)
        self.assertEqual(len(samples[1].articles[0].sentences), 8)
        self.assertEqual(samples[1].articles[0].sentences[7], "")
        self.assertNotIn("ONLY IN ORIGINAL", json.dumps(samples[1].state()))

    def test_identical_duplicate_article_is_not_counted_twice(self):
        record = fixture()
        articles = record["groups"][0]["articles"]
        articles.append(deepcopy(articles[0]))
        self.assertEqual(len(data.samples_from_record(record)[0].articles), 2)
        articles[-1]["title"] = "Different title"
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            data.samples_from_record(record)

    def test_kept_evidence_requires_visible_sentence(self):
        record = fixture()
        record["groups"][0]["evidence"][1]["kept"] = True
        with self.assertRaisesRegex(ValueError, "conflicts with available text"):
            data.samples_from_record(record)

    def test_directory_loading_expands_groups(self):
        record = fixture()
        record["groups"].append(deepcopy(record["groups"][0]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "10.json").write_text(json.dumps(record))
            self.assertEqual(len(data.load_samples(path)), 2)
            self.assertEqual(len(data.load_samples(path, max_groups=1)), 1)
            self.assertEqual(data.dataset_summary(data.load_samples(path))["positive_sentences"], 4)

    def test_flat_test_record_without_kept(self):
        record = fixture()
        group = record.pop("groups")[0]
        group["evidence"] = [
            {k: v for k, v in row.items() if k != "kept"}
            for row in group["evidence"] if row["kept"]
        ]
        record.update(group)
        sample = data.samples_from_record(record)[0]
        self.assertEqual(sample.group_index, 0)
        self.assertEqual(sample.variant, "original")
        self.assertEqual(sample.articles[0].evidence_targets, [0, 0, 1, 0])
        self.assertEqual(sample.articles[1].evidence_targets, [0, 0, 1])

    def test_original_filter_precedes_limit_and_preserves_group_indices(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for claim_id in (10, 11, 12):
                record = fixture()
                record["id"] = claim_id
                record["groups"][0]["variant"] = "original" if claim_id == 12 else "drop_1_groups"
                if claim_id == 11:
                    original = deepcopy(record["groups"][0])
                    original["variant"] = "original"
                    record["groups"].append(original)
                (path / f"{claim_id}.json").write_text(json.dumps(record))
            self.assertEqual(len(data.load_samples(path)), 4)
            originals = data.load_samples(path, original_only=True)
            self.assertEqual([(s.claim_id, s.group_index) for s in originals], [(11, 1), (12, 0)])
            limited = data.load_samples(path, max_groups=1, original_only=True)
            self.assertEqual([(s.claim_id, s.group_index) for s in limited], [(11, 1)])

    def test_original_filter_reports_empty_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "10.json").write_text(json.dumps(fixture()))
            with self.assertRaisesRegex(ValueError, "No groups with variant='original'"):
                data.load_samples(path, original_only=True)

    def test_gold_label_and_evidence_do_not_enter_state(self):
        record = fixture()
        before = data.samples_from_record(record)[0].state()
        record["groups"][0]["label"] = "NEI"
        record["groups"][0]["evidence"] = []
        after = data.samples_from_record(record)[0].state()
        self.assertEqual(before, after)
