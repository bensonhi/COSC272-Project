"""Synthetic multi-article examples with original-index gaps."""


def fixture():
    return {
        "id": 10,
        "groups": [{
            "claim": "The intervention reduces the outcome.",
            "label": "SUPPORT",
            "articles": [
                {"doc_id": 100, "title": "First article", "sentences": [
                    {"original_index": 0, "new_index": 999, "sentence": "Background."},
                    {"original_index": 2, "new_index": 0, "sentence": "Evidence."},
                ]},
                {"doc_id": 200, "title": "Second article", "sentences": [
                    {"original_index": 0, "new_index": 0, "sentence": "Other background."},
                    {"original_index": 2, "new_index": 1, "sentence": "Other evidence."},
                ]},
            ],
            "deleted_sentences": [
                {"doc_id": 100, "original_index": 1, "sentence": "DELETED SECRET ONE"},
                {"doc_id": 100, "original_index": 3, "sentence": "DELETED SECRET TWO"},
            ],
            "evidence": [
                {"doc_id": 100, "original_index": 2, "kept": True, "evidence_label": "SUPPORT"},
                {"doc_id": 100, "original_index": 3, "kept": False, "evidence_label": "SUPPORT"},
                {"doc_id": 200, "original_index": 2, "kept": True, "evidence_label": "SUPPORT_more"},
            ],
        }],
    }
