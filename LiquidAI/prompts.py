"""Native d1 question definitions, shared by training and evaluation."""

STANCE_QUESTION = {
    "type": "choice",
    "instructions": (
        "Using only the claim and all provided articles together, determine "
        "whether the available evidence supports or contradicts the claim, or "
        "is insufficient. Empty sentence slots are unavailable text."
    ),
    "criteria": {
        "NEI": "The available articles provide insufficient evidence to support or contradict the claim.",
        "SUPPORT": "The available evidence supports the claim.",
        "CONTRADICT": "The available evidence contradicts the claim.",
    },
}

EVIDENCE_INSTRUCTIONS = (
    "Considering the claim and the full context of all articles, is the sentence "
    "with original_index={original_index} in article doc_id={doc_id} part of "
    "evidence supporting or contradicting the claim? Include sentences that "
    "jointly form evidence with other sentences. Mere topical relevance is not evidence."
)

def evidence_question(doc_id: int, original_index: int) -> dict:
    return {
        "type": "noul",
        "instructions": EVIDENCE_INSTRUCTIONS.format(doc_id=doc_id, original_index=original_index),
    }

