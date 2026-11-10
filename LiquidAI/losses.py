"""Stance CE, masked sentence-evidence BCE and optional paired hinge loss."""

from __future__ import annotations

from .modeling import pool_options
from .config import LABEL_TO_ID


def pair_hinge_loss(sufficient_scores, insufficient_scores, label_id: int, margin: float):
    """Compare label-vs-NEI log odds using the ORIGINAL label for both inputs.

    Inputs are the three pooled stance logits, not full-vocabulary logits.
    Gradients flow through both versions; NEI is never an original pair label.
    """
    import torch.nn.functional as F

    nei_id = LABEL_TO_ID["NEI"]
    if label_id not in (LABEL_TO_ID["SUPPORT"], LABEL_TO_ID["CONTRADICT"]):
        raise ValueError("A pair's original label must be SUPPORT or CONTRADICT")
    sufficient_logp = F.log_softmax(sufficient_scores.float(), dim=-1)
    insufficient_logp = F.log_softmax(insufficient_scores.float(), dim=-1)
    sufficient_margin = sufficient_logp[label_id] - sufficient_logp[nei_id]
    insufficient_margin = insufficient_logp[label_id] - insufficient_logp[nei_id]
    return F.relu(margin - (sufficient_margin - insufficient_margin))


def joint_loss(vocab_logits, questions, codec, evidence_count, evidence_weight, pos_weight, label_weights=None):
    import torch
    import torch.nn.functional as F

    zero = vocab_logits.sum() * 0.0
    stance_loss, evidence_loss = zero, zero
    stance_rows = [i for i, q in enumerate(questions) if q.is_stance]
    evidence_rows = [i for i, q in enumerate(questions) if not q.is_stance]
    if stance_rows:
        scores = pool_options(vocab_logits[stance_rows], codec.stance_ids)
        target = torch.tensor([questions[i].target for i in stance_rows], device=scores.device)
        # 'sum' retains optional class weighting even with one stance per group.
        stance_loss = F.cross_entropy(scores, target, weight=label_weights, reduction="sum")
    if evidence_rows:
        scores = pool_options(vocab_logits[evidence_rows], codec.evidence_ids)
        target = torch.tensor([questions[i].target for i in evidence_rows], dtype=torch.float32, device=scores.device)
        # sigmoid(yes - no) equals softmax([yes, no])[0]. Work on logits for stability.
        evidence_loss = F.binary_cross_entropy_with_logits(
            scores[:, 0] - scores[:, 1], target,
            pos_weight=scores.new_tensor(pos_weight), reduction="sum",
        ) / evidence_count
    return stance_loss + evidence_weight * evidence_loss, stance_loss, evidence_loss
