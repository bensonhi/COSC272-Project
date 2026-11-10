"""B0: label-prior baseline. No model, no GPU.

Predicts the claim-level label distribution of the training split (original groups only,
so augmented NEI copies do not inflate the prior) for every test claim. Its argmax is the
majority class. Evidence probability is the training base rate of evidence sentences,
which is below 0.5, so no sentence is selected (evidence F1 = 0 by construction).
"""

import argparse
from collections import Counter

from common import LABELS, TRAIN_DIR, load_samples, run_baseline, sentence_keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    train = [s for s in load_samples(TRAIN_DIR) if s.variant == "original"]
    counts = Counter(s.label for s in train)
    prior = [counts[label] / len(train) for label in LABELS]
    positive = sum(sum(a.evidence_targets) for s in train for a in s.articles)
    valid = sum(sum(a.sentence_mask) for s in train for a in s.articles)
    rate = positive / valid

    def predict(sample):
        return prior, {key: rate for key in sentence_keys(sample)}

    run_baseline("b0_label_prior", predict, {
        "description": "training-split claim-level label prior; majority class = argmax",
        "prior": dict(zip(LABELS, prior)), "evidence_base_rate": rate,
    }, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
