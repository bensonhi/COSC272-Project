"""Draw two figures for the train and dev claim-verification data.

Figure 1 is the claim-label distribution, with train and dev as separate pie
charts. Figure 2 places the evidence-abstract counts and the rationale-length
counts side by side.

Figure 2's left panel counts abstracts that have gold evidence. The count is
the number of keys in ``evidence``. An empty ``evidence`` object is 0, which
is the NEI case. ``cited_doc_ids`` is not used, and rationale groups inside
one abstract are not counted as extra abstracts.

Figure 2's right panel counts sentence indices inside each rationale object.
A rationale is one ``{"sentences": [...], "label": ...}`` entry.
``len(sentences)`` is not the length of the cited abstract.

Requires matplotlib. From the project virtualenv:

    .venv/bin/python plot_dataset_figures.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"

TRAIN_COLOR = "#245B8A"
DEV_COLOR = "#E0923A"
INK = "#1F2933"
GRID = "#E4E7EB"
LABELS = ("SUPPORT", "CONTRADICT", "NEI")
LABEL_COLORS = {
    "SUPPORT": "#245B8A",
    "CONTRADICT": "#C4493A",
    "NEI": "#8A94A0",
}


def load_jsonl(path):
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def claim_label(row):
    found = set()
    for annotations in (row.get("evidence") or {}).values():
        for annotation in annotations:
            found.add(annotation["label"])
    if not found:
        return "NEI"
    if found == {"SUPPORT"} or found == {"CONTRADICT"}:
        return next(iter(found))
    raise ValueError(f"claim {row['id']} has mixed evidence labels: {sorted(found)}")


def evidence_doc_count(row):
    """Abstracts with gold evidence. Not the length of cited_doc_ids."""
    return len(row.get("evidence") or {})


def rationale_lengths(rows):
    """Indices listed in each rationale, not sentences in the abstract."""
    lengths = []
    for row in rows:
        for annotations in (row.get("evidence") or {}).values():
            for annotation in annotations:
                lengths.append(len(annotation["sentences"]))
    return lengths


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9AA5B1")
    ax.spines["bottom"].set_color("#9AA5B1")
    ax.tick_params(colors=INK, length=0)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")


def label_vertical_bars(ax, bars, texts):
    for bar, text in zip(bars, texts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
            linespacing=1.15,
        )


def save_figure(fig, name):
    FIGURE_DIR.mkdir(exist_ok=True)
    png_path = FIGURE_DIR / f"{name}.png"
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path


def plot_label_pies(train_rows, dev_rows):
    splits = (("Train", train_rows), ("Dev", dev_rows))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.8))
    for ax, (name, rows) in zip(axes, splits):
        counts = [sum(claim_label(row) == label for row in rows) for label in LABELS]
        total = len(rows)
        colors = [LABEL_COLORS[label] for label in LABELS]
        wedges, _, autotexts = ax.pie(
            counts,
            colors=colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.62, "edgecolor": "white", "linewidth": 1.5},
            autopct=lambda pct: f"{pct:.1f}%",
            pctdistance=0.72,
        )
        for text in autotexts:
            text.set_color("white")
            text.set_fontsize(10)
            text.set_fontweight("medium")
        ax.legend(
            wedges,
            [f"{label}  {count}" for label, count in zip(LABELS, counts)],
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=1,
        )
        ax.set_title(f"{name}  (n={total})", color=INK, pad=8)
        ax.set_aspect("equal")
    fig.tight_layout()
    return save_figure(fig, "fig1_claim_labels")


def plot_evidence_and_rationales(train_rows, dev_rows):
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8))
    left, right = axes

    buckets = ("0", "1", "2", "3+")

    def bucket_counts(rows):
        counts = Counter()
        for row in rows:
            count = evidence_doc_count(row)
            counts["3+" if count >= 3 else str(count)] += 1
        return [counts[bucket] for bucket in buckets]

    train_counts = bucket_counts(train_rows)
    dev_counts = bucket_counts(dev_rows)
    x = np.arange(len(buckets))
    width = 0.36
    train_bars = left.bar(
        x - width / 2, train_counts, width, label="Train", color=TRAIN_COLOR, zorder=2
    )
    dev_bars = left.bar(
        x + width / 2, dev_counts, width, label="Dev", color=DEV_COLOR, zorder=2
    )
    left.set_ylim(0, max(train_counts + dev_counts) * 1.18)
    label_vertical_bars(left, train_bars, [str(count) for count in train_counts])
    label_vertical_bars(left, dev_bars, [str(count) for count in dev_counts])
    left.set_xticks(x, buckets)
    left.set_xlabel("Evidence abstracts per claim")
    left.set_ylabel("Claims")
    left.legend(frameon=False)
    style_axes(left)

    train_lengths = Counter(rationale_lengths(train_rows))
    dev_lengths = Counter(rationale_lengths(dev_rows))
    sentence_counts = sorted(set(train_lengths) | set(dev_lengths))
    train_rationale_counts = [train_lengths[count] for count in sentence_counts]
    dev_rationale_counts = [dev_lengths[count] for count in sentence_counts]
    totals = [train + dev for train, dev in zip(train_rationale_counts, dev_rationale_counts)]
    y = np.arange(len(sentence_counts))
    right.barh(y, train_rationale_counts, color=TRAIN_COLOR, label="Train", zorder=2)
    right.barh(
        y,
        dev_rationale_counts,
        left=train_rationale_counts,
        color=DEV_COLOR,
        label="Dev",
        zorder=2,
    )
    pad = max(totals) * 0.015
    for index, total in enumerate(totals):
        right.text(total + pad, index, f"{total:,}", va="center", ha="left", fontsize=10, color=INK)
    right.set_yticks(y, [str(count) for count in sentence_counts])
    right.set_xlabel("Rationales")
    right.set_ylabel("Sentences per rationale")
    right.set_xlim(0, max(totals) * 1.18)
    right.invert_yaxis()
    right.legend(frameon=False, loc="lower right")
    style_axes(right)

    fig.tight_layout()
    return save_figure(fig, "fig2_evidence_and_rationales")


def main():
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    train_rows = load_jsonl(DATA_DIR / "claims_train.jsonl")
    dev_rows = load_jsonl(DATA_DIR / "claims_dev.jsonl")

    print(plot_label_pies(train_rows, dev_rows))
    print(plot_evidence_and_rationales(train_rows, dev_rows))


if __name__ == "__main__":
    main()
