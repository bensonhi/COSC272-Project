#!/usr/bin/env python3
"""Audit/export all test metrics, and plot only final results for three runs.

Run with: python figures/plot_test_metrics.py
Dependencies: numpy, matplotlib. No model inference is performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"liquidai-matplotlib-{os.getuid()}"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent))
from LiquidAI.metrics import TestMetrics

RUNS = ("d1-augment-1e-4", "d1-pair", "d1-scifact")
DISPLAY_NAMES = {
    "d1-augment-1e-4": "d1-augment",
    "d1-pair": "d1-pair",
    "d1-scifact": "d1-original",
}
STAGES = ("epoch-1", "epoch-2", "epoch-3", "final")
LABELS = ("NEI", "SUPPORT", "CONTRADICT")
COLORS = ("#0072B2", "#D55E00", "#009E73")
SCORES = (
    ("label_accuracy", "Label accuracy", "higher"),
    ("label_macro_f1", "Label macro-F1", "higher"),
    ("sentence_level_evidence_f1", "Evidence micro-F1", "higher"),
    ("evidence_precision", "Evidence precision", "higher"),
    ("evidence_recall", "Evidence recall", "higher"),
    ("label_ece", "Label ECE", "lower"),
)


def dump_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def leaves(value, prefix=""):
    """Keep every JSON leaf, including nulls and configuration metadata."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def assert_matches(expected, actual, context):
    """Compare recomputed metrics recursively, without elapsed/config fields."""
    if isinstance(expected, dict):
        for key, value in expected.items():
            assert_matches(value, actual[key], f"{context}.{key}")
    elif isinstance(expected, list):
        assert len(expected) == len(actual), context
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_matches(left, right, f"{context}[{index}]")
    elif isinstance(expected, float):
        assert math.isclose(expected, actual, rel_tol=1e-10, abs_tol=1e-12), context
    else:
        assert expected == actual, context


def audit_predictions(path, metrics):
    scorer = TestMetrics(metrics["evidence"]["threshold"], metrics["calibration"]["num_bins"])
    identities = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sentences = [s for a in record["articles"] for s in a["sentences"] if s["available"]]
            scorer.update(
                record["gold_label"],
                [record["label_probabilities"][label] for label in LABELS],
                [s["gold_evidence"] for s in sentences],
                [s["evidence_probability"] for s in sentences],
            )
            identity = {
                "claim": record["claim"], "gold_label": record["gold_label"],
                "articles": [
                    {"doc_id": a["doc_id"], "title": a["title"], "sentences": [
                        {key: s[key] for key in ("original_index", "sentence", "available", "gold_evidence")}
                        for s in a["sentences"]
                    ]} for a in record["articles"]
                ],
            }
            key = record["sample_id"]
            assert key not in identities, f"Duplicate sample: {path}: {key}"
            identities[key] = hashlib.sha256(
                json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
    assert_matches(scorer.result(), metrics, str(path))
    return identities


def collect(root):
    records, reference_inputs, reference_config = [], None, None
    audit = []
    config_keys = (
        "base_model", "dtype", "split", "test_dir", "summary", "evidence_threshold", "ece_bins",
        "max_length", "question_batch_size", "stance_question", "evidence_instructions",
        "gold_evidence_used_as_input", "evidence_f1_conditioned_on_predicted_label",
    )
    for run in RUNS:
        run_dir = root / "outputs" / run
        found = set(run_dir.glob("**/test/metrics.json"))
        expected = {
            run_dir / ("test" if stage == "final" else f"{stage}/test") / "metrics.json"
            for stage in STAGES
        }
        assert found == expected, f"Unexpected or missing test metric files in {run_dir}: {found ^ expected}"
        epoch_log = {
            row["epoch"]: row["test"]
            for row in (json.loads(line) for line in (run_dir / "epoch_metrics.jsonl").read_text().splitlines())
        }
        for stage in STAGES:
            path = run_dir / ("test" if stage == "final" else f"{stage}/test") / "metrics.json"
            metrics = json.loads(path.read_text())
            config = metrics["evaluation"]
            assert tuple(metrics["label_order"]) == LABELS
            assert_matches(config, json.loads(path.with_name("test_config.json").read_text()), str(path))
            inputs = audit_predictions(path.with_name("predictions.jsonl"), metrics)
            comparable = {key: config[key] for key in config_keys}
            if reference_inputs is None:
                reference_inputs, reference_config = inputs, comparable
            assert inputs == reference_inputs, f"Test inputs differ: {path}"
            assert comparable == reference_config, f"Evaluation settings differ: {path}"
            epoch = None if stage == "final" else int(stage.split("-")[1])
            if epoch is not None:
                assert_matches(metrics, epoch_log[epoch], f"epoch_metrics.jsonl: {run}/{stage}")
            meta = {
                "run": run, "display_name": DISPLAY_NAMES[run], "stage": stage, "epoch": epoch,
                "global_step": config.get("global_step"), "source": str(path.relative_to(root)),
            }
            records.append({**meta, "metrics": metrics})
            audit.append({
                **meta, "prediction_records": len(inputs), "recomputed_metrics_match": True,
                "same_test_inputs": True, "same_evaluation_settings": True,
                "test_config_matches": True, "epoch_log_matches": True if epoch is not None else None,
            })
        final, epoch3 = records[-1]["metrics"], records[-2]["metrics"]
        assert_matches(
            {k: v for k, v in epoch3.items() if k not in ("evaluation", "elapsed_seconds")},
            final, f"{run}: epoch-3 versus final",
        )
    return records, audit


def export_tables(out, records, audit):
    summary, per_label, confusion, bins, flat = [], [], [], [], []
    for record in records:
        meta = {k: v for k, v in record.items() if k != "metrics"}
        metrics = record["metrics"]
        row = {**meta, **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}}
        row.update({f"evidence_{k}": v for k, v in metrics["evidence"].items()})
        row.update({
            "device": metrics["evaluation"]["device"],
            "adapter_dir": metrics["evaluation"]["adapter_dir"],
            "ece_num_bins": metrics["calibration"]["num_bins"],
            "ece_type": metrics["calibration"]["type"],
            "ece_binning": metrics["calibration"]["binning"],
        })
        summary.append(row)
        for label, values in metrics["per_label"].items():
            per_label.append({**meta, "label": label, **values})
        for i, truth in enumerate(LABELS):
            for j, prediction in enumerate(LABELS):
                count = metrics["label_confusion_matrix"][i][j]
                confusion.append({
                    **meta, "gold_label": truth, "predicted_label": prediction, "count": count,
                    "row_fraction": count / metrics["per_label"][truth]["support"],
                })
        for index, values in enumerate(metrics["calibration"]["bins"]):
            bins.append({**meta, "bin_index": index, **values})
        for key, value in leaves(metrics):
            flat.append({
                **meta, "metric_path": key, "value": value,
                "json_value": json.dumps(value, ensure_ascii=False), "json_type": type(value).__name__,
            })
    dump_csv(out / "test_metrics_summary.csv", summary)
    dump_csv(out / "test_metrics_final.csv", [row for row in summary if row["stage"] == "final"])
    dump_csv(out / "test_per_label.csv", per_label)
    dump_csv(out / "test_confusion_matrices.csv", confusion)
    dump_csv(out / "test_calibration_bins.csv", bins)
    dump_csv(out / "test_metrics_all_fields.csv", flat)
    (out / "test_metrics_all.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8",
    )
    (out / "verification.json").write_text(
        json.dumps({
            "files_checked": len(records), "prediction_records_checked": sum(a["prediction_records"] for a in audit),
            "all_final_scores_match_epoch_3": True, "checks": audit,
        }, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def save(fig, out, name):
    for extension in ("png", "pdf"):
        fig.savefig(out / f"{name}.{extension}", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.labelcolor": "#334155", "text.color": "#0f172a",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#cbd5e1",
        "grid.alpha": 0.25, "grid.color": "#94a3b8", "axes.axisbelow": True,
        "pdf.fonttype": 42,
    })


def decorate(fig, title, subtitle, bottom=0.06):
    fig.suptitle(title, x=0.05, y=0.99, ha="left", fontsize=19, fontweight="bold")
    subtitle_y = 0.99 - 32 / (fig.get_figheight() * 72)
    fig.text(0.05, subtitle_y, subtitle, ha="left", va="top", fontsize=10, color="#475569")
    fig.subplots_adjust(top=0.87, bottom=bottom, left=0.075, right=0.97, hspace=0.46, wspace=0.31)


def plot_final(out, rows):
    final = [row for row in rows if row["stage"] == "final"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (key, title, direction) in zip(axes.flat, SCORES):
        values = [100 * row[key] for row in final]
        best = max(values) if direction == "higher" else min(values)
        bars = ax.barh(np.arange(3), values, color=COLORS, height=0.55)
        ax.set_yticks(np.arange(3), [DISPLAY_NAMES[run] for run in RUNS])
        ax.invert_yaxis()
        ax.set_xlim(0, 108 if direction == "higher" else 19)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Percent" if key != "label_ece" else "ECE × 100")
        ax.grid(axis="x")
        for bar, value in zip(bars, values):
            ax.text(value + (1 if direction == "higher" else 0.25), bar.get_y() + bar.get_height()/2,
                    f"{value:.2f}", va="center", fontweight="bold", fontsize=10,
                    color="#d62728" if math.isclose(value, best, rel_tol=0, abs_tol=1e-12) else "#0f172a")
    decorate(fig, "Final test metrics", "Final evaluation only · 333 samples · 3,515 valid sentences")
    fig.subplots_adjust(left=0.14, wspace=0.85)
    save(fig, out, "01_final_metrics")


def plot_per_label(out, records):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    labels = [DISPLAY_NAMES[r['run']] for r in records]
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        values = np.array([[r["metrics"]["per_label"][label][metric] * 100 for label in LABELS] for r in records])
        image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
        ax.set_title(metric.capitalize())
        ax.set_xticks(range(3), LABELS, rotation=18)
        ax.set_yticks(range(len(records)), labels if ax is axes[0] else [""] * len(records))
        for i in range(len(records)):
            for j in range(3):
                ax.text(j, i, f"{values[i,j]:.2f}", ha="center", va="center",
                        color="white" if values[i,j] > 65 else "#0f172a", fontsize=9)
    decorate(fig, "Final per-label precision, recall and F1", "Final evaluation only · Class support: NEI = 126, SUPPORT = 137, CONTRADICT = 70")
    fig.subplots_adjust(left=0.17, right=0.90, bottom=0.19, top=0.80, wspace=0.12)
    cax = fig.add_axes([0.92, 0.20, 0.015, 0.59])
    fig.colorbar(image, cax=cax, label="Percent (higher is better)")
    save(fig, out, "03_per_label_metrics")


def plot_confusion(out, records):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    for j, run in enumerate(RUNS):
        record = next(r for r in records if r["run"] == run)
        values = np.array(record["metrics"]["label_confusion_matrix"])
        normalized = values / values.sum(axis=1, keepdims=True) * 100
        ax = axes[j]
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{values[a,b]}\n{normalized[a,b]:.1f}%", ha="center", va="center",
                        color="white" if normalized[a,b] > 55 else "#0f172a", fontsize=10)
        ax.set_title(DISPLAY_NAMES[run], fontsize=11)
        ax.set_xticks(range(3), LABELS, rotation=15, fontsize=8)
        ax.set_yticks(range(3), LABELS, fontsize=8)
        if j == 0:
            ax.set_ylabel("Gold label")
        ax.set_xlabel("Predicted label")
    decorate(fig, "Final label confusion matrices", "Final evaluation only · Rows = gold · Columns = predicted · Cell = count and percent within gold label")
    fig.subplots_adjust(top=0.80, bottom=0.18, right=0.90, wspace=0.50)
    cax = fig.add_axes([0.93, 0.24, 0.014, 0.49])
    fig.colorbar(image, cax=cax, label="Percent within gold label")
    save(fig, out, "04_confusion_matrices")


def plot_evidence_counts(out, rows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = ("True positives", "False positives", "False negatives", "True negatives")
    for ax, key, title in zip(axes.flat, ("tp", "fp", "fn", "tn"), titles):
        values = [next(r for r in rows if r["run"] == run)[f"evidence_{key}"] for run in RUNS]
        bars = ax.bar(range(3), values, width=0.55, color=COLORS)
        ax.bar_label(bars, fontsize=10, padding=3)
        ax.set_title(title, loc="left")
        ax.set_xticks(range(3), [DISPLAY_NAMES[run] for run in RUNS])
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1]*1.14)
        ax.set_ylabel("Sentences")
        ax.grid(axis="y")
    decorate(fig, "Final sentence evidence decisions", "Final evaluation only · 528 positive + 2,987 negative sentences · Evidence threshold = 0.5 · Separate count scales", bottom=0.10)
    save(fig, out, "05_evidence_counts")


def plot_reliability(out, records):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    for i, (run, color) in enumerate(zip(RUNS, COLORS)):
        record = next(r for r in records if r["run"] == run)
        metrics = record["metrics"]
        populated = [b for b in metrics["calibration"]["bins"] if b["count"]]
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "--", color="#94a3b8", lw=1.2)
        ax.scatter([b["mean_confidence"] for b in populated], [b["accuracy"] for b in populated],
                   s=[18+1.4*b["count"] for b in populated], color=color, alpha=0.7,
                   edgecolor="white", linewidth=0.6, zorder=3)
        for b in populated:
            ax.plot([b["mean_confidence"]]*2, [b["mean_confidence"], b["accuracy"]], color=color, alpha=0.45)
        ax.set(xlim=(0, 1.03), ylim=(-0.03, 1.03))
        ax.set_title(f"{DISPLAY_NAMES[run]} | ECE × 100 = {100*metrics['label_ece']:.2f}", fontsize=10)
        ax.grid()
        ax.set_xlabel("Mean confidence in bin")
        if i == 0:
            ax.set_ylabel("Accuracy in bin")
    decorate(fig, "Final label reliability diagrams", "Final evaluation only · 10 equal-width bins · Marker area grows with sample count · Diagonal = perfect calibration")
    fig.subplots_adjust(left=0.08, top=0.80, bottom=0.15, wspace=0.25)
    save(fig, out, "06_reliability_diagrams")


def plot_calibration_bins(out, records):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, key, title, cmap, fmt in zip(
        axes, ("count", "ece_contribution"), ("Samples per confidence bin", "Contribution to ECE × 100"),
        ("Blues", "OrRd"), (".0f", ".2f"),
    ):
        scale = 100 if key == "ece_contribution" else 1
        values = np.array([[b[key]*scale for b in r["metrics"]["calibration"]["bins"]] for r in records])
        ax.imshow(values, cmap=cmap, aspect="auto", norm=Normalize(vmin=0, vmax=values.max()))
        ax.set_title(title)
        ax.set_xticks(range(10), [f"{x/10:.1f}–{(x+1)/10:.1f}" for x in range(10)], rotation=65, fontsize=8)
        ax.set_yticks(range(len(records)), [DISPLAY_NAMES[r['run']] for r in records] if ax is axes[0] else [""]*len(records))
        for i in range(len(records)):
            for j in range(10):
                ax.text(j, i, format(values[i,j], fmt), ha="center", va="center", fontsize=8,
                        color="white" if values[i,j] > values.max()*0.55 else "#0f172a")
        ax.set_xlabel("Confidence interval (last bin includes 1.0)")
    decorate(fig, "Final calibration bin counts and error contributions", "Final evaluation only · Empty bins have zero ECE contribution; their accuracy and mean confidence are null")
    fig.subplots_adjust(left=0.15, top=0.80, bottom=0.27, wspace=0.09)
    save(fig, out, "07_calibration_bins")


def plot_runtime(out, rows):
    fig, ax = plt.subplots(figsize=(11, 6))
    selected = [next(r for r in rows if r["run"] == run) for run in RUNS]
    bars = ax.bar(range(3), [r["elapsed_seconds"] for r in selected], width=0.55, color=COLORS)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=11)
    ax.set_xticks(range(3), [f"{DISPLAY_NAMES[r['run']]}\n({r['device']})" for r in selected])
    ax.set_ylabel("Elapsed seconds")
    ax.set_ylim(0, max(r['elapsed_seconds'] for r in selected) * 1.23)
    ax.grid(axis="y")
    decorate(fig, "Final test evaluation time", "Final evaluation only · Different CUDA devices; timings do not establish an intrinsic model speed ranking", bottom=0.17)
    save(fig, out, "08_evaluation_runtime")


def write_report(out, rows):
    final = {r["run"]: r for r in rows if r["stage"] == "final"}
    lines = [
        "LiquidAI：三个实验的 final test metric 比较", "="*60, "",
        "显示名称与数据来源", "-----------",
        *[f"{DISPLAY_NAMES[run]}：outputs/{run}/test/metrics.json" for run in RUNS], "",
        "范围与核验", "-----------",
        "所有图表仅展示三个实验的 final 结果，取自各自的 test/metrics.json，不包含各 epoch 的结果。",
        "完整原始数据表保留当前三个实验的全部 12 份评估，便于追溯；图表仅使用其中 3 份 final 记录。",
        "读取当前三个源目录的 epoch-1/2/3/test 与 test（final），共 12 份 metrics.json。",
        "CSV/JSON 的 run 字段保留真实目录名，display_name 字段保存图表显示名称，source 字段记录具体来源。",
        "逐条读取全部 3,996 条预测记录，用项目原有 TestMetrics 重算所有指标（含类别、混淆矩阵和校准分箱），均与原文件一致。",
        "已核验 12 份评估的 sample_id、输入文本、gold label、句子 mask 和 gold evidence 一致；评估参数一致。",
        "每份评估：333 条样本，3,515 个有效句子；正证据句 528，负证据句 2,987。",
        "类别支持数：NEI=126，SUPPORT=137，CONTRADICT=70。",
        "证据阈值 0.5；证据指标为正类 micro 指标，不依赖预测 stance；ECE 为 10 个等宽分箱的 top-label ECE。",
        "final 固定取各目录的 test/metrics.json。", "",
        "Final 对比（以下得分均为原始值 ×100；ECE 越低越好，其余越高越好）", "-----------",
        f"{'run':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Evidence-F1':>12} {'Ev-Prec.':>10} {'Ev-Recall':>10} {'ECE':>9}",
    ]
    for run in RUNS:
        row = final[run]
        lines.append(f"{DISPLAY_NAMES[run]:<20} " + " ".join(f"{100*row[key]:10.2f}" for key, _, _ in SCORES))
    lines += ["", "指标最优值（主指标图中以红色数字标出，柱子保持各实验颜色）", "-----------"]
    for key, title, direction in SCORES:
        best = (max if direction == "higher" else min)(final[run][key] for run in RUNS)
        winners = [DISPLAY_NAMES[run] for run in RUNS if math.isclose(final[run][key], best, rel_tol=0, abs_tol=1e-12)]
        lines.append(f"{title}：{'、'.join(winners)}，{100*best:.2f}")
    lines += ["", "Final 证据计数与记录耗时", "-----------"]
    for run in RUNS:
        row = final[run]
        lines.append(
            f"{DISPLAY_NAMES[run]}：TP={row['evidence_tp']}，FP={row['evidence_fp']}，"
            f"FN={row['evidence_fn']}，TN={row['evidence_tn']}；"
            f"评估耗时={row['elapsed_seconds']:.2f}s，设备={row['device']}。"
        )
    lines += ["",
        "每个实验仅有一条训练轨迹，未提供多个随机种子的重复运行；这些差异是观察值，不能据此认定统计显著。",
        "三者使用不同 CUDA 设备，硬件/负载未受控，记录耗时不能直接解释为模型本身的速度差异。",
    ]
    lines += ["", "文件说明", "-----------",
        "01_final_metrics.png/pdf        final 六项主要指标的水平条形图（条形从零开始）",
        "03_per_label_metrics.png/pdf    三个 final 的类别 precision/recall/F1 热图",
        "04_confusion_matrices.png/pdf   三个 final 的标签混淆矩阵（计数 + 按真实类别归一化百分比）",
        "05_evidence_counts.png/pdf      三个 final 的证据 TP/FP/FN/TN 计数",
        "06_reliability_diagrams.png/pdf 三个 final 的校准图（点面积随分箱样本数增大）",
        "07_calibration_bins.png/pdf     三个 final 的分箱样本数与 ECE 贡献值热图",
        "08_evaluation_runtime.png/pdf   三个 final 的评估耗时（标明设备）",
        "test_metrics_summary.csv       全部 12 份评估的主要指标、证据计数、样本量、耗时与配置",
        "test_metrics_final.csv         3 份 final 主指标",
        "test_per_label.csv             36 行类别指标，包含 support",
        "test_confusion_matrices.csv    108 个混淆矩阵单元格，包含计数和行比例",
        "test_calibration_bins.csv      120 个分箱的边界、计数、accuracy、mean_confidence 与 ECE 贡献",
        "test_metrics_all_fields.csv    所有原始 JSON 叶字段的长表（含配置元数据；json_value 保留类型及 null）",
        "test_metrics_all.json          保留完整原始指标对象及源路径的汇总文件",
        "verification.json             逐文件核验结果",
        "plot_test_metrics.py           可复现脚本（依赖 numpy、matplotlib）", "",
        "单位：CSV/JSON 中得分保持原始 0–1 精度；图表中主要得分与 ECE 乘以 100；计数与耗时保持原单位。",
        "图中文字使用英文以保证通用字体可读，中文说明位于本文件。所有图同时提供 PNG 和矢量 PDF。", "",
        "重新生成：", "  cd /jumbo/yaoqingyang/xuyh/courses/CS272/project/LiquidAI",
        "  python figures/plot_test_metrics.py", "",
    ]
    (out / "test_metrics_comparison.txt").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records, audit = collect(args.project_root.resolve())
    rows = export_tables(out, records, audit)
    final_records = [r for r in records if r["stage"] == "final"]
    final_rows = [r for r in rows if r["stage"] == "final"]
    assert len(final_records) == len(final_rows) == len(RUNS)
    style()
    plot_final(out, final_rows)
    plot_per_label(out, final_records)
    plot_confusion(out, final_records)
    plot_evidence_counts(out, final_rows)
    plot_reliability(out, final_records)
    plot_calibration_bins(out, final_records)
    plot_runtime(out, final_rows)
    write_report(out, rows)
    for extension in ("png", "pdf"):
        (out / f"02_epoch_trends.{extension}").unlink(missing_ok=True)
    print(f"Verified {len(records)} test metric files and {sum(a['prediction_records'] for a in audit):,} predictions.")
    print(f"Saved 7 final-only charts (PNG + PDF), 6 CSV tables, full JSON, verification, report and script to {out}")


if __name__ == "__main__":
    main()
