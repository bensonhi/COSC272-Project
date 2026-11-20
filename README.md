# CS272 · Evidence-Sensitive Scientific Claim Verification

Can a model tell when a scientific abstract no longer contains enough evidence to judge a claim? This project studies that question on SciFact. Given a claim and one or more cited abstracts, the model predicts `SUPPORT`, `CONTRADICT`, or `NEI` (not enough information). It also scores each sentence so readers can see which passages it identifies as evidence.

## Experiments and findings

We fine-tune LiquidAI/d1-3B with LoRA under three conditions. The first uses original abstracts. The second adds versions with annotated evidence removed. The third adds a paired loss that encourages the model to favor the original verdict more strongly when the evidence is present. Comparisons include a label-prior baseline, zero-shot d1, DeBERTa natural language inference, MultiVerS, and Qwen.

On the 333-claim test set, the original and augmented d1 runs at seed 42 scored similarly on claim classification (macro-F1 of 0.851 and 0.850, respectively). The difference was clearer when we edited the abstracts. The table shows how often each model predicted `NEI` on the 207 test claims with annotated evidence. Both edits left visible gaps in the text.

| d1 training condition | All annotated evidence removed | Random background sentences removed |
| --- | ---: | ---: |
| Original abstracts | 35.7% | 6.8% |
| Evidence-removal augmentation | 80.2% | 15.9% |

Augmentation made d1 more likely to abstain after either edit, with a larger increase when the evidence itself disappeared. The full comparison also reports sentence-level evidence F1, calibration, latency, and a version of the deletion experiment that closes the gaps. [Baseline and stress-test results](baselines/results/summary.md) · [Three-way d1 comparison](LiquidAI/figures/test_metrics_final.csv)

## Data and evaluation scope

The project uses a custom split of the labeled SciFact training and development data. The prepared directories contain 698 training claims, 78 validation claims, and 333 test claims. The split is by claim ID. Two test claims repeat training claim text exactly, and 205 test claims cite at least one article also present in training. The reported scores therefore cover a mix of familiar and unseen articles.

## Running the project

For training, use a GPU environment with PyTorch, Transformers, PEFT, and Accelerate. This command fine-tunes on the augmented examples and evaluates the resulting adapter. Use `--original` for the original-abstract condition or `--loss pair` for the paired objective. Choose a separate output directory for each run.

```bash
python3 -m LiquidAI --mode train_test --output-dir runs/d1-augmented
```

- [Baseline guide](baselines/README.md) lists model-specific setup and evaluation commands.
- [Error analysis](baselines/results/error_analysis.md) examines where the models disagree or miss evidence.
- [Interactive demo](demo/README.md) lets you compare predictions, remove individual sentences, and see how the verdict changes.
