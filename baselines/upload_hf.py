"""Upload trained weights to the team's Hugging Face repo (they are too large for git).

    HF_HOME=/scratch/f008k3y/cosc272/hf_cache python upload_hf.py [--repo COSC272-Project/COSC272-Project]

Uploads, when present: the d1-3B LoRA adapters (augmented, counterfactual and original-data SFT, final
adapter only, no intermediate checkpoints) and the fine-tuned MultiVerS state dict, each in its own
folder, plus a README. Teammates fetch one folder with, e.g.

    hf download COSC272-Project/COSC272-Project --include "d1-augmented-sft-seed42/*" --local-dir hf
    python -m LiquidAI --mode test --adapter-dir hf/d1-augmented-sft-seed42
"""

import argparse

from huggingface_hub import HfApi

from common import ROOT, SCRATCH

OUT = SCRATCH / "outputs"
ARTIFACTS = {
    "d1-augmented-sft-seed42": OUT / "d1-augmented-sft" / "final",
    "d1-augmented-sft-seed43": OUT / "d1-augmented-sft-seed43" / "final",
    **{f"d1-cf-all-seed{s}": OUT / f"d1-cf-all-seed{s}" / "final" for s in (42, 43)},
    **{f"d1-cf-evidence-only-seed{s}": OUT / f"d1-cf-evidence_flip+evidence_weaken-seed{s}" / "final"
       for s in (42, 43, 44)},
    **{f"d1-original-sft-seed{s}": OUT / f"d1-original-sft-seed{s}" / "final" for s in (42, 43, 44)},
    "multivers-finetuned-seed42": OUT / "multivers-finetuned-seed42",
}
EXTRA = ("training_config.json", "train_metrics.jsonl")  # training logs live next to final/

README = """---
library_name: peft
base_model: LiquidAI/d1-3B
tags: [claim-verification, scifact, lora]
---
# COSC272 evidence-sensitive claim verification: trained weights

Code: https://github.com/bensonhi/COSC272-Project (`LiquidAI/` training and evaluation, `baselines/`, `demo/`).
Test-set results for every system are in `baselines/results/summary.md` there.

| folder | what |
|---|---|
| `d1-augmented-sft-seed{42,43}` | d1-3B LoRA trained on original + evidence-deleted groups (our method) |
| `d1-cf-evidence-only-seed{42,43,44}` | augmented SFT + evidence-side counterfactual pairs (`counterfactual/`) |
| `d1-cf-all-seed{42,43}` | augmented SFT + all counterfactual pairs, claim negations included |
| `d1-original-sft-seed{42,43,44}` | same LoRA recipe, original groups only (baseline B2) |
| `multivers-finetuned-seed42` | MultiVerS (Longformer-large) state dict, from `fever_sci`, fine-tuned on our train split (baseline B4b) |

All of these were trained on the 776-claim training split from before `data/validation` was split off.
Each d1 folder is a PEFT adapter for `LiquidAI/d1-3B`; `training_config.json` records the exact arguments.

```bash
hf download COSC272-Project/COSC272-Project --include "d1-augmented-sft-seed42/*" --local-dir hf
python -m LiquidAI --mode test --adapter-dir hf/d1-augmented-sft-seed42      # evaluation
python demo/app.py --adapter "Augmented SFT (ours)=hf/d1-augmented-sft-seed42"  # demo
python baselines/multivers_zero_shot.py --weights hf/multivers-finetuned-seed42/model.pt --name b4b_multivers_finetuned
```
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="COSC272-Project/COSC272-Project")
    args = parser.parse_args()
    api = HfApi()
    api.upload_file(path_or_fileobj=README.encode(), path_in_repo="README.md", repo_id=args.repo,
                    commit_message="Update README")
    for name, path in ARTIFACTS.items():
        path = path.resolve()
        if not path.is_dir():
            print("skip (missing):", name)
            continue
        api.upload_folder(folder_path=path, path_in_repo=name, repo_id=args.repo,
                          ignore_patterns=["checkpoint-*", "README.md"], commit_message=f"Upload {name}")
        for extra in EXTRA:
            if path.name == "final" and (path.parent / extra).exists():
                api.upload_file(path_or_fileobj=path.parent / extra, path_in_repo=f"{name}/{extra}",
                                repo_id=args.repo, commit_message=f"Upload {name}/{extra}")
        print("uploaded:", name)


if __name__ == "__main__":
    main()
