# Interactive demo

A Gradio app around d1-3B for evidence-sensitive claim verification.

* **Verify a claim**: pick a curated example (representative / challenging / failure), any of the
  333 test claims, or paste your own claim and abstract. The app shows the three label
  probabilities, every sentence's evidence probability (selected at p >= 0.5, gold evidence marked
  with a star), and whether the prediction matches the gold label.
  *Remove sentences* turns chosen sentences into unavailable slots, which is exactly the training
  augmentation format, so you can watch the judgement change as evidence is withdrawn.
  *Remove predicted evidence & re-verify* does this in one click; a model that tracks evidence
  should move to NOT ENOUGH INFO. *Compare all models* runs zero-shot, original-data SFT and
  augmented SFT on the same input.
* **Batch mode**: 20 fixed test claims (8 SUPPORT, 8 NEI, 4 CONTRADICT), verified one at a time
  with per-claim latency, next to Qwen3.8-27B (live through a vLLM server, or its recorded run).
* **About**: results table read from `baselines/results`, intended use and limitations.

## Permanent public version
https://huggingface.co/spaces/COSC272-Project/claim-verification-demo is a static page (free hosting, no GPU):
every verdict it shows was computed in advance by the real models. For each test claim and edited variant
it covers the full abstract, each model's predicted evidence removed (once and twice), all gold evidence
removed, and every single sentence removed; a Results tab repeats the slide 5 table and the stress test.
```bash
python demo/precompute.py --adapter "d1-original=<dir>" --adapter "d1-augment=<dir>" --adapter "d1-pair=<dir>"   # GPU, resumable
python demo/precompute.py --finish --adapter "d1-original=x" --adapter "d1-augment=x" --adapter "d1-pair=x"
python demo/export_static.py --out site    # then upload site/ (plus the Space README) to the Space
```

## Files
| file | purpose |
|---|---|
| `engine.py` | loads d1-3B once with every LoRA adapter mounted; one tree pass reads the abstract once for all questions |
| `check_equivalence.py` | checks the tree pass against the evaluation code path (`LiquidAI.evaluation.predict_sample`) |
| `app.py` | the Gradio interface |
| `pick_examples.py` | chooses `examples.json` from a model's test predictions with fixed rules |
| `run_demo.sh` | launches the app (and optionally the vLLM server) on the cluster |
| `record_walkthrough.py` | records the silent backup video with headless Chromium |
| `precompute.py`, `cache.py` | run the engine on every input the static page offers and store the verdicts (`cache/`, not in git) |
| `export_static.py`, `static/` | build the static site: `static/index.html` (the page) and `static/results.json` (Results tab) |

The demo uses the same prompts, answer tokens and adapters as evaluation; only the forward pass is
batched differently. On 30 test claims the label never changed and probabilities differed by at
most 0.026 (label) / 0.059 (evidence), which is bf16 rounding.

## Running
```bash
# on l40sx8, GPU ids from `scontrol show job -d <jobid>` (IDX), e.g. 4 and 5
AUGMENTED_ADAPTER=/path/to/augmented/final demo/run_demo.sh 4 5     # app on GPU 4, Qwen server on GPU 5
demo/run_demo.sh 4                                                 # app only; LLM comparison uses the recorded run
# from your laptop
ssh -N -L 7860:localhost:7860 <netid>@l40sx8.hpcc.dartmouth.edu    # then open http://localhost:7860
```
`--share` (third argument) creates a temporary public gradio.live link instead of a tunnel.

Backup recording (after the app is running):
```bash
PLAYWRIGHT_BROWSERS_PATH=/scratch/f008k3y/cosc272/pw python demo/record_walkthrough.py --url http://127.0.0.1:7860
```
