"""Inference engine for the demo: d1-3B with any number of LoRA adapters mounted at once.

Prompts, answer tokens and the input state are built by the LiquidAI package (the same code
used for training and evaluation), so the demo shows exactly the evaluated model behaviour.
The only difference is speed: instead of re-reading the abstract once per question, the
state is read once and every question branches off it (d1's native `answer` tree pass).
`check_equivalence.py` verifies both paths give the same probabilities.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from LiquidAI.config import DEFAULT_MODEL, LABELS  # noqa: E402
from LiquidAI.data import Article, Sample  # noqa: E402
from LiquidAI.encoding import DecisionCodec  # noqa: E402
from LiquidAI.modeling import pool_options  # noqa: E402
from LiquidAI.prompts import STANCE_QUESTION, evidence_question  # noqa: E402

ZERO_SHOT = "d1-3B zero-shot"


@dataclass
class Verdict:
    label: str
    label_probabilities: dict[str, float]
    # (doc_id, original_index) -> P(evidence); only for sentences that were available
    sentence_probabilities: dict[tuple[int, int], float]
    seconds: float
    input_tokens: int


def split_sentences(text: str) -> list[str]:
    """One sentence per line if the text has line breaks, else a simple punctuation split."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        return lines
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text.strip()) if s.strip()]


class D1Verifier:
    def __init__(self, adapters: dict[str, str | Path] | None = None, model_id: str = DEFAULT_MODEL,
                 device: str = "cuda:0", token_budget: int = 32768):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        base = AutoModel.from_pretrained(model_id, trust_remote_code=True, dtype=torch.bfloat16)
        base.config.use_cache = False
        self.codec = DecisionCodec(base, self.tokenizer)
        self.adapters = {name: Path(path) for name, path in (adapters or {}).items()}
        self.model = base
        if self.adapters:
            from peft import PeftModel

            names = list(self.adapters)
            self.model = PeftModel.from_pretrained(base, self.adapters[names[0]], adapter_name=names[0])
            for name in names[1:]:
                self.model.load_adapter(self.adapters[name], adapter_name=name)
        self.model.to(self.device).eval().requires_grad_(False)
        self.base = base  # LoRA layers are injected in place, so base.answer() sees the active adapter
        self.token_budget = token_budget
        self.bos = self.tokenizer.bos_token or ""
        prompt = self.codec.prompt
        self._stance_suffix = self._encode(prompt.suffix_text(
            self.tokenizer, prompt.as_question(STANCE_QUESTION), lead=self.codec.lead))

    @property
    def model_names(self) -> list[str]:
        return [ZERO_SHOT, *self.adapters]

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _adapter(self, name: str):
        if name == ZERO_SHOT:
            return self.model.disable_adapter() if self.adapters else nullcontext()
        if name not in self.adapters:
            raise ValueError(f"unknown model {name!r}; choose from {self.model_names}")
        self.model.set_adapter(name)
        return nullcontext()

    def verify(self, claim: str, articles: list[dict], model: str = ZERO_SHOT) -> Verdict:
        """`articles`: [{"doc_id", "title", "sentences": [str]}]; an empty string is a removed sentence."""
        torch, prompt = self.torch, self.codec.prompt
        started = time.perf_counter()
        sample = Sample(-1, 0, "demo", claim, "NEI", [
            Article(int(a["doc_id"]), a["title"], list(a["sentences"]), [0] * len(a["sentences"]),
                    [bool(s.strip()) for s in a["sentences"]])
            for a in articles
        ], "demo")
        trunk = self._encode(prompt.prefix_text(self.tokenizer, sample.state(), bos=self.bos))
        keys, branches = [None], [self._stance_suffix]
        for article in sample.articles:
            for index, available in enumerate(article.sentence_mask):
                if available:
                    keys.append((article.doc_id, index))
                    branches.append(self._encode(prompt.suffix_text(
                        self.tokenizer, prompt.as_question(evidence_question(article.doc_id, index)),
                        lead=self.codec.lead)))
        label_probabilities, sentence_probabilities = None, {}
        trunk_ids = torch.tensor([trunk], device=self.device)
        with torch.inference_mode(), self._adapter(model):
            start = 0
            while start < len(branches):
                end, used = start, 0
                while end < len(branches) and (end == start or used + len(branches[end]) <= self.token_budget):
                    used += len(branches[end])
                    end += 1
                chunk = branches[start:end]
                packed = torch.tensor([t for b in chunk for t in b], device=self.device)
                lengths = torch.tensor([len(b) for b in chunk], device=self.device)
                logits = self.base.answer(trunk_ids, packed, lengths).float()
                for key, row in zip(keys[start:end], logits):
                    if key is None:
                        label_probabilities = pool_options(row, self.codec.stance_ids).softmax(-1).tolist()
                    else:
                        sentence_probabilities[key] = pool_options(row, self.codec.evidence_ids).softmax(-1)[0].item()
                start = end
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        best = max(range(len(LABELS)), key=label_probabilities.__getitem__)
        return Verdict(LABELS[best], dict(zip(LABELS, label_probabilities)), sentence_probabilities,
                       time.perf_counter() - started, len(trunk) + sum(map(len, branches)))
