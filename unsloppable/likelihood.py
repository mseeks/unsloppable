"""Layer 2 — likelihood signals from small local language models.

Stylometric tells are era-locked (they collapse on modern models — see the RAID
eval). Likelihood signals are *model-relative*, not lexical, so they generalize
across eras far better. We compute, per text:

  * perplexity        — machine text sits in a model's high-likelihood (low-PPL)
                        region; human text is "burstier".  (low = AI)
  * mean_logprob      — average per-token log-prob under the observer.  (high = AI)
  * surprisal_stdev   — burstiness: stdev of per-token surprisal.       (low = AI)
  * binoculars        — Hans et al. 2024: log-PPL / cross-PPL between an observer
                        and a performer model (same tokenizer). Normalizes "how
                        predictable is this text in general", which makes it
                        robust to prompt/topic. (low = AI)

Needs the `ml` extra (torch + transformers); imported lazily so the stdlib path
never pays for it. Runs on MPS/CUDA/CPU. Two small models (default Qwen2.5-0.5B
base + instruct) load once and score many texts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# default observer/performer pair — same tokenizer family, small enough for MPS
OBSERVER = "Qwen/Qwen2.5-0.5B"
PERFORMER = "Qwen/Qwen2.5-0.5B-Instruct"
# Must match the value used to compute the cached features the combiner trained on
# (eval/score_likelihood.py default) — otherwise inference perplexity drifts from
# training on long texts.
MAX_TOKENS = 384

# feature directions, for the harness
DIRECTION = {
    "ll_perplexity": "low=AI",
    "ll_mean_logprob": "high=AI",
    "ll_surprisal_stdev": "low=AI",
    "ll_binoculars": "low=AI",
}
FEATURES = list(DIRECTION)


@dataclass
class _Loaded:
    tok: object
    observer: object
    performer: object
    device: str


class LikelihoodScorer:
    """Loads the model pair once; scores texts into likelihood features."""

    def __init__(self, observer: str = OBSERVER, performer: str = PERFORMER,
                 device: str | None = None, max_tokens: int = MAX_TOKENS):
        self.observer_name = observer
        self.performer_name = performer
        self.max_tokens = max_tokens
        self._device = device
        self._m: _Loaded | None = None

    def _pick_device(self) -> str:
        import torch
        if self._device:
            return self._device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load(self) -> _Loaded:
        if self._m is not None:
            return self._m
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = self._pick_device()
        dtype = torch.float32 if device == "cpu" else torch.float16
        tok = AutoTokenizer.from_pretrained(self.observer_name)
        obs = AutoModelForCausalLM.from_pretrained(
            self.observer_name, torch_dtype=dtype).to(device).eval()
        perf = AutoModelForCausalLM.from_pretrained(
            self.performer_name, torch_dtype=dtype).to(device).eval()
        self._m = _Loaded(tok=tok, observer=obs, performer=perf, device=device)
        return self._m

    def score(self, text: str) -> dict[str, float]:
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str], batch_size: int = 16) -> list[dict[str, float]]:
        import torch
        m = self._load()
        if m.tok.pad_token is None:
            m.tok.pad_token = m.tok.eos_token
        out: list[dict[str, float]] = []
        nan = {k: math.nan for k in FEATURES}
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            # Empty/blank texts tokenize to length 0/1 and would crash the forward
            # pass (reshape of 0 elements). Score only the non-trivial ones; the rest
            # get all-NaN (handled downstream as "no signal").
            idx = [i for i, t in enumerate(chunk) if len(t.strip()) > 0]
            results_for_chunk: list[dict | None] = [None if i in idx else dict(nan)
                                                    for i in range(len(chunk))]
            if not idx:
                out.extend(dict(nan) for _ in chunk)
                continue
            sub = [chunk[i] for i in idx]
            enc = m.tok(sub, return_tensors="pt", truncation=True,
                        max_length=self.max_tokens, padding=True)
            ids = enc["input_ids"].to(m.device)
            mask = enc["attention_mask"].to(m.device)
            if ids.shape[1] < 2:
                out.extend(dict(nan) for _ in chunk)
                continue
            with torch.no_grad():
                lo = m.observer(ids, attention_mask=mask).logits
                lp = m.performer(ids, attention_mask=mask).logits
            for b in range(ids.shape[0]):
                length = int(mask[b].sum().item())
                if length < 2:
                    results_for_chunk[idx[b]] = dict(nan)
                    continue
                logits_o = lo[b, :length - 1].float()      # positions predicting 1..length-1
                logits_p = lp[b, :length - 1].float()
                targets = ids[b, 1:length]
                logprob_o = torch.log_softmax(logits_o, dim=-1)
                tok_lp = logprob_o[torch.arange(targets.shape[0]), targets]
                nll = -tok_lp
                prob_p = torch.softmax(logits_p, dim=-1)
                xent = -(prob_p * logprob_o).sum(dim=-1)
                log_ppl = nll.mean().item()
                log_xppl = xent.mean().item()
                results_for_chunk[idx[b]] = {
                    "ll_perplexity": math.exp(min(log_ppl, 50)),
                    "ll_mean_logprob": tok_lp.mean().item(),
                    "ll_surprisal_stdev": nll.std(unbiased=False).item() if nll.numel() > 1 else 0.0,
                    "ll_binoculars": log_ppl / log_xppl if log_xppl > 1e-6 else math.nan,
                }
            out.extend(r if r is not None else dict(nan) for r in results_for_chunk)
        return out


_DEFAULT: LikelihoodScorer | None = None


def default_scorer() -> LikelihoodScorer:
    """Process-wide singleton so the models load once."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = LikelihoodScorer()
    return _DEFAULT
