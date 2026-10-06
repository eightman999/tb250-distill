"""Teacher scorer: llama-server（Qwen3, thinking 無効）の次トークン logprob から候補ごとの logit を得る。

方式 label_logprob_perm{N}:
  - Qwen3 chat 形式。assistant 側は空の <think></think> の後 "Answer: "（ja は "回答番号: "）で止め、
    n_predict=1 で次トークンの top_logprobs を取る。ラベル "1".."K" は Qwen3 tokenizer で単一トークン
    （数字は 1 桁ずつ分割され、直前の空白は別トークンなので "Answer: " の後に素の数字が来る。tb250 で確認済み）。
  - 位置バイアス除去のため候補順序を変えた N 通りの permutation（既定 N=2: 恒等 + 逆順）で採点し、
    候補ごとに logprob を平均 → K 候補で再正規化した log-prob を logits、softmax を probs とする。
  - top-n に無いラベルは「top-n の観測最小 logprob − floor_margin」で補完し raw に記録する。
  - context+question を共通 prefix（候補一覧が最後）に置き、cache_prompt=true で prefix cache を効かせる。
  - prompt が max_prompt_tokens を超えたら context を先頭側から切る。

llama-server b11384 /completion のレスポンス（tb250 で実測）:
  completion_probabilities[0] = {id, token, bytes, logprob, top_logprobs:[{id, token, bytes, logprob}, ...]}
  （post_sampling_probs=false のとき "logprob"。true だと "prob" になる）
  timings.{prompt_n, cache_n, prompt_ms, ...}, tokens_evaluated, tokens_cached。
"""
from __future__ import annotations

import hashlib
import math
import random
import re
import time
from dataclasses import dataclass
from typing import Sequence

import requests

METHOD_PREFIX = "label_logprob_perm"
_JA_RE = re.compile(r"[぀-ヿ一-鿿]")

SYSTEM = {
    "en": "You are the decision engine of a resident agent. Read the context and the question, then answer with the number of the best option only.",
    "ja": "あなたは常駐エージェントの判断エンジンです。文脈と質問を読み、最も適切な選択肢の番号だけを答えてください。",
}
LABELS = {
    "en": ("Context:", "Question:", "Options:", "Answer: "),
    "ja": ("文脈:", "質問:", "選択肢:", "回答番号: "),
}


class ScorerError(RuntimeError):
    """item 固有の失敗（プロンプト過長・応答にラベルが無い等）。リトライしても同じ。"""


class ScorerConnectionError(ScorerError):
    """llama-server に繋がらない／5xx。サーバ復旧待ちの対象。"""


@dataclass
class ScorerConfig:
    base_url: str = "http://127.0.0.1:18080"
    max_prompt_tokens: int = 256
    n_probs: int = 50
    perms: int | Sequence[Sequence[int]] = 2   # 数 or 明示的な順列（K 依存なので数指定が普通）
    floor_margin: float = 1.0
    timeout: float = 60.0
    cache_prompt: bool = True


def detect_lang(*texts: str) -> str:
    return "ja" if any(_JA_RE.search(t) for t in texts) else "en"


def logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def make_perms(k: int, n: int, seed_text: str) -> list[list[int]]:
    """position j に表示する候補 index のリストを n 個。1 個目=恒等、2 個目=逆順、以降は seed 付き乱数（重複除外）。"""
    perms = [list(range(k))]
    if n >= 2:
        perms.append(list(range(k - 1, -1, -1)))
    if n > 2:
        rng = random.Random(int(hashlib.sha1(seed_text.encode()).hexdigest()[:12], 16))
        tries = 0
        while len(perms) < n and tries < 200:
            p = list(range(k))
            rng.shuffle(p)
            if p not in perms:
                perms.append(p)
            tries += 1
    return perms[:n]


class Scorer:
    def __init__(self, cfg: ScorerConfig | None = None):
        self.cfg = cfg or ScorerConfig()
        self.http = requests.Session()
        self.model_name: str | None = None

    # -- low-level -------------------------------------------------------
    def _post(self, path: str, payload: dict) -> dict:
        try:
            r = self.http.post(self.cfg.base_url + path, json=payload, timeout=self.cfg.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise ScorerConnectionError(str(e)) from e
        if r.status_code >= 500:
            raise ScorerConnectionError(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code != 200:
            raise ScorerError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def count_tokens(self, text: str) -> int:
        return len(self._post("/tokenize", {"content": text, "parse_special": True})["tokens"])

    def model_file(self) -> str:
        """teacher_model 列に入れる gguf ファイル名。"""
        if self.model_name is None:
            try:
                r = self.http.get(self.cfg.base_url + "/props", timeout=10).json()
                path = r.get("model_path") or r.get("model_alias") or ""
            except Exception:
                path = ""
            if not path:
                try:
                    path = self.http.get(self.cfg.base_url + "/v1/models", timeout=10).json()["data"][0]["id"]
                except Exception:
                    path = "unknown"
            self.model_name = path.rsplit("/", 1)[-1]
        return self.model_name

    @property
    def method(self) -> str:
        n = self.cfg.perms if isinstance(self.cfg.perms, int) else len(self.cfg.perms)
        return f"{METHOD_PREFIX}{n}"

    # -- prompt ----------------------------------------------------------
    @staticmethod
    def build_prompt(context: str, question: str, ordered: Sequence[str], lang: str) -> str:
        sysm = SYSTEM[lang]
        lc, lq, lo, ans = LABELS[lang]
        opts = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(ordered))
        return (
            f"<|im_start|>system\n{sysm}<|im_end|>\n"
            f"<|im_start|>user\n{lc} {context}\n{lq} {question}\n{lo}\n{opts}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n{ans}"
        )

    def _fit_context(self, context: str, question: str, cands: Sequence[str], lang: str) -> tuple[str, int, int]:
        """prompt が上限内になるよう context を先頭側から切る。(context, 切った文字数, prompt tokens)。"""
        cut_total = 0
        n = self.count_tokens(self.build_prompt(context, question, cands, lang))
        for _ in range(8):
            if n <= self.cfg.max_prompt_tokens:
                return context, cut_total, n
            if not context:
                break
            n_ctx = max(1, self.count_tokens(context))
            excess = n - self.cfg.max_prompt_tokens
            cut = min(len(context), max(1, math.ceil(excess * len(context) / n_ctx * 1.1) + 1))
            context = context[cut:]
            cut_total += cut
            n = self.count_tokens(self.build_prompt(context, question, cands, lang))
        if n > self.cfg.max_prompt_tokens:
            raise ScorerError(f"prompt too long even after truncation ({n} > {self.cfg.max_prompt_tokens})")
        return context, cut_total, n

    # -- scoring ---------------------------------------------------------
    def _label_logprobs(self, prompt: str, k: int) -> dict:
        res = self._post("/completion", {
            "prompt": prompt, "n_predict": 1, "n_probs": self.cfg.n_probs, "temperature": 0.0,
            "cache_prompt": self.cfg.cache_prompt, "post_sampling_probs": False,
        })
        cps = res.get("completion_probabilities")
        if not cps:
            raise ScorerError("no completion_probabilities in response")
        top = cps[0].get("top_logprobs")
        if top is None:
            raise ScorerError(f"unexpected response keys: {list(cps[0])}")
        seen: dict[str, float] = {}
        for t in top:
            lp = t.get("logprob")
            if lp is None:
                raise ScorerError("top_logprobs entry without 'logprob' (post_sampling_probs?)")
            tok = t["token"]
            if tok.isdigit() and len(tok) == 1 and tok in seen:
                seen[tok] = max(seen[tok], lp)
            elif tok.isdigit() and len(tok) == 1:
                seen[tok] = lp
        observed_min = min(t["logprob"] for t in top)
        floor = observed_min - self.cfg.floor_margin
        lps, filled = [], []
        for j in range(k):
            lab = str(j + 1)
            if lab in seen:
                lps.append(seen[lab])
            else:
                lps.append(floor)
                filled.append(j)
        found = [seen[str(j + 1)] for j in range(k) if str(j + 1) in seen]
        mass = logsumexp(found) if found else float("-inf")
        if len(filled) == k:
            raise ScorerError("none of the label tokens appeared in top_logprobs")
        t = res.get("timings", {})
        return {
            "label_lp": lps, "floor_filled": filled, "floor": floor if filled else None,
            "label_mass_lp": mass, "top1": cps[0].get("token"),
            "n_prompt": res.get("tokens_evaluated"), "cache_n": t.get("cache_n"),
            "prompt_ms": t.get("prompt_ms"),
        }

    def score(self, context: str, question: str, candidates: Sequence[str], lang: str | None = None) -> dict:
        k = len(candidates)
        if not 2 <= k <= 9:
            raise ScorerError(f"unsupported number of candidates: {k}")
        t0 = time.perf_counter()
        lang = lang or detect_lang(context, question, *candidates)
        ctx, cut, n_prompt = self._fit_context(context, question, candidates, lang)
        if isinstance(self.cfg.perms, int):
            perms = make_perms(k, self.cfg.perms, context + "|" + question)
        else:
            perms = [list(p) for p in self.cfg.perms]
        per_perm = []
        sums = [0.0] * k
        for order in perms:
            ordered = [candidates[i] for i in order]
            info = self._label_logprobs(self.build_prompt(ctx, question, ordered, lang), k)
            info["order"] = order
            per_perm.append(info)
            for j, cand_idx in enumerate(order):
                sums[cand_idx] += info["label_lp"][j]
        mean_lp = [s / len(perms) for s in sums]
        z = logsumexp(mean_lp)
        logits = [x - z for x in mean_lp]
        probs = [math.exp(x) for x in logits]
        total = sum(probs)
        probs = [p / total for p in probs]
        return {
            "logits": logits, "probs": probs,
            "raw": {
                "method": self.method, "lang": lang, "n_probs": self.cfg.n_probs,
                "max_prompt_tokens": self.cfg.max_prompt_tokens, "n_prompt_tokens": n_prompt,
                "truncated_chars": cut, "mean_label_lp": mean_lp, "perms": per_perm,
            },
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
