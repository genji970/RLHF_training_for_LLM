from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMRanker
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelConfig, RewardConfig
from data import PromptFormatter
from debug import Debugger


class RewardEnsemble:
    """One frozen LLM, two small trainable heads: scalar neural RM and feature projector -> LightGBM."""
    def __init__(self, model_cfg: ModelConfig, cfg: RewardConfig, debug_config=None):
        self.cfg = cfg
        self.debug = Debugger(debug_config, "reward")
        self.debug.environment("reward", model=(model_cfg.reward_name or model_cfg.policy_name))
        name = model_cfg.reward_name or model_cfg.policy_name
        # BF16-only torch path. LightGBM itself is CPU/NumPy and receives FP32
        # copies only at its API boundary.
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with self.debug.stage("reward", "reward_tokenizer_load", model=name):
            self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=model_cfg.trust_remote_code)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        self.backbone = AutoModelForCausalLM.from_pretrained(name, dtype=self.dtype,
                                                             trust_remote_code=model_cfg.trust_remote_code).to(self.device)
        self.backbone.eval()
        for p in self.backbone.parameters(): p.requires_grad_(False)
        h = self.backbone.config.hidden_size
        self.neural = nn.Linear(h, 1).to(device=self.device, dtype=self.dtype)
        self.projector = nn.Linear(h, cfg.feature_dim).to(device=self.device, dtype=self.dtype)
        self.feature_head = nn.Linear(cfg.feature_dim, 1).to(device=self.device, dtype=self.dtype)

        for module_name, module in (
            ("backbone", self.backbone),
            ("neural", self.neural),
            ("projector", self.projector),
            ("feature_head", self.feature_head),
        ):
            bad = [
                (name, p.dtype)
                for name, p in module.named_parameters()
                if p.is_floating_point() and p.dtype != torch.bfloat16
            ]
            if bad:
                raise RuntimeError(f"{module_name} contains non-BF16 floating parameters: {bad[:5]}")

        self.opt = torch.optim.AdamW([*self.neural.parameters(), *self.projector.parameters(),
                                      *self.feature_head.parameters()], lr=cfg.learning_rate)
        self.lgbm = LGBMRanker(objective="lambdarank", n_estimators=120, learning_rate=0.05,
                               num_leaves=31, verbosity=-1)
        self.lgbm_ready = False
        self.cache, self.history = {}, []
        self.groups = 0

    def score(self, group):
        gid = group.get("id")
        with self.debug.stage("reward", "score_encode", group_id=gid, responses=len(group["responses"])):
            hidden = self._encode(group["prompt"], group["responses"]).to(dtype=self.dtype)
        with self.debug.stage("reward", "score_heads", group_id=gid):
            with torch.no_grad():
                neural = self.neural(hidden).squeeze(-1)
                z = self.projector(hidden)
        if self.lgbm_ready:
            with self.debug.stage("reward", "score_lgbm", group_id=gid):
                lgbm = self.lgbm.predict(z.float().cpu().numpy()).tolist()
        else:
            lgbm = None
        self.cache[group["id"]] = hidden.detach().cpu()
        out = {"neural": neural.float().cpu().tolist(), "lgbm": lgbm,
               "ready": self.groups >= self.cfg.warmup_groups, "lgbm_ready": self.lgbm_ready}
        self.debug.log("reward", "score_done", group_id=gid, ready=out["ready"], lgbm_ready=out["lgbm_ready"])
        return out

    def decide(self, prediction, human_scores):
        h_top, h_bottom = self._ends(human_scores)
        n_top, n_bottom = self._ends(prediction["neural"])
        l_ends = self._ends(prediction["lgbm"]) if prediction["lgbm"] is not None else (None, None)
        eligible = {
            "all": True,
            "neural_agree": prediction["ready"] and (h_top, h_bottom) == (n_top, n_bottom),
            "lgbm_agree": prediction["ready"] and prediction["lgbm_ready"] and (h_top, h_bottom) == l_ends,
            "triple_agree": prediction["ready"] and prediction["lgbm_ready"] and
                            (h_top, h_bottom) == (n_top, n_bottom) == l_ends,
            "neural_only": prediction["ready"],
            "lgbm_only": prediction["ready"] and prediction["lgbm_ready"],
        }
        mode = self.cfg.filter_mode
        top, bottom = (n_top, n_bottom) if mode == "neural_only" else l_ends if mode == "lgbm_only" else (h_top, h_bottom)
        result = {"selected": bool(eligible[mode]), "top": top, "bottom": bottom,
                  "eligible": eligible, "human": [h_top, h_bottom], "neural": [n_top, n_bottom],
                  "lgbm": list(l_ends)}
        self.debug.log("reward", "decision", mode=mode, selected=result["selected"], top=top, bottom=bottom)
        return result

    def update(self, group_id, human_scores):
        self.debug.log("reward", "update_begin", group_id=group_id, groups=self.groups)
        hidden = self.cache.pop(group_id).to(device=self.device, dtype=self.dtype)
        pairs = [(i, j) for i, j in combinations(range(len(human_scores)), 2) if human_scores[i] != human_scores[j]]
        for _ in range(self.cfg.train_steps):
            ns = self.neural(hidden).squeeze(-1)
            z = self.projector(hidden); fs = self.feature_head(z).squeeze(-1)
            losses = []
            for i, j in pairs:
                a, b = (i, j) if human_scores[i] > human_scores[j] else (j, i)
                losses += [-F.logsigmoid(ns[a] - ns[b]), -F.logsigmoid(fs[a] - fs[b])]
            if losses:
                self.opt.zero_grad(); torch.stack(losses).mean().backward(); self.opt.step()
        self.history.append((hidden.detach().cpu(), list(map(float, human_scores))))
        self.groups += 1
        if self.groups >= self.cfg.warmup_groups and self.groups % self.cfg.lgbm_refit_every == 0:
            with self.debug.stage("reward", "lgbm_refit", groups=self.groups):
                self._fit_lgbm()
        result = {"groups": self.groups, "lgbm_ready": self.lgbm_ready}
        self.debug.log("reward", "update_done", group_id=group_id, **result)
        return result

    def _fit_lgbm(self):
        hs = torch.cat([x for x, _ in self.history]).to(device=self.device, dtype=self.dtype)
        # LightGBM does not train in torch BF16; convert only at this CPU API boundary.
        with torch.no_grad():
            x = self.projector(hs).to(torch.float32).cpu().numpy()
        labels = []
        for _, scores in self.history:
            order = np.argsort(scores)
            relevance = np.empty(len(scores), dtype=np.int32)
            relevance[order] = np.arange(len(scores), dtype=np.int32)
            labels.append(relevance)
        y = np.concatenate(labels)
        self.lgbm.fit(x, y, group=[len(s) for _, s in self.history])
        self.lgbm_ready = True

    def _encode(self, prompt, responses):
        texts = [PromptFormatter.chat(self.tok, prompt, r) for r in responses]
        batch = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.backbone(**batch, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[-1]
            idx = batch["attention_mask"].sum(1) - 1
            return h[torch.arange(h.size(0), device=self.device), idx].to(dtype=self.dtype)

    @staticmethod
    def _ends(scores):
        if scores is None: return None, None
        return int(np.argmax(scores)), int(np.argmin(scores))


class HumanFeedback:
    def score(self, group):
        print(f"\n\n=== policy v{group['policy_version']} ===\n{group['prompt']}\n")
        for i, r in enumerate(group["responses"]): print(f"[{i}] {r}\n")
        while True:
            raw = input(f"scores ({len(group['responses'])} numbers, q=quit): ").strip()
            if raw.lower() == "q": return None
            try:
                scores = [float(x) for x in raw.replace(",", " ").split()]
                if len(scores) != len(group["responses"]): raise ValueError
                if scores.count(max(scores)) > 1 or scores.count(min(scores)) > 1:
                    print("top/bottom score must each be unique."); continue
                return scores
            except ValueError:
                print("Example: 3 5 1 4")
