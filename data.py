import json
import time
import uuid
from typing import Any

import torch
import transfer_queue as tq
from datasets import load_dataset
from tensordict import TensorDict

from config import DataConfig


class Canonicalizer:
    Q = ("question", "question_stem", "prompt", "instruction", "query", "input")
    A = ("answer", "answerKey", "correct_answer", "label", "target")
    R = ("reference", "context", "passage", "article", "document", "fact1", "support")

    def __init__(self, task_type="auto"):
        self.task_type = task_type

    @staticmethod
    def _first(row, keys, default=""):
        return next((row[k] for k in keys if k in row and row[k] is not None), default)

    def __call__(self, row):
        task = self.task_type if self.task_type != "auto" else self._detect(row)
        question = str(self._first(row, self.Q))
        reference = str(self._first(row, self.R))
        choices, labels = self._choices(row)
        raw_answer = self._first(row, self.A)
        answer = self._answer(raw_answer, choices, labels)
        if task == "conversation":
            question, answer = self._conversation(row, question, answer)
        return {"task_type": task, "question": question, "choices": choices,
                "answer": str(answer), "reference": reference}

    def _detect(self, row):
        if any(k in row for k in ("choices", "options", "distractor1", "distractors")):
            return "mcqa"
        if any(k in row for k in ("messages", "conversations", "response", "completion")):
            return "conversation"
        return "qa"

    def _conversation(self, row, question, answer):
        messages = row.get("messages") or row.get("conversations")
        if not isinstance(messages, list):
            return question, str(row.get("response", row.get("completion", answer)))
        q = a = None
        for m in messages:
            role = str(m.get("role", m.get("from", ""))).lower()
            text = str(m.get("content", m.get("value", "")))
            if role in {"user", "human"}: q = text
            elif q is not None and role in {"assistant", "gpt", "bot"}: a = text; break
        return q or question, a or answer

    def _choices(self, row):
        raw = row.get("choices", row.get("options"))
        if isinstance(raw, dict):
            return [str(x) for x in raw.get("text", raw.get("choices", []))], [str(x) for x in raw.get("label", [])]
        if isinstance(raw, list):
            if raw and isinstance(raw[0], dict):
                texts = [str(x.get("text", x.get("content", x.get("value", "")))) for x in raw]
                labels = [str(x.get("label", chr(65 + i))) for i, x in enumerate(raw)]
                return texts, labels
            return [str(x) for x in raw], [chr(65 + i) for i in range(len(raw))]
        correct = row.get("correct_answer")
        distractors = [str(row[k]) for k in ("distractor1", "distractor2", "distractor3") if k in row]
        choices = ([str(correct)] + distractors) if correct is not None else []
        return choices, [chr(65 + i) for i in range(len(choices))]

    @staticmethod
    def _answer(answer, choices, labels):
        if answer is None: return ""
        if isinstance(answer, int) and 0 <= answer < len(choices): return choices[answer]
        answer = str(answer)
        if answer in labels and labels.index(answer) < len(choices): return choices[labels.index(answer)]
        return answer


class PromptFormatter:
    @staticmethod
    def user_prompt(example):
        parts = []
        if example.get("reference"): parts.append(f"Reference:\n{example['reference']}")
        parts.append(f"Question:\n{example['question']}")
        if example.get("choices"):
            parts.append("Choices:\n" + "\n".join(f"{chr(65+i)}. {x}" for i, x in enumerate(example["choices"])))
        return "\n\n".join(parts)

    @staticmethod
    def chat(tokenizer, prompt, response=None):
        messages = [{"role": "user", "content": prompt}]
        if response is not None: messages.append({"role": "assistant", "content": response})
        if tokenizer.chat_template:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=response is None)
        return f"User: {prompt}\nAssistant:" + (f" {response}" if response is not None else "")


class DatasetManager:
    def __init__(self, config: DataConfig):
        self.config = config
        self.normalizer = Canonicalizer(config.task_type)
        self.cache = {}

    def load(self, split):
        if split not in self.cache:
            kwargs = dict(path=self.config.dataset_name, split=split, cache_dir=self.config.cache_dir)
            if self.config.dataset_subset is not None: kwargs["name"] = self.config.dataset_subset
            ds = load_dataset(**kwargs)
            if self.config.max_samples: ds = ds.select(range(min(self.config.max_samples, len(ds))))
            self.cache[split] = ds
        return self.cache[split]

    def examples(self, split, count, offset=0):
        ds = self.load(split)
        return [self.normalizer(ds[(offset + i) % len(ds)]) for i in range(count)]


class TransferQueueBroker:
    """
    Owns the TransferQueue runtime in exactly one Ray actor.

    This avoids calling tq.init() from every rollout/train worker, which can try to
    recreate the same named TransferQueueStorageUnit actors.

    Logical queue IDs are converted to isolated TransferQueue partition IDs:
        <run_id>:<queue_id>
    """

    def __init__(self, run_id: str, rollout_id: str = "rollout", preference_id: str = "preference"):
        self.run_id = str(run_id)
        self.queue_ids = {
            "rollout": str(rollout_id),
            "preference": str(preference_id),
        }
        tq.init()

    def _partition(self, queue_name: str) -> str:
        if queue_name not in self.queue_ids:
            raise KeyError(f"Unknown queue {queue_name!r}; valid={sorted(self.queue_ids)}")
        return f"{self.run_id}:{self.queue_ids[queue_name]}"

    def describe(self):
        return {
            "run_id": self.run_id,
            "queues": {name: self._partition(name) for name in self.queue_ids},
        }

    def put(self, queue_name: str, item, tag=None):
        partition = self._partition(queue_name)
        item_id = str(item.get("id") or uuid.uuid4().hex)

        # Key is also globally namespaced, not just the partition.
        key = f"{self.run_id}:{self.queue_ids[queue_name]}:{item_id}"
        payload = torch.tensor(
            list(json.dumps(item).encode("utf-8")),
            dtype=torch.uint8,
        ).unsqueeze(0)

        metadata = {
            "run_id": self.run_id,
            "queue_id": self.queue_ids[queue_name],
            "item_id": item_id,
            **(tag or {}),
        }
        tq.kv_put(
            key=key,
            partition_id=partition,
            fields=TensorDict({"payload": payload}, batch_size=[1]),
            tag=metadata,
        )
        return key

    def pop(self, queue_name: str, n=1, current_version=None, max_lag=None, exact=False):
        partition = self._partition(queue_name)
        info = tq.kv_list(partition_id=partition).get(partition, {})
        keys, stale = [], []

        for key, tag in info.items():
            if current_version is not None and max_lag is not None:
                policy_version = int(tag.get("policy_version", current_version))
                if current_version - policy_version > max_lag:
                    stale.append(key)
                    continue
            keys.append(key)
            if len(keys) == n:
                break

        if stale:
            tq.kv_clear(keys=stale, partition_id=partition)

        if exact and len(keys) < n:
            return [], len(stale)

        items = []
        for key in keys:
            field = tq.kv_batch_get(
                keys=key,
                partition_id=partition,
                select_fields="payload",
            )["payload"]
            raw = field[0].detach().cpu().tolist()
            items.append(json.loads(bytes(raw).decode("utf-8")))

        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
        return items, len(stale)

    def size(self, queue_name: str):
        partition = self._partition(queue_name)
        return len(tq.kv_list(partition_id=partition).get(partition, {}))

    def clear(self, queue_name: str):
        partition = self._partition(queue_name)
        info = tq.kv_list(partition_id=partition).get(partition, {})
        keys = list(info.keys())
        if keys:
            tq.kv_clear(keys=keys, partition_id=partition)
        return len(keys)

    def close(self):
        try:
            tq.close()
        except Exception:
            pass


class DPOCollator:
    def __init__(self, tokenizer, max_length):
        self.tok, self.max_length = tokenizer, max_length

    def __call__(self, samples):
        seqs, masks, refs = [], [], []
        for side in ("chosen", "rejected"):
            for s in samples:
                prompt = PromptFormatter.chat(self.tok, s["prompt"])
                full = PromptFormatter.chat(self.tok, s["prompt"], s[side])
                p = self.tok(prompt, add_special_tokens=False)["input_ids"]
                f = self.tok(full, add_special_tokens=False, truncation=True, max_length=self.max_length)["input_ids"]
                prefix = 0
                for a, b in zip(p, f):
                    if a != b: break
                    prefix += 1
                mask = [0] * prefix + [1] * (len(f) - prefix)
                seqs.append(f); masks.append(mask); refs.append(float(s[f"ref_{side}_logp"]))
        width = max(map(len, seqs)); pad = self.tok.pad_token_id
        ids = torch.full((len(seqs), width), pad, dtype=torch.long)
        attn = torch.zeros_like(ids); response = torch.zeros_like(ids, dtype=torch.bfloat16)
        for i, (x, m) in enumerate(zip(seqs, masks)):
            ids[i, :len(x)] = torch.tensor(x); attn[i, :len(x)] = 1; response[i, :len(m)] = torch.tensor(m, dtype=torch.bfloat16)
        return {"input_ids": ids, "attention_mask": attn, "response_mask": response,
                "ref_logps": torch.tensor(refs, dtype=torch.bfloat16), "pairs": len(samples)}
