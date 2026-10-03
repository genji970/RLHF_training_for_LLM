import os

os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

import re
import uuid

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from config import InferenceConfig, ModelConfig
from data import PromptFormatter


class VLLMRolloutWorker:
    def __init__(self, model: ModelConfig, config: InferenceConfig):
        self.config, self.version = config, 0
        self.tokenizer = AutoTokenizer.from_pretrained(model.policy_name, trust_remote_code=model.trust_remote_code)
        self.llm = LLM(model=model.policy_name, tensor_parallel_size=config.inference_gpus,
                       dtype="bfloat16", gpu_memory_utilization=config.gpu_memory_utilization,
                       trust_remote_code=model.trust_remote_code)
        self.sampling = SamplingParams(n=config.n_responses, temperature=config.temperature,
                                       top_p=config.top_p, max_tokens=config.max_new_tokens, logprobs=1)
        self.greedy = SamplingParams(temperature=0.0, max_tokens=config.max_new_tokens)

    def ready(self):
        return {"status": "ready", "policy_version": self.version}

    def generate(self, examples):
        prompts = [PromptFormatter.user_prompt(x) for x in examples]
        texts = [PromptFormatter.chat(self.tokenizer, p) for p in prompts]
        outputs = self.llm.generate(texts, self.sampling, use_tqdm=False)
        groups = []
        for ex, prompt, out in zip(examples, prompts, outputs):
            groups.append({
                "id": uuid.uuid4().hex,
                "prompt": prompt,
                "answer": ex.get("answer", ""),
                "responses": [x.text.strip() for x in out.outputs],
                "ref_logps": [float(x.cumulative_logprob or 0.0) for x in out.outputs],
                "policy_version": self.version,
            })
        return groups

    def reload(self, checkpoint_path, version):
        self.llm.collective_rpc("reload_weights", kwargs={"weights_path": checkpoint_path})
        self.version = int(version)
        return self.version

    def evaluate(self, examples):
        prompts = [PromptFormatter.user_prompt(x) for x in examples]
        outputs = self.llm.generate([PromptFormatter.chat(self.tokenizer, p) for p in prompts], self.greedy, use_tqdm=False)
        correct = 0
        for ex, out in zip(examples, outputs):
            gold, pred = self._norm(ex.get("answer", "")), self._norm(out.outputs[0].text)
            correct += int(bool(gold) and (pred == gold or gold in pred))
        return correct / max(len(examples), 1)

    @staticmethod
    def _norm(text):
        return re.sub(r"\s+", " ", str(text).strip().lower())
