from typing import Dict, List

from transformers import (
    AutoTokenizer,
)

from vllm import (
    LLM,
    SamplingParams,
)

from config import (
    ModelConfig,
    InferenceConfig,
)

from data_load import (
    CanonicalPromptFormatter,
)


class VLLMInferenceWorker:

    def __init__(
        self,
        model_config: ModelConfig,
        inference_config:
            InferenceConfig,
    ):

        self.model_config = (
            model_config
        )

        self.inference_config = (
            inference_config
        )

        self.formatter = (
            CanonicalPromptFormatter()
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                model_config.model_name,
                trust_remote_code=(
                    model_config
                    .trust_remote_code
                ),
            )
        )

        self.sampling_params = (
            SamplingParams(
                temperature=(
                    inference_config
                    .temperature
                ),
                top_p=(
                    inference_config
                    .top_p
                ),
                max_tokens=(
                    inference_config
                    .max_new_tokens
                ),
            )
        )

        self.llm = LLM(
            model=(
                model_config.model_name
            ),

            tensor_parallel_size=(
                inference_config
                .inference_gpus
            ),

            dtype=model_config.dtype,

            gpu_memory_utilization=(
                inference_config
                .gpu_memory_utilization
            ),

            trust_remote_code=(
                model_config
                .trust_remote_code
            ),
        )

    def ready(self):

        return {
            "status": "ready",
            "model":
                self.model_config
                .model_name,

            "gpus":
                self.inference_config
                .inference_gpus,
        }

    def generate(
        self,
        examples:
            List[Dict],
    ):

        prompts = []

        for example in examples:

            user_prompt = (
                self.formatter
                .build_user_prompt(
                    example
                )
            )

            prompts.append(
                self._apply_chat_template(
                    user_prompt
                )
            )

        outputs = self.llm.generate(
            prompts,
            self.sampling_params,
            use_tqdm=False,
        )

        generated_examples = []

        for example, output in zip(
            examples,
            outputs,
        ):

            generated_text = (
                output
                .outputs[0]
                .text
                .strip()
            )

            generated_examples.append(
                self._build_generated_example(
                    example,
                    generated_text,
                )
            )

        return generated_examples

    def _apply_chat_template(
        self,
        prompt,
    ):

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        if self.tokenizer.chat_template:

            return (
                self.tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        return (
            f"User: {prompt}\n"
            "Assistant:"
        )

    def _build_generated_example(
        self,
        original,
        generated_text,
    ):

        task_type = original[
            "task_type"
        ]

        result = {
            "task_type": task_type,

            "question":
                original["question"],

            "choices":
                original.get(
                    "choices",
                    [],
                ),

            "reference":
                original.get(
                    "reference",
                    "",
                ),

            "response": "",
            "answer": "",
        }

        if task_type == "conversation":

            result["response"] = (
                generated_text
            )

        else:

            result["answer"] = (
                generated_text
            )

        return result