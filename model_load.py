import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from config import ModelConfig


class ModelLoader:

    def __init__(
        self,
        config: ModelConfig,
    ):

        self.config = config

    def load_tokenizer(self):

        tokenizer = (
            AutoTokenizer
            .from_pretrained(
                self.config.model_name,
                trust_remote_code=(
                    self.config
                    .trust_remote_code
                ),
            )
        )

        if tokenizer.pad_token is None:

            tokenizer.pad_token = (
                tokenizer.eos_token
            )

        tokenizer.padding_side = "right"

        return tokenizer

    def load_train_model(self):

        model = (
            AutoModelForCausalLM
            .from_pretrained(
                self.config.model_name,
                torch_dtype=(
                    self._get_dtype()
                ),
                trust_remote_code=(
                    self.config
                    .trust_remote_code
                ),
            )
        )

        model.config.use_cache = False

        return model

    def _get_dtype(self):

        if self.config.dtype == "auto":
            return "auto"

        if self.config.dtype == "float16":
            return torch.float16

        if (
            self.config.dtype
            == "bfloat16"
        ):
            return torch.bfloat16

        if self.config.dtype == "float32":
            return torch.float32

        raise ValueError(
            f"Unknown dtype: "
            f"{self.config.dtype}"
        )