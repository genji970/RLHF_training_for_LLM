from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from datasets import load_dataset

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from config import DataConfig


@dataclass
class CanonicalExample:
    task_type: str

    question: str

    response: str = ""
    answer: str = ""

    choices: Optional[List[str]] = None

    reference: str = ""

    def to_dict(self) -> Dict[str, Any]:

        return {
            "task_type": self.task_type,
            "question": self.question,
            "response": self.response,
            "answer": self.answer,
            "choices": (
                self.choices
                if self.choices is not None
                else []
            ),
            "reference": self.reference,
        }


class DatasetSchemaNormalizer:

    REFERENCE_FIELDS = [
        "reference",
        "context",
        "passage",
        "article",
        "document",
        "fact1",
        "support",
        "background",
        "source",
    ]

    QUESTION_FIELDS = [
        "question",
        "question_stem",
        "prompt",
        "instruction",
        "query",
        "input",
    ]

    RESPONSE_FIELDS = [
        "response",
        "output",
        "completion",
    ]

    ANSWER_FIELDS = [
        "answer",
        "answerKey",
        "correct_answer",
        "label",
        "target",
    ]

    def __init__(
        self,
        requested_task_type: str = "auto",
    ):

        self.requested_task_type = (
            requested_task_type
        )

    def detect_task_type(
        self,
        row: Dict[str, Any],
    ) -> str:

        if (
            self.requested_task_type
            != "auto"
        ):
            return self.requested_task_type

        if (
            "choices" in row
            or "options" in row
            or "distractor1" in row
            or "distractors" in row
        ):
            return "mcqa"

        if (
            "messages" in row
            or "conversations" in row
        ):
            return "conversation"

        if any(
            field in row
            for field in self.RESPONSE_FIELDS
        ):
            return "conversation"

        return "qa"

    def normalize(
        self,
        row: Dict[str, Any],
    ) -> Dict[str, Any]:

        task_type = self.detect_task_type(
            row
        )

        if task_type == "conversation":
            example = (
                self._normalize_conversation(
                    row
                )
            )

        elif task_type == "mcqa":
            example = self._normalize_mcqa(
                row
            )

        elif task_type == "qa":
            example = self._normalize_qa(
                row
            )

        else:
            raise ValueError(
                f"Unknown task type: {task_type}"
            )

        return example.to_dict()

    def _normalize_conversation(
        self,
        row: Dict[str, Any],
    ) -> CanonicalExample:

        messages = row.get("messages")

        if messages is None:
            messages = row.get(
                "conversations"
            )

        if isinstance(messages, list):

            question = None
            response = None

            for message in messages:

                role = str(
                    message.get(
                        "role",
                        message.get(
                            "from",
                            "",
                        ),
                    )
                ).lower()

                content = str(
                    message.get(
                        "content",
                        message.get(
                            "value",
                            "",
                        ),
                    )
                )

                if role in {
                    "user",
                    "human",
                }:
                    question = content

                elif (
                    role in {
                        "assistant",
                        "gpt",
                        "bot",
                    }
                    and question is not None
                ):
                    response = content
                    break

            if (
                question is None
                or response is None
            ):
                raise ValueError(
                    "Could not extract "
                    "user/assistant pair."
                )

        else:

            question = self._find_first(
                row,
                self.QUESTION_FIELDS,
            )

            response = self._find_first(
                row,
                self.RESPONSE_FIELDS,
            )

            if not response:

                response = str(
                    row.get(
                        "answer",
                        "",
                    )
                )

        reference = (
            self._extract_reference(row)
        )

        return CanonicalExample(
            task_type="conversation",
            question=str(question),
            response=str(response),
            reference=reference,
        )

    def _normalize_mcqa(
        self,
        row: Dict[str, Any],
    ) -> CanonicalExample:

        question = self._find_first(
            row,
            self.QUESTION_FIELDS,
        )

        choices, labels = (
            self._extract_choices(row)
        )

        raw_answer = self._find_first(
            row,
            self.ANSWER_FIELDS,
        )

        answer = self._resolve_answer(
            raw_answer=raw_answer,
            choices=choices,
            labels=labels,
        )

        reference = (
            self._extract_reference(row)
        )

        return CanonicalExample(
            task_type="mcqa",
            question=str(question),
            answer=answer,
            choices=choices,
            reference=reference,
        )

    def _normalize_qa(
        self,
        row: Dict[str, Any],
    ) -> CanonicalExample:

        question = self._find_first(
            row,
            self.QUESTION_FIELDS,
        )

        answer = self._find_first(
            row,
            self.ANSWER_FIELDS,
        )

        reference = (
            self._extract_reference(row)
        )

        return CanonicalExample(
            task_type="qa",
            question=str(question),
            answer=str(answer),
            reference=reference,
        )

    def _extract_reference(
        self,
        row: Dict[str, Any],
    ) -> str:

        for field in self.REFERENCE_FIELDS:

            value = row.get(field)

            if value is None:
                continue

            if isinstance(value, str):
                if value.strip():
                    return value

            else:
                return str(value)

        return ""

    def _extract_choices(
        self,
        row: Dict[str, Any],
    ):

        raw_choices = row.get(
            "choices"
        )

        if raw_choices is None:
            raw_choices = row.get(
                "options"
            )

        choices = []
        labels = []

        if isinstance(
            raw_choices,
            dict,
        ):

            texts = raw_choices.get(
                "text",
                raw_choices.get(
                    "choices",
                    [],
                ),
            )

            raw_labels = raw_choices.get(
                "label",
                [],
            )

            choices = [
                str(x)
                for x in texts
            ]

            labels = [
                str(x)
                for x in raw_labels
            ]

        elif isinstance(
            raw_choices,
            list,
        ):

            if (
                len(raw_choices) > 0
                and isinstance(
                    raw_choices[0],
                    dict,
                )
            ):

                for index, item in enumerate(
                    raw_choices
                ):

                    text = item.get(
                        "text",
                        item.get(
                            "content",
                            item.get(
                                "value",
                                "",
                            ),
                        ),
                    )

                    label = item.get(
                        "label",
                        chr(
                            ord("A")
                            + index
                        ),
                    )

                    choices.append(
                        str(text)
                    )

                    labels.append(
                        str(label)
                    )

            else:

                choices = [
                    str(x)
                    for x in raw_choices
                ]

                labels = [
                    chr(
                        ord("A") + i
                    )
                    for i in range(
                        len(choices)
                    )
                ]

        # SciQ 같은 구조
        if not choices:

            correct = row.get(
                "correct_answer"
            )

            distractors = []

            for key in [
                "distractor1",
                "distractor2",
                "distractor3",
            ]:
                if key in row:
                    distractors.append(
                        str(row[key])
                    )

            if correct is not None:

                choices = [
                    str(correct),
                    *distractors,
                ]

                labels = [
                    chr(
                        ord("A") + i
                    )
                    for i in range(
                        len(choices)
                    )
                ]

        return choices, labels

    def _resolve_answer(
        self,
        raw_answer,
        choices,
        labels,
    ) -> str:

        if raw_answer is None:
            return ""

        if isinstance(
            raw_answer,
            bool,
        ):
            return str(raw_answer)

        if isinstance(
            raw_answer,
            int,
        ):

            if (
                0
                <= raw_answer
                < len(choices)
            ):
                return choices[
                    raw_answer
                ]

            return str(raw_answer)

        raw_answer = str(
            raw_answer
        )

        if raw_answer in labels:

            index = labels.index(
                raw_answer
            )

            if index < len(choices):
                return choices[index]

        if (
            raw_answer.isdigit()
            and choices
        ):

            index = int(raw_answer)

            if (
                0 <= index < len(choices)
            ):
                return choices[index]

        return raw_answer

    def _find_first(
        self,
        row,
        candidates,
    ):

        for key in candidates:

            if key not in row:
                continue

            value = row[key]

            if value is None:
                continue

            return value

        return ""


class CanonicalPromptFormatter:

    def build_user_prompt(
        self,
        example: Dict[str, Any],
    ) -> str:

        parts = []

        reference = example.get(
            "reference",
            "",
        )

        if reference:

            parts.append(
                "Reference:\n"
                f"{reference}"
            )

        question = example[
            "question"
        ]

        parts.append(
            f"Question:\n{question}"
        )

        choices = example.get(
            "choices",
            [],
        )

        if choices:

            choice_lines = []

            for index, choice in enumerate(
                choices
            ):

                label = chr(
                    ord("A") + index
                )

                choice_lines.append(
                    f"{label}. {choice}"
                )

            parts.append(
                "Choices:\n"
                + "\n".join(
                    choice_lines
                )
            )

        return "\n\n".join(parts)

    def get_target(
        self,
        example: Dict[str, Any],
    ) -> str:

        if (
            example["task_type"]
            == "conversation"
        ):
            return example["response"]

        return example["answer"]


class CanonicalHFDataset(Dataset):

    def __init__(
        self,
        dataset,
        normalizer,
    ):

        self.dataset = dataset
        self.normalizer = normalizer

    def __len__(self):

        return len(self.dataset)

    def __getitem__(self, index):

        raw = self.dataset[index]

        return self.normalizer.normalize(
            raw
        )


class SFTCollator:

    def __init__(
        self,
        tokenizer,
        max_length,
    ):

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.formatter = (
            CanonicalPromptFormatter()
        )

    def __call__(
        self,
        examples,
    ):

        input_ids_list = []
        labels_list = []

        for example in examples:

            user_prompt = (
                self.formatter
                .build_user_prompt(
                    example
                )
            )

            target = (
                self.formatter
                .get_target(
                    example
                )
            )

            prompt_text = (
                self._format_prompt(
                    user_prompt
                )
            )

            full_text = (
                self._format_full(
                    user_prompt,
                    target,
                )
            )

            prompt_ids = (
                self.tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=(
                        self.max_length
                    ),
                )["input_ids"]
            )

            full_ids = (
                self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=(
                        self.max_length
                    ),
                )["input_ids"]
            )

            labels = full_ids.copy()

            prompt_length = min(
                len(prompt_ids),
                len(labels),
            )

            labels[
                :prompt_length
            ] = (
                [-100]
                * prompt_length
            )

            input_ids_list.append(
                full_ids
            )

            labels_list.append(
                labels
            )

        return self._pad(
            input_ids_list,
            labels_list,
        )

    def _format_prompt(
        self,
        user_prompt,
    ):

        messages = [
            {
                "role": "user",
                "content": user_prompt,
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
            f"User: {user_prompt}\n"
            "Assistant:"
        )

    def _format_full(
        self,
        user_prompt,
        target,
    ):

        messages = [
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": target,
            },
        ]

        if self.tokenizer.chat_template:

            return (
                self.tokenizer
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        eos = (
            self.tokenizer.eos_token
            or ""
        )

        return (
            f"User: {user_prompt}\n"
            f"Assistant: {target}"
            f"{eos}"
        )

    def _pad(
        self,
        input_ids_list,
        labels_list,
    ):

        max_length = max(
            len(x)
            for x in input_ids_list
        )

        batch_size = len(
            input_ids_list
        )

        pad_token_id = (
            self.tokenizer.pad_token_id
        )

        input_ids = torch.full(
            (
                batch_size,
                max_length,
            ),
            pad_token_id,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (
                batch_size,
                max_length,
            ),
            dtype=torch.long,
        )

        labels = torch.full(
            (
                batch_size,
                max_length,
            ),
            -100,
            dtype=torch.long,
        )

        for index, (
            ids,
            target_labels,
        ) in enumerate(
            zip(
                input_ids_list,
                labels_list,
            )
        ):

            length = len(ids)

            input_ids[
                index,
                :length,
            ] = torch.tensor(
                ids,
                dtype=torch.long,
            )

            attention_mask[
                index,
                :length,
            ] = 1

            labels[
                index,
                :length,
            ] = torch.tensor(
                target_labels,
                dtype=torch.long,
            )

        return {
            "input_ids": input_ids,
            "attention_mask":
                attention_mask,
            "labels": labels,
        }


class HFDatasetManager:

    def __init__(
        self,
        config: DataConfig,
        tokenizer,
        batch_size: int,
        max_length: int,
    ):

        self.config = config
        self.tokenizer = tokenizer

        self.batch_size = batch_size
        self.max_length = max_length

        self.normalizer = (
            DatasetSchemaNormalizer(
                requested_task_type=(
                    config.task_type
                )
            )
        )

        self.dataset = None

    def load(self):

        kwargs = {
            "path":
                self.config.dataset_name,

            "split":
                self.config.dataset_split,

            "cache_dir":
                self.config.cache_dir,
        }

        if (
            self.config.dataset_subset
            is not None
        ):
            kwargs["name"] = (
                self.config.dataset_subset
            )

        raw_dataset = load_dataset(
            **kwargs
        )

        if (
            self.config.max_samples
            is not None
        ):

            sample_count = min(
                self.config.max_samples,
                len(raw_dataset),
            )

            raw_dataset = (
                raw_dataset.select(
                    range(sample_count)
                )
            )

        if len(raw_dataset) == 0:
            raise ValueError(
                "Dataset is empty."
            )

        detected = (
            self.normalizer
            .detect_task_type(
                raw_dataset[0]
            )
        )

        print(
            "Dataset:",
            self.config.dataset_name,
        )

        print(
            "Detected task type:",
            detected,
        )

        self.dataset = (
            CanonicalHFDataset(
                dataset=raw_dataset,
                normalizer=(
                    self.normalizer
                ),
            )
        )

        return self.dataset

    def create_dataloader(self):

        if self.dataset is None:
            self.load()

        collator = SFTCollator(
            tokenizer=self.tokenizer,
            max_length=(
                self.max_length
            ),
        )

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=(
                self.config
                .dataloader_workers
            ),
            collate_fn=collator,
        )

    def get_rollout_examples(
        self,
        count,
        offset=0,
    ):

        if self.dataset is None:
            self.load()

        examples = []

        size = len(self.dataset)

        for index in range(count):

            dataset_index = (
                offset + index
            ) % size

            examples.append(
                self.dataset[
                    dataset_index
                ]
            )

        return examples