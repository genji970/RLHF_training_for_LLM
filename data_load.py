from __future__ import annotations

from typing import Any, Callable

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)


DatasetLike = Dataset | DatasetDict | IterableDataset | IterableDatasetDict


# ============================================================
# Normalized QA schema
# ============================================================
#
# Datasets with a real question + answer are included, including subjective/open-ended QA.
#
# Every loader returns:
#
#   question: str
#
#   reference: dict | None
#       Supporting information ONLY.
#       It never contains the gold answer by itself and never contains
#       GPT/model-generated evaluation outputs.
#
#       Examples:
#         {"choices": ["choice A", "choice B", ...]}
#         {"context": "...", "choices": [...]}
#         {"reasoning": "..."}
#         None
#
#   answer: str | list[str]
#       Gold answer ONLY.
#
#       For multiple-choice datasets:
#           "A", "B", "C", ...
#
#       For open-ended datasets:
#           "18"
#           ["14 December 1972 UTC", "December 1972"]
#
# Excluded on purpose:
##   - AlpacaEval 2.0: comparison/reference model output, not gold QA
#   - MT-Bench: judge-based open-ended evaluation, usually no gold answer
# ============================================================


def _letter(index: int) -> str:
    if index < 0:
        raise ValueError(f"Negative answer index: {index}")
    return chr(ord("A") + index)


def _map_normalized(
    ds: DatasetLike,
    fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> DatasetLike:
    if isinstance(ds, (Dataset, IterableDataset)):
        columns = ds.column_names
        return ds.map(fn, remove_columns=columns) if columns else ds.map(fn)

    if isinstance(ds, (DatasetDict, IterableDatasetDict)):
        out = {}
        for split_name, split_ds in ds.items():
            columns = split_ds.column_names
            out[split_name] = (
                split_ds.map(fn, remove_columns=columns)
                if columns
                else split_ds.map(fn)
            )

        if isinstance(ds, DatasetDict):
            return DatasetDict(out)
        return IterableDatasetDict(out)

    raise TypeError(f"Unsupported dataset type: {type(ds)}")


def _mcq_reference(choices: list[str]) -> dict[str, list[str]]:
    return {"choices": choices}



# ============================================================
# 1. OpenAssistant OASST1
#    Human-authored open-ended instruction/answer data
# ============================================================

def load_oasst1(
    split: str = "train",
    *,
    num_examples: int | None = 3200,
    seed: int = 42,
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Dataset:
    """
    OpenAssistant OASST1 -> open-ended question/answer pairs.

    Paper IFT filtering:
      - English
      - first conversational turn
      - highest-ranked assistant response (rank 0)
      - sample 3200 examples

    Output:
        question  = human prompter message
        reference = None
        answer    = human-authored assistant response

    This is subjective/open-ended QA, so there are no choices.
    """
    raw = load_dataset(
        "OpenAssistant/oasst1",
        split=split,
        cache_dir=cache_dir,
        **kwargs,
    )

    by_id = {row["message_id"]: row for row in raw}
    examples: list[dict[str, Any]] = []

    for row in raw:
        if row["role"] != "assistant":
            continue
        if row["lang"] != "en":
            continue
        if row.get("deleted", False):
            continue
        if row.get("rank") != 0:
            continue

        parent_id = row.get("parent_id")
        if parent_id is None or parent_id not in by_id:
            continue

        parent = by_id[parent_id]

        # First conversational turn: root user prompt -> assistant answer.
        if parent["role"] != "prompter":
            continue
        if parent["lang"] != "en":
            continue
        if parent.get("deleted", False):
            continue
        if parent.get("parent_id") is not None:
            continue

        examples.append(
            {
                "question": parent["text"],
                "reference": None,
                "answer": row["text"],
            }
        )

    ds = Dataset.from_list(examples)

    if num_examples is not None and len(ds) > num_examples:
        ds = ds.shuffle(seed=seed).select(range(num_examples))

    return ds


# ============================================================
# 2. ARC-Easy
# ============================================================

def load_arc_easy(
    split: str = "test",
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "allenai/ai2_arc",
        "ARC-Easy",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )
    return _normalize_arc(raw)


# ============================================================
# 3. ARC-Challenge
# ============================================================

def load_arc_challenge(
    split: str = "test",
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "allenai/ai2_arc",
        "ARC-Challenge",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )
    return _normalize_arc(raw)


def _normalize_arc(ds: DatasetLike) -> DatasetLike:
    def normalize(x: dict[str, Any]) -> dict[str, Any]:
        choices = list(x["choices"]["text"])
        labels = [str(v) for v in x["choices"]["label"]]
        answer_key = str(x["answerKey"])

        if answer_key not in labels:
            raise ValueError(
                f"ARC answerKey={answer_key!r} not found in labels={labels!r}"
            )

        correct_idx = labels.index(answer_key)

        return {
            "question": x["question"],
            "reference": _mcq_reference(choices),
            "answer": _letter(correct_idx),
        }

    return _map_normalized(ds, normalize)


# ============================================================
# 4. HellaSwag
# ============================================================

def load_hellaswag(
    split: str = "validation",
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "Rowan/hellaswag",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )

    def normalize(x: dict[str, Any]) -> dict[str, Any]:
        choices = list(x["endings"])

        try:
            idx = int(x["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid HellaSwag label: {x.get('label')!r}") from exc

        if not 0 <= idx < len(choices):
            raise ValueError(
                f"HellaSwag label {idx} outside {len(choices)} choices"
            )

        return {
            "question": x["ctx"],
            "reference": _mcq_reference(choices),
            "answer": _letter(idx),
        }

    return _map_normalized(raw, normalize)


# ============================================================
# 5. Social IQa (SIQA)
# ============================================================

def load_siqa(
    split: str = "validation",
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "allenai/social_i_qa",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        trust_remote_code=True,
        **kwargs,
    )

    def normalize(x: dict[str, Any]) -> dict[str, Any]:
        choices = [x["answerA"], x["answerB"], x["answerC"]]

        try:
            # Original SIQA labels are 1, 2, 3.
            idx = int(x["label"]) - 1
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid SIQA label: {x.get('label')!r}") from exc

        if not 0 <= idx < len(choices):
            raise ValueError(f"SIQA label {idx} outside {len(choices)} choices")

        return {
            "question": x["question"],
            "reference": {
                "context": x["context"],
                "choices": choices,
            },
            "answer": _letter(idx),
        }

    return _map_normalized(raw, normalize)


# ============================================================
# 6. PIQA
# ============================================================

def load_piqa(
    split: str = "validation",
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "ybisk/piqa",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        trust_remote_code=True,
        **kwargs,
    )

    def normalize(x: dict[str, Any]) -> dict[str, Any]:
        choices = [x["sol1"], x["sol2"]]

        try:
            idx = int(x["label"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid PIQA label: {x.get('label')!r}") from exc

        # PIQA test labels are unavailable (-1), so a split without
        # gold labels should not be treated as question+answer data.
        if idx < 0:
            raise ValueError(
                "This PIQA split has no gold answers. "
                "Use train or validation, not test."
            )

        if not 0 <= idx < len(choices):
            raise ValueError(f"PIQA label {idx} outside {len(choices)} choices")

        return {
            "question": x["goal"],
            "reference": _mcq_reference(choices),
            "answer": _letter(idx),
        }

    return _map_normalized(raw, normalize)


# ============================================================
# 7. GSM8K
# ============================================================

def load_gsm8k(
    split: str = "test",
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "openai/gsm8k",
        "main",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )

    def normalize(x: dict[str, Any]) -> dict[str, Any]:
        full_solution = x["answer"]

        if "####" in full_solution:
            reasoning, final_answer = full_solution.rsplit("####", 1)
            reasoning = reasoning.strip()
            final_answer = final_answer.strip()
        else:
            # Defensive fallback if formatting ever changes.
            reasoning = None
            final_answer = full_solution.strip()

        return {
            "question": x["question"],
            "reference": (
                {"reasoning": reasoning}
                if reasoning
                else None
            ),
            "answer": final_answer,
        }

    return _map_normalized(raw, normalize)


# ============================================================
# 8. MMLU
# ============================================================

def load_mmlu(
    split: str = "test",
    *,
    subset: str = "all",
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "cais/mmlu",
        subset,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )

    def normalize(x: dict[str, Any]) -> dict[str, Any]:
        choices = list(x["choices"])

        try:
            idx = int(x["answer"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid MMLU answer: {x.get('answer')!r}") from exc

        if not 0 <= idx < len(choices):
            raise ValueError(f"MMLU answer {idx} outside {len(choices)} choices")

        return {
            "question": x["question"],
            "reference": _mcq_reference(choices),
            "answer": _letter(idx),
        }

    return _map_normalized(raw, normalize)


# ============================================================
# 9. OpenBookQA
# ============================================================

def load_openbookqa(
    split: str = "test",
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "allenai/openbookqa",
        "main",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )

    def normalize(x: dict[str, Any]) -> dict[str, Any]:
        choices = list(x["choices"]["text"])
        labels = [str(v) for v in x["choices"]["label"]]
        answer_key = str(x["answerKey"])

        if answer_key not in labels:
            raise ValueError(
                f"OpenBookQA answerKey={answer_key!r} "
                f"not found in labels={labels!r}"
            )

        idx = labels.index(answer_key)

        reference: dict[str, Any] = {"choices": choices}

        # Some versions/configurations may expose a supporting fact.
        # Keep it as reference if present, but never treat it as the answer.
        if x.get("fact1"):
            reference["context"] = x["fact1"]

        return {
            "question": x["question_stem"],
            "reference": reference,
            "answer": _letter(idx),
        }

    return _map_normalized(raw, normalize)


# ============================================================
# 10. Natural Questions Open
# ============================================================

def load_nq_open(
    split: str = "validation",
    *,
    cache_dir: str | None = None,
    streaming: bool = False,
    **kwargs: Any,
) -> DatasetLike:
    raw = load_dataset(
        "google-research-datasets/nq_open",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        **kwargs,
    )

    def normalize(x: dict[str, Any]) -> dict[str, Any]:
        answers = x["answer"]

        if isinstance(answers, list):
            gold_answers = [str(a) for a in answers]
        else:
            gold_answers = [str(answers)]

        if not gold_answers:
            raise ValueError("NQ-Open example has no gold answer")

        return {
            "question": x["question"],
            "reference": None,
            "answer": gold_answers,
        }

    return _map_normalized(raw, normalize)


# ============================================================
# Registry
# ============================================================

DATASET_LOADERS = {
    "oasst1": load_oasst1,
    "arc_easy": load_arc_easy,
    "arc_challenge": load_arc_challenge,
    "hellaswag": load_hellaswag,
    "siqa": load_siqa,
    "piqa": load_piqa,
    "gsm8k": load_gsm8k,
    "mmlu": load_mmlu,
    "openbookqa": load_openbookqa,
    "nq_open": load_nq_open,
}


# ============================================================
# Sample / sanity check
# ============================================================

if __name__ == "__main__":
    test_loaders = [
        ("OASST1", lambda: load_oasst1(num_examples=3200)),
        ("ARC-Easy", lambda: load_arc_easy(split="validation")),
        ("ARC-Challenge", lambda: load_arc_challenge(split="validation")),
        ("HellaSwag", lambda: load_hellaswag(split="validation")),
        ("SIQA", lambda: load_siqa(split="validation")),
        ("PIQA", lambda: load_piqa(split="validation")),
        ("GSM8K", lambda: load_gsm8k(split="test")),
        ("MMLU", lambda: load_mmlu(split="test", subset="all")),
        ("OpenBookQA", lambda: load_openbookqa(split="test")),
        ("NQ-Open", lambda: load_nq_open(split="validation")),
    ]

    separator = "=" * 100

    for dataset_name, loader in test_loaders:
        print(f"\n{separator}")
        print(f"[{dataset_name}]")
        print(separator)

        try:
            ds = loader()
            sample = next(iter(ds))

            print(f"Dataset object: {ds}")
            print("\nNormalized sample:")
            print(f"question  : {sample.get('question')}")
            print(f"reference : {sample.get('reference')}")
            print(f"answer    : {sample.get('answer')}")

            # MCQ consistency check.
            reference = sample.get("reference")
            answer = sample.get("answer")

            if (
                isinstance(reference, dict)
                and reference.get("choices") is not None
                and isinstance(answer, str)
                and len(answer) == 1
                and "A" <= answer <= "Z"
            ):
                choices = reference["choices"]
                idx = ord(answer) - ord("A")

                print("\nMCQ sanity check:")

                if 0 <= idx < len(choices):
                    print(f"{answer} -> {choices[idx]}")
                    print("MATCH: answer points to a valid choice.")
                else:
                    print(
                        f"WARNING: answer={answer} is outside "
                        f"{len(choices)} choices."
                    )

        except Exception as exc:
            print(f"FAILED to load/check {dataset_name}")
            print(f"{type(exc).__name__}: {exc}")
