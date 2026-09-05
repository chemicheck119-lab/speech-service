"""Dependency-free Korean ASR metrics used by the evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
import statistics
import unicodedata


_NON_TEXT = re.compile(r"[^0-9a-zA-Z가-힣\s]")
_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = _NON_TEXT.sub(" ", normalized)
    return _SPACE.sub(" ", normalized).strip()


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, reference_item in enumerate(reference, start=1):
        current = [row]
        for column, hypothesis_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_item != hypothesis_item),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class RecordMetric:
    character_edits: int
    reference_characters: int
    word_edits: int
    reference_words: int

    @property
    def cer(self) -> float:
        return self.character_edits / max(1, self.reference_characters)


def score_record(reference: str, hypothesis: str) -> RecordMetric:
    normalized_reference = normalize_text(reference)
    normalized_hypothesis = normalize_text(hypothesis)
    reference_characters = normalized_reference.replace(" ", "")
    hypothesis_characters = normalized_hypothesis.replace(" ", "")
    reference_words = normalized_reference.split()
    hypothesis_words = normalized_hypothesis.split()
    return RecordMetric(
        character_edits=edit_distance(reference_characters, hypothesis_characters),
        reference_characters=len(reference_characters),
        word_edits=edit_distance(reference_words, hypothesis_words),
        reference_words=len(reference_words),
    )


def term_presence_counts(
    references: list[str], hypotheses: list[str], terms: list[str]
) -> dict[str, float | int | None]:
    true_positive = false_negative = false_positive = 0
    normalized_terms = [normalize_text(term).replace(" ", "") for term in terms]
    if len(references) != len(hypotheses):
        raise ValueError("reference and hypothesis counts must match")
    for reference, hypothesis in zip(references, hypotheses):
        normalized_reference = normalize_text(reference).replace(" ", "")
        normalized_hypothesis = normalize_text(hypothesis).replace(" ", "")
        for term in normalized_terms:
            in_reference = term in normalized_reference
            in_hypothesis = term in normalized_hypothesis
            true_positive += int(in_reference and in_hypothesis)
            false_negative += int(in_reference and not in_hypothesis)
            false_positive += int(not in_reference and in_hypothesis)
    recall_denominator = true_positive + false_negative
    precision_denominator = true_positive + false_positive
    return {
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_insertion": false_positive,
        "recall": (
            true_positive / recall_denominator if recall_denominator else None
        ),
        "precision": (
            true_positive / precision_denominator if precision_denominator else None
        ),
    }


def paired_bootstrap_cer_delta(
    baseline: list[RecordMetric],
    hinted: list[RecordMetric],
    *,
    samples: int = 2_000,
    seed: int = 119,
) -> dict[str, float | int]:
    if len(baseline) != len(hinted) or not baseline:
        raise ValueError("paired non-empty metric lists are required")
    rng = random.Random(seed)
    deltas: list[float] = []
    record_count = len(baseline)
    for _ in range(samples):
        selected = [rng.randrange(record_count) for _ in range(record_count)]
        baseline_edits = sum(baseline[index].character_edits for index in selected)
        hinted_edits = sum(hinted[index].character_edits for index in selected)
        reference_length = sum(
            baseline[index].reference_characters for index in selected
        )
        denominator = max(1, reference_length)
        deltas.append((hinted_edits - baseline_edits) / denominator)
    deltas.sort()
    low = deltas[int(samples * 0.025)]
    high = deltas[min(samples - 1, int(samples * 0.975))]
    return {
        "metric": "hinted_cer_minus_baseline_cer",
        "estimate": statistics.fmean(deltas),
        "ci95_low": low,
        "ci95_high": high,
        "samples": samples,
        "seed": seed,
    }
