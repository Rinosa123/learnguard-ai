"""Answer-span ranking and selection for LearnGuard AI."""

import re
from typing import Any


VAGUE_ANSWERS = {
    "a child",
    "another person",
    "help",
    "one",
    "something",
    "someone",
    "the thing",
    "what",
}


def normalize_answer(answer: str) -> str:
    """Create a normalized form used for duplicate detection."""

    normalized = answer.lower().strip()

    normalized = re.sub(
        r"[^a-z0-9\s'-]",
        "",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized


def assign_story_section(
    sentence_index: int,
    total_sentences: int,
) -> str:
    """Assign a sentence to the beginning, middle or ending."""

    if total_sentences <= 0:
        return "beginning"

    position = sentence_index / total_sentences

    if position < 1 / 3:
        return "beginning"

    if position < 2 / 3:
        return "middle"

    return "ending"


def is_useful_answer(answer: str) -> bool:
    """Reject empty, vague, extremely short or long answer spans."""

    normalized = normalize_answer(answer)

    if not normalized:
        return False

    if normalized in VAGUE_ANSWERS:
        return False

    word_count = len(normalized.split())

    return 1 <= word_count <= 8


def remove_duplicate_spans(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep the highest-scoring candidate for each answer concept."""

    sorted_candidates = sorted(
        candidates,
        key=lambda item: float(
            item.get("score", 0.0)
        ),
        reverse=True,
    )

    unique_candidates = []
    observed_answers = set()

    for candidate in sorted_candidates:
        answer = str(
            candidate.get(
                "answer",
                candidate.get(
                    "answer_span",
                    "",
                ),
            )
        ).strip()

        normalized_answer = normalize_answer(
            answer
        )

        if not is_useful_answer(answer):
            continue

        if normalized_answer in observed_answers:
            continue

        observed_answers.add(
            normalized_answer
        )

        cleaned_candidate = candidate.copy()
        cleaned_candidate["answer"] = answer
        cleaned_candidate[
            "concept_key"
        ] = normalized_answer

        unique_candidates.append(
            cleaned_candidate
        )

    return unique_candidates


def select_balanced_spans(
    candidates: list[dict[str, Any]],
    total_sentences: int,
    maximum_spans: int = 12,
) -> list[dict[str, Any]]:
    """Select high-quality spans from all three story sections."""

    unique_candidates = remove_duplicate_spans(
        candidates
    )

    section_order = (
        "beginning",
        "middle",
        "ending",
    )

    grouped_candidates = {
        section: []
        for section in section_order
    }

    for candidate in unique_candidates:
        sentence_index = int(
            candidate.get(
                "support_sentence_index",
                0,
            )
        )

        story_section = candidate.get(
            "story_section"
        )

        if story_section not in grouped_candidates:
            story_section = assign_story_section(
                sentence_index,
                total_sentences,
            )

        candidate["story_section"] = (
            story_section
        )

        grouped_candidates[
            story_section
        ].append(candidate)

    for section in section_order:
        grouped_candidates[section].sort(
            key=lambda item: float(
                item.get("score", 0.0)
            ),
            reverse=True,
        )

    base_quota = maximum_spans // 3
    selected_spans = []

    for section in section_order:
        selected_spans.extend(
            grouped_candidates[section][
                :base_quota
            ]
        )

    selected_keys = {
        item["concept_key"]
        for item in selected_spans
    }

    remaining_candidates = [
        candidate
        for candidate in unique_candidates
        if candidate["concept_key"]
        not in selected_keys
    ]

    remaining_candidates.sort(
        key=lambda item: float(
            item.get("score", 0.0)
        ),
        reverse=True,
    )

    available_places = (
        maximum_spans
        - len(selected_spans)
    )

    selected_spans.extend(
        remaining_candidates[
            :available_places
        ]
    )

    section_position = {
        "beginning": 0,
        "middle": 1,
        "ending": 2,
    }

    selected_spans.sort(
        key=lambda item: (
            section_position.get(
                item["story_section"],
                3,
            ),
            int(
                item.get(
                    "support_sentence_index",
                    0,
                )
            ),
            -float(
                item.get(
                    "score",
                    0.0,
                )
            ),
        )
    )

    for rank, candidate in enumerate(
        selected_spans,
        start=1,
    ):
        candidate["rank"] = rank

    return selected_spans