"""Question-generation utilities for LearnGuard AI."""

import re
from typing import Any


QUESTION_TYPE_BY_ENTITY = {
    "PERSON": "who",
    "GPE": "where",
    "LOC": "where",
    "FAC": "where",
    "DATE": "when",
    "TIME": "when",
}


def infer_question_type(
    entity_label: str,
) -> str:
    """Infer the expected WH-question type from an entity label."""

    return QUESTION_TYPE_BY_ENTITY.get(
        entity_label.upper(),
        "what",
    )


def build_question_prompt(
    story: str,
    answer: str,
) -> str:
    """Create the input format used to fine-tune the T5 QG model."""

    return (
        "<story>\n"
        f"{story.strip()}\n"
        "</story>\n"
        "<answer>\n"
        f"{answer.strip()}\n"
        "</answer>\n"
        "<ask>\n"
    )


def clean_question(
    question: str,
) -> str:
    """Clean and standardize a generated question."""

    cleaned = re.sub(
        r"\s+",
        " ",
        question,
    ).strip()

    cleaned = cleaned.strip(
        "\"' "
    )

    if not cleaned:
        return ""

    cleaned = (
        cleaned[0].upper()
        + cleaned[1:]
    )

    if not cleaned.endswith("?"):
        cleaned += "?"

    return cleaned


def question_start_word(
    question: str,
) -> str:
    """Return the first word of a generated question."""

    words = re.findall(
        r"[a-zA-Z]+",
        question.lower(),
    )

    return words[0] if words else ""


def is_compatible_question_type(
    question: str,
    expected_type: str,
) -> bool:
    """Check whether a generated WH-word is suitable for an answer."""

    start_word = question_start_word(
        question
    )

    compatible_types = {
        "who": {"who", "what"},
        "where": {"where", "what"},
        "when": {"when", "what"},
        "what": {"what", "which", "who", "where"},
    }

    accepted_words = compatible_types.get(
        expected_type,
        {"what", "which"},
    )

    return start_word in accepted_words


def is_valid_question(
    question: str,
    answer: str,
    expected_type: str,
) -> bool:
    """Apply basic format and relevance checks to a question."""

    cleaned_question = clean_question(
        question
    )

    if not cleaned_question:
        return False

    if not cleaned_question.endswith("?"):
        return False

    word_count = len(
        cleaned_question.split()
    )

    if not 4 <= word_count <= 25:
        return False

    if not is_compatible_question_type(
        cleaned_question,
        expected_type,
    ):
        return False

    normalized_question = (
        cleaned_question.lower()
    )

    normalized_answer = answer.lower().strip()

    # Reject questions that simply contain the full answer.
    if (
        len(normalized_answer.split()) > 1
        and normalized_answer
        in normalized_question
    ):
        return False

    return True


def score_question(
    question: str,
    expected_type: str,
) -> float:
    """Assign a simple ranking score to a generated question."""

    score = 0.0

    if question.endswith("?"):
        score += 1.0

    if is_compatible_question_type(
        question,
        expected_type,
    ):
        score += 3.0

    word_count = len(question.split())

    if 5 <= word_count <= 15:
        score += 1.0

    if not re.search(
        r"\b(thing|something|anything)\b",
        question.lower(),
    ):
        score += 1.0

    length_penalty = abs(
        word_count - 9
    ) * 0.01

    return round(
        score - length_penalty,
        4,
    )


def generate_question_candidates(
    story: str,
    answer_span: dict[str, Any],
    tokenizer: Any,
    model: Any,
    device: str,
    number_of_candidates: int = 8,
) -> list[dict[str, Any]]:
    """Generate and rank questions using the fine-tuned T5 model."""

    answer = str(
        answer_span["answer"]
    )

    entity_label = str(
        answer_span.get(
            "entity_label",
            "NOUN_PHRASE",
        )
    )

    expected_type = (
        infer_question_type(
            entity_label
        )
    )

    prompt = build_question_prompt(
        story,
        answer,
    )

    model_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(device)

    generated_outputs = model.generate(
        **model_inputs,
        max_new_tokens=64,
        num_beams=number_of_candidates,
        num_return_sequences=(
            number_of_candidates
        ),
        early_stopping=True,
        no_repeat_ngram_size=3,
    )

    decoded_questions = (
        tokenizer.batch_decode(
            generated_outputs,
            skip_special_tokens=True,
        )
    )

    unique_questions = []
    observed_questions = set()

    for generated_question in decoded_questions:
        cleaned_question = clean_question(
            generated_question
        )

        normalized_question = (
            cleaned_question.lower()
        )

        if normalized_question in observed_questions:
            continue

        observed_questions.add(
            normalized_question
        )

        if not is_valid_question(
            cleaned_question,
            answer,
            expected_type,
        ):
            continue

        unique_questions.append(
            {
                "answer_span": answer,
                "expected_question_type": (
                    expected_type
                ),
                "question": cleaned_question,
                "question_quality_score": (
                    score_question(
                        cleaned_question,
                        expected_type,
                    )
                ),
                "story_section": (
                    answer_span.get(
                        "story_section",
                        "unknown",
                    )
                ),
                "support_sentence_index": (
                    answer_span.get(
                        "support_sentence_index",
                        0,
                    )
                ),
            }
        )

    unique_questions.sort(
        key=lambda item: item[
            "question_quality_score"
        ],
        reverse=True,
    )

    return unique_questions