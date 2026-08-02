"""QA verification and quality-control functions for LearnGuard AI."""

from collections import Counter
import re
from typing import Any, Callable


QC_THRESHOLDS = (
    {
        "level": "strict",
        "f1_generative": 0.85,
        "f1_extractive": 0.50,
        "f1_agreement": 0.50,
        "extractive_confidence": 0.20,
    },
    {
        "level": "medium",
        "f1_generative": 0.85,
        "f1_extractive": 0.45,
        "f1_agreement": 0.45,
        "extractive_confidence": 0.15,
    },
    {
        "level": "mild",
        "f1_generative": 0.80,
        "f1_extractive": 0.40,
        "f1_agreement": 0.40,
        "extractive_confidence": 0.10,
    },
)


def normalize_answer(
    answer: str,
) -> str:
    """Apply SQuAD-style answer normalization."""

    normalized = (answer or "").lower()

    normalized = re.sub(
        r"[^a-z0-9\s]",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\b(a|an|the)\b",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def normalized_token_f1(
    prediction: str,
    reference: str,
) -> float:
    """Calculate token F1 after normalization."""

    predicted_tokens = (
        normalize_answer(
            prediction
        ).split()
    )

    reference_tokens = (
        normalize_answer(
            reference
        ).split()
    )

    if not predicted_tokens or not reference_tokens:
        return float(
            predicted_tokens
            == reference_tokens
        )

    predicted_counts = Counter(
        predicted_tokens
    )

    reference_counts = Counter(
        reference_tokens
    )

    shared_count = sum(
        min(
            predicted_counts[token],
            reference_counts[token],
        )
        for token in set(reference_counts)
    )

    if shared_count == 0:
        return 0.0

    precision = (
        shared_count
        / len(predicted_tokens)
    )

    recall = (
        shared_count
        / len(reference_tokens)
    )

    return (
        2
        * precision
        * recall
        / (precision + recall)
    )


def first_answer_sentence(
    answer: str,
) -> str:
    """Retain the first sentence from a verbose model answer."""

    parts = re.split(
        r"(?<=[.!?])\s+",
        (answer or "").strip(),
    )

    return (
        parts[0].strip()
        if parts
        else ""
    )


def contains_answer_tokens(
    prediction: str,
    reference: str,
) -> bool:
    """Check whether reference tokens occur consecutively."""

    prediction_tokens = (
        normalize_answer(
            prediction
        ).split()
    )

    reference_tokens = (
        normalize_answer(
            reference
        ).split()
    )

    if not prediction_tokens or not reference_tokens:
        return False

    reference_length = len(
        reference_tokens
    )

    maximum_start = (
        len(prediction_tokens)
        - reference_length
        + 1
    )

    for start in range(
        maximum_start
    ):
        window = prediction_tokens[
            start:start + reference_length
        ]

        if window != reference_tokens:
            continue

        if (
            reference_length == 1
            and len(prediction_tokens)
            > reference_length + 2
        ):
            return False

        return True

    return False


def robust_answer_score(
    prediction: str,
    reference: str,
) -> float:
    """Compare full, concise and contained forms of an answer."""

    concise_prediction = (
        first_answer_sentence(
            prediction
        )
    )

    scores = [
        normalized_token_f1(
            prediction,
            reference,
        ),
        normalized_token_f1(
            concise_prediction,
            reference,
        ),
    ]

    if contains_answer_tokens(
        concise_prediction,
        reference,
    ):
        scores.append(1.0)

    return max(scores)


def assign_qc_level(
    f1_generative: float,
    f1_extractive: float,
    f1_agreement: float,
    extractive_confidence: float,
) -> str:
    """Assign strict, medium, mild or reject status."""

    for thresholds in QC_THRESHOLDS:
        passed = (
            f1_generative
            >= thresholds[
                "f1_generative"
            ]
            and f1_extractive
            >= thresholds[
                "f1_extractive"
            ]
            and f1_agreement
            >= thresholds[
                "f1_agreement"
            ]
            and extractive_confidence
            >= thresholds[
                "extractive_confidence"
            ]
        )

        if passed:
            return str(
                thresholds["level"]
            )

    return "reject"


def build_qa_prompt(
    story: str,
    question: str,
) -> str:
    """Create the input used by the fine-tuned T5 QA model."""

    return (
        "Answer concisely and factually "
        "based on the story.\n"
        "<story>\n"
        f"{story.strip()}\n"
        "</story>\n"
        "<question>\n"
        f"{question.strip()}\n"
        "</question>\n"
        "<answer>\n"
    )


def generate_generative_answer(
    story: str,
    question: str,
    tokenizer: Any,
    model: Any,
    device: str,
) -> str:
    """Generate an answer using the fine-tuned T5 QA model."""

    prompt = build_qa_prompt(
        story,
        question,
    )

    encoded_input = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    ).to(device)

    output = model.generate(
        **encoded_input,
        num_beams=1,
        do_sample=False,
        max_new_tokens=64,
    )

    return tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    ).strip()


def generate_extractive_answer(
    story: str,
    question: str,
    verifier: Callable[..., dict],
) -> tuple[str, float]:
    """Generate an answer using an independent QA verifier."""

    output = verifier(
        question=question,
        context=story,
    )

    return (
        str(
            output.get(
                "answer",
                "",
            )
        ).strip(),
        float(
            output.get(
                "score",
                0.0,
            )
        ),
    )


def verify_qa_candidate(
    intended_answer: str,
    generative_answer: str,
    extractive_answer: str,
    extractive_confidence: float,
) -> dict:
    """Verify one candidate using both QA model outputs."""

    f1_generative = robust_answer_score(
        generative_answer,
        intended_answer,
    )

    f1_extractive = robust_answer_score(
        extractive_answer,
        intended_answer,
    )

    f1_agreement = robust_answer_score(
        generative_answer,
        extractive_answer,
    )

    qc_level = assign_qc_level(
        f1_generative=f1_generative,
        f1_extractive=f1_extractive,
        f1_agreement=f1_agreement,
        extractive_confidence=(
            extractive_confidence
        ),
    )

    return {
        "generative_answer": (
            generative_answer
        ),
        "extractive_answer": (
            extractive_answer
        ),
        "extractive_confidence": round(
            extractive_confidence,
            4,
        ),
        "robust_f1_generative": round(
            f1_generative,
            4,
        ),
        "robust_f1_extractive": round(
            f1_extractive,
            4,
        ),
        "robust_f1_agreement": round(
            f1_agreement,
            4,
        ),
        "qc_level": qc_level,
        "approved": qc_level != "reject",
    }