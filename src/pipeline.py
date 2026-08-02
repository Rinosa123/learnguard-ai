"""Main LearnGuard pipeline orchestration."""

from typing import Any

from src.config import settings
from src.story_quality import evaluate_story_quality


DEMO_TOPIC = (
    "honesty when a child finds "
    "a lost wallet"
)

DEMO_STORY = """
It was a sunny day in the park when Emily noticed something shiny on the ground. She picked it up and discovered a small leather wallet. Inside were some money, a business card and an identification card. Emily knew that taking something that belonged to another person would be wrong. She decided to ask her friend Timmy to help her find the owner of the lost wallet.

Emily and Timmy examined the business card and found a telephone number. With help from Emily's mother, they called the number. A worried man named Mr. Johnson answered. He explained that he had lost his wallet while walking through the park earlier that day. Emily's mother arranged for everyone to meet at the nearby community centre, where the wallet could be returned safely.

When Mr. Johnson arrived with his wife, Mrs. Johnson, he correctly described the wallet and everything inside it. Emily then handed it to him. Mr. Johnson thanked Emily and Timmy for protecting his belongings and trying so hard to locate him. Emily felt proud because she had made an honest and responsible decision.

On their way home, Emily and Timmy talked about what had happened. They understood that honesty helps people trust one another and makes the community safer. Emily learned that when a child finds something that belongs to someone else, returning it is more valuable than keeping it. The experience reminded both children that doing the right thing can bring relief and happiness to others.
""".strip()


DEMO_QA_PAIRS = [
    {
        "pair_id": "QA01",
        "question": (
            "Where did the man walk through "
            "earlier that day?"
        ),
        "answer": "the park",
        "story_section": "beginning",
        "qc_level": "strict",
    },
    {
        "pair_id": "QA02",
        "question": (
            "What did Emily discover "
            "on the ground?"
        ),
        "answer": "a small leather wallet",
        "story_section": "beginning",
        "qc_level": "strict",
    },
    {
        "pair_id": "QA03",
        "question": (
            "Who did Emily ask to help her "
            "find the owner of the wallet?"
        ),
        "answer": "Timmy",
        "story_section": "beginning",
        "qc_level": "strict",
    },
    {
        "pair_id": "QA04",
        "question": (
            "What did Emily and Timmy look "
            "for on the business card?"
        ),
        "answer": "a telephone number",
        "story_section": "middle",
        "qc_level": "strict",
    },
    {
        "pair_id": "QA05",
        "question": (
            "What did Emily feel proud of?"
        ),
        "answer": (
            "an honest and responsible decision"
        ),
        "story_section": "ending",
        "qc_level": "strict",
    },
]


def run_demo_pipeline(
    requested_topic: str,
    age_group: str,
) -> dict[str, Any]:
    """Return the human-approved portfolio demonstration."""

    quality_report = evaluate_story_quality(
        DEMO_TOPIC,
        DEMO_STORY,
    )

    return {
        "mode": "demo",
        "requested_topic": (
            requested_topic.strip()
        ),
        "demonstrated_topic": DEMO_TOPIC,
        "age_group": age_group,
        "story": DEMO_STORY,
        "story_quality": quality_report,
        "qa_pairs": DEMO_QA_PAIRS,
        "approved_question_count": len(
            DEMO_QA_PAIRS
        ),
        "pipeline_status": "COMPLETE",
        "notice": (
            "This local portfolio demonstration "
            "uses a saved, human-approved result. "
            "Full model inference requires a "
            "GPU environment and configured "
            "model paths."
        ),
    }


def run_gpu_pipeline(
    topic: str,
    age_group: str,
) -> dict[str, Any]:
    """Entry point reserved for GPU model inference."""

    if not settings.gpu_models_configured:
        raise RuntimeError(
            "GPU mode requires all three "
            "fine-tuned model paths."
        )

    raise NotImplementedError(
        "The GPU model-loading adapter will "
        "be connected after the local demo "
        "interface is complete."
    )


def run_pipeline(
    topic: str,
    age_group: str = "8–10",
) -> dict[str, Any]:
    """Run LearnGuard in its configured mode."""

    clean_topic = topic.strip()

    if not clean_topic:
        raise ValueError(
            "Please enter a topic."
        )

    if settings.app_mode == "demo":
        return run_demo_pipeline(
            requested_topic=clean_topic,
            age_group=age_group,
        )

    if settings.app_mode == "gpu":
        return run_gpu_pipeline(
            topic=clean_topic,
            age_group=age_group,
        )

    raise ValueError(
        "LEARNGUARD_MODE must be "
        "'demo' or 'gpu'."
    )