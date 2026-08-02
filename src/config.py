"""Central configuration for the LearnGuard AI application."""

from dataclasses import dataclass
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def optional_path(environment_variable: str) -> Path | None:
    """Return a configured model path or None when it is unavailable."""

    value = os.getenv(environment_variable)

    if not value:
        return None

    return Path(value).expanduser()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    app_mode: str
    story_model_path: Path | None
    question_model_path: Path | None
    answer_model_path: Path | None
    verifier_model_name: str
    maximum_questions: int

    @property
    def gpu_models_configured(self) -> bool:
        """Check whether all three fine-tuned model paths are configured."""

        required_paths = (
            self.story_model_path,
            self.question_model_path,
            self.answer_model_path,
        )

        return all(
            path is not None
            for path in required_paths
        )


def load_settings() -> Settings:
    """Load LearnGuard settings without exposing local model paths in code."""

    return Settings(
        app_mode=os.getenv(
            "LEARNGUARD_MODE",
            "demo",
        ).lower(),
        story_model_path=optional_path(
            "LEARNGUARD_STORY_MODEL_PATH"
        ),
        question_model_path=optional_path(
            "LEARNGUARD_QUESTION_MODEL_PATH"
        ),
        answer_model_path=optional_path(
            "LEARNGUARD_ANSWER_MODEL_PATH"
        ),
        verifier_model_name=os.getenv(
            "LEARNGUARD_VERIFIER_MODEL",
            "deepset/roberta-base-squad2",
        ),
        maximum_questions=int(
            os.getenv(
                "LEARNGUARD_MAX_QUESTIONS",
                "5",
            )
        ),
    )


settings = load_settings()