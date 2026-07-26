"""Readability analysis functions for LearnGuard AI."""

import textstat


def interpret_reading_ease(score: float) -> str:
    """Convert a Flesch Reading Ease score into a simple description."""

    if score >= 90:
        return "Very easy to read"
    if score >= 80:
        return "Easy to read"
    if score >= 70:
        return "Fairly easy to read"
    if score >= 60:
        return "Suitable for most readers"
    if score >= 50:
        return "Fairly difficult to read"
    if score >= 30:
        return "Difficult to read"
    return "Very difficult to read"


def analyse_story(story: str) -> dict:
    """Calculate basic statistics and readability scores for a story."""

    cleaned_story = story.strip()

    if not cleaned_story:
        raise ValueError("The story cannot be empty.")

    return {
        "word_count": textstat.lexicon_count(
            cleaned_story,
            removepunct=True,
        ),
        "sentence_count": textstat.sentence_count(cleaned_story),
        "reading_ease": round(
            textstat.flesch_reading_ease(cleaned_story),
            2,
        ),
        "grade_level": round(
            textstat.flesch_kincaid_grade(cleaned_story),
            2,
        ),
    }