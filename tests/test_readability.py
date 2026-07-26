"""Tests for the LearnGuard AI readability module."""

import pytest

from src.readability import analyse_story, interpret_reading_ease


@pytest.mark.parametrize(
    ("score", "expected_description"),
    [
        (95, "Very easy to read"),
        (85, "Easy to read"),
        (75, "Fairly easy to read"),
        (65, "Suitable for most readers"),
        (55, "Fairly difficult to read"),
        (40, "Difficult to read"),
        (20, "Very difficult to read"),
    ],
)
def test_interpret_reading_ease(score, expected_description):
    """Each score range should return the correct description."""

    assert interpret_reading_ease(score) == expected_description


def test_analyse_story_returns_expected_counts():
    """A simple story should return correct word and sentence counts."""

    story = "Amina found a kitten. She gave it milk."

    results = analyse_story(story)

    assert results["word_count"] == 8
    assert results["sentence_count"] == 2
    assert isinstance(results["reading_ease"], float)
    assert isinstance(results["grade_level"], float)


def test_analyse_story_rejects_empty_text():
    """An empty story should raise a clear error."""

    with pytest.raises(ValueError, match="story cannot be empty"):
        analyse_story("   ")