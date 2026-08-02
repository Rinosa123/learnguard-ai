"""Story quality-control functions for LearnGuard AI."""

from difflib import SequenceMatcher
import re


TOPIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but",
    "when", "where", "who", "why", "how",
    "in", "on", "at", "to", "from", "for",
    "with", "without", "of", "by", "about",
    "is", "are", "was", "were", "be",
    "being", "been", "this", "that",
}

EXCLUDED_NAME_WORDS = {
    "One", "Once", "The", "When", "While",
    "With", "Without", "After", "Before",
    "From", "Then", "Suddenly", "Seeing",
    "Feeling", "Every", "Everyone", "People",
    "Inside", "Outside", "Just", "That",
    "This", "There", "They", "His", "Her",
    "Their", "And", "But", "So", "Thank",
}


def check_story_structure(story_text: str) -> dict:
    """Calculate basic structural information about a story."""

    clean_story = story_text.strip()

    words = re.findall(
        r"\b[\w'-]+\b",
        clean_story,
    )

    sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            clean_story,
        )
        if sentence.strip()
    ]

    return {
        "not_empty": bool(clean_story),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "within_usable_word_range": (
            220 <= len(words) <= 450
        ),
        "has_clear_ending": len(sentences) >= 5,
    }


def extract_topic_keywords(topic: str) -> list[str]:
    """Extract meaningful keywords from the requested topic."""

    words = re.findall(
        r"\b[a-zA-Z][a-zA-Z'-]*\b",
        topic.lower(),
    )

    return [
        word
        for word in words
        if (
            word not in TOPIC_STOPWORDS
            and len(word) > 2
        )
    ]


def check_topic_preservation(
    topic: str,
    story_text: str,
    minimum_coverage: float = 0.40,
) -> dict:
    """Check whether the generated story preserves its topic."""

    keywords = extract_topic_keywords(topic)

    story_words = set(
        re.findall(
            r"\b[a-zA-Z][a-zA-Z'-]*\b",
            story_text.lower(),
        )
    )

    present_keywords = [
        keyword
        for keyword in keywords
        if keyword in story_words
    ]

    missing_keywords = [
        keyword
        for keyword in keywords
        if keyword not in story_words
    ]

    coverage = (
        len(present_keywords) / len(keywords)
        if keywords
        else 1.0
    )

    critical_keyword = (
        keywords[-1]
        if keywords
        else None
    )

    critical_keyword_preserved = (
        critical_keyword in story_words
        if critical_keyword
        else True
    )

    topic_preserved = (
        coverage >= minimum_coverage
        and critical_keyword_preserved
    )

    return {
        "topic_keywords": keywords,
        "present_topic_keywords": present_keywords,
        "missing_topic_keywords": missing_keywords,
        "topic_keyword_coverage": round(
            coverage,
            3,
        ),
        "critical_keyword": critical_keyword,
        "critical_keyword_preserved": (
            critical_keyword_preserved
        ),
        "topic_preserved": topic_preserved,
    }


def extract_possible_names(
    story_text: str,
) -> list[str]:
    """Extract words that may represent character names."""

    candidates = re.findall(
        r"\b[A-Z][a-z]{2,}\b",
        story_text,
    )

    return sorted(
        {
            word
            for word in candidates
            if word not in EXCLUDED_NAME_WORDS
        }
    )


def find_similar_name_variants(
    story_text: str,
    similarity_threshold: float = 0.72,
) -> list[dict]:
    """Find names that may be inconsistent spellings."""

    names = extract_possible_names(
        story_text
    )

    possible_variants = []

    for first_index in range(len(names)):
        for second_index in range(
            first_index + 1,
            len(names),
        ):
            first_name = names[first_index]
            second_name = names[second_index]

            similarity = SequenceMatcher(
                None,
                first_name.lower(),
                second_name.lower(),
            ).ratio()

            same_beginning = (
                first_name[:3].lower()
                == second_name[:3].lower()
            )

            if (
                first_name != second_name
                and same_beginning
                and similarity
                >= similarity_threshold
            ):
                possible_variants.append(
                    {
                        "name_1": first_name,
                        "name_2": second_name,
                        "similarity": round(
                            similarity,
                            3,
                        ),
                    }
                )

    return possible_variants


def check_name_consistency(
    story_text: str,
) -> dict:
    """Check for possible character-name inconsistencies."""

    detected_names = extract_possible_names(
        story_text
    )

    possible_variants = (
        find_similar_name_variants(
            story_text
        )
    )

    return {
        "detected_names": detected_names,
        "possible_name_variants": (
            possible_variants
        ),
        "name_consistency_passed": (
            len(possible_variants) == 0
        ),
    }


def evaluate_story_quality(
    topic: str,
    story_text: str,
) -> dict:
    """Combine structural, topic and name checks."""

    structural_checks = (
        check_story_structure(story_text)
    )

    topic_checks = (
        check_topic_preservation(
            topic,
            story_text,
        )
    )

    name_checks = (
        check_name_consistency(story_text)
    )

    rejection_reasons = []
    review_reasons = []

    if not structural_checks["not_empty"]:
        rejection_reasons.append(
            "The story is empty."
        )

    if not structural_checks[
        "within_usable_word_range"
    ]:
        rejection_reasons.append(
            "The story is outside the "
            "220–450 word range."
        )

    if not structural_checks[
        "has_clear_ending"
    ]:
        rejection_reasons.append(
            "The story may be incomplete."
        )

    if not topic_checks[
        "critical_keyword_preserved"
    ]:
        rejection_reasons.append(
            "The critical topic keyword "
            "was not preserved."
        )

    if not topic_checks["topic_preserved"]:
        rejection_reasons.append(
            "The story has insufficient "
            "topic-keyword coverage."
        )

    if not name_checks[
        "name_consistency_passed"
    ]:
        review_reasons.append(
            "Possible inconsistent character "
            "names were detected."
        )

    if rejection_reasons:
        decision = "REJECT"
    elif review_reasons:
        decision = "REVIEW"
    else:
        decision = "PASS"

    return {
        "decision": decision,
        "rejection_reasons": rejection_reasons,
        "review_reasons": review_reasons,
        "structural_checks": structural_checks,
        "topic_checks": topic_checks,
        "name_checks": name_checks,
    }