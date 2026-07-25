"""Streamlit application for the LearnGuard AI project."""

import streamlit as st
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

    return {
        "word_count": textstat.lexicon_count(story, removepunct=True),
        "sentence_count": textstat.sentence_count(story),
        "reading_ease": round(textstat.flesch_reading_ease(story), 2),
        "grade_level": round(textstat.flesch_kincaid_grade(story), 2),
    }


st.set_page_config(
    page_title="LearnGuard AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 LearnGuard AI")

st.subheader("Reliable Reading-Comprehension Content Generation")

st.write(
    "Enter a children's story below. LearnGuard AI will analyse its "
    "length, readability and estimated grade level."
)

story_text = st.text_area(
    "Children's story",
    height=300,
    placeholder=(
        "Example: Amina found a small kitten near the school gate. "
        "The kitten was hungry, so Amina gave it some milk."
    ),
)

analyse_button = st.button("Analyse Story", type="primary")

if analyse_button:
    if not story_text.strip():
        st.warning("Please enter a story before clicking Analyse Story.")
    else:
        results = analyse_story(story_text)

        st.success("Story analysis completed.")

        column1, column2, column3, column4 = st.columns(4)

        column1.metric("Words", results["word_count"])
        column2.metric("Sentences", results["sentence_count"])
        column3.metric("Reading Ease", results["reading_ease"])
        column4.metric("Grade Level", results["grade_level"])

        difficulty = interpret_reading_ease(results["reading_ease"])

        st.subheader("Readability Interpretation")
        st.info(difficulty)

        st.caption(
            "These readability measurements are estimates. Teachers should "
            "also consider vocabulary, cultural context and learner needs."
        )
