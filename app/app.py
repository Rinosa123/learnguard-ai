"""Streamlit application for the LearnGuard AI project."""

import sys
from pathlib import Path

import streamlit as st

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.readability import analyse_story, interpret_reading_ease


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
        try:
            results = analyse_story(story_text)

            st.success("Story analysis completed.")

            column1, column2, column3, column4 = st.columns(4)

            column1.metric("Words", results["word_count"])
            column2.metric("Sentences", results["sentence_count"])
            column3.metric("Reading Ease", results["reading_ease"])
            column4.metric("Grade Level", results["grade_level"])

            difficulty = interpret_reading_ease(
                results["reading_ease"]
            )

            st.subheader("Readability Interpretation")
            st.info(difficulty)

            st.caption(
                "These readability measurements are estimates. "
                "Teachers should also consider vocabulary, "
                "cultural context and learner needs."
            )

        except ValueError as error:
            st.error(str(error))