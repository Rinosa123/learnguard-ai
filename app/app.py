"""Streamlit interface for the LearnGuard AI project."""

import json
import sys
from pathlib import Path

import streamlit as st

# Allow imports from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline
from src.readability import analyse_story, interpret_reading_ease


st.set_page_config(
    page_title="LearnGuard AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 LearnGuard AI")

st.subheader(
    "Reliable Children's Story and "
    "Reading-Comprehension Generation"
)

st.write(
    "Enter a topic and age group. LearnGuard AI produces a "
    "quality-checked story and verified question-answer pairs."
)

st.info(
    "Portfolio Demo Mode: This local version displays a saved, "
    "human-approved result from the complete LearnGuard pipeline. "
    "Full model inference requires a GPU environment."
)

topic = st.text_input(
    "Story topic",
    value="honesty when a child finds a lost wallet",
    placeholder="Example: kindness to a new student",
)

age_group = st.selectbox(
    "Age group",
    options=["6–8", "8–10", "10–12"],
    index=1,
)

generate_button = st.button(
    "Generate Learning Activity",
    type="primary",
)

if generate_button:
    if not topic.strip():
        st.warning("Please enter a story topic.")
    else:
        try:
            with st.spinner("Running the LearnGuard pipeline..."):
                result = run_pipeline(
                    topic=topic,
                    age_group=age_group,
                )

            st.success("Learning activity generated successfully.")

            st.caption(result["notice"])

            if (
                result["requested_topic"].lower()
                != result["demonstrated_topic"].lower()
            ):
                st.warning(
                    "Demo Mode currently uses the validated topic: "
                    f"'{result['demonstrated_topic']}'. "
                    "Your requested topic will be supported when "
                    "GPU inference is connected."
                )

            st.subheader("Generated Story")

            st.write(result["story"])

            readability = analyse_story(result["story"])

            column1, column2, column3, column4 = st.columns(4)

            column1.metric(
                "Words",
                readability["word_count"],
            )
            column2.metric(
                "Sentences",
                readability["sentence_count"],
            )
            column3.metric(
                "Reading Ease",
                readability["reading_ease"],
            )
            column4.metric(
                "Grade Level",
                readability["grade_level"],
            )

            st.subheader("Story Quality Control")

            quality_decision = result["story_quality"]["decision"]

            quality_column1, quality_column2, quality_column3 = (
                st.columns(3)
            )

            quality_column1.metric(
                "QC Decision",
                quality_decision,
            )
            quality_column2.metric(
                "Pipeline Status",
                result["pipeline_status"],
            )
            quality_column3.metric(
                "Approved Questions",
                result["approved_question_count"],
            )

            if quality_decision == "PASS":
                st.success(
                    "The story passed the automated quality checks."
                )
            elif quality_decision == "REVIEW":
                st.warning(
                    "The story requires human review."
                )
            else:
                st.error(
                    "The story did not pass the quality checks."
                )

            difficulty = interpret_reading_ease(
                readability["reading_ease"]
            )

            st.info(f"Readability: {difficulty}")

            st.subheader("Verified Questions and Answers")

            for pair in result["qa_pairs"]:
                with st.expander(
                    f"{pair['pair_id']}: {pair['question']}"
                ):
                    st.write(f"**Answer:** {pair['answer']}")
                    st.write(
                        f"**Story section:** "
                        f"{pair['story_section'].title()}"
                    )
                    st.write(
                        f"**Quality level:** "
                        f"{pair['qc_level'].title()}"
                    )

            downloadable_result = {
                "topic": result["demonstrated_topic"],
                "age_group": result["age_group"],
                "story": result["story"],
                "qa_pairs": result["qa_pairs"],
                "pipeline_status": result["pipeline_status"],
            }

            st.download_button(
                label="Download Activity as JSON",
                data=json.dumps(
                    downloadable_result,
                    indent=2,
                    ensure_ascii=False,
                ),
                file_name="learnguard_activity.json",
                mime="application/json",
            )

            st.caption(
                "Readability measurements are estimates. Teachers "
                "should also consider vocabulary, cultural context "
                "and individual learner needs."
            )

        except (ValueError, RuntimeError, NotImplementedError) as error:
            st.error(str(error))
        except Exception as error:
            st.error(
                "An unexpected error occurred: "
                f"{error}"
            )