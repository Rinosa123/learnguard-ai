# LearnGuard AI

An AI-powered pipeline for generating reliable reading-comprehension
questions with automated quality checking.

> **Project status:** Under active development

## Overview

LearnGuard AI helps teachers generate reading-comprehension activities
from children's stories.

The system is designed to generate questions and answers, evaluate their
quality, and reject unsuitable outputs before they are shown to the user.

## Problem

Large language models can generate educational content quickly, but the
generated questions may be:

- unanswerable from the supplied story;
- factually inconsistent;
- duplicated or unclear;
- inappropriate for the learner's reading level;
- based on information not present in the story.

LearnGuard AI addresses this problem by adding automated validation and
quality-control stages to the generation pipeline.

## Planned Workflow

1. Accept a children's story.
2. Analyse the story and its reading difficulty.
3. Identify suitable answer spans.
4. Generate comprehension questions.
5. Generate or extract expected answers.
6. Evaluate question answerability and relevance.
7. Assign a quality score.
8. Accept, reject, or flag each question for review.
9. Export the approved questions as a worksheet.

## Planned Features

- Story input and document upload
- Reading-level analysis
- Answer-span identification
- Question generation
- Question-answer validation
- Duplicate-question detection
- Quality scoring and gating
- Teacher review interface
- Exportable comprehension worksheets
- Model evaluation dashboard

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- scikit-learn
- FastAPI
- Streamlit or Gradio
- pytest
- Docker
- GitHub Actions

## Proposed Project Structure

```text
learnguard-ai/
├── app/
├── configs/
├── data/
│   └── sample/
├── notebooks/
├── reports/
│   └── figures/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   └── pipeline/
├── tests/
├── .github/
│   └── workflows/
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
