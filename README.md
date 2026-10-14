# TruthCluster AI — Unsupervised Fake News Verification

## Overview

TruthCluster AI is an LLM-powered news verification project that retrieves online evidence for a user-provided claim, organizes related evidence using unsupervised machine learning, and uses language-model analysis to help assess whether the claim is supported, contradicted, or inconclusive.

The goal is to help users examine evidence behind viral news instead of trusting a headline alone.

## Problem Statement

Misleading news can spread rapidly through social media and online news platforms. Repeated publication does not necessarily mean a claim is true. Users need a way to find relevant evidence, compare sources, and understand why a claim may or may not be supported.

## Proposed Solution

TruthCluster AI combines:

- **Evidence retrieval:** Searches for articles related to a submitted claim.
- **Article extraction:** Attempts to extract readable article content.
- **Unsupervised clustering:** Uses TF-IDF features and K-Means to group related evidence.
- **Evidence analysis:** Examines relevance, stance, duplication, and evidence quality.
- **LLM-assisted assessment:** Produces a reasoned assessment based on retrieved evidence.
- **Evidence-based reporting:** Presents the findings to help users inspect the supporting information.

## Unsupervised Learning Approach

The project uses TF-IDF text representations and K-Means clustering to organize similar evidence into groups. This helps structure the retrieved information before the language model evaluates the evidence.

Clustering identifies similarity patterns; it does not independently prove that a claim is true or false.

## Technology Stack

- Python
- Streamlit
- Scikit-learn
- Groq API
- Requests
- Beautiful Soup
- Trafilatura
- Google News link decoding

## Project Workflow

1. User enters a news claim.
2. The system retrieves related evidence.
3. Article content is extracted when available.
4. Evidence is analyzed and clustered.
5. Duplicate evidence and evidence quality are reviewed.
6. The LLM analyzes the available evidence.
7. The application presents an assessment and reasoning.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/truthcluster-ai.git
cd truthcluster-ai
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project directory:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Replace the placeholder with your own Groq API key. Never commit `.env` or share your key publicly.

### 5. Start the application

```bash
streamlit run app.py
```

## Deployment

The application is intended for deployment using Streamlit Community Cloud. Configure `GROQ_API_KEY` in the deployed app's Secrets settings rather than committing credentials to GitHub.

## Limitations

- Search results may be incomplete, outdated, or unavailable.
- Some websites block automated article extraction.
- LLM-generated assessments can be incorrect.
- Similarity clustering groups related text but does not establish factual truth.
- Results should be reviewed alongside the original sources.

## Responsible Use

TruthCluster AI is an evidence-assistance tool, not an infallible fact-checker. Users should inspect cited sources and seek independent confirmation before making important decisions.

## Project Status

Under development. Features and assessment quality may change as the evidence retrieval, clustering, and verification pipeline improves.

## Author

Manojkumar  
CSE (Artificial Intelligence and Machine Learning)  
KPR Institute of Engineering and Technology
