# TruthCluster AI — Unsupervised Fake News Verification

**Verify the claim. Explore the evidence. Understand the uncertainty.**

TruthCluster AI is an AI-powered news verification web application that helps users investigate online claims by retrieving related articles, organizing evidence with unsupervised machine learning, and using a large language model (LLM) to assess the available evidence.

**Team:** The Outliers  
**Team Members:** Manojkumar R, Jesvanthkavi A  
**Domain:** Artificial Intelligence, Machine Learning, Natural Language Processing

<p align="center">

  **[🚀 Try the Live Demo](https://truthcluster-ai-sysba84agb2rtbaybnxzi8.streamlit.app/)**

  **[💻 GitHub Repository](https://github.com/manoj-xai/truthcluster-ai)**

</p>

---

## 1. Problem Statement

Misinformation and misleading claims spread rapidly through social media and online news platforms. Users often find it difficult to distinguish reliable information from unsupported claims.

Traditional manual fact-checking requires users to search multiple sources, compare articles, identify repeated reporting, and understand conflicting evidence. Finding the same claim on several websites does not necessarily mean the claim is true.

TruthCluster AI aims to make this process more structured by collecting related evidence, grouping similar articles, and presenting an evidence-based assessment.

## 2. Proposed Solution

TruthCluster AI combines online evidence retrieval, unsupervised machine learning, evidence processing, and LLM-assisted reasoning in one workflow.

Users enter a news claim, and the system searches for related articles, extracts available content, organizes similar articles into clusters, selects relevant evidence, and generates an assessment with reasoning.

Possible assessment outcomes include:

- **SUPPORTED** — the available evidence supports the claim.
- **CONTRADICTED** — the available evidence contradicts the claim.
- **UNDETERMINED** — the evidence is insufficient, unclear, or inconclusive.

These labels represent the system's assessment of the retrieved evidence, not a guarantee of absolute truth.

## 3. Key Features

- **News Claim Input:** Enter a claim to investigate.
- **Online Evidence Retrieval:** Search for related news articles and supporting information.
- **Article Extraction:** Collect available article text and metadata.
- **Unsupervised Clustering:** Use TF-IDF and K-Means to group similar articles.
- **Evidence Relevance Analysis:** Identify articles and sentences that appear relevant to the claim.
- **Duplicate Screening:** Detect potential duplicate or highly similar articles.
- **Evidence Quality Review:** Screen retrieved evidence for relevance and quality indicators.
- **LLM-Assisted Assessment:** Use a Groq-powered language model to analyze selected evidence.
- **Reasoning and Sources:** Present an assessment, explanation, and available source information.
- **Undetermined Results:** Allow inconclusive outcomes when the evidence does not support a confident assessment.

## 4. System Architecture

The application follows a multi-stage evidence-processing pipeline.

```text
          User Enters a News Claim
                     |
                     v
          Online Evidence Search
                     |
                     v
          Article Content Extraction
                     |
                     v
       Text Processing and Relevance Analysis
                     |
                     v
       TF-IDF Representation of Article Text
                     |
                     v
           K-Means Clustering
                     |
                     v
       Claim and Stance Analysis
                     |
                     v
        Duplicate and Quality Screening
                     |
                     v
          Relevant Evidence Selection
                     |
                     v
       Groq-Powered LLM Evidence Analysis
                     |
                     v
         Final Assessment and Reasoning
                     |
                     v
          Results in Streamlit Web App
```

### How the workflow operates

1. The user submits a news claim through the web interface.
2. The search stage retrieves related articles.
3. The extraction stage collects readable article content where accessible.
4. Text analysis estimates relevance between the claim and retrieved articles.
5. TF-IDF converts text into numerical feature representations.
6. K-Means groups articles with similar textual characteristics.
7. Additional stages analyze claim relevance, possible stance, duplicates, and evidence quality.
8. A selection stage chooses evidence for the final LLM analysis.
9. The language model generates an assessment and explanation based on the selected evidence.
10. The application displays the result to the user.

## 5. Unsupervised Machine Learning Approach

A central component of TruthCluster AI is the use of unsupervised learning to organize related evidence.

### TF-IDF

Term Frequency–Inverse Document Frequency (TF-IDF) represents article text numerically according to the importance of terms within documents and across the document collection.

### K-Means Clustering

K-Means groups articles according to similarities in their numerical text representations. This helps organize related articles into clusters for further examination.

### Why unsupervised learning?

The clustering stage does not require a manually assigned true/fake label for every article. Instead, it discovers groups of textually similar articles in the retrieved collection.

**Important distinction:** Clustering identifies similarity, not truth. Similar articles may repeat the same original report, and different articles may discuss the same topic without independently verifying a claim.

## 6. What Makes TruthCluster AI Different?

TruthCluster AI combines several stages rather than relying only on a text classifier or a single LLM response.

Its key design ideas are:

- **Evidence-first workflow:** Collect relevant information before generating the final assessment.
- **Unsupervised organization:** Group similar articles to make a large collection easier to inspect.
- **Multi-stage evidence processing:** Combine relevance analysis, stance screening, duplicate detection, and evidence selection.
- **LLM-assisted reasoning:** Use selected evidence to produce an understandable assessment.
- **Uncertainty awareness:** Preserve an undetermined outcome when the evidence is insufficient.

The project's intended contribution is the integration of unsupervised evidence organization with an evidence-based LLM verification workflow.

## 7. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Web interface | Streamlit |
| Text vectorization | Scikit-learn TF-IDF |
| Unsupervised learning | Scikit-learn K-Means |
| HTTP requests | Requests |
| HTML parsing | BeautifulSoup |
| Article extraction | Trafilatura |
| News link processing | Google News RSS and `googlenewsdecoder` |
| LLM integration | Groq API |
| Environment configuration | python-dotenv |
| Version control | Git and GitHub |
| Deployment | Streamlit Community Cloud |

## 8. Getting Started

Follow these steps to run TruthCluster AI locally.

### Prerequisites

- Python installed on your computer
- Git
- A Groq API key
- Internet access

### Step 1: Clone the repository

```bash
git clone https://github.com/manoj-xai/truthcluster-ai.git
cd truthcluster-ai
```

### Step 2: Create a virtual environment

Windows:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Configure the API key

Create a `.env` file in the project root, next to `app.py`.

Add the following line, replacing the placeholder with your own Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Never commit your actual `.env` file or expose your API key in screenshots, public repositories, or logs.

### Step 5: Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`. Open that address in your browser.

**Note:** Install all dependencies from `requirements.txt`. If a dependency fails on your Python version, use a compatible Python environment and resolve the dependency issue before deployment.

## 9. Repository Structure

```text
truthcluster-ai/
│
├── app.py
├── demo_runner.py
├── evidence_search.py
├── extract_articles.py
├── analyze_evidence.py
├── cluster_evidence.py
├── analyze_claim.py
├── analyze_stance.py
├── deduplicate_evidence.py
├── build_evidence_report.py
├── analyze_evidence_quality.py
├── select_quality_evidence.py
├── llm_evidence_analysis_v6.py
├── assessment_rules.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

The repository also contains supporting scripts for evidence inspection, evaluation, and alternative pipeline versions.

Generated evidence files and evaluation outputs are excluded from Git where configured in `.gitignore`.

## 10. Testing and Evaluation

The pipeline has been exercised with claims such as:

- “The Sun is far away from Earth.”
- “The Moon is made entirely of cheese.”
- “Elon Musk has the largest satellite internet network in the world.”

These are example demonstration claims, not a formal benchmark.

The pipeline produces intermediate evidence files and a final assessment. However, an assessment that appears plausible does not establish that the system is consistently accurate.

Future evaluation should use a manually verified dataset and report suitable metrics, such as precision, recall, F1-score, and the proportion of claims for which the system returns an undetermined result.

No benchmark accuracy is claimed in this README.

## 11. Limitations and Responsible Use

TruthCluster AI is a decision-support tool, not an infallible truth detector.

- Search engines may return irrelevant or incomplete articles.
- Some websites restrict automated article extraction.
- Duplicate detection may miss copies or flag articles that share similar content.
- TF-IDF and K-Means capture textual similarity, not factual correctness.
- Retrieved sources may repeat the same original report and therefore may not provide independent confirmation.
- LLM-generated assessments can be incomplete or incorrect.
- An undetermined outcome may be appropriate when reliable evidence is unavailable.

Users should inspect the cited sources and consult trustworthy primary sources when possible, especially for high-impact claims.

## 12. Future Improvements

Planned areas for improvement include:

- Improving search relevance and article extraction reliability.
- Better identifying original reporting and independent corroboration.
- Strengthening evidence quality and source reliability analysis.
- Evaluating the system with a verified benchmark dataset.
- Improving evidence citations and explainability.
- Supporting additional languages.
- Improving robustness when online sources are unavailable or contradictory.

## 13. Expected Real-World Impact

TruthCluster AI aims to help students, readers, and the public investigate viral claims more efficiently.

By organizing related evidence and explaining the basis of an assessment, the project encourages critical thinking, source checking, and more responsible information sharing.

The long-term goal is to make evidence-based news investigation more accessible and understandable.

## 14. Team

**Team Name:** The Outliers

- **Manojkumar R**
- **Jesvanthkavi A**

**Hackathon:** ZERO ORIGIN 2026 — Round 2: Build & Prove

## 15. Project Links

- **Live Application:** https://truthcluster-ai-sysba84agb2rtbaybnxzi8.streamlit.app/
- **GitHub Repository:** https://github.com/manoj-xai/truthcluster-ai

---

<p align="center">

  **TruthCluster AI**

  *Verify the claim. Explore the evidence. Think before you share.*

  **IDEATE. INNOVATE. IMPACT.**

</p>