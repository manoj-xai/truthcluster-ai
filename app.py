
import json
import subprocess
import sys
import threading
from pathlib import Path

import streamlit as st

BASE = Path(__file__).resolve().parent
RUNNER = BASE / "demo_runner.py"

RESULT_FILE = (
    BASE
    / "demo_results"
    / "latest"
    / "llm_evidence_analysis_v6.json"
)


st.set_page_config(
    page_title="TruthCluster AI",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f5f7fb; }
    .block-container { max-width: 1150px; padding-top: 2rem; }
    .hero {
        padding: 25px;
        border-radius: 18px;
        background: linear-gradient(120deg, #101b35, #2456a6);
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 { color: white; margin-bottom: 5px; }
    .hero p { color: #dce7ff; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## 🛡️ TruthCluster AI")
    st.caption("AI-powered news verification")
    st.divider()
    page = st.radio("Navigation", ["Verify Claim", "About Project"])
    st.divider()
    st.caption("Unsupervised clustering + LLM assessment")

st.markdown(
    """
    <div class="hero">
        <h1>TruthCluster AI</h1>
        <p>Cluster the evidence. Verify the truth.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if "verification_result" not in st.session_state:
    st.session_state.verification_result = None

if "verification_claim" not in st.session_state:
    st.session_state.verification_claim = ""

if page == "Verify Claim":
    st.subheader("Verify a news claim")
    st.write(
        "Enter a claim. TruthCluster AI will search for evidence, "
        "group related articles, and assess the available evidence."
    )

    claim = st.text_area(
        "News claim",
        placeholder="Example: The Moon is made entirely of cheese.",
        height=120,
        key="claim_input",
    )

    if st.button(
        "🔎 Verify News",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("verification_running", False),
    ):
        if not claim.strip():
            st.warning("Please enter a news claim first.")
        elif not RUNNER.exists():
            st.error(f"Backend runner not found: {RUNNER}")
        else:
            st.session_state.verification_running = True
            st.session_state.verification_result = None
            st.session_state.verification_claim = claim.strip()

            with st.spinner(
                "Verifying claim: retrieving evidence, clustering articles, "
                "and running the LLM assessment. Please wait..."
            ):
                try:
                    process = subprocess.run(
                        [
                            sys.executable,
                            str(RUNNER),
                            "--claim",
                            claim.strip(),
                        ],
                        cwd=str(BASE),
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=900,
                    )

                    if process.returncode != 0:
                        st.error(
                            "The verification pipeline failed. "
                            "Check the output below for details."
                        )
                        if process.stdout:
                            with st.expander("Backend output"):
                                st.code(process.stdout[-12000:])
                        if process.stderr:
                            with st.expander("Error details"):
                                st.code(process.stderr[-12000:])
                    elif not RESULT_FILE.exists():
                        st.error(
                            "The pipeline finished but the result file "
                            "was not found."
                        )
                    else:
                        with RESULT_FILE.open(
                            "r", encoding="utf-8"
                        ) as file:
                            result = json.load(file)

                        st.session_state.verification_result = result
                        st.success("Verification pipeline completed!")

                        if process.stdout:
                            with st.expander("Pipeline log"):
                                st.code(process.stdout[-12000:])

                except subprocess.TimeoutExpired:
                    st.error(
                        "Verification exceeded 15 minutes and was stopped. "
                        "Try again later or check the backend logs."
                    )
                except Exception as error:
                    st.error(f"Could not run verification: {error}")
                finally:
                    st.session_state.verification_running = False

    result = st.session_state.verification_result

    if result:
        st.divider()
        st.subheader("Verification result")

        verdict = (
            result.get("assessment")
            or result.get("verdict")
            or result.get("final_verdict")
            or result.get("status")
            or "UNDETERMINED"
        )

        if isinstance(verdict, dict):
            verdict = (
                verdict.get("verdict")
                or verdict.get("label")
                or str(verdict)
            )

        st.markdown(f"### Assessment: **{verdict}**")
        st.caption(
            f"Claim submitted: {st.session_state.verification_claim}"
        )

        reasoning = result.get("reasoning")
        if reasoning:
            st.markdown("#### Why?")
            if isinstance(reasoning, (dict, list)):
                st.json(reasoning)
            else:
                st.write(reasoning)

        explanation = result.get("explanation")
        if explanation:
            st.markdown("#### Explanation")
            if isinstance(explanation, (dict, list)):
                st.json(explanation)
            else:
                st.write(explanation)

        st.markdown("#### Evidence details")
        with st.expander("View complete assessment data"):
            st.json(result)

        st.info(
            "This assessment depends on the evidence retrieved and analyzed. "
            "A verdict is not a guarantee of truth; review the original "
            "sources before relying on it."
        )

else:
    st.subheader("About TruthCluster AI")
    st.write(
        """
        TruthCluster AI is an undergraduate AI/ML project that retrieves
        news evidence, groups related articles using unsupervised learning,
        and uses an LLM to assess a claim against the evidence.
        """
    )

    st.markdown("### Verification workflow")
    st.markdown(
        """
        1. **Claim input** — enter a news statement.
        2. **Evidence retrieval** — search for related articles.
        3. **Unsupervised clustering** — group related evidence.
        4. **Evidence analysis** — examine relevance and quality.
        5. **LLM assessment** — assess the claim against the evidence.
        6. **Result** — review the assessment and evidence.
        """
    )

    st.warning(
        "Clustering organizes related evidence; it does not independently "
        "prove whether a claim is true."
    )
