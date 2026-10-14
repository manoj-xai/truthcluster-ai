import argparse
import json
import os
from pathlib import Path
from assessment_rules import validate_assessment
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DEFAULT_INPUT_FILE = Path("quality_selected_evidence.json")
DEFAULT_OUTPUT_FILE = Path("llm_evidence_analysis_v6.json")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze selected evidence for a news claim."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to the quality-selected evidence JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to save the LLM assessment JSON file.",
    )
    return parser.parse_args()

ALLOWED_ASSESSMENTS = {
    "SUPPORTED",
    "CONTRADICTED",
    "MIXED",
    "UNDETERMINED",
}

ALLOWED_RELATIONSHIPS = {
    "supports",
    "contradicts",
    "context",
    "unrelated",
}

ALLOWED_CATEGORIES = {
    "DIRECT_TREATMENT_EVIDENCE",
    "PRECLINICAL_EVIDENCE",
    "POTENTIAL_TARGET_OR_MECHANISM",
    "OBSERVATIONAL_ASSOCIATION",
    "UNRELATED_OR_INSUFFICIENT",
}


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)



def parse_model_json(raw):
    text = raw.strip()

    # Remove Markdown code fences such as ```json ... ```
    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Handle any extra text surrounding the JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object found in model response.")

    text = text[start:end + 1]
    return json.loads(text)


def main():
    args = parse_args()
    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        print("ERROR: GROQ_API_KEY is missing from your .env file.")
        return

    if not input_file.exists():
        print("ERROR: selected_evidence.json was not found.")
        print("Run py select_relevant_evidence.py first.")
        return

    data = load_json(input_file)
    claim = str(data.get("claim", "")).strip()
    articles = data.get("articles", [])

    if not claim or not isinstance(articles, list) or not articles:
        print("ERROR: The claim or selected articles are missing.")
        return


    evidence = []

    for index, article in enumerate(articles, start=1):
        if not isinstance(article, dict):
            continue

        article_text = article.get("article_text", "")
        if not isinstance(article_text, str):
            article_text = ""

        if len(article_text.strip()) < 100:
            continue

        evidence.append({
            "id": index,
            "title": article.get("title", ""),
            "source": article.get("source", ""),
            "date": article.get("date", ""),
            "url": article.get(
                "url",
                article.get("resolved_url", "")
            ),
            "cluster_id": article.get("cluster_id"),
            "selection_score": article.get("selection_score"),
            "tfidf_similarity": article.get("tfidf_similarity"),
            "evidence_category_hints": article.get(
                "category_counts", {}
            ),
            "article_text": article_text[:1200],
        })

        if len(evidence) >= 8:
            break


    if not evidence:
        print("ERROR: No usable article text was found.")
        return

    system_prompt = """
You are an evidence-grounded assistant for a fake-news verification
research project.

Analyze the exact claim using ONLY the supplied article excerpts.
Never invent facts, quotations, dates, sources, or URLs.

Classify each article by its actual evidential role:

DIRECT_TREATMENT_EVIDENCE:
The article describes a specific intervention tested as a cancer
treatment and reports relevant results. State whether results are
preclinical or from human studies.

PRECLINICAL_EVIDENCE:
The intervention was studied in cells, laboratory experiments, or
animals. This does not establish effectiveness in humans.

POTENTIAL_TARGET_OR_MECHANISM:
A protein, enzyme, pathway, or biological mechanism was discovered,
but the supplied text does not demonstrate a treatment intervention.

OBSERVATIONAL_ASSOCIATION:
The article reports an association between an exposure and patient
outcomes. Do not treat association alone as proof of causation or
treatment effectiveness.

UNRELATED_OR_INSUFFICIENT:
The article is unrelated to the exact claim or lacks enough relevant
information to support it.

Important rules:
1. Topic similarity is not proof that a claim is true.
2. A possible future application is not an established treatment.
3. A biological target is not automatically a treatment.
4. Animal or cell results must be identified as preclinical.
5. Observational findings do not alone establish a treatment effect.
6. Do not require regulatory approval unless the claim asserts approval.
7. Do not assume early-stage findings are false.
8. Multiple articles may report the same underlying study.
9. Automated evidence-level hints may be wrong; inspect article text.
10. Lack of evidence does not prove a claim is false.
11. Do not claim source verification unless the supplied evidence
    actually establishes it.
12. Do not invent facts or citations.

Assess the exact wording fairly:
SUPPORTED: relevant evidence directly supports the claim at the level
asserted.
CONTRADICTED: reliable supplied evidence directly conflicts with it.
MIXED: meaningful support exists alongside conflicting or materially
limiting evidence.
UNDETERMINED: the supplied evidence is insufficient to decide.

For "Scientists discover a new treatment for cancer", do not require
regulatory approval. However, a discovery of a biological mechanism or
an observational association alone is not sufficient to demonstrate
a new treatment. A relevant preclinical intervention can support a
carefully qualified claim about a potential treatment, but must not be
described as a proven human treatment.

Return ONLY valid JSON:
{
  "assessment": "SUPPORTED | CONTRADICTED | MIXED | UNDETERMINED",
  "claim_interpretation": "What the exact wording asserts",
  "reasoning": "Evidence-based explanation",
  "evidence_used": [
    {
      "id": 1,
      "relationship": "supports | contradicts | context | unrelated",
      "evidence_category": "DIRECT_TREATMENT_EVIDENCE | PRECLINICAL_EVIDENCE | POTENTIAL_TARGET_OR_MECHANISM | OBSERVATIONAL_ASSOCIATION | UNRELATED_OR_INSUFFICIENT",
      "explanation": "What the article establishes and what it does not"
    }
  ],
  "limitations": ["Uncertainties or missing evidence"]
}

Use only supplied article IDs. Never invent references.
Explain the evidence, not just the article title.
"""

    try:
        client = Groq(api_key=api_key)


        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "claim": claim,
                            "selection_method": data.get("method", ""),
                            "evidence": evidence,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
                temperature=0.1,
                max_completion_tokens=4000,

        )
	

        raw = response.choices[0].message.content or ""

        try:
            result = parse_model_json(raw)

        except (json.JSONDecodeError, ValueError):
            print("WARNING: Model returned invalid JSON.")
            print("Raw response:")
            print(raw[:3000])

            result = {
                "assessment": "UNDETERMINED",
                "claim_interpretation": claim,
                "reasoning": "The model did not return valid JSON.",
                "evidence_used": [],
                "limitations": [
                    "Inspect the raw model response and retry if needed."
                ],
                "raw_model_response": raw,
            }


        if not isinstance(result, dict):
            result = {
                "assessment": "UNDETERMINED",
                "reasoning": "The model response was not a JSON object.",
                "evidence_used": [],
                "limitations": ["Manual review is required."],
            }

        if result.get("assessment") not in ALLOWED_ASSESSMENTS:
            result["assessment"] = "UNDETERMINED"

        valid_sources = {
            str(item["id"]): item
            for item in evidence
        }

        validated = []

        for item in result.get("evidence_used", []):
            if not isinstance(item, dict):
                continue

            source = valid_sources.get(str(item.get("id")))
            if source is None:
                continue

            relationship = item.get("relationship", "context")
            if relationship not in ALLOWED_RELATIONSHIPS:
                relationship = "context"

            category = item.get(
                "evidence_category",
                "UNRELATED_OR_INSUFFICIENT",
            )
            if category not in ALLOWED_CATEGORIES:
                category = "UNRELATED_OR_INSUFFICIENT"

            validated.append({
                "id": source["id"],
                "title": source["title"],
                "url": source["url"],
                "cluster_id": source["cluster_id"],
                "selection_score": source["selection_score"],
                "tfidf_similarity": source["tfidf_similarity"],
                "evidence_category_hints": source["evidence_category_hints"],
                "relationship": relationship,
                "evidence_category": category,
                "explanation": item.get("explanation", ""),
            })

        result["evidence_used"] = validated

        # Preserve the original model assessment for comparison.
        result["llm_assessment"] = result.get(
            "assessment", "UNDETERMINED"
        )

        # Apply the independent basic consistency checks.
        rule_check = validate_assessment(result)

        result["validated_assessment"] = rule_check["assessment"]
        result["assessment_note"] = rule_check["assessment_note"]

        # Use the rule-checked label as the final assessment.
        result["assessment"] = rule_check["assessment"]
        result["claim"] = claim
        result["model"] = MODEL
        result["articles_sent_to_llm"] = len(evidence)
        result["selection_method"] = data.get("method", "")
        result["notice"] = (
            "AI-assisted assessment, not definitive proof. "
            "Evidence categories are model-generated and may be wrong. "
            "Review the original article and study before drawing conclusions."
        )

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)

        print("\nLLM evidence analysis v5 completed.")
        print("Claim:", claim)
        print("Articles sent:", len(evidence))
        print("Assessment:", result["assessment"])
        print("Reasoning:", result.get("reasoning", "Not provided"))
        print("Validated references:", len(validated))
        print("Saved to:", output_file)

    except Exception as error:
        print("LLM analysis failed.")
        print(type(error).__name__ + ":", error)


if __name__ == "__main__":
    main()
