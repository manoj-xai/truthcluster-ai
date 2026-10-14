
ALLOWED_ASSESSMENTS = {
    "SUPPORTED",
    "CONTRADICTED",
    "MIXED",
    "UNDETERMINED",
}


def validate_assessment(result):
    """
    Apply basic consistency checks to an LLM assessment.
    This does not independently establish whether a claim is true.
    """
    if not isinstance(result, dict):
        return {
            "assessment": "UNDETERMINED",
            "assessment_note": "Invalid assessment format.",
        }

    assessment = result.get("assessment", "UNDETERMINED")

    if assessment not in ALLOWED_ASSESSMENTS:
        assessment = "UNDETERMINED"

    evidence = result.get("evidence_used", [])

    if not isinstance(evidence, list):
        evidence = []

    relationships = {
        item.get("relationship")
        for item in evidence
        if isinstance(item, dict)
    }

    # No usable evidence means the claim cannot be decided.
    if not evidence:
        assessment = "UNDETERMINED"
        note = "No validated evidence references were returned."

    # Related context alone does not establish support.
    elif relationships and relationships <= {"context", "unrelated"}:
        assessment = "UNDETERMINED"
        note = "The returned evidence does not directly support or contradict the claim."

    else:
        note = (
            "Assessment retained after basic consistency checks. "
            "Manual review of the original sources is still required."
        )

    return {
        "assessment": assessment,
        "assessment_note": note,
    }


if __name__ == "__main__":
    sample = {
        "assessment": "MIXED",
        "evidence_used": [],
    }

    print(validate_assessment(sample))
