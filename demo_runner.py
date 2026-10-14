import argparse
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "demo_results"

STAGES = [
    ("evidence_search.py", "evidence_results.json"),
    ("extract_articles.py", "article_content.json"),
    ("analyze_evidence.py", "analyzed_evidence.json"),
    ("cluster_evidence.py", "clustered_evidence.json"),
    ("analyze_claim.py", "claim_evidence_analysis.json"),
    ("analyze_stance.py", "stance_analysis.json"),
    ("deduplicate_evidence.py", "deduplicated_evidence.json"),
    ("build_evidence_report.py", "final_evidence_report.json"),
    ("analyze_evidence_quality.py", "evidence_quality_review.json"),
    ("select_quality_evidence.py", "quality_selected_evidence.json"),
    ("llm_evidence_analysis_v6.py", "llm_evidence_analysis_v6.json"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the TruthCluster AI verification pipeline."
    )
    parser.add_argument(
        "--claim",
        type=str,
        default=None,
        help="News claim to verify. If omitted, prompt in the terminal.",
    )
    args = parser.parse_args()

    claim = (
        args.claim
        if args.claim is not None
        else input("Enter a news claim to verify: ")
    ).strip()

    if not claim:
        print("ERROR: Claim cannot be empty.")
        sys.exit(1)

    RESULTS.mkdir(parents=True, exist_ok=True)
    run_dir = RESULTS / "latest"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Remove outputs from the previous run to prevent stale results.
    for _, output_name in STAGES:
        old_output = run_dir / output_name
        if old_output.is_file():
            old_output.unlink()

    print(f"\nStarting a fresh verification for: {claim}")

    for script, output_name in STAGES:
        script_path = BASE / script
        output_path = run_dir / output_name

        if not script_path.is_file():
            print(f"\nSTOPPED: Script not found: {script_path}")
            sys.exit(1)

        command = [sys.executable, str(script_path)]

        if script == "evidence_search.py":
            command += [
                "--claim", claim,
                "--output", str(output_path),
            ]

        elif script == "extract_articles.py":
            command += [
                "--input", str(run_dir / "evidence_results.json"),
                "--output", str(output_path),
            ]

        elif script == "analyze_evidence.py":
            command += [
                "--input", str(run_dir / "article_content.json"),
                "--output", str(output_path),
            ]

        elif script == "cluster_evidence.py":
            command += [
                "--input", str(run_dir / "analyzed_evidence.json"),
                "--output", str(output_path),
            ]

        elif script == "analyze_claim.py":
            command += [
                "--input", str(run_dir / "clustered_evidence.json"),
                "--output", str(output_path),
            ]

        elif script == "analyze_stance.py":
            command += [
                "--input", str(run_dir / "claim_evidence_analysis.json"),
                "--output", str(output_path),
            ]

        elif script == "deduplicate_evidence.py":
            command += [
                "--input", str(run_dir / "article_content.json"),
                "--output", str(output_path),
            ]

        elif script == "build_evidence_report.py":
            command += [
                "--folder", str(run_dir),
                "--output", str(output_path),
            ]

        elif script == "analyze_evidence_quality.py":
            command += [
                "--input", str(run_dir / "final_evidence_report.json"),
                "--output", str(output_path),
            ]

        elif script == "select_quality_evidence.py":
            command += [
                "--input", str(run_dir / "clustered_evidence.json"),
                "--quality", str(run_dir / "evidence_quality_review.json"),
                "--output", str(output_path),
            ]

        elif script == "llm_evidence_analysis_v6.py":
            command += [
                "--input", str(run_dir / "quality_selected_evidence.json"),
                "--output", str(output_path),
            ]

        print(f"\nRunning {script}...", flush=True)

        # Ensure this stage cannot accidentally reuse an old output.
        if output_path.exists():
            output_path.unlink()

        try:
            process = subprocess.run(
                command,
                cwd=str(BASE),
                check=False,
            )
        except OSError as error:
            print(f"\nSTOPPED: Could not run {script}: {error}")
            sys.exit(1)

        # Check immediately, inside the loop, after every stage.
        if process.returncode != 0:
            print(
                f"\nSTOPPED: {script} failed "
                f"with exit code {process.returncode}."
            )
            sys.exit(process.returncode)

        if not output_path.is_file():
            print(
                f"\nSTOPPED: {script} finished without creating "
                f"the expected file:\n{output_path}"
            )
            sys.exit(1)

    # Read only the final result generated by this run.
    result_path = run_dir / "llm_evidence_analysis_v6.json"

    try:
        with result_path.open("r", encoding="utf-8") as file:
            result = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"\nERROR: Could not read final assessment: {error}")
        sys.exit(1)

    print("\n========== FINAL ASSESSMENT ==========")
    print("Claim:", claim)
    print("Assessment:", result.get("assessment", "UNDETERMINED"))
    print("Reasoning:", result.get("reasoning") or "No reasoning was returned.")
    print("Saved results:", run_dir)


if __name__ == "__main__":
    main()