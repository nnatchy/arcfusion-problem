"""
CI/CD Accuracy Threshold Checker

Reads evaluation_results.json and checks if accuracy meets minimum threshold.
Returns exit code 0 if passing, 1 if failing (for CI/CD pipelines).
"""

import sys
import json
import argparse
from pathlib import Path


def check_accuracy_threshold(results_file: str = "evaluation_results.json", min_accuracy: float = 0.70):
    """
    Check if evaluation results meet minimum accuracy threshold.

    Args:
        results_file: Path to evaluation_results.json
        min_accuracy: Minimum accuracy threshold (default 0.70 = 70%)

    Returns:
        Exit code: 0 if passing, 1 if failing
    """
    results_path = Path(results_file)

    if not results_path.exists():
        print(f"❌ ERROR: Results file not found: {results_file}")
        print("   Run 'uv run python scripts/evaluate_golden_qa.py' first")
        return 1

    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    # Extract metrics
    accuracy = results["summary"]["accuracy"]
    passed = results["summary"]["passed_tests"]
    total = results["summary"]["total_tests"]
    accuracy_pct = accuracy * 100
    threshold_pct = min_accuracy * 100

    # Check threshold
    meets_threshold = accuracy >= min_accuracy

    # Print results
    print("\n" + "="*60)
    print("🎯 Accuracy Threshold Check")
    print("="*60)
    print(f"Accuracy:        {accuracy_pct:.1f}% ({passed}/{total} tests passed)")
    print(f"Threshold:       {threshold_pct:.1f}%")
    print(f"Status:          {'✅ PASS' if meets_threshold else '❌ FAIL'}")
    print("="*60 + "\n")

    if meets_threshold:
        print(f"✅ Accuracy {accuracy_pct:.1f}% meets threshold {threshold_pct:.1f}%")
        return 0
    else:
        print(f"❌ Accuracy {accuracy_pct:.1f}% below threshold {threshold_pct:.1f}%")
        print(f"   Need to pass {int((min_accuracy * total) - passed)} more tests")
        return 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Check if RAG evaluation meets accuracy threshold (for CI/CD)"
    )
    parser.add_argument(
        "--results",
        default="evaluation_results.json",
        help="Path to evaluation results JSON (default: evaluation_results.json)"
    )
    parser.add_argument(
        "--min",
        type=float,
        default=0.70,
        help="Minimum accuracy threshold 0.0-1.0 (default: 0.70 = 70%%)"
    )

    args = parser.parse_args()

    # Validate threshold
    if not 0.0 <= args.min <= 1.0:
        print(f"❌ ERROR: Threshold must be between 0.0 and 1.0, got {args.min}")
        sys.exit(1)

    # Run check
    exit_code = check_accuracy_threshold(args.results, args.min)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
