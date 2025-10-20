"""
Golden Q&A Evaluation Script for ArcFusion RAG System

Evaluates system performance against assignment test cases.
Provides quantifiable metrics for accuracy, latency, and source attribution.
"""

import sys
import json
import asyncio
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.utils.logging_config import setup_logging
from src.config import config
from src.agents.orchestrator import orchestrator


@dataclass
class TestResult:
    """Result for a single test case"""
    test_id: str
    category: str
    query: str
    passed: bool
    latency_seconds: float
    status: str
    answer: str
    expected_keywords: List[str] = None
    found_keywords: List[str] = None
    missing_keywords: List[str] = None
    sources_count: int = 0
    pdf_sources: int = 0
    web_sources: int = 0
    error: str = None
    quality_score: float = None
    clarification_detected: bool = False
    multi_step_planning: bool = False


class GoldenQAEvaluator:
    """Evaluates RAG system against golden Q&A test cases"""

    def __init__(self, test_cases_path: str = None):
        if test_cases_path is None:
            test_cases_path = config.evaluation.test_cases_file
        self.test_cases_path = Path(test_cases_path)
        self.test_cases = self._load_test_cases()
        self.results: List[TestResult] = []

    def _load_test_cases(self) -> List[Dict]:
        """Load test cases from JSON file"""
        if not self.test_cases_path.exists():
            raise FileNotFoundError(f"Test cases file not found: {self.test_cases_path}")

        with open(self.test_cases_path, 'r') as f:
            data = json.load(f)

        logger.info(f"📋 Loaded {len(data['test_cases'])} test cases from {self.test_cases_path.name}")
        return data['test_cases']

    async def run_single_test(self, test_case: Dict) -> TestResult:
        """Run a single test case"""
        test_id = test_case['id']
        category = test_case['category']
        query = test_case['query']

        logger.info(f"\n{'='*80}")
        logger.info(f"🧪 Running Test: {test_id}")
        logger.info(f"📝 Query: {query}")
        logger.info(f"🏷️  Category: {category}")
        logger.info(f"{'='*80}")

        # Run query through orchestrator
        start_time = time.time()

        try:
            result = await orchestrator.ask(
                query=query,
                session_id=f"eval_{test_id}"
            )

            latency = time.time() - start_time

            # Evaluate based on category
            if category == "ambiguous":
                test_result = self._evaluate_ambiguous(test_case, result, latency)
            elif category == "pdf_only":
                test_result = self._evaluate_pdf_only(test_case, result, latency)
            elif category == "autonomous":
                test_result = self._evaluate_autonomous(test_case, result, latency)
            elif category == "web_only":
                test_result = self._evaluate_web_only(test_case, result, latency)
            else:
                test_result = TestResult(
                    test_id=test_id,
                    category=category,
                    query=query,
                    passed=False,
                    latency_seconds=latency,
                    status="unknown_category",
                    answer=result.get('answer', ''),
                    error=f"Unknown category: {category}"
                )

            # Log result
            status_emoji = "✅" if test_result.passed else "❌"
            logger.info(f"{status_emoji} {test_id}: {test_result.status}")
            logger.info(f"⏱️  Latency: {latency:.2f}s")

            if not test_result.passed:
                logger.warning(f"❌ Failed: {test_result.error or test_result.status}")

            # Clear session memory between tests to prevent history bleed
            session_id = f"eval_{test_id}"
            logger.info(f"🧹 Clearing memory for session: {session_id}")
            orchestrator.clear_memory(session_id)

            return test_result

        except Exception as e:
            latency = time.time() - start_time
            logger.error(f"❌ Test {test_id} crashed: {str(e)}")

            return TestResult(
                test_id=test_id,
                category=category,
                query=query,
                passed=False,
                latency_seconds=latency,
                status="error",
                answer="",
                error=str(e)
            )

    def _evaluate_ambiguous(self, test_case: Dict, result: Dict, latency: float) -> TestResult:
        """Evaluate ambiguous query test case"""
        test_id = test_case['id']

        # Check if system detected ambiguity
        status_value = result.get('status')
        reason = result.get('reason', '')
        questions = result.get('questions', [])

        # Success if system asked for clarification
        clarification_detected = (
            status_value in ['needs_clarification', 'clarification_needed'] or
            len(questions) > 0 or
            'clarification' in reason.lower() or
            'ambiguous' in reason.lower() or
            'vague' in reason.lower()
        )

        passed = clarification_detected

        return TestResult(
            test_id=test_id,
            category=test_case['category'],
            query=test_case['query'],
            passed=passed,
            latency_seconds=latency,
            status="detected_ambiguity" if passed else "missed_ambiguity",
            answer=result.get('answer', ''),
            clarification_detected=clarification_detected,
            error=None if passed else "System did not detect ambiguity"
        )

    def _evaluate_pdf_only(self, test_case: Dict, result: Dict, latency: float) -> TestResult:
        """Evaluate PDF-only query test case"""
        test_id = test_case['id']
        answer = result.get('answer', '')
        sources = result.get('sources', [])

        # Check if expected keywords are in answer
        expected_keywords = test_case.get('expected_keywords', [])
        found_keywords = []
        missing_keywords = []

        for keyword in expected_keywords:
            if keyword.lower() in answer.lower():
                found_keywords.append(keyword)
            else:
                missing_keywords.append(keyword)

        # Check if answer uses PDF sources
        pdf_sources = [s for s in sources if s.get('type') == 'pdf']
        web_sources = [s for s in sources if s.get('type') == 'web']

        # Check if expected paper is cited
        expected_paper = test_case.get('expected_source_paper', '')
        paper_cited = any(
            expected_paper.lower() in s.get('source', '').lower()
            for s in pdf_sources
        )

        # Validation
        keyword_match = len(found_keywords) >= len(expected_keywords) * 0.75  # At least 75% keywords
        has_pdf_sources = len(pdf_sources) > 0
        no_web_if_restricted = not test_case.get('should_not_use_web', False) or len(web_sources) == 0

        passed = keyword_match and has_pdf_sources and no_web_if_restricted

        if passed and expected_paper and not paper_cited:
            passed = False
            error = f"Expected paper '{expected_paper}' not in sources"
        else:
            error = None
            if not keyword_match:
                error = f"Missing keywords: {missing_keywords}"
            elif not has_pdf_sources:
                error = "No PDF sources used"
            elif not no_web_if_restricted:
                error = "Web sources used when should use PDF only"

        return TestResult(
            test_id=test_id,
            category=test_case['category'],
            query=test_case['query'],
            passed=passed,
            latency_seconds=latency,
            status="correct_answer" if passed else "incorrect_answer",
            answer=answer,
            expected_keywords=expected_keywords,
            found_keywords=found_keywords,
            missing_keywords=missing_keywords,
            sources_count=len(sources),
            pdf_sources=len(pdf_sources),
            web_sources=len(web_sources),
            quality_score=result.get('quality_score'),
            error=error
        )

    def _evaluate_autonomous(self, test_case: Dict, result: Dict, latency: float) -> TestResult:
        """Evaluate autonomous multi-step query"""
        test_id = test_case['id']
        sources = result.get('sources', [])

        # Check if system used hybrid approach (PDF + web)
        pdf_sources = [s for s in sources if s.get('type') == 'pdf']
        web_sources = [s for s in sources if s.get('type') == 'web']

        has_pdf = len(pdf_sources) > 0
        has_web = len(web_sources) > 0
        is_hybrid = has_pdf and has_web

        # Check routing decision
        routing = result.get('routing_decision', '') or ''  # Ensure string not None
        used_hybrid_routing = 'hybrid' in routing.lower() or (has_pdf and has_web)

        passed = is_hybrid and used_hybrid_routing

        error = None
        if not has_pdf:
            error = "No PDF sources - should search PDFs for SOTA approach"
        elif not has_web:
            error = "No web sources - should search web for author info"

        return TestResult(
            test_id=test_id,
            category=test_case['category'],
            query=test_case['query'],
            passed=passed,
            latency_seconds=latency,
            status="multi_step_success" if passed else "multi_step_failed",
            answer=result.get('answer', ''),
            sources_count=len(sources),
            pdf_sources=len(pdf_sources),
            web_sources=len(web_sources),
            multi_step_planning=is_hybrid,
            quality_score=result.get('quality_score'),
            error=error
        )

    def _evaluate_web_only(self, test_case: Dict, result: Dict, latency: float) -> TestResult:
        """Evaluate web-only query"""
        test_id = test_case['id']
        sources = result.get('sources', [])

        # Check if system used web search
        pdf_sources = [s for s in sources if s.get('type') == 'pdf']
        web_sources = [s for s in sources if s.get('type') == 'web']

        has_web = len(web_sources) > 0
        no_pdf = len(pdf_sources) == 0

        # Check routing decision
        routing = result.get('routing_decision', '') or ''  # Ensure string not None
        routed_to_web = 'web' in routing.lower()

        passed = has_web and (no_pdf or not test_case.get('should_not_use_pdf', False))

        error = None
        if not has_web:
            error = "No web sources - should route to web search"
        elif not no_pdf and test_case.get('should_not_use_pdf', False):
            error = "PDF sources used when should use web only"

        return TestResult(
            test_id=test_id,
            category=test_case['category'],
            query=test_case['query'],
            passed=passed,
            latency_seconds=latency,
            status="web_search_success" if passed else "web_search_failed",
            answer=result.get('answer', ''),
            sources_count=len(sources),
            pdf_sources=len(pdf_sources),
            web_sources=len(web_sources),
            quality_score=result.get('quality_score'),
            error=error
        )

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases and generate report"""
        logger.info("\n" + "="*80)
        logger.info("🧪 ArcFusion RAG - Golden Q&A Evaluation")
        logger.info("="*80)
        logger.info(f"📋 Running {len(self.test_cases)} test cases from assignment\n")

        self.results = []

        for i, test_case in enumerate(self.test_cases, 1):
            logger.info(f"[{i}/{len(self.test_cases)}] {test_case['id']}")
            result = await self.run_single_test(test_case)
            self.results.append(result)

            # Small delay between tests
            await asyncio.sleep(0.5)

        # Generate report
        report = self._generate_report()

        # Save results
        self._save_results(report)

        # Print summary
        self._print_summary(report)

        return report

    def _generate_report(self) -> Dict[str, Any]:
        """Generate evaluation report"""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        accuracy = passed_tests / total_tests if total_tests > 0 else 0.0

        # Category breakdown
        categories = {}
        for result in self.results:
            cat = result.category
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0}
            categories[cat]['total'] += 1
            if result.passed:
                categories[cat]['passed'] += 1

        # Latency metrics
        latencies = [r.latency_seconds for r in self.results]
        latencies_sorted = sorted(latencies)

        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        p50_latency = latencies_sorted[len(latencies)//2] if latencies else 0
        p95_latency = latencies_sorted[int(len(latencies)*0.95)] if latencies else 0
        p99_latency = latencies_sorted[int(len(latencies)*0.99)] if latencies else 0

        # Failed tests
        failed_tests = [r for r in self.results if not r.passed]

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": len(failed_tests),
                "accuracy": accuracy,
                "avg_latency_seconds": avg_latency,
                "p50_latency": p50_latency,
                "p95_latency": p95_latency,
                "p99_latency": p99_latency
            },
            "categories": {
                cat: {
                    "accuracy": data['passed'] / data['total'] if data['total'] > 0 else 0,
                    "passed": data['passed'],
                    "total": data['total']
                }
                for cat, data in categories.items()
            },
            "test_results": [asdict(r) for r in self.results],
            "failed_tests": [
                {
                    "test_id": r.test_id,
                    "query": r.query,
                    "error": r.error,
                    "missing_keywords": r.missing_keywords
                }
                for r in failed_tests
            ]
        }

        return report

    def _save_results(self, report: Dict):
        """Save results to JSON files"""
        # Save current results
        results_file = Path(config.evaluation.results_file)
        with open(results_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\n💾 Results saved to: {results_file}")

        # Append to history
        history_file = Path(config.evaluation.history_file)
        history = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history = json.load(f)

        history.append({
            "timestamp": report["timestamp"],
            "accuracy": report["summary"]["accuracy"],
            "avg_latency": report["summary"]["avg_latency_seconds"],
            "passed": report["summary"]["passed_tests"],
            "total": report["summary"]["total_tests"]
        })

        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"📜 History updated: {history_file}")

    def _print_summary(self, report: Dict):
        """Print evaluation summary to console"""
        summary = report["summary"]
        categories = report["categories"]
        failed = report["failed_tests"]

        logger.info("\n" + "="*80)
        logger.info("📊 Evaluation Results")
        logger.info("="*80)

        # Overall metrics
        accuracy_pct = summary['accuracy'] * 100
        status_emoji = "✅" if summary['accuracy'] >= 0.70 else "⚠️"
        logger.info(f"\n{status_emoji} Overall Accuracy:        {accuracy_pct:.1f}% ({summary['passed_tests']}/{summary['total_tests']})")
        logger.info(f"⏱️  Avg Response Time:       {summary['avg_latency_seconds']:.2f}s")
        logger.info(f"📈 P50 Latency:             {summary['p50_latency']:.2f}s")
        logger.info(f"📈 P95 Latency:             {summary['p95_latency']:.2f}s")
        logger.info(f"📈 P99 Latency:             {summary['p99_latency']:.2f}s")

        # Category breakdown
        logger.info("\n📂 Category Breakdown:")
        for cat_name, cat_data in categories.items():
            cat_accuracy = cat_data['accuracy'] * 100
            cat_emoji = "✅" if cat_data['accuracy'] >= 0.70 else "❌"
            logger.info(f"  {cat_emoji} {cat_name:20s}: {cat_accuracy:5.1f}% ({cat_data['passed']}/{cat_data['total']})")

        # Failed tests details
        if failed:
            logger.info(f"\n❌ Failed Test Cases ({len(failed)}):")
            for fail in failed:
                logger.info(f"\n  • {fail['test_id']}")
                logger.info(f"    Query: {fail['query'][:80]}...")
                logger.info(f"    Error: {fail['error']}")
                if fail.get('missing_keywords'):
                    logger.info(f"    Missing: {fail['missing_keywords']}")
        else:
            logger.info("\n✅ All test cases passed!")

        # Status message
        logger.info("\n" + "="*80)
        if summary['accuracy'] >= 0.70:
            logger.success("✅ EVALUATION PASSED (Accuracy >= 70%)")
        else:
            logger.warning(f"⚠️  EVALUATION BELOW THRESHOLD (Need 70%, got {accuracy_pct:.1f}%)")
        logger.info("="*80 + "\n")


async def main():
    """Main entry point"""
    # Setup logging
    setup_logging(config)

    # Create evaluator
    evaluator = GoldenQAEvaluator()

    # Run evaluation
    try:
        report = await evaluator.run_all_tests()

        # Exit with appropriate code for CI/CD
        if report["summary"]["accuracy"] >= 0.70:
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
