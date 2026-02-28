"""
Statistical Analysis Module for BDH Science Fest Sprint
========================================================

This module provides statistical tools for analyzing and comparing
performance metrics between baseline and improved models.

Features:
1. Statistical significance testing (t-test, Wilcoxon, Mann-Whitney U)
2. Effect size calculation (Cohen's d, Cliff's delta)
3. Confidence intervals
4. Power analysis
5. Multiple comparison correction
"""

import numpy as np
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from scipy import stats
from scipy.stats import norm
import warnings


@dataclass
class StatisticalTest:
    """Results from a statistical test"""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    interpretation: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "test_name": self.test_name,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "is_significant": bool(self.is_significant),
            "alpha": float(self.alpha),
            "effect_size": float(self.effect_size) if self.effect_size is not None else None,
            "confidence_interval": list(self.confidence_interval) if self.confidence_interval else None,
            "interpretation": self.interpretation
        }


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for comparing model performance.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}

    def independent_t_test(self, baseline: List[float], improved: List[float],
                           alternative: str = "two-sided") -> StatisticalTest:
        """
        Perform independent samples t-test.

        Tests whether the means of two groups are significantly different.
        Assumes normal distribution and equal variances.
        """
        baseline = np.array(baseline)
        improved = np.array(improved)

        # Remove NaN values
        baseline = baseline[~np.isnan(baseline)]
        improved = improved[~np.isnan(improved)]

        if len(baseline) < 2 or len(improved) < 2:
            warnings.warn("Insufficient data for t-test")
            return StatisticalTest(
                test_name="Independent t-test",
                statistic=0,
                p_value=1,
                is_significant=False,
                alpha=self.alpha,
                interpretation="Insufficient data"
            )

        # Perform t-test
        statistic, p_value = stats.ttest_ind(baseline, improved, alternative=alternative)

        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(baseline) - 1) * np.var(baseline, ddof=1) +
                             (len(improved) - 1) * np.var(improved, ddof=1)) /
                            (len(baseline) + len(improved) - 2))

        cohens_d = (np.mean(improved) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0

        # Calculate confidence interval for difference
        se_diff = pooled_std * np.sqrt(1/len(baseline) + 1/len(improved))
        ci_lower = (np.mean(improved) - np.mean(baseline)) - stats.t.ppf(1 - self.alpha/2, len(baseline) + len(improved) - 2) * se_diff
        ci_upper = (np.mean(improved) - np.mean(baseline)) + stats.t.ppf(1 - self.alpha/2, len(baseline) + len(improved) - 2) * se_diff

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"

        interpretation = f"Effect size: {effect_interpretation} (Cohen's d = {cohens_d:.3f})"
        if p_value < self.alpha:
            interpretation += f". Statistically significant at α={self.alpha}."
        else:
            interpretation += f". Not statistically significant at α={self.alpha}."

        return StatisticalTest(
            test_name="Independent t-test",
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=cohens_d,
            confidence_interval=(ci_lower, ci_upper),
            interpretation=interpretation
        )

    def welch_t_test(self, baseline: List[float], improved: List[float],
                     alternative: str = "two-sided") -> StatisticalTest:
        """
        Perform Welch's t-test (does not assume equal variances).
        """
        baseline = np.array(baseline)
        improved = np.array(improved)

        baseline = baseline[~np.isnan(baseline)]
        improved = improved[~np.isnan(improved)]

        if len(baseline) < 2 or len(improved) < 2:
            return StatisticalTest(
                test_name="Welch's t-test",
                statistic=0, p_value=1, is_significant=False, alpha=self.alpha
            )

        statistic, p_value = stats.ttest_ind(baseline, improved, equal_var=False, alternative=alternative)

        # Welch's t-test effect size (Hedges' g)
        pooled_std = np.sqrt((np.var(baseline, ddof=1) / len(baseline) +
                             np.var(improved, ddof=1) / len(improved)))
        hedges_g = (np.mean(improved) - np.mean(baseline)) / pooled_std if pooled_std > 0 else 0

        interpretation = f"Hedges' g = {hedges_g:.3f}"
        if p_value < self.alpha:
            interpretation += f". Statistically significant at α={self.alpha}."
        else:
            interpretation += f". Not statistically significant."

        return StatisticalTest(
            test_name="Welch's t-test",
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=hedges_g,
            interpretation=interpretation
        )

    def mann_whitney_test(self, baseline: List[float], improved: List[float],
                          alternative: str = "two-sided") -> StatisticalTest:
        """
        Mann-Whitney U test (non-parametric alternative to t-test).
        Tests whether one distribution is stochastically greater than the other.
        """
        baseline = np.array(baseline)
        improved = np.array(improved)

        baseline = baseline[~np.isnan(baseline)]
        improved = improved[~np.isnan(improved)]

        if len(baseline) < 2 or len(improved) < 2:
            return StatisticalTest(
                test_name="Mann-Whitney U",
                statistic=0, p_value=1, is_significant=False, alpha=self.alpha
            )

        statistic, p_value = stats.mannwhitneyu(baseline, improved, alternative=alternative)

        # Calculate rank-biserial correlation as effect size
        n1, n2 = len(baseline), len(improved)
        rank_biserial = 1 - (2 * statistic) / (n1 * n2)

        interpretation = f"Rank-biserial correlation = {rank_biserial:.3f}"
        if p_value < self.alpha:
            interpretation += f". Statistically significant at α={self.alpha}."
        else:
            interpretation += f". Not statistically significant."

        return StatisticalTest(
            test_name="Mann-Whitney U",
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=rank_biserial,
            interpretation=interpretation
        )

    def wilcoxon_signed_rank_test(self, baseline: List[float], improved: List[float]) -> StatisticalTest:
        """
        Wilcoxon signed-rank test (for paired samples).
        """
        baseline = np.array(baseline)
        improved = np.array(improved)

        # Remove NaN and ensure equal lengths
        mask = ~(np.isnan(baseline) | np.isnan(improved))
        baseline = baseline[mask]
        improved = improved[mask]

        if len(baseline) < 2:
            return StatisticalTest(
                test_name="Wilcoxon signed-rank",
                statistic=0, p_value=1, is_significant=False, alpha=self.alpha
            )

        statistic, p_value = stats.wilcoxon(baseline, improved)

        # Calculate effect size (r)
        z_score = stats.norm.ppf(p_value / 2)  # Approximate z-score
        n = len(baseline)
        r = z_score / np.sqrt(n) if n > 0 else 0

        interpretation = f"Effect size r = {abs(r):.3f}"
        if p_value < self.alpha:
            interpretation += f". Statistically significant at α={self.alpha}."
        else:
            interpretation += f". Not statistically significant."

        return StatisticalTest(
            test_name="Wilcoxon signed-rank",
            statistic=statistic,
            p_value=p_value,
            is_significant=p_value < self.alpha,
            alpha=self.alpha,
            effect_size=r,
            interpretation=interpretation
        )

    def confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the mean"""
        data = np.array(data)
        data = data[~np.isnan(data)]

        if len(data) < 2:
            return (0, 0)

        mean = np.mean(data)
        std_err = stats.sem(data)

        # Use t-distribution for small samples
        h = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)

        return (mean - h, mean + h)

    def bootstrap_confidence_interval(self, data: List[float], confidence: float = 0.95,
                                      n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval"""
        data = np.array(data)
        data = data[~np.isnan(data)]

        if len(data) < 2:
            return (0, 0)

        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return (lower, upper)

    def multiple_comparison_correction(self, p_values: List[float],
                                       method: str = "bonferroni") -> List[float]:
        """
        Correct p-values for multiple comparisons.

        Methods:
        - bonferroni: Conservative, p_adjusted = p * n_tests
        - holm: Less conservative, step-down procedure
        - benjamini_hochberg: Controls false discovery rate
        """
        p_values = np.array(p_values)
        n = len(p_values)

        if method == "bonferroni":
            corrected = np.minimum(p_values * n, 1.0)

        elif method == "holm":
            order = np.argsort(p_values)
            corrected = np.empty_like(p_values)
            for i, idx in enumerate(order):
                corrected[idx] = min(p_values[idx] * (n - i), 1.0)

        elif method == "benjamini_hochberg":
            order = np.argsort(p_values)
            corrected = np.empty_like(p_values)
            for i, idx in enumerate(order):
                corrected[idx] = min(p_values[idx] * n / (i + 1), 1.0)

        else:
            raise ValueError(f"Unknown correction method: {method}")

        return corrected.tolist()

    def power_analysis(self, effect_size: float, alpha: float = 0.05,
                       power: float = 0.8, ratio: float = 1.0) -> int:
        """
        Calculate required sample size for given effect size and power.
        """
        from scipy.stats import norm

        # Two-tailed test
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # Required sample size per group
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        # Adjust for ratio
        n1 = int(np.ceil(n_per_group * (1 + ratio) / (2 * ratio)))
        n2 = int(np.ceil(n1 * ratio))

        return max(n1, n2)

    def cliff_delta(self, baseline: List[float], improved: List[float]) -> Tuple[float, str]:
        """
        Calculate Cliff's Delta (non-parametric effect size).

        Values:
        - 0: No effect
        - ±0.147: Negligible
        - ±0.33: Small
        - ±0.474: Medium
        - ±1: Large
        """
        baseline = np.array(baseline)
        improved = np.array(improved)

        baseline = baseline[~np.isnan(baseline)]
        improved = improved[~np.isnan(improved)]

        # Count dominance
        greater = 0
        less = 0

        for b in baseline:
            for i in improved:
                if i > b:
                    greater += 1
                elif i < b:
                    less += 1

        total = len(baseline) * len(improved)
        delta = (greater - less) / total if total > 0 else 0

        # Interpret
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            magnitude = "negligible"
        elif abs_delta < 0.33:
            magnitude = "small"
        elif abs_delta < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"

        direction = "improved > baseline" if delta > 0 else "baseline > improved"

        return delta, f"{magnitude} ({direction})"


class ComparisonReportGenerator:
    """
    Generate comparison reports with statistical analysis.
    """

    def __init__(self, output_dir: Path = None):
        if output_dir is None:
            output_dir = Path(__file__).parent / "results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = StatisticalAnalyzer()
        self.comparisons = []

    def add_comparison(self, metric_name: str, baseline: List[float],
                      improved: List[float], higher_is_better: bool = True):
        """Add a comparison metric"""
        if len(baseline) == 0 or len(improved) == 0:
            print(f"[Warning] Skipping {metric_name}: empty data")
            return

        # Run statistical tests
        t_test = self.analyzer.independent_t_test(baseline, improved)
        mann_whitney = self.analyzer.mann_whitney_test(baseline, improved)
        delta, delta_interpretation = self.analyzer.cliff_delta(baseline, improved)

        # Calculate summary statistics
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline)
        improved_mean = np.mean(improved)
        improved_std = np.std(improved)

        improvement = improved_mean - baseline_mean
        improvement_pct = (improvement / baseline_mean * 100) if baseline_mean != 0 else 0

        comparison = {
            "metric_name": metric_name,
            "baseline": {
                "mean": float(baseline_mean),
                "std": float(baseline_std),
                "min": float(np.min(baseline)),
                "max": float(np.max(baseline)),
                "n": len(baseline)
            },
            "improved": {
                "mean": float(improved_mean),
                "std": float(improved_std),
                "min": float(np.min(improved)),
                "max": float(np.max(improved)),
                "n": len(improved)
            },
            "improvement": {
                "absolute": float(improvement),
                "percentage": float(improvement_pct),
                "direction": "better" if (improvement > 0 and higher_is_better) or
                                      (improvement < 0 and not higher_is_better) else "worse"
            },
            "statistical_tests": {
                "t_test": t_test.to_dict(),
                "mann_whitney_u": mann_whitney.to_dict()
            },
            "effect_size": {
                "cliffs_delta": float(delta),
                "interpretation": delta_interpretation
            },
            "higher_is_better": higher_is_better
        }

        self.comparisons.append(comparison)

    def generate_report(self) -> str:
        """Generate comprehensive comparison report"""
        report_lines = [
            "# BDH Performance Comparison Report",
            f"# Generated: {self._get_timestamp()}",
            "",
            "## Executive Summary",
            ""
        ]

        # Count significant improvements
        significant_improvements = [c for c in self.comparisons
                                   if c["statistical_tests"]["t_test"]["is_significant"] and
                                   c["improvement"]["direction"] == "better"]

        report_lines.append(f"- Total metrics compared: {len(self.comparisons)}")
        report_lines.append(f"- Statistically significant improvements: {len(significant_improvements)}")
        report_lines.append("")

        # Key improvements table
        report_lines.extend([
            "## Key Performance Improvements",
            "",
            "| Metric | Baseline | Improved | Change | Significance | Effect Size |",
            "|--------|----------|----------|--------|--------------|-------------|"
        ])

        for comp in self.comparisons:
            metric = comp["metric_name"]
            baseline_val = comp["baseline"]["mean"]
            improved_val = comp["improved"]["mean"]
            change = comp["improvement"]["percentage"]
            sig = "✓" if comp["statistical_tests"]["t_test"]["is_significant"] else "✗"
            effect = comp["effect_size"]["cliffs_delta"]

            report_lines.append(
                f"| {metric} | {baseline_val:.4f} | {improved_val:.4f} | "
                f"{change:+.1f}% | {sig} | {effect:.3f} |"
            )

        report_lines.append("")

        # Detailed results
        report_lines.extend([
            "## Detailed Statistical Analysis",
            ""
        ])

        for comp in self.comparisons:
            metric = comp["metric_name"]

            report_lines.extend([
                f"### {metric}",
                "",
                f"**Summary Statistics:**",
                f"- Baseline: {comp['baseline']['mean']:.4f} ± {comp['baseline']['std']:.4f} (n={comp['baseline']['n']})",
                f"- Improved: {comp['improved']['mean']:.4f} ± {comp['improved']['std']:.4f} (n={comp['improved']['n']})",
                f"- Improvement: {comp['improvement']['absolute']:.4f} ({comp['improvement']['percentage']:+.2f}%)",
                ""
            ])

            t_test = comp["statistical_tests"]["t_test"]
            report_lines.extend([
                f"**Independent t-test:**",
                f"- t-statistic: {t_test['statistic']:.4f}",
                f"- p-value: {t_test['p_value']:.6f}",
                f"- Effect size (Cohen's d): {t_test['effect_size']:.4f}",
                f"- 95% CI: [{t_test['confidence_interval'][0]:.4f}, {t_test['confidence_interval'][1]:.4f}]",
                f"- **Significant**: {t_test['is_significant']}",
                f"- Interpretation: {t_test['interpretation']}",
                ""
            ])

            mw = comp["statistical_tests"]["mann_whitney_u"]
            report_lines.extend([
                f"**Mann-Whitney U test (non-parametric):**",
                f"- U statistic: {mw['statistic']:.4f}",
                f"- p-value: {mw['p_value']:.6f}",
                f"- **Significant**: {mw['is_significant']}",
                ""
            ])

            effect = comp["effect_size"]
            report_lines.extend([
                f"**Effect Size (Cliff's Delta):**",
                f"- Delta: {effect['cliffs_delta']:.4f}",
                f"- Interpretation: {effect['interpretation']}",
                ""
            ])

        # Save report
        report_content = "\n".join(report_lines)
        report_file = self.output_dir / "statistical_comparison_report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)

        # Also save JSON
        json_file = self.output_dir / "statistical_comparison.json"
        with open(json_file, 'w') as f:
            json.dump(self.comparisons, f, indent=2)

        print(f"[ReportGenerator] Statistical report saved: {report_file}")
        print(f"[ReportGenerator] JSON data saved: {json_file}")

        return report_content

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    """Test the statistical analyzer"""
    print("="*60)
    print("Testing Statistical Analysis Module")
    print("="*60)

    # Create sample data
    np.random.seed(42)
    baseline_loss = np.random.normal(3.5, 0.3, 20).tolist()
    improved_loss = np.random.normal(2.8, 0.2, 20).tolist()

    baseline_tps = np.random.normal(1000, 100, 15).tolist()
    improved_tps = np.random.normal(1500, 150, 15).tolist()

    # Generate report
    generator = ComparisonReportGenerator()
    generator.add_comparison("Loss", baseline_loss, improved_loss, higher_is_better=False)
    generator.add_comparison("Tokens/Second", baseline_tps, improved_tps, higher_is_better=True)

    report = generator.generate_report()
    print("\n" + report)

    print("\n[INFO] Statistical analysis test complete!")


if __name__ == "__main__":
    main()
