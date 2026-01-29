"""
Statistical Analysis Service
Performs statistical tests on Google Ads data
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StatisticalResult:
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str
    effect_size: Optional[float] = None
    effect_interpretation: Optional[str] = None


class StatisticsService:

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def kruskal_wallis_hourly_cpfd(self, df: pd.DataFrame) -> StatisticalResult:
        """
        Perform Kruskal-Wallis H-test to check if hour of day affects CPFD
        """
        if 'hour' not in df.columns or 'cpfd' not in df.columns:
            raise ValueError("DataFrame must contain 'hour' and 'cpfd' columns")

        # Filter out zero CPFD (no FTD hours)
        df_filtered = df[df['cpfd'] > 0].copy()

        # Group by hour
        groups = [group['cpfd'].values for name, group in df_filtered.groupby('hour')]
        groups = [g for g in groups if len(g) > 0]

        if len(groups) < 2:
            return StatisticalResult(
                test_name="Kruskal-Wallis H-test",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=0,
                interpretation="Insufficient data for analysis"
            )

        h_stat, p_value = stats.kruskal(*groups)

        # Calculate effect size (eta-squared)
        n = len(df_filtered)
        k = len(groups)
        eta_squared = (h_stat - k + 1) / (n - k) if n > k else 0

        # Interpret effect size
        if eta_squared < 0.01:
            effect_interp = "Negligible"
        elif eta_squared < 0.06:
            effect_interp = "Small"
        elif eta_squared < 0.14:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"

        is_significant = p_value < self.alpha
        confidence = (1 - p_value) * 100

        interpretation = (
            f"Hour of day {'SIGNIFICANTLY affects' if is_significant else 'does not significantly affect'} CPFD. "
            f"Effect size is {effect_interp.lower()} (η² = {eta_squared:.4f})."
        )

        return StatisticalResult(
            test_name="Kruskal-Wallis H-test (Hourly CPFD)",
            statistic=round(h_stat, 2),
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=round(confidence, 4),
            interpretation=interpretation,
            effect_size=round(eta_squared, 4),
            effect_interpretation=effect_interp
        )

    def spearman_correlation(self, df: pd.DataFrame, col1: str, col2: str) -> StatisticalResult:
        """
        Calculate Spearman rank correlation between two columns
        """
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"DataFrame must contain '{col1}' and '{col2}' columns")

        # Remove rows with missing values
        df_clean = df[[col1, col2]].dropna()

        if len(df_clean) < 3:
            return StatisticalResult(
                test_name=f"Spearman Correlation ({col1} vs {col2})",
                statistic=0,
                p_value=1.0,
                is_significant=False,
                confidence_level=0,
                interpretation="Insufficient data"
            )

        rho, p_value = stats.spearmanr(df_clean[col1], df_clean[col2])

        # Interpret correlation strength
        abs_rho = abs(rho)
        if abs_rho < 0.3:
            strength = "Weak"
        elif abs_rho < 0.7:
            strength = "Moderate"
        else:
            strength = "Strong"

        direction = "positive" if rho > 0 else "negative"
        is_significant = p_value < self.alpha

        interpretation = (
            f"{strength} {direction} correlation (ρ = {rho:.3f}). "
            f"{'Statistically significant' if is_significant else 'Not statistically significant'}."
        )

        return StatisticalResult(
            test_name=f"Spearman Correlation ({col1} vs {col2})",
            statistic=round(rho, 4),
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=round((1 - p_value) * 100, 4),
            interpretation=interpretation
        )

    def linear_regression(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict[str, Any]:
        """
        Perform linear regression analysis
        """
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{x_col}' and '{y_col}' columns")

        df_clean = df[[x_col, y_col]].dropna()

        if len(df_clean) < 3:
            return {"error": "Insufficient data for regression"}

        x = df_clean[x_col].values
        y = df_clean[y_col].values

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared = r_value ** 2

        return {
            "slope": round(slope, 6),
            "intercept": round(intercept, 4),
            "r_squared": round(r_squared, 4),
            "r_value": round(r_value, 4),
            "p_value": p_value,
            "std_error": round(std_err, 6),
            "is_significant": p_value < self.alpha,
            "equation": f"{y_col} = {intercept:.2f} + {slope:.4f} × {x_col}",
            "interpretation": f"Every $100 increase in {x_col} → +{slope * 100:.2f} {y_col}"
        }

    def calculate_hourly_stats(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculate statistics for each hour
        """
        if 'hour' not in df.columns:
            return []

        hourly_stats = []

        for hour in range(24):
            hour_data = df[df['hour'] == hour]

            if len(hour_data) == 0:
                continue

            stats_dict = {
                "hour": hour,
                "hour_formatted": f"{hour:02d}:00",
                "data_points": len(hour_data),
            }

            # Calculate stats for available columns
            for col in ['cost', 'ftd', 'cpfd', 'registrations', 'clicks', 'impressions']:
                if col in hour_data.columns:
                    col_data = hour_data[col]
                    stats_dict[f"{col}_mean"] = round(col_data.mean(), 2)
                    stats_dict[f"{col}_median"] = round(col_data.median(), 2)
                    stats_dict[f"{col}_sum"] = round(col_data.sum(), 2)
                    stats_dict[f"{col}_std"] = round(col_data.std(), 2) if len(col_data) > 1 else 0

            # Calculate conversion rate
            if 'ftd' in hour_data.columns and 'registrations' in hour_data.columns:
                total_ftd = hour_data['ftd'].sum()
                total_regs = hour_data['registrations'].sum()
                stats_dict['conversion_rate'] = round((total_ftd / total_regs * 100) if total_regs > 0 else 0, 2)

            hourly_stats.append(stats_dict)

        return sorted(hourly_stats, key=lambda x: x['hour'])

    def get_tier_classification(self, df: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Classify hours into performance tiers based on CPFD
        Tier 1: Best (lowest CPFD)
        Tier 2: Medium
        Tier 3: Worst (highest CPFD)
        """
        hourly_stats = self.calculate_hourly_stats(df)

        if not hourly_stats:
            return {"tier_1": [], "tier_2": [], "tier_3": []}

        # Sort by median CPFD
        sorted_hours = sorted(
            [h for h in hourly_stats if h.get('cpfd_median', 0) > 0],
            key=lambda x: x.get('cpfd_median', float('inf'))
        )

        n = len(sorted_hours)
        tier_1_count = max(1, n // 3)
        tier_2_count = max(1, n // 3)

        tier_1 = [h['hour'] for h in sorted_hours[:tier_1_count]]
        tier_3 = [h['hour'] for h in sorted_hours[-(n - tier_1_count - tier_2_count):]]
        tier_2 = [h['hour'] for h in sorted_hours if h['hour'] not in tier_1 and h['hour'] not in tier_3]

        return {
            "tier_1": tier_1,  # Best hours (lowest CPFD)
            "tier_2": tier_2,  # Medium hours
            "tier_3": tier_3,  # Worst hours (highest CPFD)
            "tier_1_budget_allocation": "45%",
            "tier_2_budget_allocation": "40%",
            "tier_3_budget_allocation": "15%"
        }

    def get_peak_vs_offpeak_comparison(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare peak hours (evening) vs off-peak hours (early morning)
        """
        peak_hours = [19, 20, 21, 22, 23]  # 7 PM - 11 PM
        offpeak_hours = [0, 1, 2, 3, 4, 5, 6]  # 12 AM - 6 AM

        peak_data = df[df['hour'].isin(peak_hours)]
        offpeak_data = df[df['hour'].isin(offpeak_hours)]

        result = {
            "peak_hours": peak_hours,
            "offpeak_hours": offpeak_hours,
        }

        for label, data in [("peak", peak_data), ("offpeak", offpeak_data)]:
            if len(data) > 0:
                result[f"{label}_median_cpfd"] = round(data['cpfd'].median(), 2) if 'cpfd' in data.columns else 0
                result[f"{label}_total_ftd"] = int(data['ftd'].sum()) if 'ftd' in data.columns else 0
                result[f"{label}_total_cost"] = round(data['cost'].sum(), 2) if 'cost' in data.columns else 0
                result[f"{label}_avg_conversion_rate"] = round(
                    (data['ftd'].sum() / data['registrations'].sum() * 100)
                    if 'registrations' in data.columns and data['registrations'].sum() > 0 else 0, 2
                )

        # Calculate savings
        if result.get('peak_median_cpfd', 0) > 0 and result.get('offpeak_median_cpfd', 0) > 0:
            result['cpfd_difference'] = round(result['offpeak_median_cpfd'] - result['peak_median_cpfd'], 2)
            result['cpfd_reduction_percent'] = round(
                (result['cpfd_difference'] / result['offpeak_median_cpfd']) * 100, 1
            )

        return result


# Singleton instance
statistics_service = StatisticsService()
