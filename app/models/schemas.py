"""
Pydantic models for API responses
"""
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class MetricsSummary(BaseModel):
    total_spend: float
    total_registrations: int
    total_ftd: int
    average_cpfd: float
    average_conversion_rate: float
    data_points: int
    date_range: Optional[Dict[str, str]] = None


class HourlyData(BaseModel):
    hour: int
    hour_formatted: str
    cost_sum: float
    cost_mean: float
    ftd_sum: int
    ftd_mean: float
    cpfd_median: float
    cpfd_mean: float
    registrations_sum: int
    conversion_rate: float
    data_points: int


class StatisticalTest(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    interpretation: str
    effect_size: Optional[float] = None
    effect_interpretation: Optional[str] = None


class RegressionResult(BaseModel):
    slope: float
    intercept: float
    r_squared: float
    r_value: float
    p_value: float
    is_significant: bool
    equation: str
    interpretation: str


class TierClassification(BaseModel):
    tier_1: List[int]
    tier_2: List[int]
    tier_3: List[int]
    tier_1_budget_allocation: str
    tier_2_budget_allocation: str
    tier_3_budget_allocation: str


class ComparisonData(BaseModel):
    peak_hours: List[int]
    offpeak_hours: List[int]
    peak_median_cpfd: float
    peak_total_ftd: int
    peak_total_cost: float
    peak_avg_conversion_rate: float
    offpeak_median_cpfd: float
    offpeak_total_ftd: int
    offpeak_total_cost: float
    offpeak_avg_conversion_rate: float
    cpfd_difference: float
    cpfd_reduction_percent: float


class SyncStatus(BaseModel):
    last_sync: Optional[datetime] = None
    last_sync_formatted: Optional[str] = None
    status: str
    data_rows: int
    timezone: str = "Asia/Manila"


class Recommendation(BaseModel):
    priority: int
    priority_label: str
    title: str
    action: str
    rationale: str
    metrics: Dict[str, Any]


class DashboardData(BaseModel):
    metrics: MetricsSummary
    hourly_data: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    tiers: TierClassification
    comparison: Dict[str, Any]
    recommendations: List[Recommendation]
    sync_status: SyncStatus
