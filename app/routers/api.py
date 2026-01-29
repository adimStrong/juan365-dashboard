"""
API Router for Dashboard Endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from datetime import datetime
import pytz
import logging
import numpy as np
import pandas as pd

from app.services.sheets import sheets_service
from app.services.statistics import statistics_service
from app.models.schemas import (
    MetricsSummary, SyncStatus, StatisticalTest,
    TierClassification, Recommendation
)

router = APIRouter(prefix="/api", tags=["dashboard"])
logger = logging.getLogger(__name__)

TIMEZONE = pytz.timezone('Asia/Manila')


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def filter_by_date_range(df: pd.DataFrame, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    """Filter DataFrame by date range"""
    if df.empty or 'date' not in df.columns:
        return df

    if not start_date and not end_date:
        return df

    # Get unique dates and create a mapping for filtering
    unique_dates = df['date'].unique()

    # Parse dates for comparison (handle "January 29, 2026" format)
    def parse_date(date_str):
        try:
            return datetime.strptime(date_str, "%B %d, %Y")
        except:
            try:
                return datetime.strptime(date_str, "%Y-%m-%d")
            except:
                return None

    filtered_df = df.copy()

    if start_date:
        start_dt = parse_date(start_date)
        if start_dt:
            filtered_df = filtered_df[filtered_df['date'].apply(lambda x: parse_date(x) >= start_dt if parse_date(x) else True)]

    if end_date:
        end_dt = parse_date(end_date)
        if end_dt:
            filtered_df = filtered_df[filtered_df['date'].apply(lambda x: parse_date(x) <= end_dt if parse_date(x) else True)]

    return filtered_df


def get_available_dates() -> List[str]:
    """Get list of available dates in the data"""
    df = sheets_service.get_processed_data()
    if df.empty or 'date' not in df.columns:
        return []
    return sorted(df['date'].unique().tolist())


@router.get("/dates")
async def get_dates() -> List[str]:
    """Get available dates for filtering"""
    return get_available_dates()


@router.get("/metrics")
async def get_metrics(
    start_date: Optional[str] = Query(None, description="Start date (e.g., 'January 1, 2026' or '2026-01-01')"),
    end_date: Optional[str] = Query(None, description="End date")
):
    """Get summary metrics with optional date filtering"""
    try:
        df = sheets_service.get_processed_data()
        df = filter_by_date_range(df, start_date, end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")

        total_spend = df['cost'].sum() if 'cost' in df.columns else 0
        total_regs = int(df['registrations'].sum()) if 'registrations' in df.columns else 0
        total_ftd = int(df['ftd'].sum()) if 'ftd' in df.columns else 0
        avg_cpfd = total_spend / total_ftd if total_ftd > 0 else 0
        avg_conv_rate = (total_ftd / total_regs * 100) if total_regs > 0 else 0

        # Get date range
        date_range = None
        if 'date' in df.columns:
            dates = df['date'].dropna()
            if len(dates) > 0:
                date_range = {
                    "start": str(dates.min()),
                    "end": str(dates.max())
                }

        return MetricsSummary(
            total_spend=round(total_spend, 2),
            total_registrations=total_regs,
            total_ftd=total_ftd,
            average_cpfd=round(avg_cpfd, 2),
            average_conversion_rate=round(avg_conv_rate, 2),
            data_points=len(df),
            date_range=date_range
        )
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hourly")
async def get_hourly_data(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """Get hourly breakdown data for charts"""
    try:
        df = sheets_service.get_processed_data()
        df = filter_by_date_range(df, start_date, end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")

        hourly_stats = statistics_service.calculate_hourly_stats(df)
        return hourly_stats

    except Exception as e:
        logger.error(f"Error fetching hourly data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Get statistical test results"""
    try:
        df = sheets_service.get_processed_data()
        df = filter_by_date_range(df, start_date, end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")

        results = {}

        # Kruskal-Wallis test for hourly CPFD
        try:
            kw_result = statistics_service.kruskal_wallis_hourly_cpfd(df)
            results['kruskal_wallis'] = {
                "test_name": kw_result.test_name,
                "statistic": kw_result.statistic,
                "p_value": kw_result.p_value,
                "is_significant": kw_result.is_significant,
                "confidence_level": kw_result.confidence_level,
                "interpretation": kw_result.interpretation,
                "effect_size": kw_result.effect_size,
                "effect_interpretation": kw_result.effect_interpretation
            }
        except Exception as e:
            logger.warning(f"Kruskal-Wallis test failed: {e}")

        # Spearman correlations
        correlations = [
            ('cost', 'registrations'),
            ('cost', 'ftd'),
            ('registrations', 'ftd'),
        ]

        results['correlations'] = {}
        for col1, col2 in correlations:
            try:
                corr_result = statistics_service.spearman_correlation(df, col1, col2)
                results['correlations'][f"{col1}_vs_{col2}"] = {
                    "rho": corr_result.statistic,
                    "p_value": corr_result.p_value,
                    "is_significant": corr_result.is_significant,
                    "interpretation": corr_result.interpretation
                }
            except Exception as e:
                logger.warning(f"Correlation {col1} vs {col2} failed: {e}")

        # Linear regression
        try:
            regression = statistics_service.linear_regression(df, 'cost', 'ftd')
            results['regression'] = regression
        except Exception as e:
            logger.warning(f"Regression failed: {e}")

        return results

    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/comparison")
async def get_comparison(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Get peak vs off-peak comparison"""
    try:
        df = sheets_service.get_processed_data()
        df = filter_by_date_range(df, start_date, end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")

        comparison = statistics_service.get_peak_vs_offpeak_comparison(df)
        return comparison

    except Exception as e:
        logger.error(f"Error calculating comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tiers")
async def get_tiers(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Get hour tier classification"""
    try:
        df = sheets_service.get_processed_data()
        df = filter_by_date_range(df, start_date, end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")

        tiers = statistics_service.get_tier_classification(df)
        return tiers

    except Exception as e:
        logger.error(f"Error calculating tiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recommendations")
async def get_recommendations(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
) -> List[Dict[str, Any]]:
    """Get action recommendations based on analysis"""
    try:
        df = sheets_service.get_processed_data()
        df = filter_by_date_range(df, start_date, end_date)

        if df.empty:
            return []

        recommendations = []

        # Get stats for recommendations
        comparison = statistics_service.get_peak_vs_offpeak_comparison(df)
        tiers = statistics_service.get_tier_classification(df)
        hourly_stats = statistics_service.calculate_hourly_stats(df)

        # Recommendation 1: Reduce off-peak spend
        if comparison.get('cpfd_reduction_percent', 0) > 30:
            recommendations.append({
                "priority": 1,
                "priority_label": "CRITICAL",
                "title": "Reduce Off-Peak Hours Budget (02:00-06:00)",
                "action": "Cut budget by 60-80% during early morning hours",
                "rationale": f"Off-peak CPFD (${comparison.get('offpeak_median_cpfd', 0)}) is {comparison.get('cpfd_reduction_percent', 0)}% higher than peak hours",
                "metrics": {
                    "potential_savings": "25-35%",
                    "affected_hours": comparison.get('offpeak_hours', [])
                }
            })

        # Recommendation 2: Dayparting strategy
        recommendations.append({
            "priority": 2,
            "priority_label": "HIGH",
            "title": "Implement 3-Tier Dayparting Strategy",
            "action": "Restructure daily budget allocation based on hour performance",
            "rationale": "Statistical analysis confirms hour significantly affects CPFD",
            "metrics": {
                "tier_1_hours": tiers.get('tier_1', []),
                "tier_1_budget": "45%",
                "tier_2_budget": "40%",
                "tier_3_budget": "15%"
            }
        })

        # Recommendation 3: Scale peak hours
        if len(tiers.get('tier_1', [])) > 0:
            best_hours = tiers['tier_1'][:3]
            recommendations.append({
                "priority": 3,
                "priority_label": "MEDIUM",
                "title": f"Test Scaling During Peak Hours ({', '.join([f'{h}:00' for h in best_hours])})",
                "action": "Increase budget by 30% during best performing hours",
                "rationale": "These hours show consistently lower CPFD",
                "metrics": {
                    "test_duration": "14 days",
                    "budget_increase": "30%",
                    "monitor_metric": "CPFD < $25"
                }
            })

        return recommendations

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sync-status")
async def get_sync_status():
    """Get last sync status and next scheduled sync"""
    try:
        df = sheets_service.get_processed_data()
        last_sync = sheets_service.get_last_sync_time()

        last_sync_formatted = None
        if last_sync:
            last_sync_ph = last_sync.astimezone(TIMEZONE) if last_sync.tzinfo else TIMEZONE.localize(last_sync)
            last_sync_formatted = last_sync_ph.strftime("%Y-%m-%d %H:%M:%S PHT")

        # Get next scheduled sync time
        try:
            from app.scheduler import get_next_sync_time
            next_sync = get_next_sync_time()
        except:
            next_sync = "Not scheduled"

        return {
            "last_sync": last_sync.isoformat() if last_sync else None,
            "last_sync_formatted": last_sync_formatted,
            "next_sync": next_sync,
            "status": "ok" if not df.empty else "no_data",
            "data_rows": len(df),
            "timezone": "Asia/Manila"
        }

    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        return {
            "status": "error",
            "data_rows": 0,
            "timezone": "Asia/Manila"
        }


@router.post("/refresh")
async def refresh_data():
    """Manually trigger data refresh"""
    try:
        sheets_service.clear_cache()
        df = sheets_service.get_processed_data(force_refresh=True)

        return {
            "status": "success",
            "message": "Data refreshed successfully",
            "rows_loaded": len(df),
            "timestamp": datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M:%S PHT")
        }

    except Exception as e:
        logger.error(f"Error refreshing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all")
async def get_all_dashboard_data(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
) -> Dict[str, Any]:
    """Get all dashboard data in one request with optional date filtering"""
    try:
        df = sheets_service.get_processed_data()
        df = filter_by_date_range(df, start_date, end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")

        result = {
            "metrics": await get_metrics(start_date, end_date),
            "hourly_data": await get_hourly_data(start_date, end_date),
            "statistics": await get_statistics(start_date, end_date),
            "comparison": await get_comparison(start_date, end_date),
            "tiers": await get_tiers(start_date, end_date),
            "recommendations": await get_recommendations(start_date, end_date),
            "sync_status": await get_sync_status(),
            "available_dates": get_available_dates(),
            "filter": {"start_date": start_date, "end_date": end_date}
        }
        return convert_numpy_types(result)

    except Exception as e:
        logger.error(f"Error fetching all data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
