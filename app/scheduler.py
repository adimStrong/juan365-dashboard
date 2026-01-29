"""
Scheduler for T+1 Daily Data Sync
Runs at 02:00 AM Philippine Time (UTC+8) every day
"""
import os
import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from app.services.sheets import sheets_service

logger = logging.getLogger(__name__)

PH_TZ = pytz.timezone('Asia/Manila')

def sync_data_job():
    """Job to refresh data from Google Sheets"""
    try:
        logger.info(f"Starting scheduled data sync at {datetime.now(PH_TZ).strftime('%Y-%m-%d %H:%M:%S PHT')}")

        # Clear cache and fetch fresh data
        sheets_service.clear_cache()
        df = sheets_service.get_processed_data(force_refresh=True)

        logger.info(f"Data sync completed: {len(df)} rows loaded")

        return True
    except Exception as e:
        logger.error(f"Data sync failed: {e}")
        return False


def create_scheduler() -> BackgroundScheduler:
    """Create and configure the scheduler"""
    scheduler = BackgroundScheduler(timezone=PH_TZ)

    # Schedule daily sync at 02:00 AM PHT
    # T+1 means we sync yesterday's data, which is complete by 2 AM
    scheduler.add_job(
        sync_data_job,
        trigger=CronTrigger(hour=2, minute=0, timezone=PH_TZ),
        id='daily_sync',
        name='Daily Google Ads Data Sync',
        replace_existing=True
    )

    logger.info("Scheduler configured: Daily sync at 02:00 AM PHT")

    return scheduler


# Global scheduler instance
scheduler = None


def start_scheduler():
    """Start the scheduler"""
    global scheduler
    if scheduler is None:
        scheduler = create_scheduler()
        scheduler.start()
        logger.info("Scheduler started")


def stop_scheduler():
    """Stop the scheduler"""
    global scheduler
    if scheduler is not None:
        scheduler.shutdown()
        scheduler = None
        logger.info("Scheduler stopped")


def get_next_sync_time() -> str:
    """Get the next scheduled sync time"""
    global scheduler
    if scheduler is not None:
        job = scheduler.get_job('daily_sync')
        if job and job.next_run_time:
            return job.next_run_time.strftime('%Y-%m-%d %H:%M:%S PHT')
    return "Not scheduled"
