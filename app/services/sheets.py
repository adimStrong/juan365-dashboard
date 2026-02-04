"""
Google Sheets Data Service
Fetches FB Ads and Google Ads data from the configured spreadsheet
"""
import os
import json
import gspread
from google.oauth2.service_account import Credentials
from cachetools import TTLCache
from datetime import datetime
import pandas as pd
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Cache data for 5 minutes to reduce API calls
cache = TTLCache(maxsize=20, ttl=300)

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive.readonly'
]

# Sheet names to fetch
SHEET_NAMES = ['FB Ads', 'Google Ads']


class GoogleSheetsService:
    def __init__(self):
        self.sheet_id = os.getenv('GOOGLE_SHEET_ID', '1P6GoOQUa7FdiGKPLJiytMzYvkRJwt7jPmqqHo0p0p0c')
        self.sheet_names = SHEET_NAMES
        self.credentials_file = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', 'credentials.json')
        self._client = None
        self._last_sync = None

    def _get_client(self) -> gspread.Client:
        """Get or create gspread client"""
        if self._client is None:
            try:
                # Try to get credentials from environment variable first (for Railway/cloud)
                creds_json = os.getenv('GOOGLE_CREDENTIALS')
                if creds_json:
                    creds_dict = json.loads(creds_json)
                    creds = Credentials.from_service_account_info(
                        creds_dict,
                        scopes=SCOPES
                    )
                    logger.info("Using credentials from GOOGLE_CREDENTIALS env var")
                else:
                    # Fall back to file for local development
                    creds = Credentials.from_service_account_file(
                        self.credentials_file,
                        scopes=SCOPES
                    )
                    logger.info("Using credentials from file")
                self._client = gspread.authorize(creds)
                logger.info("Google Sheets client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Google Sheets client: {e}")
                raise
        return self._client

    def fetch_sheet_data(self, sheet_name: str, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch data from a specific sheet
        Returns a pandas DataFrame with 'platform' column added
        """
        cache_key = f"sheet_data_{self.sheet_id}_{sheet_name}"

        if not force_refresh and cache_key in cache:
            logger.info(f"Returning cached data for {sheet_name}")
            return cache[cache_key]

        try:
            client = self._get_client()
            spreadsheet = client.open_by_key(self.sheet_id)
            worksheet = spreadsheet.worksheet(sheet_name)

            # Get all values as a list of lists
            all_values = worksheet.get_all_values()

            if len(all_values) < 2:
                logger.warning(f"Sheet {sheet_name} has no data rows")
                return pd.DataFrame()

            # First row is headers
            headers = all_values[0]
            data_rows = all_values[1:]

            # Clean headers: remove empty ones and make unique
            clean_headers = []
            seen = set()
            for i, h in enumerate(headers):
                h = str(h).strip()
                if not h:
                    h = f"unnamed_{i}"
                # Make unique
                original_h = h
                counter = 1
                while h in seen:
                    h = f"{original_h}_{counter}"
                    counter += 1
                seen.add(h)
                clean_headers.append(h)

            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=clean_headers)

            # Remove completely empty rows
            df = df.dropna(how='all')

            # Remove unnamed columns that are entirely empty
            cols_to_drop = [c for c in df.columns if c.startswith('unnamed_') and df[c].astype(str).str.strip().eq('').all()]
            df = df.drop(columns=cols_to_drop)

            # Add platform column
            df['platform'] = sheet_name

            # Store in cache
            cache[cache_key] = df

            logger.info(f"Fetched {len(df)} rows from {sheet_name}")
            return df

        except gspread.WorksheetNotFound:
            logger.warning(f"Worksheet '{sheet_name}' not found")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching data from {sheet_name}: {e}")
            raise

    def fetch_all_platforms(self, force_refresh: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all platform sheets
        Returns a dict with platform name as key and DataFrame as value
        """
        result = {}
        for sheet_name in self.sheet_names:
            try:
                df = self.fetch_sheet_data(sheet_name, force_refresh)
                if not df.empty:
                    result[sheet_name] = df
            except Exception as e:
                logger.error(f"Failed to fetch {sheet_name}: {e}")

        self._last_sync = datetime.now()
        return result

    def fetch_combined_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch and combine data from all platform sheets
        Returns a single DataFrame with all platforms
        """
        cache_key = f"combined_data_{self.sheet_id}"

        if not force_refresh and cache_key in cache:
            logger.info("Returning cached combined data")
            return cache[cache_key]

        all_data = self.fetch_all_platforms(force_refresh)

        if not all_data:
            return pd.DataFrame()

        # Combine all DataFrames
        combined = pd.concat(all_data.values(), ignore_index=True)

        # Store in cache
        cache[cache_key] = combined

        logger.info(f"Combined data: {len(combined)} total rows from {len(all_data)} platforms")
        return combined

    def get_processed_data(self, platform: str = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch and process data with proper column types.
        Handles pivoted sheet structure where each date has columns: Time, Cost, Register, FTD, CPFD

        Args:
            platform: 'FB Ads', 'Google Ads', or None for combined
            force_refresh: Force refresh from API
        """
        if platform:
            df = self.fetch_sheet_data(platform, force_refresh)
        else:
            df = self.fetch_combined_data(force_refresh)

        if df.empty:
            return df

        # Parse the pivoted structure
        all_records = []
        headers = list(df.columns)

        # Find date columns (they contain date patterns like "January 29, 2026")
        # Also handle typos like "Februrary" instead of "February"
        date_columns = []
        for i, col in enumerate(headers):
            col_lower = col.lower()
            if any(month in col_lower for month in ['january', 'february', 'februrary', 'march', 'april', 'may', 'june',
                                                      'july', 'august', 'september', 'october', 'november', 'december']):
                date_columns.append((i, col))

        # If no date columns found, try parsing as flat data
        if not date_columns:
            logger.warning("No date columns found, attempting direct column mapping")
            return self._process_flat_data(df)

        # Process each date block
        for date_idx, date_str in date_columns:
            # Check if the date column is on Time or Cost column by looking at header row
            # Some dates have: Time | Date(Cost) | Reg | FTD | CPFD
            # Others have: Date(Time) | Cost | Reg | FTD | CPFD (off by 1)
            header_val = str(df.iloc[0, date_idx]).strip().lower() if date_idx < len(df.columns) else ''

            # If the date column header row says "Time", the actual Cost is at date_idx + 1
            cost_offset = 1 if header_val == 'time' else 0

            for row_idx, row in df.iterrows():
                if row_idx == 0:
                    continue

                # Get hour from first column (Time column)
                time_str = str(row.iloc[0]).strip()
                if not time_str or time_str.lower() == 'time':
                    continue

                try:
                    hour = int(time_str.split(':')[0])
                    if hour == 24:
                        hour = 0
                except:
                    continue

                try:
                    # Apply offset if date header is on Time column instead of Cost column
                    actual_cost_idx = date_idx + cost_offset
                    cost_val = row.iloc[actual_cost_idx] if actual_cost_idx < len(row) else ''
                    reg_val = row.iloc[actual_cost_idx + 1] if actual_cost_idx + 1 < len(row) else ''
                    ftd_val = row.iloc[actual_cost_idx + 2] if actual_cost_idx + 2 < len(row) else ''
                    cpfd_val = row.iloc[actual_cost_idx + 3] if actual_cost_idx + 3 < len(row) else ''

                    cost = self._parse_number(cost_val)
                    registrations = self._parse_number(reg_val)
                    ftd = self._parse_number(ftd_val)
                    cpfd = self._parse_number(cpfd_val)

                    if cost == 0 and registrations == 0 and ftd == 0:
                        continue

                    if cpfd == 0 and ftd > 0:
                        cpfd = cost / ftd

                    conversion_rate = (ftd / registrations * 100) if registrations > 0 else 0

                    # Get platform from the row if available
                    plat = row.get('platform', 'Unknown') if 'platform' in row.index else 'Unknown'

                    all_records.append({
                        'date': date_str,
                        'hour': hour,
                        'cost': cost,
                        'registrations': registrations,
                        'ftd': ftd,
                        'cpfd': cpfd,
                        'conversion_rate': conversion_rate,
                        'platform': plat
                    })
                except Exception as e:
                    logger.debug(f"Error parsing row {row_idx} for date {date_str}: {e}")
                    continue

        if not all_records:
            logger.warning("No records parsed from pivoted data")
            return pd.DataFrame()

        result_df = pd.DataFrame(all_records)

        # For FB Ads only: Exclude incomplete days (days without hour 0/24:00 data)
        # Hour 0 represents end of day (midnight), so missing hour 0 = partial day
        # Google Ads data is already hourly, so no filtering needed
        if 'hour' in result_df.columns and 'date' in result_df.columns and 'platform' in result_df.columns:
            fb_mask = result_df['platform'] == 'FB Ads'
            if fb_mask.any():
                fb_data = result_df[fb_mask]
                other_data = result_df[~fb_mask]

                # Find complete days for FB Ads (those with hour 0 data)
                complete_dates = fb_data[fb_data['hour'] == 0]['date'].unique()
                incomplete_dates = set(fb_data['date'].unique()) - set(complete_dates)
                if incomplete_dates:
                    logger.info(f"Excluding incomplete FB Ads days: {incomplete_dates}")
                    fb_data = fb_data[fb_data['date'].isin(complete_dates)]

                # Recombine FB Ads (filtered) with other platforms (unfiltered)
                result_df = pd.concat([fb_data, other_data], ignore_index=True)

        # Convert FB Ads cumulative data to actual hourly values
        result_df = self._convert_cumulative_to_hourly(result_df)

        logger.info(f"Processed {len(result_df)} hourly records from {len(date_columns)} dates")
        return result_df

    def _parse_number(self, value) -> float:
        """Parse a number from various string formats"""
        if pd.isna(value) or value == '':
            return 0.0
        try:
            clean_val = str(value).replace(',', '').replace('$', '').replace('₱', '').strip()
            if not clean_val:
                return 0.0
            return float(clean_val)
        except:
            return 0.0

    def _convert_cumulative_to_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert cumulative values to hourly differences for FB Ads.

        FB Ads data is cumulative (each hour shows running total for the day).
        This method converts to actual hourly values by subtracting previous hour.

        Note: Hour 0 in the data represents 24:00 (midnight/end of day), not start of day.
        Hours run chronologically: 1, 2, 3, ... 22, 23, 0 (where 0 = 24:00)

        Args:
            df: DataFrame with columns: date, hour, cost, registrations, ftd, platform

        Returns:
            DataFrame with FB Ads values converted to hourly (Google Ads unchanged)
        """
        if df.empty:
            return df

        # Separate FB Ads and other platforms
        fb_mask = df['platform'] == 'FB Ads'
        if not fb_mask.any():
            return df  # No FB Ads data, return as-is

        fb_df = df[fb_mask].copy()
        other_df = df[~fb_mask].copy()

        # Columns to convert from cumulative to hourly
        cumulative_cols = ['cost', 'registrations', 'ftd']

        # Process each date separately
        converted_records = []

        for date in fb_df['date'].unique():
            date_data = fb_df[fb_df['date'] == date].copy()

            # Sort by hour chronologically: 1,2,3...22,23,0 (where 0 = 24:00/midnight)
            # Create sort key: hour 0 becomes 24 for sorting
            date_data['_sort_hour'] = date_data['hour'].apply(lambda h: 24 if h == 0 else h)
            date_data = date_data.sort_values('_sort_hour').reset_index(drop=True)
            date_data = date_data.drop(columns=['_sort_hour'])

            for idx, row in date_data.iterrows():
                record = row.to_dict()

                if idx == 0:
                    # First hour of the day (chronologically) - keep as-is
                    pass
                else:
                    # Get previous row (previous hour)
                    prev_row = date_data.iloc[idx - 1]

                    # Calculate hourly values by subtracting previous cumulative
                    for col in cumulative_cols:
                        current_val = row[col]
                        prev_val = prev_row[col]
                        hourly_val = current_val - prev_val

                        # Handle negative values (could indicate data reset or error)
                        if hourly_val < 0:
                            logger.warning(
                                f"Negative hourly value for {col} on {date} hour {row['hour']}: "
                                f"{current_val} - {prev_val} = {hourly_val}"
                            )

                        record[col] = hourly_val

                # Recalculate derived columns
                cost = record['cost']
                ftd = record['ftd']
                registrations = record['registrations']

                record['cpfd'] = cost / ftd if ftd > 0 else 0
                record['conversion_rate'] = (ftd / registrations * 100) if registrations > 0 else 0

                converted_records.append(record)

        if not converted_records:
            return other_df if not other_df.empty else df

        # Create DataFrame from converted FB Ads records
        converted_fb_df = pd.DataFrame(converted_records)

        # Combine with other platforms (Google Ads stays unchanged)
        if not other_df.empty:
            result_df = pd.concat([converted_fb_df, other_df], ignore_index=True)
        else:
            result_df = converted_fb_df

        logger.info(f"Converted {len(converted_records)} FB Ads records from cumulative to hourly")
        return result_df

    def _process_flat_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback for flat data structure"""
        # Preserve platform column if it exists
        platform_col = df['platform'].copy() if 'platform' in df.columns else None

        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

        column_mappings = {
            'hour_of_day': 'hour',
            'spend': 'cost',
            'regs': 'registrations',
            'register': 'registrations',
            'first_time_deposits': 'ftd',
            'ftds': 'ftd',
        }

        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})

        numeric_cols = ['cost', 'registrations', 'ftd', 'hour', 'cpfd']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        if 'cpfd' not in df.columns and 'cost' in df.columns and 'ftd' in df.columns:
            df['cpfd'] = df.apply(lambda row: row['cost'] / row['ftd'] if row['ftd'] > 0 else 0, axis=1)

        if 'conversion_rate' not in df.columns and 'ftd' in df.columns and 'registrations' in df.columns:
            df['conversion_rate'] = df.apply(
                lambda row: (row['ftd'] / row['registrations'] * 100) if row['registrations'] > 0 else 0, axis=1
            )

        # Restore platform column
        if platform_col is not None:
            df['platform'] = platform_col

        return df

    def get_platform_summary(self, force_refresh: bool = False) -> Dict[str, Dict]:
        """
        Get summary statistics for each platform
        """
        all_data = self.fetch_all_platforms(force_refresh)

        summaries = {}
        for platform, df in all_data.items():
            processed = self.get_processed_data(platform, force_refresh=False)
            if not processed.empty:
                summaries[platform] = {
                    'total_cost': processed['cost'].sum() if 'cost' in processed.columns else 0,
                    'total_registrations': processed['registrations'].sum() if 'registrations' in processed.columns else 0,
                    'total_ftd': processed['ftd'].sum() if 'ftd' in processed.columns else 0,
                    'avg_cpfd': processed['cpfd'].mean() if 'cpfd' in processed.columns else 0,
                    'avg_conversion_rate': processed['conversion_rate'].mean() if 'conversion_rate' in processed.columns else 0,
                    'row_count': len(processed)
                }

        return summaries

    def get_last_sync_time(self) -> Optional[datetime]:
        """Return the last sync timestamp"""
        return self._last_sync

    def clear_cache(self):
        """Clear the data cache"""
        cache.clear()
        logger.info("Cache cleared")


# Singleton instance
sheets_service = GoogleSheetsService()
