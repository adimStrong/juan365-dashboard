"""
Google Sheets Data Service
Fetches Google Ads data from the configured spreadsheet
"""
import os
import json
import gspread
from google.oauth2.service_account import Credentials
from cachetools import TTLCache
from datetime import datetime
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Cache data for 5 minutes to reduce API calls
cache = TTLCache(maxsize=10, ttl=300)

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets.readonly',
    'https://www.googleapis.com/auth/drive.readonly'
]


class GoogleSheetsService:
    def __init__(self):
        self.sheet_id = os.getenv('GOOGLE_SHEET_ID', '13oDZjGctd8mkVik2_kUxSPpIQQUyC_iIuIHplIFWeUM')
        self.sheet_name = os.getenv('GOOGLE_SHEET_NAME', 'Google Ads')
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

    def fetch_raw_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch all data from Google Ads sheet
        Returns a pandas DataFrame
        """
        cache_key = f"raw_data_{self.sheet_id}"

        if not force_refresh and cache_key in cache:
            logger.info("Returning cached data")
            return cache[cache_key]

        try:
            client = self._get_client()
            spreadsheet = client.open_by_key(self.sheet_id)
            worksheet = spreadsheet.worksheet(self.sheet_name)

            # Get all values as a list of lists
            all_values = worksheet.get_all_values()

            if len(all_values) < 2:
                logger.warning("Sheet has no data rows")
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

            # Store in cache
            cache[cache_key] = df
            self._last_sync = datetime.now()

            logger.info(f"Fetched {len(df)} rows from Google Sheets")
            return df

        except Exception as e:
            logger.error(f"Error fetching data from Google Sheets: {e}")
            raise

    def get_processed_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch and process data with proper column types.
        Handles pivoted sheet structure where each date has columns: Time, Cost, Register, FTD, CPFD
        """
        df = self.fetch_raw_data(force_refresh)

        if df.empty:
            return df

        # Parse the pivoted structure
        # Sheet has dates as column groups, hours as rows
        # Each date block has: Time, Cost, Register, FTD, CPFD (or CFPD)

        all_records = []
        headers = list(df.columns)

        # Find date columns (they contain date patterns like "January 29, 2026")
        date_columns = []
        for i, col in enumerate(headers):
            col_lower = col.lower()
            # Check if this looks like a date column
            if any(month in col_lower for month in ['january', 'february', 'march', 'april', 'may', 'june',
                                                      'july', 'august', 'september', 'october', 'november', 'december']):
                date_columns.append((i, col))

        # If no date columns found, try parsing the first few columns as summary data
        if not date_columns:
            logger.warning("No date columns found, attempting direct column mapping")
            return self._process_flat_data(df)

        # Process each date block
        for date_idx, date_str in date_columns:
            # Each date block typically has 5 columns: Time (or empty), Cost, Register, FTD, CPFD
            # The date column itself is usually the "Cost" column for that date

            for row_idx, row in df.iterrows():
                # Skip header row (row 0 contains column labels like "Cost", "Register", etc.)
                if row_idx == 0:
                    continue

                # Get hour from first column (Time column)
                time_str = str(row.iloc[0]).strip()
                if not time_str or time_str.lower() == 'time':
                    continue

                # Parse hour from time string (e.g., "1:00" -> 1, "23:00" -> 23, "0:00" -> 0)
                try:
                    hour = int(time_str.split(':')[0])
                    if hour == 24:
                        hour = 0
                except:
                    continue

                # Get data from this date's columns (date_idx is Cost, +1 is Register, +2 is FTD, +3 is CPFD)
                try:
                    cost_val = row.iloc[date_idx] if date_idx < len(row) else ''
                    reg_val = row.iloc[date_idx + 1] if date_idx + 1 < len(row) else ''
                    ftd_val = row.iloc[date_idx + 2] if date_idx + 2 < len(row) else ''
                    cpfd_val = row.iloc[date_idx + 3] if date_idx + 3 < len(row) else ''

                    # Clean and convert values
                    cost = self._parse_number(cost_val)
                    registrations = self._parse_number(reg_val)
                    ftd = self._parse_number(ftd_val)
                    cpfd = self._parse_number(cpfd_val)

                    # Skip rows with no data
                    if cost == 0 and registrations == 0 and ftd == 0:
                        continue

                    # Calculate CPFD if not provided
                    if cpfd == 0 and ftd > 0:
                        cpfd = cost / ftd

                    # Calculate conversion rate
                    conversion_rate = (ftd / registrations * 100) if registrations > 0 else 0

                    all_records.append({
                        'date': date_str,
                        'hour': hour,
                        'cost': cost,
                        'registrations': registrations,
                        'ftd': ftd,
                        'cpfd': cpfd,
                        'conversion_rate': conversion_rate
                    })
                except Exception as e:
                    logger.debug(f"Error parsing row {row_idx} for date {date_str}: {e}")
                    continue

        if not all_records:
            logger.warning("No records parsed from pivoted data")
            return pd.DataFrame()

        result_df = pd.DataFrame(all_records)
        logger.info(f"Processed {len(result_df)} hourly records from {len(date_columns)} dates")
        return result_df

    def _parse_number(self, value) -> float:
        """Parse a number from various string formats"""
        if pd.isna(value) or value == '':
            return 0.0
        try:
            # Remove commas and convert
            clean_val = str(value).replace(',', '').replace('$', '').strip()
            if not clean_val:
                return 0.0
            return float(clean_val)
        except:
            return 0.0

    def _process_flat_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback for flat data structure"""
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

        return df

    def get_last_sync_time(self) -> Optional[datetime]:
        """Return the last sync timestamp"""
        return self._last_sync

    def clear_cache(self):
        """Clear the data cache"""
        cache.clear()
        logger.info("Cache cleared")


# Singleton instance
sheets_service = GoogleSheetsService()
