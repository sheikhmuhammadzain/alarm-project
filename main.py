# ISA-18.2 Compliant Alarm Rationalization Application
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime, time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    layout="wide", 
    page_title="Alarm Rationalization - ISA-18.2 Compliant", 
    initial_sidebar_state="expanded",
    page_icon="🚨"
)
sns.set_style("whitegrid")

# -----------------------
# ISA-18.2 Benchmark Constants
# -----------------------
ISA_BENCHMARKS = {
    'target_alarms_per_hour': 6,  # Good performance
    'max_alarms_per_hour': 12,    # Maximum acceptable
    'flood_threshold': 10,        # Alarms per 10-min window
    'flood_target_percent': 1.0,  # Target < 1% of time in flood
    'top_10_contribution_target': 5.0,  # Target < 5% from top 10
    'priority_target': {'P1': 5, 'P2': 15, 'P3': 80},  # Target distribution %
    'chattering_threshold': 5,    # Repeats in window
    'standing_alarm_hours': 24    # Hours to consider "standing"
}

# -----------------------
# Enhanced Utility Helpers
# -----------------------
@st.cache_data
def list_subfolders(root_path: str):
    p = Path(root_path)
    if not p.exists():
        return []
    # return only directories
    return [f for f in sorted(p.iterdir()) if f.is_dir()]

@st.cache_data
def list_data_files(folder_path: str, extensions=(".csv", ".xlsx", ".xls")):
    p = Path(folder_path)
    if not p.exists():
        return []
    files = [f for f in sorted(p.iterdir()) if f.suffix.lower() in extensions and f.is_file()]
    return files

@st.cache_data
def read_table(file_path: str, nrows=None):
    p = Path(file_path)
    suffix = p.suffix.lower()
    
    try:
        if suffix == ".csv":
            # Enhanced encoding detection for problematic files
            import re
            with open(p, 'rb') as fb:
                head = fb.read(8192)  # Read more bytes for better detection

            # Detect encoding more aggressively
            text_encoding = None
            
            # Check for BOM markers first
            if head.startswith(b"\xff\xfe"):
                text_encoding = 'utf-16le'
            elif head.startswith(b"\xfe\xff"):
                text_encoding = 'utf-16be'
            elif head.startswith(b"\xef\xbb\xbf"):
                text_encoding = 'utf-8-sig'
            elif head.count(b"\x00") > max(1, len(head)//20):
                # Likely UTF-16 without BOM
                even_nulls = sum(1 for i, b in enumerate(head[:2000]) if (i % 2 == 0 and b == 0))
                odd_nulls = sum(1 for i, b in enumerate(head[:2000]) if (i % 2 == 1 and b == 0))
                text_encoding = 'utf-16le' if odd_nulls > even_nulls else 'utf-16be'
            else:
                # For files with problematic bytes like 0xb3, try common encodings
                # Order matters: try most specific first
                encoding_candidates = [
                    'windows-1252',  # Most common for Windows exports
                    'cp1252',        # Alternative name
                    'iso-8859-1',    # Latin-1
                    'latin-1',       # Fallback that accepts all bytes
                    'utf-8',         # Try UTF-8 last for non-BOM files
                ]
                
                for enc in encoding_candidates:
                    try:
                        # Test decode the entire head to ensure it works
                        test_decode = head.decode(enc)
                        text_encoding = enc
                        break
                    except UnicodeDecodeError:
                        continue
                
                # Ultimate fallback
                if text_encoding is None:
                    text_encoding = 'latin-1'

            # Decode sample for delimiter detection
            try:
                sample_text = head.decode(text_encoding, errors='replace')
            except Exception:
                sample_text = head.decode('latin-1', errors='replace')
                text_encoding = 'latin-1'

            # Skip direct read for files that likely have metadata - go straight to header detection

            # Enhanced header detection for files with metadata
            try:
                with open(p, 'r', encoding=text_encoding, errors='replace') as f:
                    probe_lines = [f.readline() for _ in range(60)]
            except Exception:
                probe_lines = []

            detected_sep = None
            detected_header_idx = None
            candidate_seps = [',', ';', '\t', '|']
            
            # Look for the actual header row containing column names
            for i, line in enumerate(probe_lines):
                line_stripped = line.strip()
                if not line_stripped:
                    continue
                    
                lo = line.lower()
                
                # Skip obvious metadata lines more aggressively
                metadata_indicators = [
                    'date/time of report', 'requester', 'filter applied', 
                    'sort criteria', 'server', 'tip:', 'vcma', 'stn07',
                    'mngr', 'before', 'descending', 'excel', 'custom'
                ]
                if any(indicator in lo for indicator in metadata_indicators):
                    continue
                
                # Look for header patterns - must contain "event" and "time" in the same line
                # Also check for specific column patterns
                if ('event' in lo and 'time' in lo) or ('location' in lo and 'tag' in lo):
                    # Try different separators to see which one works best
                    for sep in candidate_seps:
                        if line.count(sep) >= 4:  # Need at least 4 separators for typical alarm columns
                            cells = [c.strip() for c in line.split(sep)]
                            if len(cells) >= 5:  # Need at least 5 columns
                                # Check if this looks like a header row
                                cell_text = " ".join(cells).lower()
                                header_score = 0
                                header_keywords = ['event', 'time', 'location', 'tag', 'source', 'condition', 'action', 'priority', 'description', 'value', 'units']
                                for keyword in header_keywords:
                                    if keyword in cell_text:
                                        header_score += 1
                                
                                # Also check if cells look like column names (not data)
                                data_like = any(cell.replace(':', '').replace('-', '').replace('.', '').isdigit() for cell in cells[:3])
                                
                                if header_score >= 4 and not data_like:  # Must match keywords and not look like data
                                    detected_sep = sep
                                    detected_header_idx = i
                                    break
                    
                    if detected_sep is not None:
                        break

            # Try reading with detected header and separator
            if detected_sep is not None and detected_header_idx is not None:
                try:
                    # Use the detected header row as the header, skip all rows before it
                    df = pd.read_csv(
                        p,
                        encoding=text_encoding,
                        sep=detected_sep,
                        header=detected_header_idx,  # Use the detected row as header
                        nrows=nrows,
                        engine='python',
                        on_bad_lines='skip',
                    )
                    # Clean column names and handle duplicates
                    df.columns = [" ".join(str(c).split()) for c in df.columns]
                    
                    # Handle duplicate column names by adding suffixes
                    cols = df.columns.tolist()
                    seen = {}
                    for i, col in enumerate(cols):
                        if col in seen:
                            seen[col] += 1
                            cols[i] = f"{col}_{seen[col]}"
                        else:
                            seen[col] = 0
                    df.columns = cols
                    
                    df = _normalize_columns_to_canonical(df)
                    if hasattr(df, 'attrs'):
                        df.attrs['encoding_used'] = f"{text_encoding} (header detected at row {detected_header_idx}, sep={repr(detected_sep)})"
                    return df
                except Exception as e:
                    # If that fails, try with skiprows approach
                    try:
                        df = pd.read_csv(
                            p,
                            encoding=text_encoding,
                            sep=detected_sep,
                            header=0,
                            skiprows=detected_header_idx,
                            nrows=nrows,
                            engine='python',
                            on_bad_lines='skip',
                        )
                        # Clean column names and handle duplicates
                        df.columns = [" ".join(str(c).split()) for c in df.columns]
                        
                        # Handle duplicate column names by adding suffixes
                        cols = df.columns.tolist()
                        seen = {}
                        for i, col in enumerate(cols):
                            if col in seen:
                                seen[col] += 1
                                cols[i] = f"{col}_{seen[col]}"
                            else:
                                seen[col] = 0
                        df.columns = cols
                        
                        df = _normalize_columns_to_canonical(df)
                        if hasattr(df, 'attrs'):
                            df.attrs['encoding_used'] = f"{text_encoding} (header detected with skiprows, sep={repr(detected_sep)})"
                        return df
                    except Exception:
                        pass

            # Final fallback: try with the detected encoding and default settings
            try:
                df = pd.read_csv(p, encoding=text_encoding, nrows=nrows, low_memory=False, on_bad_lines='skip')
                # Clean column names and handle duplicates
                df.columns = [" ".join(str(c).split()) for c in df.columns]
                
                # Handle duplicate column names by adding suffixes
                cols = df.columns.tolist()
                seen = {}
                for i, col in enumerate(cols):
                    if col in seen:
                        seen[col] += 1
                        cols[i] = f"{col}_{seen[col]}"
                    else:
                        seen[col] = 0
                df.columns = cols
                
                df = _normalize_columns_to_canonical(df)
                if hasattr(df, 'attrs'):
                    df.attrs['encoding_used'] = f"{text_encoding} (fallback)"
                return df
            except Exception:
                # Ultimate fallback with latin-1
                df = pd.read_csv(p, encoding='latin-1', nrows=nrows, low_memory=False, on_bad_lines='skip')
                # Clean column names and handle duplicates
                df.columns = [" ".join(str(c).split()) for c in df.columns]
                
                # Handle duplicate column names by adding suffixes
                cols = df.columns.tolist()
                seen = {}
                for i, col in enumerate(cols):
                    if col in seen:
                        seen[col] += 1
                        cols[i] = f"{col}_{seen[col]}"
                    else:
                        seen[col] = 0
                df.columns = cols
                
                df = _normalize_columns_to_canonical(df)
                if hasattr(df, 'attrs'):
                    df.attrs['encoding_used'] = 'latin-1 (ultimate fallback)'
                return df
                
        elif suffix in (".xlsx", ".xls"):
            # Use robust header detection to skip report metadata rows
            df = _read_excel_with_header_detection(p, nrows=nrows)
        else:
            raise ValueError("Unsupported file type")
            
    except Exception as e:
        raise RuntimeError(f"Failed to read {file_path}: {e}")
    
    return df

def parse_space_separated_csv(file_path: Path, nrows=None):
    """Parse CSV files where each character is separated by spaces"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        # Find the header line (usually contains "Event Time" or similar)
        header_idx = None
        for i, line in enumerate(lines):
            if 'E v e n t   T i m e' in line or 'Event Time' in line:
                header_idx = i
                break
        
        if header_idx is None:
            # Fallback: assume first non-empty line is header
            for i, line in enumerate(lines):
                if line.strip():
                    header_idx = i
                    break
        
        if header_idx is None:
            raise ValueError("Could not find header line")
        
        # Parse header
        header_line = lines[header_idx].strip()
        # Remove spaces between characters and split by multiple spaces
        header_clean = ''.join(header_line.split())
        
        # Try to identify column boundaries by looking for common patterns
        columns = []
        if 'EventTime' in header_clean:
            # Standard alarm log format
            columns = ['Event_Time', 'Location_Tag', 'Source', 'Condition', 'Action', 'Priority', 'Description', 'Value', 'Units']
        else:
            # Generic parsing - split by multiple spaces in original
            import re
            # Split by 2+ spaces to identify columns
            columns = re.split(r'\s{2,}', header_line.strip())
            columns = [''.join(col.split()) for col in columns if col.strip()]
        
        # Parse data rows
        data_rows = []
        start_idx = header_idx + 1
        end_idx = min(start_idx + nrows, len(lines)) if nrows else len(lines)
        
        for line in lines[start_idx:end_idx]:
            line = line.strip()
            if not line:
                continue
            
            # Remove character-level spaces and split by multiple spaces
            # This is tricky - we need to preserve spaces within values
            # Strategy: split by patterns that look like column boundaries
            import re
            
            # First pass: remove single-character spaces but preserve multi-character spaces
            processed_line = re.sub(r'(?<!\s)\s(?!\s)', '', line)
            
            # Split by multiple spaces (2+)
            values = re.split(r'\s{2,}', processed_line)
            values = [v.strip() for v in values if v.strip()]
            
            # Pad or truncate to match column count
            while len(values) < len(columns):
                values.append('')
            values = values[:len(columns)]
            
            data_rows.append(values)
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=columns)
        
        # Clean up timestamp column if detected
        if 'Event_Time' in df.columns:
            df['Event_Time'] = df['Event_Time'].str.replace(r'(\d)/(\d)/(\d{4})', r'\1/\2/\3', regex=True)
        
        # Add metadata
        if hasattr(df, 'attrs'):
            df.attrs['encoding_used'] = 'space-separated-format'
            df.attrs['original_format'] = 'industrial-space-separated'
        
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to parse space-separated CSV {file_path}: {e}")

def _normalize_columns_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to a canonical set (Event Time, Location Tag, Source, Condition, Action, Priority, Description, Value, Units)."""
    import re
    if df is None or df.empty:
        return df
    def norm(s: str) -> str:
        s = str(s)
        s = s.replace('_', ' ')
        s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
        s = " ".join(s.split()).strip().lower()
        return s

    canonical = {
        'event time': ['event time', 'eventtime', 'time stamp', 'timestamp', 'date time', 'datetime'],
        'location tag': ['location tag', 'locationtag', 'tag', 'tagname', 'point', 'source tag'],
        'source': ['source', 'orig', 'origin', 'area source'],
        'condition': ['condition', 'state', 'status'],
        'action': ['action', 'ack', 'acknowledge', 'operator action'],
        'priority': ['priority', 'prio', 'severity', 'level'],
        'description': ['description', 'descrip', 'descriptio', 'message', 'text'],
        'value': ['value', 'val'],
        'units': ['units', 'unit']
    }

    mapping = {}
    for c in df.columns:
        n = norm(c)
        # Try exact/alias matches
        matched = None
        for canon, aliases in canonical.items():
            if n == canon or n in aliases:
                matched = canon
                break
            # prefix match for truncated headers like 'descriptio'
            if any(n.startswith(a) for a in aliases):
                matched = canon
                break
        # Final cleanup: keep original if nothing matched
        if matched:
            # Title-case canonical for display consistency
            mapping[c] = matched.title()
        else:
            mapping[c] = c if isinstance(c, str) else str(c)

    df = df.rename(columns=mapping)
    return df

def _read_excel_with_header_detection(p: Path, nrows=None) -> pd.DataFrame:
    """Read Excel files that contain report metadata above the real header row.
    Detects the header row by searching for a row containing tokens like 'Event' and 'Time'."""
    try:
        preview = pd.read_excel(p, header=None, nrows=50)
    except Exception:
        # Fallback straight read
        df = pd.read_excel(p, nrows=nrows)
        df.columns = [" ".join(str(c).split()) for c in df.columns]
        return _normalize_columns_to_canonical(df)

    header_idx = None
    for i in range(len(preview)):
        row = preview.iloc[i].astype(str).fillna("")
        joined = " ".join([x for x in row.tolist() if x.strip()])
        lo = joined.lower()
        # Look for a plausible header row with multiple tokens
        non_empty_cells = (row.str.strip() != "").sum()
        if ('event' in lo and 'time' in lo) and non_empty_cells >= 4:
            header_idx = i
            break
    if header_idx is None:
        # Secondary pass: look for row containing many typical header fields
        keywords = ['event', 'time', 'location', 'tag', 'source', 'condition', 'priority', 'description']
        best_i = None
        best_score = 0
        for i in range(len(preview)):
            lo = " ".join(preview.iloc[i].astype(str).tolist()).lower()
            score = sum(1 for k in keywords if k in lo)
            if score > best_score and score >= 3:
                best_i, best_score = i, score
        header_idx = best_i if best_i is not None else 0

    df = pd.read_excel(p, header=header_idx, nrows=nrows)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    df = _normalize_columns_to_canonical(df)
    return df

def detect_timestamp_column(df: pd.DataFrame):
    """Detect the most likely timestamp column, robust to variations like 'Event Time'."""
    if df is None or df.empty:
        return None

    # Normalize column names for matching
    norm = {c: str(c).strip().lower().replace('_', ' ').replace('-', ' ') for c in df.columns}

    # Priority names: exact or near-exact timestamp indicators
    priority_keys = ['timestamp', 'date time', 'datetime', 'time stamp', 'alarm time', 'event time']
    secondary_tokens = ['time', 'date', 'occurred', 'generated']

    # Order columns by priority
    ordered_cols = []
    # 1) Names containing both 'event' and 'time' strongly preferred
    ordered_cols += [c for c, n in norm.items() if 'event' in n and 'time' in n]
    # 2) Known priority keys
    ordered_cols += [c for c, n in norm.items() if any(k in n for k in priority_keys) and c not in ordered_cols]
    # 3) Secondary tokens
    ordered_cols += [c for c, n in norm.items() if any(t in n for t in secondary_tokens) and c not in ordered_cols]

    best_col = None
    best_ratio = 0.0
    for c in ordered_cols:
        try:
            # If column already a datetime dtype, prefer it immediately
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
            # If many values are datetime.time objects, accept it
            try:
                time_obj_ratio = df[c].map(lambda v: isinstance(v, time)).mean()
            except Exception:
                time_obj_ratio = 0
            if time_obj_ratio >= 0.60:
                return c
            ser = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
            ratio = ser.notna().mean()
            # Accept if >= 40% parseable; keep the best
            if ratio >= 0.40 and ratio > best_ratio:
                best_col, best_ratio = c, ratio
        except Exception:
            continue

    # As a last resort, scan all object columns to find the most parseable
    if not best_col:
        for c in df.columns:
            if df[c].dtype == 'object':
                try:
                    ser = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
                    ratio = ser.notna().mean()
                    if ratio >= 0.60 and ratio > best_ratio:
                        best_col, best_ratio = c, ratio
                except Exception:
                    continue

    return best_col

def detect_tag_column(df: pd.DataFrame):
    """Detect alarm tag/point name column"""
    candidates = ['tag', 'tagname', 'point', 'alarm_tag', 'location tag', 'location', 'source', 'item', 'name']
    for c in df.columns:
        if c.lower() in candidates:
            return c
    # Fallback: column with high cardinality string values
    for c in df.columns:
        if df[c].dtype == 'object' and df[c].nunique() > len(df) * 0.1:
            return c
    return None

def detect_priority_column(df: pd.DataFrame):
    """Detect priority/severity column"""
    candidates = ['priority', 'severity', 'prio', 'level', 'class']
    for c in df.columns:
        if c.lower() in candidates:
            return c
    return None

# -----------------------
# ISA-18.2 Compliant Analysis Functions
# -----------------------
def calculate_isa_metrics(df, ts_col, tag_col=None, priority_col=None):
    """Calculate comprehensive ISA-18.2 metrics"""
    metrics = {}
    
    # Prepare data
    df_clean = df.copy()
    ts_series = df_clean[ts_col]
    # First attempt: normal parsing
    parsed = pd.to_datetime(ts_series, errors='coerce', infer_datetime_format=True)

    # If parsing failed for majority, try handling time-only values
    try:
        parsed_ratio = parsed.notna().mean()
    except Exception:
        parsed_ratio = 0

    if parsed_ratio < 0.40:
        # Case A: python datetime.time objects
        try:
            is_time_obj_ratio = ts_series.map(lambda v: isinstance(v, time)).mean()
        except Exception:
            is_time_obj_ratio = 0
        if is_time_obj_ratio >= 0.50:
            base = pd.Timestamp.today().normalize()
            parsed = ts_series.map(
                lambda t: base + pd.to_timedelta(getattr(t, 'hour', 0), unit='h')
                + pd.to_timedelta(getattr(t, 'minute', 0), unit='m')
                + pd.to_timedelta(getattr(t, 'second', 0), unit='s') if isinstance(t, time) else pd.NaT
            )
        else:
            # Case B: strings like HH:MM[:SS] possibly with AM/PM
            s = ts_series.astype(str).str.strip()
            # Extract H, M, S (with optional fractional seconds) and AM/PM
            parts = s.str.extract(r"^(?P<a>\d{1,3})(?::(?P<b>\d{1,2}(?:\.\d+)?))(?::(?P<c>\d{1,2}(?:\.\d+)?))?(?:\s*(?P<ap>(?:AM|PM|am|pm)))?$")
            if not parts.isna().all().all():
                A = pd.to_numeric(parts['a'], errors='coerce')
                B = pd.to_numeric(parts['b'], errors='coerce')
                # seconds may have fraction
                C = pd.to_numeric(parts['c'], errors='coerce')
                ap = parts['ap'].astype(str).str.lower().where(parts['ap'].notna(), '')
                base = pd.Timestamp.today().normalize()
                # Decide whether A is hours or minutes
                # If AM/PM present, A is hours
                is_hours = (ap.isin(['am','pm'])) | (A <= 23)
                hours = np.where(is_hours, A.fillna(0), 0)
                minutes = np.where(is_hours, B.fillna(0), A.fillna(0))
                seconds = np.where(is_hours, C.fillna(0), B.fillna(0))
                # Convert 12-hour to 24-hour when AM/PM present
                hours = pd.to_numeric(hours, errors='coerce').fillna(0)
                if (ap == 'pm').any() or (ap == 'am').any():
                    h_adj = hours.copy()
                    h_adj = np.where(ap == 'pm', np.where((hours % 12) == 0, 12, (hours % 12) + 12), h_adj)
                    h_adj = np.where(ap == 'am', np.where(hours == 12, 0, hours), h_adj)
                    hours = pd.to_numeric(h_adj, errors='coerce').fillna(0)
                parsed = base + pd.to_timedelta(hours * 3600 + minutes * 60 + seconds, unit='s')

            # Case C: Excel serial time/day fractions (numbers)
            if parsed.isna().mean() > 0.50:
                def parse_scalar(v):
                    try:
                        if isinstance(v, (int, float)):
                            # If within a day fraction
                            frac = float(v)
                            base = pd.Timestamp.today().normalize()
                            return base + pd.to_timedelta(frac, unit='D')
                        return pd.NaT
                    except Exception:
                        return pd.NaT
                parsed_alt = ts_series.map(parse_scalar)
                # Use where parsed is NaT and parsed_alt is valid
                parsed = parsed.where(parsed.notna(), parsed_alt)

    # Ensure no duplicate columns before assigning timestamp
    if ts_col in df_clean.columns:
        # Create a temporary column name to avoid conflicts
        temp_ts_col = f"_temp_timestamp_{hash(ts_col) % 10000}"
        df_clean[temp_ts_col] = parsed
        df_clean = df_clean.dropna(subset=[temp_ts_col])
        
        if df_clean.empty:
            return metrics
        
        # Basic metrics
        total_alarms = len(df_clean)
        time_span = (df_clean[temp_ts_col].max() - df_clean[temp_ts_col].min()).total_seconds() / 3600
        avg_alarms_per_hour = total_alarms / time_span if time_span > 0 else 0
        
        metrics.update({
            'total_alarms': total_alarms,
            'time_span_hours': time_span,
            'avg_alarms_per_hour': avg_alarms_per_hour,
            'isa_performance': 'Good' if avg_alarms_per_hour <= 6 else 'Acceptable' if avg_alarms_per_hour <= 12 else 'Poor'
        })
        
        # Hourly analysis - use temporary column for indexing
        try:
            df_indexed = df_clean.set_index(temp_ts_col)
        except Exception:
            # If set_index fails, create a copy with reset index
            df_indexed = df_clean.copy()
            df_indexed.index = df_clean[temp_ts_col]
    else:
        return metrics  # Column not found
    hourly_counts = df_indexed.resample('1H').size()
    metrics['hourly_counts'] = hourly_counts
    
    # Flood analysis (10-min windows)
    ten_min_counts = df_indexed.resample('10T').size()
    flood_periods = ten_min_counts[ten_min_counts > ISA_BENCHMARKS['flood_threshold']]
    flood_percentage = (len(flood_periods) / len(ten_min_counts)) * 100 if len(ten_min_counts) > 0 else 0
    
    metrics.update({
        'flood_periods': len(flood_periods),
        'flood_percentage': flood_percentage,
        'flood_performance': 'Good' if flood_percentage < 1.0 else 'Poor'
    })
    
    # Top contributors analysis
    if tag_col and tag_col in df_clean.columns:
        tag_counts = df_clean[tag_col].value_counts()
        top_10_count = tag_counts.head(10).sum()
        top_10_percentage = (top_10_count / total_alarms) * 100
        
        metrics.update({
            'top_10_contribution': top_10_percentage,
            'top_10_performance': 'Good' if top_10_percentage < 5.0 else 'Poor',
            'tag_counts': tag_counts
        })
    
    # Priority distribution
    if priority_col and priority_col in df_clean.columns:
        prio_dist = df_clean[priority_col].astype(str).value_counts(normalize=True) * 100
        metrics['priority_distribution'] = prio_dist
    
    return metrics

def alarms_per_hour(df, ts_col):
    """Legacy function for backward compatibility"""
    s = df.copy()
    s[ts_col] = pd.to_datetime(s[ts_col], errors='coerce')
    s = s.dropna(subset=[ts_col])
    s = s.set_index(ts_col)
    counts = s.resample('1H').size()
    return counts

def top_n_tags(df, tag_col='tag', n=10):
    if tag_col not in df.columns:
        # try common names
        for cand in ['tag', 'alarm', 'message', 'tagname']:
            if cand in df.columns:
                tag_col = cand
                break
    if tag_col in df.columns:
        return df[tag_col].value_counts().head(n)
    return pd.Series(dtype=int)

def priority_distribution(df, priority_col='priority'):
    # normalize priority values if numeric strings like P1,P2 or numeric 1,2,3
    if priority_col not in df.columns:
        # try common names
        for cand in ['priority', 'prio', 'severity']:
            if cand in df.columns:
                priority_col = cand
                break
    if priority_col in df.columns:
        return df[priority_col].astype(str).value_counts()
    return pd.Series(dtype=int)

def flood_detection(df, ts_col, window_minutes=10, threshold=10):
    ser = pd.to_datetime(df[ts_col], errors='coerce').dropna()
    if ser.empty:
        return pd.DataFrame()
    counts = ser.dt.floor('T').groupby(ser.dt.floor('T')).size()
    counts = counts.resample(f'{window_minutes}T').sum()
    floods = counts[counts > threshold]
    return floods

def chattering_detection(df, ts_col, tag_col='tag', repeats_threshold=5, window_minutes=10):
    # count repeats of same tag in sliding window
    if tag_col not in df.columns:
        for cand in ['tag', 'alarm', 'message', 'tagname']:
            if cand in df.columns:
                tag_col = cand
                break
    if ts_col not in df.columns or tag_col not in df.columns:
        return pd.DataFrame()
    s = df[[ts_col, tag_col]].copy()
    s[ts_col] = pd.to_datetime(s[ts_col], errors='coerce')
    s = s.dropna(subset=[ts_col])
    s = s.sort_values(ts_col)
    # rolling window count per tag
    s['window_start'] = (s[ts_col] - pd.to_timedelta(window_minutes, unit='m')).astype('datetime64[ns]')
    # we'll use groupby + apply: for each tag, compute count in rolling
    out = []
    for tag, sub in s.groupby(tag_col):
        idx = sub[ts_col]
        # for each event, count how many in trailing window_minutes
        counts = idx.rolling(f'{window_minutes}min').count()
        max_count = int(counts.max()) if not counts.empty else 0
        if max_count >= repeats_threshold:
            out.append({'tag': tag, 'max_repeats_in_window': max_count})
    if out:
        return pd.DataFrame(out).sort_values('max_repeats_in_window', ascending=False)
    return pd.DataFrame()

def generate_recommendations(df, ts_col, tag_col=None, priority_col=None):
    """Generate ISA-18.2 compliant recommendations with confidence scoring"""
    recommendations = []
    
    if not tag_col or tag_col not in df.columns:
        return pd.DataFrame(recommendations)
    
    # Chattering analysis for deadband recommendations
    chatter_tags = chattering_detection(df, ts_col, tag_col, repeats_threshold=5)
    for _, row in chatter_tags.iterrows():
        confidence = min(0.95, 0.5 + (row['max_repeats_in_window'] - 5) * 0.1)
        recommendations.append({
            'tag': row['tag'],
            'issue': 'Chattering alarm',
            'recommendation': 'Increase deadband/time delay',
            'confidence': confidence,
            'evidence': f"{row['max_repeats_in_window']} repeats in 10-min window",
            'action_type': 'deadband_tuning',
            'priority': 'High' if row['max_repeats_in_window'] > 10 else 'Medium'
        })
    
    # Priority rebalancing recommendations
    if priority_col and priority_col in df.columns:
        prio_dist = df[priority_col].astype(str).value_counts(normalize=True) * 100
        high_prio_tags = df[df[priority_col].astype(str).str.contains('1|high|critical', case=False, na=False)]
        
        if not high_prio_tags.empty:
            # Analyze high priority alarms for potential downgrading
            tag_response_times = high_prio_tags.groupby(tag_col).size()
            for tag, count in tag_response_times.items():
                if count > 50:  # Frequent high priority alarms
                    confidence = min(0.9, 0.6 + (count - 50) * 0.005)
                    recommendations.append({
                        'tag': tag,
                        'issue': 'Over-prioritized alarm',
                        'recommendation': 'Consider priority downgrade',
                        'confidence': confidence,
                        'evidence': f"{count} high-priority occurrences",
                        'action_type': 'priority_adjustment',
                        'priority': 'Medium'
                    })
    
    return pd.DataFrame(recommendations)

def standing_alarms_heuristic(df, ts_col, tag_col=None, min_duration_hours=24):
    """Enhanced standing alarm detection"""
    if not tag_col or tag_col not in df.columns:
        return pd.DataFrame()
    
    ts = pd.to_datetime(df[ts_col], errors='coerce')
    if ts.isna().all():
        return pd.DataFrame()
    
    tmp = df.copy()
    tmp[ts_col] = ts
    tmp = tmp.dropna(subset=[ts_col])
    
    # Calculate span per tag
    spans = tmp.groupby(tag_col)[ts_col].agg(['min', 'max', 'count'])
    spans['span_hours'] = (spans['max'] - spans['min']).dt.total_seconds() / 3600.0
    spans['avg_interval_hours'] = spans['span_hours'] / spans['count']
    
    # Filter for potential standing alarms
    standing = spans[spans['span_hours'] >= min_duration_hours]
    standing = standing.sort_values('span_hours', ascending=False)
    
    return standing.head(50)

# -----------------------
# Enhanced Streamlit UI Layout
# -----------------------
st.title("🚨 Alarm Rationalization Application")
st.markdown("""
**ISA-18.2 Compliant Alarm Management System**  
*Digital Operations / Process Control*

Analyze historical alarm data, evaluate performance against ISA-18.2 benchmarks, and generate actionable recommendations.
""")

# Create tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📁 File Explorer", "📈 Analytics", "💡 Recommendations"])

with st.sidebar:
    st.header("🔍 Data Source Configuration")
    
    # Auto-detect current project path
    current_path = os.getcwd()
    st.info(f"Current project: {os.path.basename(current_path)}")
    
    root_path = st.text_input(
        "Root folder path", 
        value=current_path,
        help="Path containing alarm data folders (PVC-I, PVC-II, etc.)"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Scan Folders"):
            st.rerun()
    with col2:
        if st.button("📂 Use Current"):
            root_path = current_path
            st.rerun()

    folders = list_subfolders(root_path)
    if not folders:
        st.warning("No subfolders found. Ensure your alarm data is organized in subfolders.")
        st.stop()
    
    # Enhanced folder display with file counts
    st.subheader("📁 Available Data Sources")
    folder_info = []
    for folder in folders:
        files = list_data_files(folder)
        folder_info.append({
            'folder': folder.name,
            'files': len(files),
            'path': str(folder)
        })
    
    folder_df = pd.DataFrame(folder_info)
    st.dataframe(folder_df, use_container_width=True)
    
    # Folder selection
    folder_names = [f.name for f in folders]
    selected = st.selectbox("Select Data Source", folder_names, key="folder_select")
    selected_folder = next((f for f in folders if f.name == selected), None)
    
    st.markdown("---")
    
    # File list for selected folder
    files = list_data_files(selected_folder)
    if files:
        st.write(f"**{len(files)} files in {selected}:**")
        for f in files:
            file_size = f.stat().st_size / 1024  # KB
            st.write(f"📄 {f.name} ({file_size:.1f} KB)")
    else:
        st.info("No data files in this folder.")
    
    # Analysis settings
    st.subheader("⚙️ Analysis Settings")
    sample_size = st.slider("Sample size (rows)", 1000, 50000, 10000)
    show_advanced = st.checkbox("Show advanced analytics")

# Main analysis panels
if files:
    # Dashboard Tab
    with tab1:
        st.header("📊 ISA-18.2 Performance Dashboard")
        
        # Aggregate metrics across all files
        all_metrics = []
        for file_path in files:
            try:
                df = read_table(str(file_path), nrows=sample_size)
                ts_col = detect_timestamp_column(df)
                tag_col = detect_tag_column(df)
                priority_col = detect_priority_column(df)
                
                if ts_col:
                    metrics = calculate_isa_metrics(df, ts_col, tag_col, priority_col)
                    metrics['source_file'] = file_path.name
                    all_metrics.append(metrics)
            except Exception as e:
                st.error(f"Error processing {file_path.name}: {e}")
        
        if all_metrics:
            # Overall KPI cards
            col1, col2, col3, col4 = st.columns(4)
            
            total_alarms = sum(m.get('total_alarms', 0) for m in all_metrics)
            avg_rate = np.mean([m.get('avg_alarms_per_hour', 0) for m in all_metrics if m.get('avg_alarms_per_hour')])
            avg_flood = np.mean([m.get('flood_percentage', 0) for m in all_metrics if m.get('flood_percentage')])
            
            with col1:
                st.metric(
                    "Total Alarms", 
                    f"{total_alarms:,}",
                    help="Total alarm count across all files"
                )
            
            with col2:
                rate_status = "🟢" if avg_rate <= 6 else "🟡" if avg_rate <= 12 else "🔴"
                st.metric(
                    "Avg Alarms/Hour", 
                    f"{avg_rate:.1f} {rate_status}",
                    delta=f"{avg_rate - 6:.1f} vs ISA target",
                    help="ISA-18.2 target: ≤6 good, ≤12 acceptable"
                )
            
            with col3:
                flood_status = "🟢" if avg_flood < 1.0 else "🔴"
                st.metric(
                    "Flood %", 
                    f"{avg_flood:.1f}% {flood_status}",
                    delta=f"{avg_flood - 1.0:.1f}% vs target",
                    help="% of time in alarm flood (>10 alarms/10min)"
                )
            
            with col4:
                st.metric(
                    "Data Sources", 
                    f"{len(all_metrics)}",
                    help="Number of processed alarm files"
                )
            
            # Enhanced Performance Dashboard with Multiple Charts
            st.subheader("🎯 ISA-18.2 Performance Overview")
            
            # Performance summary chart
            if len(all_metrics) > 1:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Alarm Rate Performance Chart
                    fig1 = go.Figure()
                    
                    file_names = [m['source_file'] for m in all_metrics]
                    alarm_rates = [m.get('avg_alarms_per_hour', 0) for m in all_metrics]
                    
                    fig1.add_trace(go.Bar(
                        x=file_names,
                        y=alarm_rates,
                        name='Alarms/Hour',
                        marker_color=['#2E8B57' if rate <= 6 else '#FF8C00' if rate <= 12 else '#DC143C' for rate in alarm_rates],
                        text=[f"{rate:.1f}" for rate in alarm_rates],
                        textposition='outside'
                    ))
                    
                    fig1.add_hline(y=6, line_dash="dash", line_color="#2E8B57", annotation_text="ISA Good (≤6)")
                    fig1.add_hline(y=12, line_dash="dash", line_color="#FF8C00", annotation_text="ISA Max (≤12)")
                    
                    fig1.update_layout(
                        title="Alarm Rate Performance by Data Source",
                        xaxis_title="Data Source",
                        yaxis_title="Alarms per Hour",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Flood Performance Chart
                    fig2 = go.Figure()
                    
                    flood_rates = [m.get('flood_percentage', 0) for m in all_metrics]
                    
                    fig2.add_trace(go.Bar(
                        x=file_names,
                        y=flood_rates,
                        name='Flood %',
                        marker_color=['#2E8B57' if rate < 1.0 else '#DC143C' for rate in flood_rates],
                        text=[f"{rate:.1f}%" for rate in flood_rates],
                        textposition='outside'
                    ))
                    
                    fig2.add_hline(y=1.0, line_dash="dash", line_color="#DC143C", annotation_text="ISA Target (<1%)")
                    
                    fig2.update_layout(
                        title="Alarm Flood Performance",
                        xaxis_title="Data Source",
                        yaxis_title="Flood Percentage (%)",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Time-based Analysis
            st.subheader("📈 Time-based Analysis")
            
            # Combine hourly data from all sources
            combined_hourly = pd.Series(dtype=int)
            for m in all_metrics:
                if 'hourly_counts' in m:
                    combined_hourly = combined_hourly.add(m['hourly_counts'], fill_value=0)
            
            if not combined_hourly.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Hourly alarm pattern
                    fig3 = go.Figure()
                    fig3.add_trace(go.Scatter(
                        x=combined_hourly.index,
                        y=combined_hourly.values,
                        mode='lines+markers',
                        name='Alarms per Hour',
                        line=dict(color='#1f77b4', width=2),
                        marker=dict(size=4)
                    ))
                    
                    fig3.add_hline(y=6, line_dash="dash", line_color="#2E8B57", annotation_text="Good")
                    fig3.add_hline(y=12, line_dash="dash", line_color="#FF8C00", annotation_text="Max")
                    
                    fig3.update_layout(
                        title="Hourly Alarm Pattern",
                        xaxis_title="Time",
                        yaxis_title="Alarms per Hour",
                        height=400
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                with col2:
                    # Alarm distribution histogram
                    fig4 = go.Figure()
                    fig4.add_trace(go.Histogram(
                        x=combined_hourly.values,
                        nbinsx=20,
                        name='Distribution',
                        marker_color='#1f77b4',
                        opacity=0.7
                    ))
                    
                    fig4.add_vline(x=6, line_dash="dash", line_color="#2E8B57", annotation_text="Good")
                    fig4.add_vline(x=12, line_dash="dash", line_color="#FF8C00", annotation_text="Max")
                    
                    fig4.update_layout(
                        title="Alarm Rate Distribution",
                        xaxis_title="Alarms per Hour",
                        yaxis_title="Frequency",
                        height=400
                    )
                    st.plotly_chart(fig4, use_container_width=True)
            
            # Top Contributors Analysis
            st.subheader("🔝 Top Alarm Contributors")
            
            # Combine tag data from all sources
            all_tag_counts = pd.Series(dtype=int)
            for file_path in files:
                try:
                    df = read_table(str(file_path), nrows=sample_size)
                    tag_col = detect_tag_column(df)
                    if tag_col:
                        tag_counts = df[tag_col].value_counts()
                        all_tag_counts = all_tag_counts.add(tag_counts, fill_value=0)
                except Exception:
                    continue
            
            if not all_tag_counts.empty:
                top_tags = all_tag_counts.head(15)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Top contributors bar chart
                    fig5 = go.Figure()
                    fig5.add_trace(go.Bar(
                        y=top_tags.index[::-1],  # Reverse for horizontal bar
                        x=top_tags.values[::-1],
                        orientation='h',
                        marker_color='#ff7f0e',
                        text=top_tags.values[::-1],
                        textposition='outside'
                    ))
                    
                    fig5.update_layout(
                        title="Top 15 Alarm Contributors",
                        xaxis_title="Alarm Count",
                        yaxis_title="Tag/Location",
                        height=500,
                        showlegend=False
                    )
                    st.plotly_chart(fig5, use_container_width=True)
                
                with col2:
                    # Top 10 contribution percentage
                    top_10_count = top_tags.head(10).sum()
                    total_count = all_tag_counts.sum()
                    top_10_pct = (top_10_count / total_count) * 100 if total_count > 0 else 0
                    
                    st.metric(
                        "Top 10 Contribution",
                        f"{top_10_pct:.1f}%",
                        delta=f"{top_10_pct - 5:.1f}% vs ISA target",
                        help="ISA-18.2 target: <5% from top 10 contributors"
                    )
                    
                    # Show top contributors table
                    st.write("**Top Contributors:**")
                    top_df = pd.DataFrame({
                        'Tag': top_tags.head(10).index,
                        'Count': top_tags.head(10).values,
                        'Percentage': (top_tags.head(10).values / total_count * 100).round(1)
                    })
                    st.dataframe(top_df, use_container_width=True)
            
            # ISA-18.2 Compliance Summary
            st.subheader("✅ ISA-18.2 Compliance Summary")
            
            compliance_data = []
            for m in all_metrics:
                rate = m.get('avg_alarms_per_hour', 0)
                flood = m.get('flood_percentage', 0)
                
                rate_status = "✅ Good" if rate <= 6 else "⚠️ Acceptable" if rate <= 12 else "❌ Poor"
                flood_status = "✅ Good" if flood < 1.0 else "❌ Poor"
                
                compliance_data.append({
                    'Data Source': m['source_file'],
                    'Alarm Rate': f"{rate:.1f}/hr",
                    'Rate Status': rate_status,
                    'Flood %': f"{flood:.1f}%",
                    'Flood Status': flood_status,
                    'Total Alarms': f"{m.get('total_alarms', 0):,}"
                })
            
            if compliance_data:
                compliance_df = pd.DataFrame(compliance_data)
                st.dataframe(compliance_df, use_container_width=True)
            
            # Key Insights
            st.subheader("💡 Key Insights")
            
            insights = []
            
            # Performance insights
            good_sources = sum(1 for m in all_metrics if m.get('avg_alarms_per_hour', 0) <= 6)
            poor_sources = sum(1 for m in all_metrics if m.get('avg_alarms_per_hour', 0) > 12)
            
            if good_sources > 0:
                insights.append(f"✅ **{good_sources}/{len(all_metrics)}** data sources meet ISA-18.2 'Good' performance (≤6 alarms/hour)")
            
            if poor_sources > 0:
                insights.append(f"❌ **{poor_sources}/{len(all_metrics)}** data sources exceed ISA-18.2 maximum (>12 alarms/hour)")
            
            # Flood insights
            flood_sources = sum(1 for m in all_metrics if m.get('flood_percentage', 0) >= 1.0)
            if flood_sources > 0:
                insights.append(f"🌊 **{flood_sources}/{len(all_metrics)}** data sources experience alarm floods (≥1% of time)")
            
            # Top contributor insights
            if not all_tag_counts.empty and total_count > 0:
                if top_10_pct > 5:
                    insights.append(f"🔝 Top 10 contributors generate **{top_10_pct:.1f}%** of alarms (ISA target: <5%)")
                else:
                    insights.append(f"✅ Top 10 contributors generate **{top_10_pct:.1f}%** of alarms (within ISA target)")
            
            for insight in insights:
                st.markdown(f"• {insight}")
            
            if not insights:
                st.info("Complete analysis to generate insights")
        else:
            # No metrics available; show guidance and quick diagnostics
            st.warning("No charts to display. Could not compute metrics from the selected files.")
            diag_expander = st.expander("Show quick diagnostics", expanded=True)
            with diag_expander:
                for file_path in files:
                    try:
                        df = read_table(str(file_path), nrows=min(2000, sample_size))
                        ts_col = detect_timestamp_column(df)
                        tag_col = detect_tag_column(df)
                        priority_col = detect_priority_column(df)
                        st.write(f"• {file_path.name} — Timestamp: {ts_col or 'not detected'}, Tag: {tag_col or 'not detected'}, Priority: {priority_col or 'not detected'}")
                    except Exception as e:
                        st.write(f"• {file_path.name} — read error: {e}")
            st.info("Tips: 1) Click 'Scan Folders' in the sidebar to rerun. 2) Ensure files export include 'Event Time' and are readable. 3) Try reducing sample size.")
    
    # File Explorer Tab Content
    with tab2:
        st.header("📁 Data File Explorer")
        for file_path in files:
            with st.expander(f"📄 {file_path.name}", expanded=False):
                try:
                    df = read_table(str(file_path), nrows=sample_size)
                    
                    # Auto-detect columns
                    ts_col = detect_timestamp_column(df)
                    tag_col = detect_tag_column(df)
                    priority_col = detect_priority_column(df)
                    
                    # Column detection summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Timestamp:** {ts_col or 'Not detected'}")
                    with col2:
                        st.write(f"**Tag/Point:** {tag_col or 'Not detected'}")
                    with col3:
                        st.write(f"**Priority:** {priority_col or 'Not detected'}")
                    
                    # Quick preview
                    st.dataframe(df.head(5), use_container_width=True)
                    
                    # Basic stats
                    st.write(f"**Rows:** {len(df):,} | **Columns:** {len(df.columns)}")
                    
                except Exception as e:
                    st.error(f"Unable to read file: {e}")
    
    # Analytics Tab - Individual File Analysis
    with tab3:
        st.header("📈 Individual File Analytics")
        
        if not files:
            st.info("No files selected for analysis.")
        else:
            # Analyze each file separately
            for file_path in files:
                with st.expander(f"📊 Analysis: {file_path.name}", expanded=True):
                    try:
                        df = read_table(str(file_path), nrows=sample_size)
                        
                        # Auto-detect columns
                        ts_col = detect_timestamp_column(df)
                        tag_col = detect_tag_column(df)
                        priority_col = detect_priority_column(df)
                        
                        # File info header
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        with info_col1:
                            st.metric("Total Records", f"{len(df):,}")
                        with info_col2:
                            st.metric("Columns", len(df.columns))
                        with info_col3:
                            file_size = file_path.stat().st_size / (1024*1024)  # MB
                            st.metric("File Size", f"{file_size:.1f} MB")
                        with info_col4:
                            encoding_info = getattr(df, 'attrs', {}).get('encoding_used', 'auto-detected')
                            st.write(f"**Encoding:** {encoding_info}")
                        
                        # Column detection status
                        st.write("**Column Detection:**")
                        det_col1, det_col2, det_col3 = st.columns(3)
                        with det_col1:
                            ts_status = "✅" if ts_col else "❌"
                            st.write(f"{ts_status} **Timestamp:** {ts_col or 'Not found'}")
                        with det_col2:
                            tag_status = "✅" if tag_col else "❌"
                            st.write(f"{tag_status} **Tag/Point:** {tag_col or 'Not found'}")
                        with det_col3:
                            prio_status = "✅" if priority_col else "❌"
                            st.write(f"{prio_status} **Priority:** {priority_col or 'Not found'}")
                        
                        # Data preview
                        st.subheader("📋 Data Preview")
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        if ts_col:
                            # Calculate metrics for this file
                            metrics = calculate_isa_metrics(df, ts_col, tag_col, priority_col)
                            
                            # ISA-18.2 Performance for this file
                            st.subheader("🎯 ISA-18.2 Performance")
                            
                            kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
                            
                            with kpi_col1:
                                rate = metrics.get('avg_alarms_per_hour', 0)
                                rate_delta = rate - ISA_BENCHMARKS['target_alarms_per_hour']
                                rate_status = "🟢" if rate <= 6 else "🟡" if rate <= 12 else "🔴"
                                st.metric(
                                    "Alarms/Hour", 
                                    f"{rate:.1f} {rate_status}",
                                    delta=f"{rate_delta:+.1f} vs target (6)"
                                )
                            
                            with kpi_col2:
                                flood_pct = metrics.get('flood_percentage', 0)
                                flood_delta = flood_pct - ISA_BENCHMARKS['flood_target_percent']
                                flood_status = "🟢" if flood_pct < 1.0 else "🔴"
                                st.metric(
                                    "Flood %", 
                                    f"{flood_pct:.1f}% {flood_status}",
                                    delta=f"{flood_delta:+.1f}% vs target (1%)"
                                )
                            
                            with kpi_col3:
                                top10_pct = metrics.get('top_10_contribution', 0)
                                top10_delta = top10_pct - ISA_BENCHMARKS['top_10_contribution_target']
                                top10_status = "🟢" if top10_pct < 5.0 else "🔴"
                                st.metric(
                                    "Top-10 Contrib", 
                                    f"{top10_pct:.1f}% {top10_status}",
                                    delta=f"{top10_delta:+.1f}% vs target (5%)"
                                )
                            
                            # Time Series Chart
                            if 'hourly_counts' in metrics and not metrics['hourly_counts'].empty:
                                st.subheader("📈 Alarm Rate Trend")
                                
                                hourly_data = metrics['hourly_counts'].reset_index()
                                hourly_data.columns = ['timestamp', 'alarm_count']
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=hourly_data['timestamp'],
                                    y=hourly_data['alarm_count'],
                                    mode='lines+markers',
                                    name='Alarms/Hour',
                                    line=dict(color='blue', width=2)
                                ))
                                
                                # Add ISA benchmark lines
                                fig.add_hline(y=6, line_dash="dash", line_color="green", 
                                            annotation_text="ISA Good (≤6)", annotation_position="top right")
                                fig.add_hline(y=12, line_dash="dash", line_color="red", 
                                            annotation_text="ISA Max (≤12)", annotation_position="top right")
                                
                                fig.update_layout(
                                    title=f"Alarm Rate Over Time - {file_path.name}",
                                    xaxis_title="Time",
                                    yaxis_title="Alarms per Hour",
                                    height=400,
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Rolling average
                                if len(hourly_data) > 24:
                                    hourly_data['rolling_24h'] = hourly_data['alarm_count'].rolling(24, min_periods=1).mean()
                                    
                                    fig2 = go.Figure()
                                    fig2.add_trace(go.Scatter(
                                        x=hourly_data['timestamp'],
                                        y=hourly_data['rolling_24h'],
                                        mode='lines',
                                        name='24h Rolling Average',
                                        line=dict(color='orange', width=3)
                                    ))
                                    
                                    fig2.update_layout(
                                        title="24-Hour Rolling Average Trend",
                                        xaxis_title="Time",
                                        yaxis_title="Avg Alarms/Hour",
                                        height=300
                                    )
                                    
                                    st.plotly_chart(fig2, use_container_width=True)
                            
                            # Top Contributors Analysis
                            if 'tag_counts' in metrics and not metrics['tag_counts'].empty:
                                st.subheader("📊 Top Alarm Contributors")
                                
                                top_tags = metrics['tag_counts'].head(15)
                                
                                # Horizontal bar chart
                                fig3 = go.Figure()
                                fig3.add_trace(go.Bar(
                                    x=top_tags.values,
                                    y=top_tags.index,
                                    orientation='h',
                                    marker_color='lightblue',
                                    text=top_tags.values,
                                    textposition='outside'
                                ))
                                
                                fig3.update_layout(
                                    title=f"Top 15 Alarm Tags - {file_path.name}",
                                    xaxis_title="Alarm Count",
                                    yaxis_title="Alarm Tag",
                                    height=500,
                                    yaxis=dict(autorange="reversed")
                                )
                                
                                st.plotly_chart(fig3, use_container_width=True)
                                
                                # Pareto cumulative chart
                                cumulative_pct = (top_tags.cumsum() / top_tags.sum()) * 100
                                
                                fig4 = make_subplots(specs=[[{"secondary_y": True}]])
                                
                                fig4.add_trace(
                                    go.Bar(x=list(range(len(top_tags))), y=top_tags.values, 
                                          name="Count", marker_color='lightcoral'),
                                    secondary_y=False,
                                )
                                
                                fig4.add_trace(
                                    go.Scatter(x=list(range(len(top_tags))), y=cumulative_pct.values,
                                             mode='lines+markers', name="Cumulative %", 
                                             line=dict(color='darkgreen', width=3)),
                                    secondary_y=True,
                                )
                                
                                fig4.update_xaxes(title_text="Top Tags (Ranked)")
                                fig4.update_yaxes(title_text="Alarm Count", secondary_y=False)
                                fig4.update_yaxes(title_text="Cumulative %", secondary_y=True)
                                fig4.update_layout(title="Pareto Analysis - Cumulative Contribution")
                                
                                st.plotly_chart(fig4, use_container_width=True)
                            
                            # Priority Distribution
                            if 'priority_distribution' in metrics and not metrics['priority_distribution'].empty:
                                st.subheader("🎯 Priority Distribution Analysis")
                                
                                prio_data = metrics['priority_distribution']
                                
                                # Pie chart
                                fig5 = go.Figure(data=[go.Pie(
                                    labels=prio_data.index,
                                    values=prio_data.values,
                                    hole=0.3,
                                    textinfo='label+percent',
                                    textposition='outside'
                                )])
                                
                                fig5.update_layout(
                                    title=f"Priority Distribution - {file_path.name}",
                                    height=400
                                )
                                
                                st.plotly_chart(fig5, use_container_width=True)
                                
                                # ISA Target Comparison
                                st.write("**ISA-18.2 Target vs Actual Comparison:**")
                                target_comparison = []
                                for prio in ['P1', 'P2', 'P3', '1', '2', '3', 'High', 'Medium', 'Low']:
                                    if prio in prio_data.index:
                                        actual_pct = prio_data[prio]
                                        if prio in ['P1', '1', 'High']:
                                            target = 5
                                        elif prio in ['P2', '2', 'Medium']:
                                            target = 15
                                        else:
                                            target = 80
                                        
                                        target_comparison.append({
                                            'Priority': prio,
                                            'Target %': target,
                                            'Actual %': actual_pct,
                                            'Deviation': actual_pct - target,
                                            'Status': '🟢' if abs(actual_pct - target) < 10 else '🔴'
                                        })
                                
                                if target_comparison:
                                    comp_df = pd.DataFrame(target_comparison)
                                    st.dataframe(comp_df, use_container_width=True)
                            
                            # Advanced Pattern Detection
                            st.subheader("🔍 Pattern Detection Results")
                            
                            pattern_col1, pattern_col2 = st.columns(2)
                            
                            with pattern_col1:
                                # Flood detection
                                floods = flood_detection(df, ts_col, window_minutes=10, threshold=10)
                                st.write("**🌊 Alarm Floods:**")
                                if not floods.empty:
                                    flood_df = floods.reset_index()
                                    flood_df.columns = ['timestamp', 'alarm_count']
                                    flood_df = flood_df.sort_values('alarm_count', ascending=False)
                                    st.dataframe(flood_df.head(10), use_container_width=True)
                                    
                                    # Flood timeline chart
                                    fig_flood = px.bar(flood_df.head(20), x='timestamp', y='alarm_count',
                                                     title='Alarm Flood Events')
                                    fig_flood.add_hline(y=10, line_dash="dash", line_color="red",
                                                       annotation_text="Flood Threshold (10)")
                                    st.plotly_chart(fig_flood, use_container_width=True)
                                else:
                                    st.success("✅ No alarm floods detected")
                            
                            with pattern_col2:
                                # Chattering detection
                                if tag_col:
                                    chatter = chattering_detection(df, ts_col, tag_col, repeats_threshold=5)
                                    st.write("**🔄 Chattering Alarms:**")
                                    if not chatter.empty:
                                        st.dataframe(chatter.head(10), use_container_width=True)
                                        
                                        # Chattering chart
                                        fig_chatter = px.bar(chatter.head(10), x='max_repeats_in_window', y='tag',
                                                           orientation='h', title='Chattering Alarm Frequency')
                                        fig_chatter.add_vline(x=5, line_dash="dash", line_color="orange",
                                                             annotation_text="Threshold (5)")
                                        st.plotly_chart(fig_chatter, use_container_width=True)
                                    else:
                                        st.success("✅ No chattering alarms detected")
                            
                            # Standing Alarms
                            if tag_col:
                                standing = standing_alarms_heuristic(df, ts_col, tag_col)
                                st.write("**⏰ Standing Alarms (Long Duration):**")
                                if not standing.empty:
                                    standing_display = standing.reset_index()
                                    standing_display = standing_display.sort_values('span_hours', ascending=False)
                                    st.dataframe(standing_display.head(10), use_container_width=True)
                                    
                                    # Standing alarms duration chart
                                    fig_standing = px.bar(standing_display.head(15), 
                                                        x='span_hours', y=standing_display.index,
                                                        orientation='h', title='Standing Alarm Duration (Hours)')
                                    fig_standing.add_vline(x=24, line_dash="dash", line_color="red",
                                                         annotation_text="24h Threshold")
                                    st.plotly_chart(fig_standing, use_container_width=True)
                                else:
                                    st.success("✅ No standing alarms detected")
                            
                            # Data Quality Assessment
                            st.subheader("🔍 Data Quality Assessment")
                            
                            qual_col1, qual_col2 = st.columns(2)
                            
                            with qual_col1:
                                st.write("**Missing Values:**")
                                missing_data = df.isnull().sum().sort_values(ascending=False)
                                missing_pct = (missing_data / len(df)) * 100
                                
                                if missing_data.sum() > 0:
                                    missing_df = pd.DataFrame({
                                        'Column': missing_data.index,
                                        'Missing Count': missing_data.values,
                                        'Missing %': missing_pct.values
                                    })
                                    missing_df = missing_df[missing_df['Missing Count'] > 0]
                                    st.dataframe(missing_df, use_container_width=True)
                                else:
                                    st.success("✅ No missing values")
                            
                            with qual_col2:
                                st.write("**Column Data Types:**")
                                dtype_df = pd.DataFrame({
                                    'Column': df.columns,
                                    'Data Type': df.dtypes.astype(str),
                                    'Unique Values': [df[col].nunique() for col in df.columns]
                                })
                                st.dataframe(dtype_df, use_container_width=True)
                            
                            # Summary Statistics for Numeric Columns
                            numeric_cols = df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                st.subheader("📊 Numeric Column Statistics")
                                st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
                        
                        else:
                            st.warning("⚠️ No timestamp column detected. Limited analysis available.")
                            
                            # Still show basic info
                            st.subheader("📋 Basic Data Preview")
                            st.dataframe(df.head(10), use_container_width=True)
                            
                            # Show column info
                            st.subheader("📝 Column Information")
                            col_info = pd.DataFrame({
                                'Column': df.columns,
                                'Data Type': df.dtypes.astype(str),
                                'Non-Null Count': df.count(),
                                'Unique Values': [df[col].nunique() for col in df.columns]
                            })
                            st.dataframe(col_info, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"❌ Error analyzing {file_path.name}: {str(e)}")
                        st.write("**Troubleshooting suggestions:**")
                        st.write("- Check if file is corrupted or locked")
                        st.write("- Verify file contains proper CSV/Excel data")
                        st.write("- Try reducing sample size in sidebar settings")
    
    # Recommendations Tab
    with tab4:
        st.header("💡 Alarm Rationalization Recommendations")
        
        if not files:
            st.info("No files available for recommendation generation.")
        else:
            # Generate recommendations for all files
            all_recommendations = []
            
            for file_path in files:
                try:
                    df = read_table(str(file_path), nrows=sample_size)
                    ts_col = detect_timestamp_column(df)
                    tag_col = detect_tag_column(df)
                    priority_col = detect_priority_column(df)
                    
                    if ts_col and tag_col:
                        recs = generate_recommendations(df, ts_col, tag_col, priority_col)
                        if not recs.empty:
                            recs['source_file'] = file_path.name
                            all_recommendations.append(recs)
                except Exception as e:
                    st.error(f"Error generating recommendations for {file_path.name}: {e}")
            
            if all_recommendations:
                combined_recs = pd.concat(all_recommendations, ignore_index=True)
                
                # Filter options
                st.subheader("🔧 Recommendation Filters")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    action_filter = st.multiselect(
                        "Action Type", 
                        combined_recs['action_type'].unique(),
                        default=combined_recs['action_type'].unique()
                    )
                
                with col2:
                    priority_filter = st.multiselect(
                        "Priority", 
                        combined_recs['priority'].unique(),
                        default=combined_recs['priority'].unique()
                    )
                
                with col3:
                    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.5)
                
                # Apply filters
                filtered_recs = combined_recs[
                    (combined_recs['action_type'].isin(action_filter)) &
                    (combined_recs['priority'].isin(priority_filter)) &
                    (combined_recs['confidence'] >= min_confidence)
                ]
                
                st.subheader("📋 Actionable Recommendations")
                
                if not filtered_recs.empty:
                    # Display recommendations in cards
                    for _, rec in filtered_recs.iterrows():
                        with st.container():
                            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                            
                            with col1:
                                st.write(f"**{rec['tag']}**")
                                st.write(f"*{rec['issue']}*")
                                st.write(f"**Action:** {rec['recommendation']}")
                            
                            with col2:
                                st.write(f"**Evidence:** {rec['evidence']}")
                                st.write(f"**Source:** {rec['source_file']}")
                            
                            with col3:
                                confidence_color = "🟢" if rec['confidence'] > 0.8 else "🟡" if rec['confidence'] > 0.6 else "🔴"
                                st.metric("Confidence", f"{rec['confidence']:.2f} {confidence_color}")
                            
                            with col4:
                                priority_color = "🔴" if rec['priority'] == 'High' else "🟡"
                                st.write(f"**Priority**")
                                st.write(f"{rec['priority']} {priority_color}")
                            
                            st.markdown("---")
                    
                    # Export recommendations
                    st.subheader("📤 Export Recommendations")
                    csv_data = filtered_recs.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Recommendations CSV",
                        data=csv_data,
                        file_name=f"alarm_recommendations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No recommendations match the current filters.")
            else:
                st.info("No recommendations generated. Ensure files have proper timestamp and tag columns.")

else:
    st.info("No files to analyze in the selected folder.")
