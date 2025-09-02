from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
import os
from pathlib import Path
from datetime import timedelta, datetime, time
from typing import Optional, List, Dict, Any
import json
import base64
from pydantic import BaseModel
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import glob
from threading import Lock

app = FastAPI(
    title="ISA-18.2 Alarm Rationalization API",
    description="FastAPI backend for ISA-18.2 compliant alarm management and analysis",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ISA-18.2 Benchmark Constants
ISA_BENCHMARKS = {
    'target_alarms_per_hour': 6,
    'max_alarms_per_hour': 12,
    'flood_threshold': 10,
    'flood_target_percent': 1.0,
    'top_10_contribution_target': 5.0,
    'priority_target': {'P1': 5, 'P2': 15, 'P3': 80},
    'chattering_threshold': 5,
    'standing_alarm_hours': 24
}

# Data Models
class FileInfo(BaseModel):
    filename: str
    size_mb: float
    rows: int
    columns: int
    encoding: Optional[str] = None

class AnalysisRequest(BaseModel):
    sample_size: Optional[int] = 10000
    show_advanced: Optional[bool] = False

class MetricsResponse(BaseModel):
    total_alarms: int
    time_span_hours: float
    avg_alarms_per_hour: float
    isa_performance: str
    flood_periods: int
    flood_percentage: float
    flood_performance: str
    top_10_contribution: Optional[float] = None
    top_10_performance: Optional[str] = None
    source_file: str

# Utility Functions (from original code)
def _normalize_columns_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to a canonical set"""
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
    used_targets = set()
    
    for c in df.columns:
        n = norm(c)
        matched = None
        for canon, aliases in canonical.items():
            if n == canon or n in aliases:
                matched = canon
                break
            if any(n.startswith(a) for a in aliases):
                matched = canon
                break
        
        if matched:
            target_name = matched.title()
            counter = 1
            original_target = target_name
            while target_name in used_targets:
                target_name = f"{original_target}_{counter}"
                counter += 1
            used_targets.add(target_name)
            mapping[c] = target_name
        else:
            original_col = c if isinstance(c, str) else str(c)
            final_col = original_col
            counter = 1
            while final_col in used_targets:
                final_col = f"{original_col}_{counter}"
                counter += 1
            used_targets.add(final_col)
            mapping[c] = final_col

    df = df.rename(columns=mapping)
    return df

def _apply_detected_header(df: pd.DataFrame) -> pd.DataFrame:
    """If the file was exported with a banner and the real header appears
    a few rows below (causing columns like 'Unnamed: 1'), detect and apply
    the correct header row.

    Heuristics:
    - If many columns are 'Unnamed' or the canonical header names are present
      inside the first N rows, treat that row as the header and drop rows above it.
    - Keep this conservative so normal files are untouched.
    """
    try:
        if df is None or df.empty:
            return df

        colnames = [str(c) for c in df.columns]
        unnamed_count = sum(1 for c in colnames if str(c).strip().lower().startswith('unnamed'))

        # Only attempt fix if it's likely broken
        likely_broken = unnamed_count >= max(2, len(colnames) // 2) or 'Event Time' not in colnames
        if not likely_broken:
            return df

        # Canonical header candidates to search for in early rows
        canonical_candidates = {
            'event time', 'location tag', 'source', 'condition', 'action',
            'priority', 'description', 'value', 'units'
        }

        def _norm(s: str) -> str:
            s = str(s)
            return " ".join(s.replace('_', ' ').strip().split()).lower()

        header_row_idx = None
        max_scan = min(50, len(df))
        for i in range(max_scan):
            row_vals = [str(v).strip() for v in df.iloc[i].tolist()]
            norm_vals = [_norm(v) for v in row_vals]
            hits = sum(1 for v in norm_vals if v in canonical_candidates)
            if hits >= 3 and any(tok in norm_vals for tok in ['event time', 'timestamp', 'time stamp']):
                header_row_idx = i
                break

        if header_row_idx is None:
            return df

        # Build new header from that row
        raw_headers = [" ".join(str(v).split()) for v in df.iloc[header_row_idx].tolist()]
        # Replace empty/placeholder cells
        cleaned_headers = [h if h and h not in ['-', '—', 'None', 'nan'] else f"Column_{idx+1}" for idx, h in enumerate(raw_headers)]

        # Ensure uniqueness
        used = {}
        unique_headers = []
        for h in cleaned_headers:
            base = h
            count = used.get(base, 0)
            final = base if count == 0 else f"{base}_{count+1}"
            used[base] = count + 1
            unique_headers.append(final)

        # Drop rows up to header and set columns
        df = df.iloc[header_row_idx + 1:].copy()
        df.columns = unique_headers
        return df
    except Exception:
        # In worst case, return original frame
        return df

def detect_timestamp_column(df: pd.DataFrame):
    """Detect the most likely timestamp column"""
    if df is None or df.empty:
        return None

    norm = {c: str(c).strip().lower().replace('_', ' ').replace('-', ' ') for c in df.columns}
    priority_keys = ['timestamp', 'date time', 'datetime', 'time stamp', 'alarm time', 'event time']
    secondary_tokens = ['time', 'date', 'occurred', 'generated']

    ordered_cols = []
    ordered_cols += [c for c, n in norm.items() if 'event' in n and 'time' in n]
    ordered_cols += [c for c, n in norm.items() if any(k in n for k in priority_keys) and c not in ordered_cols]
    ordered_cols += [c for c, n in norm.items() if any(t in n for t in secondary_tokens) and c not in ordered_cols]

    best_col = None
    best_ratio = 0.0
    for c in ordered_cols:
        try:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                return c
            try:
                time_obj_ratio = df[c].map(lambda v: isinstance(v, time)).mean()
            except Exception:
                time_obj_ratio = 0
            if time_obj_ratio >= 0.60:
                return c
            ser = pd.to_datetime(df[c], errors='coerce', infer_datetime_format=True)
            ratio = ser.notna().mean()
            if ratio >= 0.40 and ratio > best_ratio:
                best_col, best_ratio = c, ratio
        except Exception:
            continue

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

def read_uploaded_file(file_content: bytes, filename: str, sample_size: int = None) -> pd.DataFrame:
    """Read uploaded file content"""
    try:
        if filename.endswith('.csv'):
            # Try multiple encodings
            for encoding in ['utf-8', 'utf-8-sig', 'windows-1252', 'latin-1']:
                try:
                    content = file_content.decode(encoding)
                    # First try fast engine
                    try:
                        df = pd.read_csv(io.StringIO(content), nrows=sample_size)
                    except Exception:
                        # Fallback: robust parser with delimiter sniffing and bad-line skipping
                        df = pd.read_csv(
                            io.StringIO(content),
                            nrows=sample_size,
                            engine='python',
                            sep=None,
                            on_bad_lines='skip',
                        )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # Ultimate fallback
                content = file_content.decode('latin-1', errors='replace')
                try:
                    df = pd.read_csv(io.StringIO(content), nrows=sample_size)
                except Exception:
                    df = pd.read_csv(
                        io.StringIO(content),
                        nrows=sample_size,
                        engine='python',
                        sep=None,
                        on_bad_lines='skip',
                    )
        
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(file_content), nrows=sample_size)
        else:
            raise ValueError("Unsupported file format")
        
        # Fix common export pattern where the real header is inside the data
        df = _apply_detected_header(df)
        # Clean and normalize column names
        df.columns = [" ".join(str(c).split()) for c in df.columns]
        df = _normalize_columns_to_canonical(df)
        
        return df
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

def calculate_isa_metrics(df: pd.DataFrame, ts_col: str, tag_col: str = None, priority_col: str = None) -> Dict[str, Any]:
    """Calculate comprehensive ISA-18.2 metrics"""
    metrics = {}
    
    df_clean = df.copy()
    ts_series = df_clean[ts_col]
    parsed = pd.to_datetime(ts_series, errors='coerce', infer_datetime_format=True)
    
    # Handle time-only values if needed
    parsed_ratio = parsed.notna().mean()
    if parsed_ratio < 0.40:
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
    
    # Set index for time-based analysis
    try:
        df_indexed = df_clean.set_index(temp_ts_col)
    except Exception:
        df_indexed = df_clean.copy()
        df_indexed.index = df_clean[temp_ts_col]
    
    # Hourly analysis
    hourly_counts = df_indexed.resample('1H').size()
    metrics['hourly_data'] = {
        'timestamps': hourly_counts.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
        'counts': hourly_counts.values.tolist()
    }
    
    # Flood analysis
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
            'tag_data': {
                'tags': tag_counts.head(15).index.tolist(),
                'counts': tag_counts.head(15).values.tolist()
            }
        })
    
    # Priority distribution
    if priority_col and priority_col in df_clean.columns:
        prio_dist = df_clean[priority_col].astype(str).value_counts(normalize=True) * 100
        metrics['priority_data'] = {
            'priorities': prio_dist.index.tolist(),
            'percentages': prio_dist.values.tolist()
        }
    
    return metrics

# Global storage for uploaded files (in production, use proper database/storage)
uploaded_files_storage = {}

# Global storage for auto-discovered files
auto_discovered_files = {}
file_discovery_lock = Lock()

# Project root directory for auto-discovery
PROJECT_ROOT = Path(__file__).parent

# Supported file extensions for auto-discovery
SUPPORTED_EXTENSIONS = ['.csv', '.xlsx', '.xls']

def scan_project_folders():
    """Scan project folders for CSV and Excel files"""
    discovered_files = {}
    
    try:
        # Get all folders in project root
        for folder_path in PROJECT_ROOT.iterdir():
            if folder_path.is_dir() and not folder_path.name.startswith('.'):
                folder_name = folder_path.name
                
                # Skip common non-data folders
                if folder_name.lower() in ['__pycache__', '.git', 'node_modules', 'venv', 'env']:
                    continue
                
                # Scan for supported files in this folder
                folder_files = []
                for ext in SUPPORTED_EXTENSIONS:
                    pattern = folder_path / f"*{ext}"
                    files = glob.glob(str(pattern))
                    for file_path in files:
                        file_path_obj = Path(file_path)
                        try:
                            # Get file info
                            stat = file_path_obj.stat()
                            file_info = {
                                'file_path': str(file_path_obj),
                                'filename': file_path_obj.name,
                                'folder': folder_name,
                                'size_bytes': stat.st_size,
                                'size_mb': stat.st_size / (1024 * 1024),
                                'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                'extension': file_path_obj.suffix.lower(),
                                'file_id': f"auto_{hash(str(file_path_obj))}_{int(stat.st_mtime)}"
                            }
                            folder_files.append(file_info)
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")
                            continue
                
                if folder_files:
                    discovered_files[folder_name] = folder_files
    
    except Exception as e:
        print(f"Error scanning project folders: {e}")
    
    return discovered_files

def load_auto_discovered_file(file_info: Dict, sample_size: int = None) -> pd.DataFrame:
    """Load an auto-discovered file into a DataFrame"""
    file_path = Path(file_info['file_path'])
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        if file_info['extension'] == '.csv':
            # Try multiple encodings for CSV files
            for encoding in ['utf-8', 'utf-8-sig', 'windows-1252', 'latin-1']:
                try:
                    # First try fast C engine
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, nrows=sample_size)
                    except Exception:
                        # Fallback to robust Python engine with delimiter sniffing and bad-line skipping
                        df = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            nrows=sample_size,
                            engine='python',
                            sep=None,
                            on_bad_lines='skip',
                        )
                    break
                except UnicodeDecodeError:
                    continue
            else:
                # Ultimate fallback
                try:
                    df = pd.read_csv(file_path, encoding='latin-1', errors='replace', nrows=sample_size)
                except Exception:
                    df = pd.read_csv(
                        file_path,
                        encoding='latin-1',
                        errors='replace',
                        nrows=sample_size,
                        engine='python',
                        sep=None,
                        on_bad_lines='skip',
                    )
        
        elif file_info['extension'] in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, nrows=sample_size)
        else:
            raise ValueError(f"Unsupported file format: {file_info['extension']}")
        
        # Fix common export pattern where the real header is inside the data
        df = _apply_detected_header(df)
        # Clean and normalize column names
        df.columns = [" ".join(str(c).split()) for c in df.columns]
        df = _normalize_columns_to_canonical(df)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {str(e)}")

def refresh_auto_discovered_files():
    """Refresh the auto-discovered files cache"""
    global auto_discovered_files
    with file_discovery_lock:
        auto_discovered_files = scan_project_folders()
    return auto_discovered_files

# Initialize auto-discovery on startup
refresh_auto_discovered_files()

# API Endpoints

@app.get("/")
async def root():
    return {"message": "ISA-18.2 Alarm Rationalization API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "benchmarks": ISA_BENCHMARKS}

@app.post("/upload", response_model=FileInfo)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process alarm data file"""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only CSV and Excel files are supported")
    
    try:
        content = await file.read()
        df = read_uploaded_file(content, file.filename)
        
        # Store file data
        file_id = f"{file.filename}_{hash(content) % 100000}"
        uploaded_files_storage[file_id] = {
            'filename': file.filename,
            'dataframe': df,
            'content': content
        }
        
        file_info = FileInfo(
            filename=file.filename,
            size_mb=len(content) / (1024 * 1024),
            rows=len(df),
            columns=len(df.columns),
            encoding="auto-detected"
        )
        
        # Add file_id to response
        response_data = file_info.dict()
        response_data['file_id'] = file_id
        
        return response_data
    
    except HTTPException as he:
        # Preserve intended HTTP status codes (e.g., 400/404)
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/files")
async def list_files():
    """List all uploaded files"""
    return [
        {
            "file_id": file_id,
            "filename": data["filename"],
            "rows": len(data["dataframe"]),
            "columns": len(data["dataframe"].columns)
        }
        for file_id, data in uploaded_files_storage.items()
    ]

@app.get("/auto-files")
async def list_auto_discovered_files():
    """List all automatically discovered files from project folders"""
    return {
        "folders": auto_discovered_files,
        "total_folders": len(auto_discovered_files),
        "total_files": sum(len(files) for files in auto_discovered_files.values()),
        "last_scan": datetime.now().isoformat()
    }

@app.post("/auto-files/refresh")
async def refresh_discovered_files():
    """Manually refresh the auto-discovered files cache"""
    try:
        discovered = refresh_auto_discovered_files()
        return {
            "message": "Auto-discovered files refreshed successfully",
            "folders": discovered,
            "total_folders": len(discovered),
            "total_files": sum(len(files) for files in discovered.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error refreshing files: {str(e)}")

@app.get("/auto-files/{folder_name}")
async def list_folder_files(folder_name: str):
    """List files in a specific auto-discovered folder"""
    if folder_name not in auto_discovered_files:
        raise HTTPException(status_code=404, detail=f"Folder '{folder_name}' not found")
    
    return {
        "folder": folder_name,
        "files": auto_discovered_files[folder_name],
        "file_count": len(auto_discovered_files[folder_name])
    }

@app.get("/all-files")
async def list_all_available_files():
    """List all available files (both uploaded and auto-discovered)"""
    all_files = []
    
    # Add uploaded files
    for file_id, data in uploaded_files_storage.items():
        all_files.append({
            "file_id": file_id,
            "filename": data["filename"],
            "source_type": "uploaded",
            "rows": len(data["dataframe"]),
            "columns": len(data["dataframe"].columns),
            "folder": None
        })
    
    # Add auto-discovered files
    for folder_name, files in auto_discovered_files.items():
        for file_info in files:
            all_files.append({
                "file_id": file_info["file_id"],
                "filename": file_info["filename"],
                "source_type": "auto-discovered",
                "folder": folder_name,
                "size_mb": file_info["size_mb"],
                "modified_time": file_info["modified_time"],
                "extension": file_info["extension"]
            })
    
    return {
        "files": all_files,
        "total_files": len(all_files),
        "uploaded_count": len(uploaded_files_storage),
        "auto_discovered_count": sum(len(files) for files in auto_discovered_files.values())
    }

def get_file_data(file_id: str, sample_size: int = None):
    """Helper function to get file data from either uploaded or auto-discovered files"""
    # Check uploaded files first
    if file_id in uploaded_files_storage:
        df = uploaded_files_storage[file_id]['dataframe'].copy()
        filename = uploaded_files_storage[file_id]['filename']
        if sample_size:
            df = df.head(sample_size)
        return df, filename, 'uploaded'
    
    # Check auto-discovered files
    for folder_name, files in auto_discovered_files.items():
        for file_info in files:
            if file_info['file_id'] == file_id:
                try:
                    df = load_auto_discovered_file(file_info, sample_size)
                    return df, file_info['filename'], 'auto-discovered'
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error loading auto-discovered file: {str(e)}")
    
    raise HTTPException(status_code=404, detail="File not found in uploaded or auto-discovered files")

@app.post("/analyze/{file_id}")
async def analyze_file(file_id: str, request: AnalysisRequest = None):
    """Analyze a specific file and return ISA-18.2 metrics"""
    try:
        sample_size = request.sample_size if request else None
        df, filename, source_type = get_file_data(file_id, sample_size)
        
        # Auto-detect columns
        ts_col = detect_timestamp_column(df)
        tag_col = detect_tag_column(df)
        priority_col = detect_priority_column(df)
        
        if not ts_col:
            raise HTTPException(status_code=400, detail="No timestamp column detected")
        
        # Calculate metrics
        metrics = calculate_isa_metrics(df, ts_col, tag_col, priority_col)
        metrics['source_file'] = filename
        metrics['source_type'] = source_type
        metrics['detected_columns'] = {
            'timestamp': ts_col,
            'tag': tag_col,
            'priority': priority_col
        }
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/alarm-rate/{file_id}")
async def get_alarm_rate_chart(file_id: str, sample_size: int | None = Query(None, ge=1)):
    """Generate alarm rate performance chart"""
    try:
        df, filename, source_type = get_file_data(file_id, sample_size)
        
        ts_col = detect_timestamp_column(df)
        if not ts_col:
            raise HTTPException(status_code=400, detail="No timestamp column detected")
        
        metrics = calculate_isa_metrics(df, ts_col)
        
        if 'hourly_data' not in metrics:
            raise HTTPException(status_code=400, detail="No hourly data available")
        
        # Create alarm rate chart
        fig = go.Figure()
        
        timestamps = pd.to_datetime(metrics['hourly_data']['timestamps'])
        counts = metrics['hourly_data']['counts']
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=counts,
            mode='lines+markers',
            name='Alarms/Hour',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Add ISA benchmark lines
        fig.add_hline(y=6, line_dash="dash", line_color="#2E8B57", 
                     annotation_text="ISA Good (≤6)")
        fig.add_hline(y=12, line_dash="dash", line_color="#FF8C00", 
                     annotation_text="ISA Max (≤12)")
        
        fig.update_layout(
            title=f"Alarm Rate Over Time - {filename} ({source_type})",
            xaxis_title="Time",
            yaxis_title="Alarms per Hour",
            height=400,
            showlegend=True
        )
        
        # Convert to JSON
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return {"chart": chart_json, "metrics": metrics, "source_type": source_type}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/top-contributors/{file_id}")
async def get_top_contributors_chart(file_id: str, top_n: int = Query(15, ge=5, le=50), sample_size: int | None = Query(None, ge=1)):
    """Generate top contributors chart"""
    try:
        df, filename, source_type = get_file_data(file_id, sample_size)
        
        tag_col = detect_tag_column(df)
        if not tag_col:
            raise HTTPException(status_code=400, detail="No tag column detected")
        
        tag_counts = df[tag_col].value_counts().head(top_n)
        
        # Create horizontal bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=tag_counts.index[::-1],
            x=tag_counts.values[::-1],
            orientation='h',
            marker_color='#ff7f0e',
            text=tag_counts.values[::-1],
            textposition='outside'
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Alarm Contributors - {filename} ({source_type})",
            xaxis_title="Alarm Count",
            yaxis_title="Tag/Location",
            height=500,
            showlegend=False
        )
        
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        # Calculate contribution percentage
        total_alarms = len(df)
        top_10_count = tag_counts.head(10).sum()
        top_10_percentage = (top_10_count / total_alarms) * 100
        
        return {
            "chart": chart_json,
            "top_10_contribution": top_10_percentage,
            "total_alarms": total_alarms,
            "source_type": source_type,
            "data": {
                "tags": tag_counts.index.tolist(),
                "counts": tag_counts.values.tolist()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/priority-distribution/{file_id}")
async def get_priority_distribution_chart(file_id: str, sample_size: int | None = Query(None, ge=1)):
    """Generate priority distribution pie chart"""
    try:
        df, filename, source_type = get_file_data(file_id, sample_size)
        
        priority_col = detect_priority_column(df)
        if not priority_col:
            raise HTTPException(status_code=400, detail="No priority column detected")
        
        prio_dist = df[priority_col].astype(str).value_counts(normalize=True) * 100
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=prio_dist.index,
            values=prio_dist.values,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title=f"Priority Distribution - {filename} ({source_type})",
            height=400
        )
        
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return {
            "chart": chart_json,
            "source_type": source_type,
            "data": {
                "priorities": prio_dist.index.tolist(),
                "percentages": prio_dist.values.tolist()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/pareto-analysis/{file_id}")
async def get_pareto_chart(file_id: str, sample_size: int | None = Query(None, ge=1)):
    """Generate Pareto analysis chart for top contributors"""
    try:
        df, filename, source_type = get_file_data(file_id, sample_size)
        
        tag_col = detect_tag_column(df)
        if not tag_col:
            raise HTTPException(status_code=400, detail="No tag column detected")
        
        top_tags = df[tag_col].value_counts().head(15)
        cumulative_pct = (top_tags.cumsum() / top_tags.sum()) * 100
        
        # Create combination chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(x=list(range(len(top_tags))), y=top_tags.values,
                  name="Count", marker_color='lightcoral'),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=list(range(len(top_tags))), y=cumulative_pct.values,
                     mode='lines+markers', name="Cumulative %",
                     line=dict(color='darkgreen', width=3)),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Top Tags (Ranked)")
        fig.update_yaxes(title_text="Alarm Count", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative %", secondary_y=True)
        fig.update_layout(title=f"Pareto Analysis - {filename} ({source_type})")
        
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return {
            "chart": chart_json,
            "source_type": source_type,
            "data": {
                "tags": top_tags.index.tolist(),
                "counts": top_tags.values.tolist(),
                "cumulative_percentages": cumulative_pct.values.tolist()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/flood-analysis/{file_id}")
async def get_flood_analysis_chart(file_id: str, sample_size: int | None = Query(None, ge=1)):
    """Generate flood analysis chart"""
    try:
        df, filename, source_type = get_file_data(file_id, sample_size)
        
        ts_col = detect_timestamp_column(df)
        if not ts_col:
            raise HTTPException(status_code=400, detail="No timestamp column detected")
        
        # Parse timestamps
        parsed_ts = pd.to_datetime(df[ts_col], errors='coerce')
        df_clean = df[parsed_ts.notna()].copy()
        df_clean[ts_col] = parsed_ts[parsed_ts.notna()]
        df_indexed = df_clean.set_index(ts_col)
        
        # 10-minute window analysis
        ten_min_counts = df_indexed.resample('10T').size()
        flood_periods = ten_min_counts[ten_min_counts > ISA_BENCHMARKS['flood_threshold']]
        
        # Create flood timeline chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ten_min_counts.index,
            y=ten_min_counts.values,
            mode='lines',
            name='Alarms per 10-min',
            line=dict(color='blue', width=1)
        ))
        
        # Highlight flood periods
        if not flood_periods.empty:
            fig.add_trace(go.Scatter(
                x=flood_periods.index,
                y=flood_periods.values,
                mode='markers',
                name='Flood Periods',
                marker=dict(color='red', size=8, symbol='triangle-up')
            ))
        
        # Add flood threshold line
        fig.add_hline(y=ISA_BENCHMARKS['flood_threshold'], line_dash="dash", 
                     line_color="red", annotation_text=f"Flood Threshold ({ISA_BENCHMARKS['flood_threshold']})")
        
        fig.update_layout(
            title=f"Alarm Flood Analysis - {filename} ({source_type})",
            xaxis_title="Time",
            yaxis_title="Alarms per 10-minute Window",
            height=400
        )
        
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        flood_percentage = (len(flood_periods) / len(ten_min_counts)) * 100 if len(ten_min_counts) > 0 else 0
        
        return {
            "chart": chart_json,
            "source_type": source_type,
            "flood_percentage": flood_percentage,
            "flood_periods_count": len(flood_periods),
            "total_windows": len(ten_min_counts),
            "flood_performance": 'Good' if flood_percentage < 1.0 else 'Poor'
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/performance-comparison")
async def get_performance_comparison_chart():
    """Generate performance comparison chart across all files"""
    if not uploaded_files_storage:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        all_metrics = []
        
        for file_id, file_data in uploaded_files_storage.items():
            df = file_data['dataframe']
            filename = file_data['filename']
            
            ts_col = detect_timestamp_column(df)
            tag_col = detect_tag_column(df)
            priority_col = detect_priority_column(df)
            
            if ts_col:
                metrics = calculate_isa_metrics(df, ts_col, tag_col, priority_col)
                metrics['source_file'] = filename
                all_metrics.append(metrics)
        
        if not all_metrics:
            raise HTTPException(status_code=400, detail="No analyzable files found")
        
        # Create performance comparison chart
        file_names = [m['source_file'] for m in all_metrics]
        alarm_rates = [m.get('avg_alarms_per_hour', 0) for m in all_metrics]
        flood_rates = [m.get('flood_percentage', 0) for m in all_metrics]
        
        # Alarm rate comparison
        fig1 = go.Figure()
        
        colors = ['#2E8B57' if rate <= 6 else '#FF8C00' if rate <= 12 else '#DC143C' for rate in alarm_rates]
        
        fig1.add_trace(go.Bar(
            x=file_names,
            y=alarm_rates,
            name='Alarms/Hour',
            marker_color=colors,
            text=[f"{rate:.1f}" for rate in alarm_rates],
            textposition='outside'
        ))
        
        fig1.add_hline(y=6, line_dash="dash", line_color="#2E8B57", annotation_text="ISA Good (≤6)")
        fig1.add_hline(y=12, line_dash="dash", line_color="#FF8C00", annotation_text="ISA Max (≤12)")
        
        fig1.update_layout(
            title="Alarm Rate Performance Comparison",
            xaxis_title="Data Source",
            yaxis_title="Alarms per Hour",
            height=400,
            showlegend=False
        )
        
        chart1_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig1))
        
        # Flood rate comparison
        fig2 = go.Figure()
        
        flood_colors = ['#2E8B57' if rate < 1.0 else '#DC143C' for rate in flood_rates]
        
        fig2.add_trace(go.Bar(
            x=file_names,
            y=flood_rates,
            name='Flood %',
            marker_color=flood_colors,
            text=[f"{rate:.1f}%" for rate in flood_rates],
            textposition='outside'
        ))
        
        fig2.add_hline(y=1.0, line_dash="dash", line_color="#DC143C", annotation_text="ISA Target (<1%)")
        
        fig2.update_layout(
            title="Alarm Flood Performance Comparison",
            xaxis_title="Data Source",
            yaxis_title="Flood Percentage (%)",
            height=400,
            showlegend=False
        )
        
        chart2_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig2))
        
        return {
            "alarm_rate_chart": chart1_json,
            "flood_rate_chart": chart2_json,
            "summary_metrics": {
                "total_files": len(all_metrics),
                "avg_alarm_rate": np.mean(alarm_rates),
                "avg_flood_rate": np.mean(flood_rates),
                "good_performers": sum(1 for rate in alarm_rates if rate <= 6),
                "poor_performers": sum(1 for rate in alarm_rates if rate > 12)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/distribution-histogram/{file_id}")
async def get_distribution_histogram(file_id: str):
    """Generate alarm rate distribution histogram (supports uploaded and auto-discovered files)"""
    try:
        # Support both uploaded and auto-discovered files
        df, filename, _source_type = get_file_data(file_id)
        
        ts_col = detect_timestamp_column(df)
        if not ts_col:
            raise HTTPException(status_code=400, detail="No timestamp column detected")
        
        metrics = calculate_isa_metrics(df, ts_col)
        
        if 'hourly_data' not in metrics:
            raise HTTPException(status_code=400, detail="No hourly data available")
        
        counts = metrics['hourly_data']['counts']
        
        # Create histogram
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=counts,
            nbinsx=20,
            name='Distribution',
            marker_color='#1f77b4',
            opacity=0.7
        ))
        
        fig.add_vline(x=6, line_dash="dash", line_color="#2E8B57", annotation_text="Good")
        fig.add_vline(x=12, line_dash="dash", line_color="#FF8C00", annotation_text="Max")
        
        fig.update_layout(
            title=f"Alarm Rate Distribution - {filename}",
            xaxis_title="Alarms per Hour",
            yaxis_title="Frequency",
            height=400
        )
        
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return {
            "chart": chart_json,
            "statistics": {
                "mean": float(np.mean(counts)),
                "median": float(np.median(counts)),
                "std": float(np.std(counts)),
                "min": float(np.min(counts)),
                "max": float(np.max(counts))
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/charts/rolling-average/{file_id}")
async def get_rolling_average_chart(file_id: str, window_hours: int = Query(24, ge=1, le=168)):
    """Generate rolling average trend chart (supports uploaded and auto-discovered files)"""
    try:
        # Support both uploaded and auto-discovered files
        df, filename, _source_type = get_file_data(file_id)
        
        ts_col = detect_timestamp_column(df)
        if not ts_col:
            raise HTTPException(status_code=400, detail="No timestamp column detected")
        
        metrics = calculate_isa_metrics(df, ts_col)
        
        if 'hourly_data' not in metrics:
            raise HTTPException(status_code=400, detail="No hourly data available")
        
        timestamps = pd.to_datetime(metrics['hourly_data']['timestamps'])
        counts = np.array(metrics['hourly_data']['counts'])
        
        # Calculate rolling average
        rolling_avg = pd.Series(counts).rolling(window_hours, min_periods=1).mean()
        # Ensure native Python lists for JSON/Plotly compatibility
        ts_list = timestamps.tolist()
        counts_list = counts.tolist() if isinstance(counts, np.ndarray) else list(counts)
        rolling_list = rolling_avg.astype(float).tolist()
        
        fig = go.Figure()
        
        # Original data
        fig.add_trace(go.Scatter(
            x=ts_list,
            y=counts_list,
            mode='lines',
            name='Hourly Alarms',
            line=dict(color='lightblue', width=1),
            opacity=0.6
        ))
        
        # Rolling average
        fig.add_trace(go.Scatter(
            x=ts_list,
            y=rolling_list,
            mode='lines',
            name=f'{window_hours}h Rolling Average',
            line=dict(color='orange', width=3)
        ))
        
        fig.update_layout(
            title=f"{window_hours}-Hour Rolling Average Trend - {filename}",
            xaxis_title="Time",
            yaxis_title="Alarms per Hour",
            height=400
        )
        
        chart_json = json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig))
        
        return {
            "chart": chart_json,
            "rolling_average_data": {
                "timestamps": pd.to_datetime(ts_list).strftime('%Y-%m-%d %H:%M:%S').tolist(),
                "rolling_values": rolling_list,
                "window_hours": window_hours
            }
        }
    
    except HTTPException as he:
        # Preserve 4xx errors like insufficient data or missing columns
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/summary")
async def get_dashboard_summary():
    """Get overall dashboard summary across all files"""
    if not uploaded_files_storage:
        return {"message": "No files uploaded", "files": []}
    
    try:
        all_metrics = []
        file_summaries = []
        
        for file_id, file_data in uploaded_files_storage.items():
            df = file_data['dataframe']
            filename = file_data['filename']
            
            ts_col = detect_timestamp_column(df)
            tag_col = detect_tag_column(df)
            priority_col = detect_priority_column(df)
            
            file_summary = {
                "file_id": file_id,
                "filename": filename,
                "rows": len(df),
                "columns": len(df.columns),
                "detected_columns": {
                    "timestamp": ts_col,
                    "tag": tag_col,
                    "priority": priority_col
                }
            }
            
            if ts_col:
                try:
                    metrics = calculate_isa_metrics(df, ts_col, tag_col, priority_col)
                    metrics['source_file'] = filename
                    all_metrics.append(metrics)
                    
                    file_summary.update({
                        "avg_alarms_per_hour": metrics.get('avg_alarms_per_hour', 0),
                        "flood_percentage": metrics.get('flood_percentage', 0),
                        "isa_performance": metrics.get('isa_performance', 'Unknown'),
                        "analyzable": True
                    })
                except Exception:
                    file_summary["analyzable"] = False
            else:
                file_summary["analyzable"] = False
            
            file_summaries.append(file_summary)
        
        # Calculate overall KPIs
        if all_metrics:
            total_alarms = sum(m.get('total_alarms', 0) for m in all_metrics)
            avg_rate = np.mean([m.get('avg_alarms_per_hour', 0) for m in all_metrics])
            avg_flood = np.mean([m.get('flood_percentage', 0) for m in all_metrics])
            
            good_sources = sum(1 for m in all_metrics if m.get('avg_alarms_per_hour', 0) <= 6)
            poor_sources = sum(1 for m in all_metrics if m.get('avg_alarms_per_hour', 0) > 12)
            flood_sources = sum(1 for m in all_metrics if m.get('flood_percentage', 0) >= 1.0)
            
            overall_kpis = {
                "total_alarms": total_alarms,
                "avg_alarms_per_hour": avg_rate,
                "avg_flood_percentage": avg_flood,
                "good_sources": good_sources,
                "poor_sources": poor_sources,
                "flood_sources": flood_sources,
                "total_sources": len(all_metrics),
                "rate_status": "Good" if avg_rate <= 6 else "Acceptable" if avg_rate <= 12 else "Poor",
                "flood_status": "Good" if avg_flood < 1.0 else "Poor"
            }
        else:
            overall_kpis = {"message": "No analyzable files"}
        
        return {
            "overall_kpis": overall_kpis,
            "file_summaries": file_summaries,
            "isa_benchmarks": ISA_BENCHMARKS
        }
    
    except HTTPException as he:
        # Preserve intended HTTP status codes (e.g., 404 for missing file)
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch")
async def analyze_batch(request: AnalysisRequest = None):
    """Analyze all uploaded files and return comprehensive metrics"""
    if not uploaded_files_storage:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    try:
        results = []
        
        for file_id, file_data in uploaded_files_storage.items():
            df = file_data['dataframe'].copy()
            filename = file_data['filename']
            
            # Apply sample size if specified
            if request and request.sample_size:
                df = df.head(request.sample_size)
            
            ts_col = detect_timestamp_column(df)
            tag_col = detect_tag_column(df)
            priority_col = detect_priority_column(df)
            
            result = {
                "file_id": file_id,
                "filename": filename,
                "detected_columns": {
                    "timestamp": ts_col,
                    "tag": tag_col,
                    "priority": priority_col
                },
                "analyzable": bool(ts_col)
            }
            
            if ts_col:
                try:
                    metrics = calculate_isa_metrics(df, ts_col, tag_col, priority_col)
                    result["metrics"] = metrics
                except Exception as e:
                    result["error"] = str(e)
            
            results.append(result)
        
        return {"results": results, "total_files": len(results)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file/{file_id}/preview")
async def get_file_preview(file_id: str, rows: int = Query(10, ge=1, le=100)):
    """Get preview of file data (supports uploaded and auto-discovered files)"""
    try:
        df, filename, _source_type = get_file_data(file_id)
        
        preview_df = df.head(rows)
        
        # Convert to JSON-serializable format
        preview_data = []
        for _, row in preview_df.iterrows():
            row_dict = {}
            for col, val in row.items():
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (pd.Timestamp, datetime)):
                    row_dict[col] = val.isoformat()
                elif isinstance(val, (np.integer, np.floating)):
                    row_dict[col] = float(val)
                else:
                    row_dict[col] = str(val)
            preview_data.append(row_dict)
        
        return {
            "filename": filename,
            "columns": list(df.columns),
            "total_rows": len(df),
            "preview_rows": len(preview_data),
            "data": preview_data,
            "detected_columns": {
                "timestamp": detect_timestamp_column(df),
                "tag": detect_tag_column(df),
                "priority": detect_priority_column(df)
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/file/{file_id}/statistics")
async def get_file_statistics(file_id: str):
    """Get detailed statistics for a file (supports uploaded and auto-discovered files)"""
    try:
        df, filename, _source_type = get_file_data(file_id)
        
        # Basic statistics
        basic_stats = {
            "filename": filename,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        # Data completeness
        completeness = ((df.notna().sum() / len(df)) * 100).round(1)
        completeness_data = {
            "column": completeness.index.tolist(),
            "completeness_percentage": completeness.values.tolist()
        }
        
        # Missing values
        missing_data = df.isnull().sum()
        missing_pct = (missing_data / len(df)) * 100
        missing_info = []
        for col, count in missing_data.items():
            if count > 0:
                missing_info.append({
                    "column": col,
                    "missing_count": int(count),
                    "missing_percentage": float(missing_pct[col])
                })
        
        # Data types
        dtype_info = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            non_null_count = int(df[col].notna().sum())
            unique_count = int(df[col].nunique())
            
            dtype_info.append({
                "column": col,
                "data_type": dtype_str,
                "non_null_count": non_null_count,
                "unique_values": unique_count
            })
        
        # Numeric statistics for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_stats = {}
        if len(numeric_cols) > 0:
            desc = df[numeric_cols].describe()
            for col in numeric_cols:
                numeric_stats[col] = {
                    "mean": float(desc.loc['mean', col]),
                    "std": float(desc.loc['std', col]),
                    "min": float(desc.loc['min', col]),
                    "max": float(desc.loc['max', col]),
                    "q25": float(desc.loc['25%', col]),
                    "q50": float(desc.loc['50%', col]),
                    "q75": float(desc.loc['75%', col])
                }
        
        return {
            "basic_statistics": basic_stats,
            "data_completeness": completeness_data,
            "missing_values": missing_info,
            "column_info": dtype_info,
            "numeric_statistics": numeric_stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/file/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file"""
    if file_id not in uploaded_files_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    filename = uploaded_files_storage[file_id]['filename']
    del uploaded_files_storage[file_id]
    
    return {"message": f"File {filename} deleted successfully"}

@app.delete("/files/all")
async def delete_all_files():
    """Delete all uploaded files"""
    count = len(uploaded_files_storage)
    uploaded_files_storage.clear()
    return {"message": f"Deleted {count} files"}

@app.get("/benchmarks")
async def get_isa_benchmarks():
    """Get ISA-18.2 benchmark values"""
    return {"benchmarks": ISA_BENCHMARKS}

@app.get("/insights/{file_id}")
async def get_insights(file_id: str):
    """Generate actionable insights for a specific file (supports uploaded and auto-discovered)"""
    try:
        df, filename, _source_type = get_file_data(file_id)
        
        ts_col = detect_timestamp_column(df)
        tag_col = detect_tag_column(df)
        priority_col = detect_priority_column(df)
        
        if not ts_col:
            return {"insights": ["No timestamp column detected - limited analysis available"]}
        
        metrics = calculate_isa_metrics(df, ts_col, tag_col, priority_col)
        insights = []
        
        # Performance insights
        rate = metrics.get('avg_alarms_per_hour', 0)
        if rate <= 6:
            insights.append(f"✅ Alarm rate ({rate:.1f}/hr) meets ISA-18.2 'Good' performance target")
        elif rate <= 12:
            insights.append(f"⚠️ Alarm rate ({rate:.1f}/hr) is acceptable but should be reduced")
        else:
            insights.append(f"❌ Alarm rate ({rate:.1f}/hr) exceeds ISA-18.2 maximum - immediate action required")
        
        # Flood insights
        flood_pct = metrics.get('flood_percentage', 0)
        if flood_pct >= 1.0:
            insights.append(f"🌊 Alarm floods occur {flood_pct:.1f}% of time (target: <1%) - investigate burst patterns")
        else:
            insights.append(f"✅ Alarm flood rate ({flood_pct:.1f}%) within ISA target")
        
        # Top contributor insights
        top_10_pct = metrics.get('top_10_contribution', 0)
        if top_10_pct and top_10_pct > 5:
            insights.append(f"🔝 Top 10 contributors generate {top_10_pct:.1f}% of alarms - focus rationalization efforts here")
        elif top_10_pct:
            insights.append(f"✅ Top 10 contributors ({top_10_pct:.1f}%) within ISA target")
        
        # Priority insights
        if 'priority_data' in metrics:
            priorities = metrics['priority_data']['priorities']
            percentages = metrics['priority_data']['percentages']
            
            high_prio_pct = sum(pct for prio, pct in zip(priorities, percentages) 
                              if prio.upper() in ['P1', '1', 'HIGH'])
            if high_prio_pct > 10:
                insights.append(f"⚠️ High priority alarms ({high_prio_pct:.1f}%) exceed typical range - review criticality")
        
        # Data quality insights
        total_rows = len(df)
        if total_rows < 1000:
            insights.append("ℹ️ Limited data sample - consider longer time period for comprehensive analysis")
        
        return {
            "insights": insights,
            "metrics_summary": {
                "alarm_rate": rate,
                "flood_percentage": flood_pct,
                "top_10_contribution": top_10_pct,
                "isa_performance": metrics.get('isa_performance', 'Unknown')
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/metrics/{file_id}")
async def export_metrics(file_id: str, format: str = Query("json", regex="^(json|csv)$")):
    """Export detailed metrics for a file (supports uploaded and auto-discovered files)"""
    try:
        # Support both uploaded and auto-discovered files
        df, filename, _source_type = get_file_data(file_id)
        
        ts_col = detect_timestamp_column(df)
        tag_col = detect_tag_column(df)
        priority_col = detect_priority_column(df)
        
        if not ts_col:
            raise HTTPException(status_code=400, detail="No timestamp column detected")
        
        metrics = calculate_isa_metrics(df, ts_col, tag_col, priority_col)
        
        if format == "json":
            return metrics
        elif format == "csv":
            # Create CSV export of key metrics
            export_data = []
            
            # Basic metrics
            export_data.append(["Metric", "Value", "ISA Target", "Status"])
            export_data.append(["Total Alarms", metrics.get('total_alarms', 0), "N/A", "Info"])
            export_data.append(["Avg Alarms/Hour", f"{metrics.get('avg_alarms_per_hour', 0):.2f}", "≤6 (Good), ≤12 (Max)", metrics.get('isa_performance', 'Unknown')])
            export_data.append(["Flood Percentage", f"{metrics.get('flood_percentage', 0):.2f}%", "<1%", metrics.get('flood_performance', 'Unknown')])
            
            if metrics.get('top_10_contribution'):
                export_data.append(["Top 10 Contribution", f"{metrics.get('top_10_contribution', 0):.2f}%", "<5%", metrics.get('top_10_performance', 'Unknown')])
            
            # Convert to CSV string
            csv_content = "\n".join([",".join(map(str, row)) for row in export_data])
            
            return JSONResponse(
                content={"csv_data": csv_content, "filename": f"{filename}_metrics.csv"},
                media_type="application/json"
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Analysis Endpoints
@app.get("/advanced/chattering/{file_id}")
async def detect_chattering_alarms(file_id: str, window_minutes: int = Query(10, ge=1, le=60), threshold: int = Query(5, ge=2, le=20)):
    """Detect chattering alarms (repeated alarms in short windows) - supports uploaded and auto-discovered"""
    try:
        df, _filename, _source_type = get_file_data(file_id)
        ts_col = detect_timestamp_column(df)
        tag_col = detect_tag_column(df)
        
        if not ts_col or not tag_col:
            raise HTTPException(status_code=400, detail="Missing required columns for chattering analysis")
        
        # Parse timestamps
        parsed_ts = pd.to_datetime(df[ts_col], errors='coerce')
        df_clean = df[parsed_ts.notna()].copy()
        df_clean[ts_col] = parsed_ts[parsed_ts.notna()]
        df_clean = df_clean.sort_values(ts_col)
        
        # Detect chattering
        chattering_tags = []
        for tag, group in df_clean.groupby(tag_col):
            timestamps = group[ts_col]
            # Count alarms in rolling windows
            max_in_window = 0
            for i, ts in enumerate(timestamps):
                window_start = ts - pd.Timedelta(minutes=window_minutes)
                count_in_window = ((timestamps >= window_start) & (timestamps <= ts)).sum()
                max_in_window = max(max_in_window, count_in_window)
            
            if max_in_window >= threshold:
                chattering_tags.append({
                    "tag": tag,
                    "max_repeats_in_window": int(max_in_window),
                    "total_occurrences": len(group)
                })
        
        # Sort by severity
        chattering_tags.sort(key=lambda x: x['max_repeats_in_window'], reverse=True)
        
        return {
            "chattering_alarms": chattering_tags[:20],  # Top 20
            "analysis_parameters": {
                "window_minutes": window_minutes,
                "threshold": threshold
            },
            "summary": {
                "total_chattering_tags": len(chattering_tags),
                "worst_offender": chattering_tags[0] if chattering_tags else None
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/advanced/standing-alarms/{file_id}")
async def detect_standing_alarms(file_id: str, min_duration_hours: int = Query(24, ge=1, le=168)):
    """Detect standing alarms (long-duration active alarms) - supports uploaded and auto-discovered"""
    try:
        df, _filename, _source_type = get_file_data(file_id)
        ts_col = detect_timestamp_column(df)
        tag_col = detect_tag_column(df)
        
        if not ts_col or not tag_col:
            raise HTTPException(status_code=400, detail="Missing required columns for standing alarm analysis")
        
        # Parse timestamps
        parsed_ts = pd.to_datetime(df[ts_col], errors='coerce')
        df_clean = df[parsed_ts.notna()].copy()
        df_clean[ts_col] = parsed_ts[parsed_ts.notna()]
        
        # Calculate spans per tag
        spans = df_clean.groupby(tag_col)[ts_col].agg(['min', 'max', 'count'])
        spans['span_hours'] = (spans['max'] - spans['min']).dt.total_seconds() / 3600.0
        spans['avg_interval_hours'] = spans['span_hours'] / spans['count']
        
        # Filter for standing alarms
        standing = spans[spans['span_hours'] >= min_duration_hours]
        standing = standing.sort_values('span_hours', ascending=False)
        
        standing_alarms = []
        for tag, row in standing.head(50).iterrows():
            standing_alarms.append({
                "tag": tag,
                "span_hours": float(row['span_hours']),
                "first_occurrence": row['min'].isoformat(),
                "last_occurrence": row['max'].isoformat(),
                "total_count": int(row['count']),
                "avg_interval_hours": float(row['avg_interval_hours'])
            })
        
        return {
            "standing_alarms": standing_alarms,
            "analysis_parameters": {
                "min_duration_hours": min_duration_hours
            },
            "summary": {
                "total_standing_alarms": len(standing_alarms),
                "longest_duration_hours": float(standing['span_hours'].max()) if not standing.empty else 0
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)