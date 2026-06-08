import sys
import io
import json
import traceback
import os
import base64
import warnings
from langchain_core.tools import tool

warnings.filterwarnings("ignore")

# ─── HELPERS injected into every code execution context ───────────────────────
HELPERS = '''
import pandas as pd
import numpy as np
import os, io, base64, warnings, json, re
from scipy import stats as scipy_stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.close("all")


# ═══════════════════════════════════════════════════════════════════
#  STRUCTURE DETECTION
# ═══════════════════════════════════════════════════════════════════

def _detect_structure(raw_df, max_rows=10):
    """Detect title, header, sub-header, and data rows in a raw DataFrame."""
    n = min(max_rows, len(raw_df))
    n_cols = max(len(raw_df.columns), 1)

    PERIOD_KWS = {"during","cumulative","total","month","quarter","fy","ytd","annual","weekly","daily","period"}

    rows_meta = []
    for i in range(n):
        row = raw_df.iloc[i]
        vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() not in ("","nan")]
        numeric   = sum(1 for v in vals if v.replace(".","",1).replace("-","",1).replace(",","").isdigit())
        text      = len(vals) - numeric
        text_join = " ".join(vals).lower()
        has_period = any(kw in text_join for kw in PERIOD_KWS)
        rows_meta.append(dict(
            i=i, filled=len(vals), ratio=len(vals)/n_cols,
            text=text, numeric=numeric, has_period=has_period,
        ))

    # Title rows: very sparse at start (report name, subtitle)
    title_rows = []
    for r in rows_meta:
        if r["ratio"] < 0.25 and r["numeric"] == 0:
            title_rows.append(r["i"])
        else:
            break

    # Main header row: first dense text-heavy row after title rows
    candidates = [r for r in rows_meta if r["i"] not in title_rows]
    main_hdr = candidates[0]["i"] if candidates else 0
    for r in candidates[:5]:
        if r["text"] >= rows_meta[main_hdr]["text"]:
            main_hdr = r["i"]
            break

    # Sub-header rows: immediately after main_hdr with period keywords OR sparse text
    sub_hdrs = []
    for r in rows_meta:
        if r["i"] <= main_hdr:
            continue
        if r["has_period"] or (0 < r["ratio"] < 0.75 and r["numeric"] == 0 and r["text"] > 0):
            sub_hdrs.append(r["i"])
        else:
            break

    data_start = max([main_hdr] + sub_hdrs) + 1
    return {
        "title_rows":   title_rows,
        "main_header":  main_hdr,
        "sub_headers":  sub_hdrs,
        "header_rows":  [main_hdr] + sub_hdrs,
        "data_start":   data_start,
    }


def _ffill_row(vals):
    """Forward-fill a list, treating empty/nan/unnamed cells as gaps (merged-cell simulation)."""
    result = list(vals)
    last = None
    for i, v in enumerate(result):
        s = str(v).strip() if v is not None else ""
        if s in ("", "nan") or s.lower().startswith("unnamed"):
            result[i] = last
        else:
            last = v
    return result


def _build_col_names(raw_df, header_rows):
    """Build clean column names from one or more header rows with forward-fill."""
    n_cols = len(raw_df.columns)
    hdr_matrix = []
    for r in header_rows:
        hdr_matrix.append(_ffill_row(list(raw_df.iloc[r])))

    col_names = []
    for j in range(n_cols):
        parts = []
        for row_vals in hdr_matrix:
            v = str(row_vals[j]).strip() if j < len(row_vals) and row_vals[j] is not None else ""
            if v and v.lower() not in ("nan", ""):
                parts.append(v)
        # Deduplicate consecutive identical parts
        deduped = []
        for p in parts:
            if not deduped or p != deduped[-1]:
                deduped.append(p)
        col_names.append(" - ".join(deduped) if deduped else f"col_{j}")

    return col_names


# ═══════════════════════════════════════════════════════════════════
#  MAIN LOADERS
# ═══════════════════════════════════════════════════════════════════

def load_complex_file(path, nrows=None, sheet=0):
    """
    Smart loader for CSV/Excel files with complex/multi-row headers.

    Automatically handles:
      - Title rows (sheet name / report name at the top)
      - Multi-row merged column headers
      - Sub-header rows (period qualifiers: "During the month", "Cumulative in FY")
      - Forward-fills gaps from merged cells
      - Converts numeric columns properly

    Returns a clean DataFrame with meaningful column names like:
      "PLHIVs Registrations - During the month"
      "PLHIVs Registrations - Cumulative in FY"

    Usage:
      df = load_complex_file("/tmp/.../data.csv", nrows=2000)
      df = load_complex_file("/tmp/.../report.xlsx", sheet="Sheet1", nrows=1000)
    """
    ext = os.path.splitext(path)[1].lower()
    load_n = max((nrows or 0) + 20, 20)

    # ── Load raw (no header interpretation) ─────────────────────────────
    raw = None
    if ext == ".csv":
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                raw = pd.read_csv(path, header=None, nrows=load_n, encoding=enc, on_bad_lines="skip")
                break
            except Exception:
                pass
    elif ext in (".xlsx", ".xls", ".xlsm"):
        try:
            raw = pd.read_excel(path, header=None, nrows=load_n, sheet_name=sheet, engine="openpyxl")
        except Exception as e:
            print(f"[load_complex_file] Excel load error: {e}")
    
    if raw is None or len(raw) == 0:
        print(f"[load_complex_file] Could not read file: {path}")
        return pd.DataFrame()

    # ── Detect structure ─────────────────────────────────────────────────
    struct      = _detect_structure(raw)
    header_rows = struct["header_rows"]
    data_start  = struct["data_start"]

    print(f"[load_complex_file] title_rows={struct['title_rows']} main_header={struct['main_header']} "
          f"sub_headers={struct['sub_headers']} data_start={data_start}")

    # ── Build column names ────────────────────────────────────────────────
    col_names = _build_col_names(raw, header_rows)

    # ── Extract data ──────────────────────────────────────────────────────
    df = raw.iloc[data_start:].copy()
    if nrows:
        df = df.head(nrows)
    df.columns = col_names[:len(df.columns)]
    df = df.dropna(how="all").reset_index(drop=True)

    # Remove rows where the first column is empty (header repeats, totals)
    if len(df.columns) > 0:
        df = df[df.iloc[:, 0].notna()].reset_index(drop=True)

    # ── Convert numeric columns ───────────────────────────────────────────
    for col in df.columns:
        s = df[col].astype(str).str.replace(",", "", regex=False)
        converted = pd.to_numeric(s, errors="coerce")
        if converted.notna().sum() > len(df) * 0.4:
            df[col] = converted

    print(f"[load_complex_file] => {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"[load_complex_file] Columns: {list(df.columns)}")
    return df


def inspect_raw(path, nrows=8, sheet=0):
    """
    Show raw file structure WITHOUT any header interpretation.
    Always call this FIRST to understand the file layout.

    Usage:
      inspect_raw("/tmp/.../data.csv")
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        raw = pd.read_csv(path, header=None, nrows=nrows, encoding="latin-1", on_bad_lines="skip")
    else:
        raw = pd.read_excel(path, header=None, nrows=nrows, sheet_name=sheet, engine="openpyxl")

    struct = _detect_structure(raw)
    print("═" * 60)
    print(f"FILE: {os.path.basename(path)}")
    print(f"Detected → title_rows={struct['title_rows']}, "
          f"header_rows={struct['header_rows']}, data_from_row={struct['data_start']}")
    print("─" * 60)
    print("RAW ROWS (no header interpretation):")
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    pd.set_option("display.max_colwidth", 40)
    print(raw.to_string())
    print("═" * 60)
    return raw


def load_file(path, nrows=None, sheet=0):
    """Simple fallback loader (only for truly simple flat files)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(path, nrows=nrows, encoding=enc, on_bad_lines="skip")
            except Exception:
                pass
    elif ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(path, nrows=nrows, sheet_name=sheet, engine="openpyxl")
    return pd.DataFrame()


def quick_profile(df):
    """Print shape, dtypes, missing %, numeric summary, and sample rows."""
    print(f"Shape: {df.shape}")
    num = df.select_dtypes(include="number")
    cat = df.select_dtypes(exclude="number")
    print(f"Numeric  ({len(num.columns)}): {list(num.columns)}")
    print(f"Categorical ({len(cat.columns)}): {list(cat.columns)}")
    missing = (df.isnull().mean() * 100).round(1)
    missing = missing[missing > 0]
    if len(missing):
        print(f"Missing %:")
        print(missing.to_string())
    if not num.empty:
        print(f"\\nNumeric stats:")
        print(num.describe().round(2).to_string())
    print(f"\\nSample rows:")
    print(df.head(5).to_string())


def top_n(df, group_col, value_col, n=10, agg="sum", pct=True):
    """Show top N groups by value. Prints a formatted table."""
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    result = df.groupby(group_col)[value_col].agg(agg).sort_values(ascending=False).head(n)
    total  = result.sum()
    print(f"Top {n} by {value_col} ({agg}):")
    print(f"{'Rank':<5} {group_col:<35} {value_col:>15}" + (f"  {'%':>7}" if pct and total>0 else ""))
    print("─" * 65)
    for rank, (name, val) in enumerate(result.items(), 1):
        pct_str = f"  {val/total*100:>6.1f}%" if pct and total > 0 else ""
        print(f"{rank:<5} {str(name):<35} {val:>15,.0f}{pct_str}")
    print(f"{'TOTAL':<5} {'':<35} {total:>15,.0f}")
    return result


def find_col(df, *keywords):
    """Find first column matching any keyword (case-insensitive). Returns column name or None."""
    for kw in keywords:
        matches = [c for c in df.columns if kw.lower() in str(c).lower()]
        if matches: return matches[0]
    return None


def save_chart(fig=None, title="Chart"):
    """Save a matplotlib figure and emit a base64 marker for the chat UI to capture."""
    if fig is None:
        fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    print(f"[CHART_B64]{b64}[/CHART_B64]")
    plt.close("all")
    return b64


def multi_file_summary(paths):
    """Quick summary of multiple files: columns, rows, potential join keys."""
    summaries = {}
    for p in paths:
        df = load_complex_file(p, nrows=5)
        summaries[os.path.basename(p)] = {
            "columns": list(df.columns),
            "shape_sample": df.shape,
        }
    # Find potential join columns (common column names)
    all_col_sets = [set(v["columns"]) for v in summaries.values()]
    if len(all_col_sets) > 1:
        common = set.intersection(*all_col_sets)
        print(f"Potential join columns (shared across files): {sorted(common)}")
    for fname, info in summaries.items():
        print(f"\\n{fname}: {info['shape_sample'][1]} cols → {info['columns'][:8]}...")
    return summaries
'''

BLOCKED = ["subprocess", "socket", "requests", "urllib", "ftplib", "smtplib", "paramiko"]


def _check_safety(code: str) -> str | None:
    for b in BLOCKED:
        if f"import {b}" in code or f"from {b}" in code:
            return f"Blocked: '{b}' not allowed in analysis code."
    return None


@tool
def python_code_executor_tool(code: str) -> str:
    """Execute Python code for data analysis. Always use print() for output.

    Pre-loaded helpers (no import needed):
    - inspect_raw(path) — show raw file layout, ALWAYS call first for any uploaded file
    - load_complex_file(path, nrows=N) — smart multi-row header loader, use for all CSV/Excel
    - load_file(path, nrows=N) — simple fallback
    - quick_profile(df) — shape, types, missing, numeric stats, sample
    - top_n(df, group_col, value_col, n=10) — ranked table with % share
    - find_col(df, "keyword") — find column by keyword
    - save_chart(fig) — capture chart to chat UI
    - pd, np, plt, sns, scipy_stats — pre-imported

    Args:
        code: Python code string. Use print() for all output.
    """
    if not code or not code.strip():
        return "ERROR: No code provided."

    err = _check_safety(code)
    if err:
        return f"SECURITY ERROR: {err}"

    buf = io.StringIO()
    old_out = sys.stdout
    sys.stdout = buf
    charts_b64 = []

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        g = {"__builtins__": __builtins__}
        exec(compile(HELPERS, "<helpers>", "exec"), g)
        exec(compile(code, "<analyst>", "exec"), g)

        sys.stdout = old_out
        output = buf.getvalue()

        import re as _re
        found = _re.findall(r"\[CHART_B64\](.*?)\[/CHART_B64\]", output, _re.DOTALL)
        charts_b64.extend(found)
        clean = _re.sub(r"\[CHART_B64\].*?\[/CHART_B64\]", "[Chart captured]", output, flags=_re.DOTALL).strip()

        for fn in plt.get_fignums():
            fig = plt.figure(fn)
            ib = io.BytesIO()
            fig.savefig(ib, format="png", dpi=150, bbox_inches="tight", facecolor="white")
            ib.seek(0)
            charts_b64.append(base64.b64encode(ib.read()).decode())
        plt.close("all")

        result = clean or "Code executed (no print output)."
        if charts_b64:
            return json.dumps({"output": result, "charts_count": len(charts_b64), "charts_b64": charts_b64[:4]})
        return result

    except Exception:
        sys.stdout = old_out
        return f"EXECUTION ERROR:\n{traceback.format_exc()}"