import os
import json
import warnings
import pandas as pd
from langchain_core.tools import tool

warnings.filterwarnings("ignore")

PERIOD_KWS = {"during", "cumulative", "total", "month", "quarter", "fy", "ytd", "annual", "period"}


def _detect_structure(raw_df, max_rows=10):
    n = min(max_rows, len(raw_df))
    n_cols = max(len(raw_df.columns), 1)
    rows = []
    for i in range(n):
        row = raw_df.iloc[i]
        vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() not in ("", "nan")]
        numeric = sum(1 for v in vals if v.replace(".", "", 1).replace("-", "", 1).replace(",", "").isdigit())
        text = len(vals) - numeric
        text_join = " ".join(vals).lower()
        has_period = any(kw in text_join for kw in PERIOD_KWS)
        rows.append(dict(i=i, filled=len(vals), ratio=len(vals)/n_cols, text=text, numeric=numeric, has_period=has_period))

    title_rows = []
    for r in rows:
        if r["ratio"] < 0.25 and r["numeric"] == 0:
            title_rows.append(r["i"])
        else:
            break

    candidates = [r for r in rows if r["i"] not in title_rows]
    main_hdr = 0
    if candidates:
        main_hdr = max(candidates[:5], key=lambda r: r["text"])["i"]

    sub_hdrs = []
    for r in rows:
        if r["i"] <= main_hdr:
            continue
        if r["has_period"] or (0 < r["ratio"] < 0.75 and r["numeric"] == 0 and r["text"] > 0):
            sub_hdrs.append(r["i"])
        else:
            break

    data_start = max([main_hdr] + sub_hdrs) + 1
    return {"title_rows": title_rows, "main_header": main_hdr, "sub_headers": sub_hdrs,
            "header_rows": [main_hdr] + sub_hdrs, "data_start": data_start}


def _ffill_row(vals):
    result = list(vals)
    last = None
    for i, v in enumerate(result):
        s = str(v).strip() if v is not None else ""
        if s in ("", "nan") or s.lower().startswith("unnamed"):
            result[i] = last
        else:
            last = v
    return result


def _smart_load(file_path: str, nrows: int, sheet_name=0) -> pd.DataFrame | None:
    ext = os.path.splitext(file_path)[1].lower()
    load_n = max(nrows + 20, 20)

    raw = None
    if ext == ".csv":
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                raw = pd.read_csv(file_path, header=None, nrows=load_n, encoding=enc, on_bad_lines="skip")
                break
            except Exception:
                pass
    elif ext in (".xlsx", ".xls", ".xlsm"):
        try:
            raw = pd.read_excel(file_path, header=None, nrows=load_n, sheet_name=sheet_name, engine="openpyxl")
        except Exception:
            pass

    if raw is None or len(raw) == 0:
        return None

    struct = _detect_structure(raw)
    header_rows = struct["header_rows"]
    data_start = struct["data_start"]

    # Build column names with forward-fill
    n_cols = len(raw.columns)
    hdr_matrix = [_ffill_row(list(raw.iloc[r])) for r in header_rows]
    col_names = []
    for j in range(n_cols):
        parts = []
        for row_vals in hdr_matrix:
            v = str(row_vals[j]).strip() if j < len(row_vals) and row_vals[j] is not None else ""
            if v and v.lower() not in ("nan", ""):
                parts.append(v)
        deduped = []
        for p in parts:
            if not deduped or p != deduped[-1]:
                deduped.append(p)
        col_names.append(" - ".join(deduped) if deduped else f"col_{j}")

    df = raw.iloc[data_start:].copy()
    if nrows:
        df = df.head(nrows)
    df.columns = col_names[:len(df.columns)]
    df = df.dropna(how="all").reset_index(drop=True)
    if len(df.columns) > 0:
        df = df[df.iloc[:, 0].notna()].reset_index(drop=True)

    for col in df.columns:
        s = df[col].astype(str).str.replace(",", "", regex=False)
        conv = pd.to_numeric(s, errors="coerce")
        if conv.notna().sum() > len(df) * 0.4:
            df[col] = conv

    return df, struct


@tool
def data_profiling_tool(file_path: str, sheet_name: str = "0", num_rows: int = 200) -> str:
    """Profile a file: real column names, dtypes, stats, missing %, sample rows.
    Auto-detects and handles multi-row headers — returns actual column names, never Unnamed:.

    Args:
        file_path: absolute path to CSV or Excel file
        sheet_name: sheet name or index for Excel (default "0")
        num_rows: rows to sample (default 200)
    """
    if not file_path or not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})
    sheet_name_parsed = int(sheet_name) if str(sheet_name).isdigit() else sheet_name

    try:
        result = _smart_load(file_path, num_rows, sheet_name_parsed)
        if result is None:
            return json.dumps({"error": "Could not load file"})
        df, struct = result

        profile = {
            "file": os.path.basename(file_path),
            "rows_sampled": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "header_structure": {
                "title_rows": struct["title_rows"],
                "main_header_row": struct["main_header"],
                "sub_header_rows": struct["sub_headers"],
                "data_starts_at_row": struct["data_start"],
            },
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing_pct": (df.isnull().mean() * 100).round(1).to_dict(),
            "unique_counts": {c: int(df[c].nunique()) for c in df.columns},
            "sample_data": df.head(5).fillna("NULL").to_dict(orient="records"),
        }

        # Numeric stats
        nc = df.select_dtypes(include="number").columns.tolist()
        if nc:
            profile["numeric_columns"] = nc
            ns = {}
            for col in nc:
                s = df[col].dropna()
                if len(s) == 0:
                    continue
                q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
                ns[col] = {
                    "sum": round(float(s.sum()), 2),
                    "mean": round(float(s.mean()), 2),
                    "median": round(float(s.median()), 2),
                    "std": round(float(s.std()), 2),
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                    "q25": round(q1, 2),
                    "q75": round(q3, 2),
                    "outlier_count": int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum()),
                }
            profile["numeric_stats"] = ns

        # Categorical stats
        cc = df.select_dtypes(exclude="number").columns.tolist()
        if cc:
            profile["categorical_columns"] = cc
            cs = {}
            for col in cc[:15]:
                vc = df[col].astype(str).value_counts()
                cs[col] = {"unique_count": int(df[col].nunique()), "top_10": dict(list(vc.head(10).items()))}
            profile["categorical_stats"] = cs

        return json.dumps(profile, indent=2, default=str)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})