import os
import json
import warnings
import pandas as pd
from langchain_core.tools import tool

warnings.filterwarnings("ignore")

PERIOD_KWS = {"during", "cumulative", "total", "month", "quarter", "fy", "ytd", "annual", "period"}


def _quick_structure(file_path: str, ext: str, sheet_name=0):
    """Quickly detect structure and return real column names."""
    try:
        if ext == ".csv":
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    raw = pd.read_csv(file_path, header=None, nrows=12, encoding=enc, on_bad_lines="skip")
                    break
                except Exception:
                    pass
            else:
                return None, None
        else:
            raw = pd.read_excel(file_path, header=None, nrows=12, sheet_name=sheet_name, engine="openpyxl")

        n_cols = max(len(raw.columns), 1)
        rows_meta = []
        for i in range(min(10, len(raw))):
            row = raw.iloc[i]
            vals = [str(v).strip() for v in row if pd.notna(v) and str(v).strip() not in ("", "nan")]
            numeric = sum(1 for v in vals if v.replace(".", "", 1).replace("-", "", 1).replace(",", "").isdigit())
            text = len(vals) - numeric
            has_period = any(kw in " ".join(vals).lower() for kw in PERIOD_KWS)
            rows_meta.append(dict(i=i, filled=len(vals), ratio=len(vals) / n_cols,
                                   text=text, numeric=numeric, has_period=has_period))

        title_rows = []
        for r in rows_meta:
            if r["ratio"] < 0.25 and r["numeric"] == 0:
                title_rows.append(r["i"])
            else:
                break

        candidates = [r for r in rows_meta if r["i"] not in title_rows]
        main_hdr = candidates[0]["i"] if candidates else 0
        for r in candidates[:5]:
            if r["text"] >= rows_meta[main_hdr]["text"]:
                main_hdr = r["i"]
                break

        sub_hdrs = []
        for r in rows_meta:
            if r["i"] <= main_hdr:
                continue
            if r["has_period"] or (0 < r["ratio"] < 0.75 and r["numeric"] == 0 and r["text"] > 0):
                sub_hdrs.append(r["i"])
            else:
                break

        data_start = max([main_hdr] + sub_hdrs) + 1

        # Forward-fill header rows to get real column names
        hdr_rows = [main_hdr] + sub_hdrs
        hdr_matrix = []
        for r in hdr_rows:
            row_vals = list(raw.iloc[r])
            filled = []
            last = None
            for v in row_vals:
                s = str(v).strip() if v is not None else ""
                if s in ("", "nan") or s.lower().startswith("unnamed"):
                    filled.append(last)
                else:
                    last = v
                    filled.append(v)
            hdr_matrix.append(filled)

        col_names = []
        for j in range(n_cols):
            parts = []
            for row_vals in hdr_matrix:
                if j < len(row_vals) and row_vals[j] is not None:
                    v = str(row_vals[j]).strip()
                    if v and v.lower() not in ("nan", ""):
                        parts.append(v)
            deduped = []
            for p in parts:
                if not deduped or p != deduped[-1]:
                    deduped.append(p)
            col_names.append(" - ".join(deduped) if deduped else f"col_{j}")

        structure = {
            "title_rows": title_rows,
            "main_header_row": main_hdr,
            "sub_header_rows": sub_hdrs,
            "data_starts_at_row": data_start,
            "note": "load_complex_file() handles this automatically"
        }

        # Estimate row count
        try:
            if ext == ".csv":
                with open(file_path, "r", encoding="latin-1") as f:
                    row_count = sum(1 for _ in f) - data_start
            else:
                row_count = "unknown (Excel)"
        except Exception:
            row_count = "unknown"

        return col_names, structure, row_count

    except Exception:
        return None, None, "unknown"


@tool
def file_discovery_tool(file_paths: str) -> str:
    """Inspect uploaded files: detect structure, return REAL column names (not Unnamed:).
    Handles multi-row headers, merged cells, title rows.
    ALWAYS call this first before any analysis.

    Args:
        file_paths: Newline-separated file paths from [UPLOADED FILES] section.

    Returns: JSON with real column names, detected structure, row count, sheet names.
    """
    paths = [p.strip() for p in file_paths.replace(",", "\n").split("\n") if p.strip()]
    if not paths:
        return json.dumps({"error": "No file paths provided"})

    results = []
    for path in paths:
        info = {"path": path, "name": os.path.basename(path), "exists": os.path.exists(path)}

        if not os.path.exists(path):
            info["error"] = "File not found"
            results.append(info)
            continue

        info["size_mb"] = round(os.path.getsize(path) / 1048576, 3)
        ext = os.path.splitext(path)[1].lower()
        info["extension"] = ext

        try:
            if ext in (".xlsx", ".xls", ".xlsm"):
                info["type"] = "Excel"
                import openpyxl
                wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
                info["sheets"] = wb.sheetnames
                wb.close()
                for sheet in info["sheets"][:3]:
                    col_names, structure, row_count = _quick_structure(path, ext, sheet)
                    info.setdefault("sheets_info", {})[sheet] = {
                        "columns": col_names[:20] if col_names else [],
                        "column_count": len(col_names) if col_names else 0,
                        "estimated_rows": row_count,
                        "structure": structure,
                    }
                # Use first sheet as default
                first_sheet = info["sheets"][0]
                si = info["sheets_info"].get(first_sheet, {})
                info["columns"] = si.get("columns", [])
                info["column_count"] = si.get("column_count", 0)
                info["structure"] = si.get("structure", {})

            elif ext == ".csv":
                info["type"] = "CSV"
                col_names, structure, row_count = _quick_structure(path, ext)
                info["columns"] = col_names[:30] if col_names else []
                info["column_count"] = len(col_names) if col_names else 0
                info["estimated_data_rows"] = row_count
                info["structure"] = structure

            else:
                info["type"] = f"Unsupported ({ext})"

        except Exception as e:
            info["error"] = str(e)

        results.append(info)

    summary = {
        "files_found": len(results),
        "files": results,
        "tip": "Use load_complex_file(path) in python_code_executor_tool — it handles multi-row headers automatically."
    }
    return json.dumps(summary, indent=2, default=str)