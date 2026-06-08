import json
import os
import warnings
import pandas as pd
from langchain_core.tools import tool

warnings.filterwarnings("ignore")

# Reuse the smart loader from data_profiling
from tools.data_profiling import _smart_load


@tool
def auto_insights_tool(file_path: str, sample_size: int = 2000, sheet_name: str = "0") -> str:
    """Automated data quality score (A-D) and key insights. Handles multi-row headers.

    Args:
        file_path: absolute path to CSV or Excel file
        sample_size: rows to sample (default 2000)
        sheet_name: sheet name or index for Excel (default "0")
    """
    fp = file_path
    n = sample_size
    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name

    if not fp or not os.path.exists(fp):
        return json.dumps({"error": f"File not found: {fp}"})

    try:
        result = _smart_load(fp, n, sheet)
        if result is None:
            return json.dumps({"error": "Could not load file"})
        df, struct = result
    except Exception as e:
        return json.dumps({"error": str(e)})

    missing_avg = df.isnull().mean().mean() * 100
    dup_pct = df.duplicated().sum() / max(len(df), 1) * 100
    nc = df.select_dtypes(include="number").columns.tolist()
    cc = df.select_dtypes(exclude="number").columns.tolist()

    score = 100
    issues = []
    if missing_avg > 30:
        score -= 30
        issues.append(f"High missing data: {missing_avg:.1f}% average")
    elif missing_avg > 10:
        score -= 15
        issues.append(f"Moderate missing data: {missing_avg:.1f}% average")

    if dup_pct > 10:
        score -= 20
        issues.append(f"High duplicate rows: {dup_pct:.1f}%")
    elif dup_pct > 1:
        score -= 5
        issues.append(f"Some duplicates: {dup_pct:.1f}%")

    grade = "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 50 else "D"

    insights = [
        f"Dataset: {len(df):,} rows × {len(df.columns)} columns (sampled {n})",
        f"Header structure: title_rows={struct['title_rows']}, header_row={struct['main_header']}, sub_headers={struct['sub_headers']}",
        f"Numeric columns ({len(nc)}): {', '.join(nc[:8])}",
    ]
    if cc:
        insights.append(f"Categorical columns ({len(cc)}): {', '.join(cc[:6])}")

    # Numeric insights
    for col in nc[:6]:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        insights.append(
            f"'{col}': sum={s.sum():,.0f}, mean={s.mean():,.1f}, max={s.max():,.0f}"
            + (f", {outliers} outliers" if outliers > 0 else "")
        )

    # Categorical insights
    for col in cc[:4]:
        n_unique = df[col].nunique()
        if n_unique == 0:
            continue
        top = df[col].value_counts().index[0]
        insights.append(f"'{col}': {n_unique} unique values, most common='{top}'")

    recs = []
    if nc:
        recs.append("Run statistical_analysis (correlation) to find related numeric columns")
    if any(k in str(c).lower() for c in df.columns for k in ["date", "time", "year", "month"]):
        recs.append("Run statistical_analysis (timeseries) — date columns detected")
    recs.append("Use python_code_executor with top_n(df, group_col, value_col) for rankings")

    return json.dumps({
        "quality_score": score,
        "quality_grade": grade,
        "rows_sampled": len(df),
        "columns": len(df.columns),
        "real_column_names": list(df.columns),
        "numeric_columns": nc,
        "categorical_columns": cc,
        "missing_avg_pct": round(missing_avg, 2),
        "duplicate_pct": round(dup_pct, 2),
        "issues": issues,
        "insights": insights,
        "recommendations": recs,
        "header_structure": struct,
    }, indent=2)