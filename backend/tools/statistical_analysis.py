import json
import os
import warnings
import pandas as pd
import numpy as np
from langchain_core.tools import tool

warnings.filterwarnings("ignore")


def _load(file_path, sample_size, sheet_name=0):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(file_path, nrows=sample_size, encoding=enc, on_bad_lines="skip")
            except Exception:
                pass
    elif ext in (".xlsx", ".xls", ".xlsm"):
        return pd.read_excel(file_path, sheet_name=sheet_name, nrows=sample_size, engine="openpyxl")
    return None


@tool
def statistical_analysis_tool(
    file_path: str,
    analysis_type: str,
    columns: str = "",
    sample_size: int = 5000,
    date_column: str = "",
    threshold: float = 0.5,
    sheet_name: str = "0",
) -> str:
    """Run statistical analysis. Auto-detects multi-row headers.

    Args:
        file_path: absolute path to file
        analysis_type: correlation | outliers | distribution | timeseries | trends | segmentation
        columns: comma-separated column names (leave empty for auto-detect)
        sample_size: rows to sample (default 5000)
        date_column: date column name for timeseries analysis
        threshold: correlation threshold 0-1 (default 0.5)
        sheet_name: Excel sheet (default "0")
    """
    if not file_path or not os.path.exists(file_path):
        return json.dumps({"error": f"File not found: {file_path}"})

    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df = _load(file_path, sample_size, sheet)
    if df is None:
        return json.dumps({"error": "Could not load file"})

    col_list = [c.strip() for c in columns.split(",") if c.strip()] if columns else []
    date_col = date_column

    try:
        if analysis_type == "correlation":
            nc = col_list if col_list else df.select_dtypes(include="number").columns.tolist()
            corr = df[nc].corr().round(3)
            # Find strong correlations
            strong = []
            for i in range(len(nc)):
                for j in range(i + 1, len(nc)):
                    val = corr.iloc[i, j]
                    if abs(val) >= threshold:
                        strong.append({"col1": nc[i], "col2": nc[j], "correlation": round(val, 4), "strength": "strong" if abs(val) >= 0.7 else "moderate"})
            strong.sort(key=lambda x: abs(x["correlation"]), reverse=True)
            return json.dumps({"analysis": "correlation", "matrix": corr.to_dict(), "strong_correlations": strong[:20], "threshold": threshold})

        elif analysis_type == "outliers":
            nc = col_list if col_list else df.select_dtypes(include="number").columns.tolist()
            results = {}
            for col in nc[:10]:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) < 4:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outlier_vals = s[(s < lower) | (s > upper)]
                results[col] = {
                    "outlier_count": int(len(outlier_vals)),
                    "outlier_pct": round(len(outlier_vals) / len(s) * 100, 2),
                    "lower_bound": round(float(lower), 4),
                    "upper_bound": round(float(upper), 4),
                    "extreme_values": sorted([round(float(v), 4) for v in outlier_vals])[:10],
                }
            return json.dumps({"analysis": "outliers", "results": results, "rows_analyzed": len(df)})

        elif analysis_type == "distribution":
            from scipy import stats as scipy_stats
            nc = col_list if col_list else df.select_dtypes(include="number").columns.tolist()
            results = {}
            for col in nc[:8]:
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(s) < 4:
                    continue
                stat, pval = scipy_stats.normaltest(s)
                skew = float(s.skew())
                kurt = float(s.kurtosis())
                results[col] = {
                    "count": int(len(s)),
                    "mean": round(float(s.mean()), 4),
                    "std": round(float(s.std()), 4),
                    "skewness": round(skew, 4),
                    "kurtosis": round(kurt, 4),
                    "is_normal": bool(pval > 0.05),
                    "normality_pvalue": round(float(pval), 6),
                    "distribution_hint": "normal" if pval > 0.05 else ("right-skewed" if skew > 0.5 else "left-skewed" if skew < -0.5 else "non-normal"),
                }
            return json.dumps({"analysis": "distribution", "results": results})

        elif analysis_type == "segmentation":
            nc = col_list if col_list else df.select_dtypes(include="number").columns.tolist()
            cc = df.select_dtypes(include=["object", "category"]).columns.tolist()
            results = {}
            for cat_col in cc[:5]:
                if df[cat_col].nunique() > 30:
                    continue
                seg = {}
                for num_col in nc[:4]:
                    grp = df.groupby(cat_col)[num_col].agg(["mean", "median", "count", "std"]).round(3)
                    seg[num_col] = grp.to_dict()
                results[cat_col] = seg
            return json.dumps({"analysis": "segmentation", "results": results})

        elif analysis_type == "timeseries":
            nc_ts = col_list if col_list else df.select_dtypes(include="number").columns.tolist()
            if not date_col:
                # Auto-detect
                for c in df.columns:
                    if any(k in str(c).lower() for k in ["date", "time", "year", "month"]):
                        date_col = c
                        break
            if not date_col:
                return json.dumps({"error": "No date_column specified and none auto-detected"})

            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).sort_values(date_col)
            results = {"date_column": date_col, "date_range": {"start": str(df[date_col].min()), "end": str(df[date_col].max()), "total_rows": len(df)}}
            for num_col in nc_ts[:4]:
                s = df[[date_col, num_col]].dropna()
                if len(s) < 2:
                    continue
                results[num_col] = {
                    "total": round(float(s[num_col].sum()), 2),
                    "mean": round(float(s[num_col].mean()), 2),
                    "trend": "up" if s[num_col].iloc[-1] > s[num_col].iloc[0] else "down",
                    "change_pct": round((float(s[num_col].iloc[-1]) - float(s[num_col].iloc[0])) / abs(float(s[num_col].iloc[0])) * 100, 2) if s[num_col].iloc[0] != 0 else 0,
                }
            return json.dumps({"analysis": "timeseries", "results": results})

        elif analysis_type == "trends":
            nc = col_list if col_list else df.select_dtypes(include="number").columns.tolist()
            from scipy import stats as scipy_stats
            results = {}
            for col in nc[:8]:
                s = pd.to_numeric(df[col], errors="coerce").dropna().reset_index(drop=True)
                if len(s) < 4:
                    continue
                slope, intercept, r, p, se = scipy_stats.linregress(range(len(s)), s)
                results[col] = {
                    "slope": round(float(slope), 6),
                    "r_squared": round(float(r ** 2), 4),
                    "p_value": round(float(p), 6),
                    "trend": "increasing" if slope > 0 else "decreasing",
                    "significant": bool(p < 0.05),
                }
            return json.dumps({"analysis": "trends", "results": results})

        else:
            return json.dumps({"error": f"Unknown analysis_type: {analysis_type}. Use: correlation, outliers, trends, distribution, segmentation, timeseries"})

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})
