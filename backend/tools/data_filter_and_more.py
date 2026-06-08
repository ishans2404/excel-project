import json, os, warnings
import pandas as pd
import numpy as np
from langchain_core.tools import tool
warnings.filterwarnings("ignore")


def _load(fp, n=5000, sheet=0):
    ext = os.path.splitext(fp)[1].lower()
    if ext == ".csv":
        for enc in ["utf-8","utf-8-sig","latin-1","cp1252"]:
            try: return pd.read_csv(fp, nrows=n, encoding=enc, on_bad_lines="skip")
            except: pass
    elif ext in (".xlsx",".xls",".xlsm"):
        return pd.read_excel(fp, sheet_name=sheet, nrows=n, engine="openpyxl")
    return None


# ─── Data Filter ──────────────────────────────────────────────────────────────

@tool
def data_filter_tool(
    file_path: str,
    filter_column: str = "",
    filter_operator: str = "eq",
    filter_value: str = "",
    group_by: str = "",
    aggregate_column: str = "",
    aggregate_func: str = "sum",
    sample_size: int = 5000,
    limit: int = 20,
    sheet_name: str = "0",
) -> str:
    """Filter and aggregate data. Auto-detects multi-row headers.

    Args:
        file_path: absolute path to file
        filter_column: column to filter on
        filter_operator: eq | ne | gt | lt | gte | lte | contains | isnull | notnull
        filter_value: value to compare
        group_by: column to group by
        aggregate_column: column to aggregate
        aggregate_func: sum | mean | count | max | min
        sample_size: rows to load
        limit: max rows returned
        sheet_name: Excel sheet
    """
    fp = file_path
    if not fp or not os.path.exists(fp): return json.dumps({"error": f"File not found: {fp}"})

    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df = _load(fp, sample_size, sheet)
    if df is None: return json.dumps({"error": "Load failed"})

    original_rows = len(df)
    if filter_column and filter_column in df.columns:
        col, op, val = filter_column, filter_operator, filter_value
        try:
            if op == "eq": df = df[df[col] == val]
            elif op == "ne": df = df[df[col] != val]
            elif op == "gt": df = df[pd.to_numeric(df[col], errors="coerce") > float(val)]
            elif op == "lt": df = df[pd.to_numeric(df[col], errors="coerce") < float(val)]
            elif op == "gte": df = df[pd.to_numeric(df[col], errors="coerce") >= float(val)]
            elif op == "lte": df = df[pd.to_numeric(df[col], errors="coerce") <= float(val)]
            elif op == "contains": df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
            elif op == "isnull": df = df[df[col].isna()]
            elif op == "notnull": df = df[df[col].notna()]
        except Exception: pass

    result = {"original_rows": original_rows, "filtered_rows": len(df)}

    if group_by and group_by in df.columns and aggregate_column and aggregate_column in df.columns:
        try:
            df[aggregate_column] = pd.to_numeric(df[aggregate_column], errors="coerce")
            agg_df = df.groupby(group_by)[aggregate_column].agg(aggregate_func).reset_index().round(4)
            agg_df = agg_df.sort_values(aggregate_column, ascending=False)
            result["grouped_data"] = agg_df.head(50).to_dict(orient="records")
            result["group_count"] = len(agg_df)
        except Exception as e:
            result["group_error"] = str(e)
    else:
        result["data"] = df.head(limit).fillna("NULL").to_dict(orient="records")
        result["columns"] = list(df.columns)

    return json.dumps(result, indent=2, default=str)


# ─── Multi File Operations ─────────────────────────────────────────────────────

@tool
def multi_file_ops_tool(
    file_paths: str,
    operation: str = "compare",
    join_column: str = "",
    how: str = "inner",
    sample_size: int = 3000,
) -> str:
    """Compare, merge, or concatenate multiple files. Auto-detects multi-row headers.

    Args:
        file_paths: newline or comma-separated absolute file paths
        operation: compare | merge | concat
        join_column: column to join on (for merge)
        how: inner | left | right | outer (for merge)
        sample_size: rows per file
    """
    op = operation.lower()
    fps = [p.strip() for p in file_paths.replace(",", "\n").split("\n") if p.strip()]
    n = sample_size

    if len(fps) < 2: return json.dumps({"error": "Need at least 2 file paths"})

    dfs = []
    for fp in fps:
        if not os.path.exists(fp): return json.dumps({"error": f"File not found: {fp}"})
        d = _load(fp, n)
        if d is not None: dfs.append((os.path.basename(fp), d))

    if not dfs: return json.dumps({"error": "Could not load any files"})

    try:
        if op == "concat":
            result_df = pd.concat([d for _,d in dfs], ignore_index=True)
            return json.dumps({"operation":"concat","result_rows":len(result_df),"result_columns":list(result_df.columns),"sample":result_df.head(5).fillna("NULL").to_dict(orient="records")}, default=str)

        elif op == "merge":
            if len(dfs) < 2: return json.dumps({"error":"Need 2 files for merge"})
            merged = pd.merge(dfs[0][1], dfs[1][1], on=join_column, how=how, suffixes=("_left","_right"))
            return json.dumps({"operation":"merge","on":join_column,"how":how,"result_rows":len(merged),"result_columns":list(merged.columns),"sample":merged.head(5).fillna("NULL").to_dict(orient="records")}, default=str)

        elif op == "compare":
            reports = []
            for name, d in dfs:
                reports.append({"file":name,"rows":len(d),"columns":list(d.columns),"numeric_cols":d.select_dtypes(include="number").columns.tolist(),"missing_avg":round(d.isnull().mean().mean()*100,2),"summary":d.describe().round(2).to_dict()})
            return json.dumps({"operation":"compare","files":reports}, indent=2, default=str)

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})


# ─── KPI Calculator ───────────────────────────────────────────────────────────

@tool
def kpi_calculator_tool(
    file_path: str,
    metric_columns: str = "",
    group_by: str = "",
    sample_size: int = 5000,
    sheet_name: str = "0",
) -> str:
    """Calculate KPIs: totals, rankings, percentiles. Auto-detects multi-row headers.

    Args:
        file_path: absolute path to file
        metric_columns: comma-separated numeric column names (empty = auto-detect)
        group_by: column to group rankings by
        sample_size: rows to load
        sheet_name: Excel sheet
    """
    fp = file_path
    if not fp or not os.path.exists(fp): return json.dumps({"error": f"File not found: {fp}"})

    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df = _load(fp, sample_size, sheet)
    if df is None: return json.dumps({"error": "Load failed"})

    metrics = [c.strip() for c in metric_columns.split(",") if c.strip()] if metric_columns else df.select_dtypes(include="number").columns.tolist()[:6]
    result = {"rows_analyzed": len(df), "kpis": {}}

    for col in metrics:
        if col not in df.columns: continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        result["kpis"][col] = {
            "total": round(float(s.sum()), 2),
            "mean": round(float(s.mean()), 2),
            "median": round(float(s.median()), 2),
            "p25": round(float(s.quantile(0.25)), 2),
            "p75": round(float(s.quantile(0.75)), 2),
            "p90": round(float(s.quantile(0.90)), 2),
            "p99": round(float(s.quantile(0.99)), 2),
            "max": round(float(s.max()), 2),
            "min": round(float(s.min()), 2),
        }

    if group_by and group_by in df.columns:
        rankings = {}
        for col in metrics[:4]:
            if col not in df.columns: continue
            grp = df.groupby(group_by)[col].sum().sort_values(ascending=False).head(10)
            rankings[col] = [{"rank":i+1, "group":str(k), "value":round(float(v),2)} for i,(k,v) in enumerate(grp.items())]
        result["rankings"] = rankings

    return json.dumps(result, indent=2, default=str)


# ─── Forecasting ──────────────────────────────────────────────────────────────

@tool
def forecasting_tool(
    file_path: str,
    value_column: str,
    method: str = "linear",
    periods: int = 10,
    date_column: str = "",
    sample_size: int = 3000,
    sheet_name: str = "0",
) -> str:
    """Forecast future values. Auto-detects multi-row headers.

    Args:
        file_path: absolute path to file
        value_column: numeric column to forecast (exact name from df.columns)
        method: linear | moving_avg | exponential
        periods: number of future periods to forecast
        date_column: optional date column for sorting
        sample_size: rows to load
        sheet_name: Excel sheet
    """
    fp = file_path
    val_col = value_column
    if not fp or not os.path.exists(fp): return json.dumps({"error": f"File not found: {fp}"})
    if not val_col: return json.dumps({"error": "value_column required"})

    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df = _load(fp, sample_size, sheet)
    if df is None: return json.dumps({"error": "Load failed"})
    if val_col not in df.columns:
        matches = [c for c in df.columns if val_col.lower() in str(c).lower()]
        if matches: val_col = matches[0]
        else: return json.dumps({"error": f"Column '{val_col}' not found. Available: {list(df.columns)[:10]}"})

    if date_column and date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.sort_values(date_column)

    s = pd.to_numeric(df[val_col], errors="coerce").dropna().values
    n = len(s)

    try:
        from scipy import stats as scipy_stats

        if method == "linear":
            x = np.arange(n)
            slope, intercept, r, p, se = scipy_stats.linregress(x, s)
            forecast = [round(float(slope*(n+i)+intercept), 4) for i in range(1, periods+1)]
            ci = [round(float(se * 1.96 * np.sqrt(1 + 1/n + (n+i-np.mean(x))**2/np.sum((x-np.mean(x))**2))), 4) for i in range(1, periods+1)]
            return json.dumps({"method":"linear","r_squared":round(r**2,4),"trend":"up" if slope>0 else "down","slope_per_period":round(float(slope),6),"forecast":forecast,"confidence_interval_95":ci,"historical_mean":round(float(np.mean(s)),4)})

        elif method == "moving_avg":
            window = min(7, n // 3)
            ma = pd.Series(s).rolling(window).mean().dropna().values
            last_ma = ma[-1]
            trend = (ma[-1] - ma[-min(window,len(ma))]) / window if len(ma) > 1 else 0
            forecast = [round(float(last_ma + trend*i), 4) for i in range(1, periods+1)]
            return json.dumps({"method":"moving_avg","window":window,"last_ma":round(float(last_ma),4),"trend_per_period":round(float(trend),6),"forecast":forecast})

        elif method == "exponential":
            alpha = 0.3
            smoothed = [s[0]]
            for v in s[1:]:
                smoothed.append(alpha*v + (1-alpha)*smoothed[-1])
            last = smoothed[-1]
            forecast = [round(float(last), 4)] * periods  # ETS flat forecast
            return json.dumps({"method":"exponential_smoothing","alpha":alpha,"last_smoothed":round(float(last),4),"forecast":forecast})

    except Exception as e:
        return json.dumps({"error": str(e)})


# ─── Data Quality Auditor ─────────────────────────────────────────────────────

@tool
def data_quality_tool(file_path: str, sample_size: int = 3000, sheet_name: str = "0") -> str:
    """Per-column quality scores and repair suggestions. Auto-detects multi-row headers.

    Args:
        file_path: absolute path to file
        sample_size: rows to sample
        sheet_name: Excel sheet
    """
    fp = file_path
    if not fp or not os.path.exists(fp): return json.dumps({"error": f"File not found: {fp}"})

    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df = _load(fp, sample_size, sheet)
    if df is None: return json.dumps({"error": "Load failed"})

    results = {}
    for col in df.columns:
        s = df[col]
        missing_pct = round(s.isnull().mean()*100, 2)
        n_unique = int(s.nunique())
        issues = []
        suggestions = []
        score = 100

        if missing_pct > 50: score -= 40; issues.append(f"{missing_pct:.1f}% missing")
        elif missing_pct > 20: score -= 20; issues.append(f"{missing_pct:.1f}% missing")
        elif missing_pct > 5: score -= 10; issues.append(f"{missing_pct:.1f}% missing")

        if missing_pct > 0:
            if s.dtype in ["float64","int64"]: suggestions.append("Fill with mean/median or use interpolation")
            else: suggestions.append("Fill with mode or 'Unknown'")

        if n_unique == 1: score -= 20; issues.append("All values identical (zero variance)")
        elif n_unique == len(df): score -= 5; issues.append("All values unique (possible ID column)")

        # Check for mixed types in object columns
        if s.dtype == "object":
            sample = s.dropna().head(100)
            numeric_count = sum(1 for v in sample if str(v).replace(".","",1).replace("-","",1).isdigit())
            if 0 < numeric_count < len(sample)*0.9:
                score -= 10; issues.append("Mixed numeric/text values")
                suggestions.append("Consider type cleaning with pd.to_numeric(errors='coerce')")

        grade = "A" if score>=85 else "B" if score>=70 else "C" if score>=50 else "D"
        results[col] = {"score":score,"grade":grade,"dtype":str(s.dtype),"missing_pct":missing_pct,"unique_count":n_unique,"issues":issues,"suggestions":suggestions}

    overall = round(sum(r["score"] for r in results.values()) / len(results), 1) if results else 0
    return json.dumps({"overall_score":overall,"overall_grade":"A" if overall>=85 else "B" if overall>=70 else "C" if overall>=50 else "D","columns":results}, indent=2)


# ─── Column Analysis ──────────────────────────────────────────────────────────

@tool
def column_analysis_tool(
    file_path: str,
    column_name: str,
    sample_size: int = 5000,
    sheet_name: str = "0",
) -> str:
    """Deep-dive on a single column: distribution, outliers, normality, value breakdown.

    Args:
        file_path: absolute path to file
        column_name: exact column name to analyze (get from data_profiling_tool first)
        sample_size: rows to load
        sheet_name: Excel sheet
    """
    fp = file_path
    col = column_name
    if not fp or not os.path.exists(fp): return json.dumps({"error": f"File not found: {fp}"})
    if not col: return json.dumps({"error": "column_name required"})

    sheet = int(sheet_name) if str(sheet_name).isdigit() else sheet_name
    df = _load(fp, sample_size, sheet)
    if df is None: return json.dumps({"error": "Load failed"})
    if col not in df.columns:
        matches = [c for c in df.columns if col.lower() in str(c).lower()]
        if matches: col = matches[0]
        else: return json.dumps({"error": f"Column '{col}' not found. Available: {list(df.columns)[:15]}"})

    s = df[col]
    result = {"column": col, "dtype": str(s.dtype), "total_rows": len(s), "missing_count": int(s.isnull().sum()), "missing_pct": round(s.isnull().mean()*100,2), "unique_count": int(s.nunique())}

    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() > len(s) * 0.5:  # numeric column
        num = num.dropna()
        q1,q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3-q1
        from scipy import stats as sc
        stat, p = sc.normaltest(num) if len(num) >= 8 else (0, 1)
        result.update({
            "type":"numeric","count":int(len(num)),
            "mean":round(float(num.mean()),4),"median":round(float(num.median()),4),
            "std":round(float(num.std()),4),"min":round(float(num.min()),4),"max":round(float(num.max()),4),
            "q1":round(float(q1),4),"q3":round(float(q3),4),"iqr":round(float(iqr),4),
            "outlier_count":int(((num<q1-1.5*iqr)|(num>q3+1.5*iqr)).sum()),
            "outlier_pct":round(int(((num<q1-1.5*iqr)|(num>q3+1.5*iqr)).sum())/len(num)*100,2),
            "skewness":round(float(num.skew()),4),"kurtosis":round(float(num.kurtosis()),4),
            "is_normal":bool(p>0.05),"normality_pvalue":round(float(p),6),
            "percentiles":{str(p):round(float(num.quantile(p/100)),4) for p in [5,10,25,50,75,90,95,99]},
        })
    else:
        vc = s.value_counts()
        result.update({
            "type":"categorical","top_20":vc.head(20).to_dict(),
            "most_common":str(vc.index[0]) if len(vc)>0 else None,
            "least_common":str(vc.index[-1]) if len(vc)>0 else None,
            "top_pct":round(float(vc.iloc[0]/len(s)*100),2) if len(vc)>0 else 0,
        })

    return json.dumps(result, indent=2, default=str)
