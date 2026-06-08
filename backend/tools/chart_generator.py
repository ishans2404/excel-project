import json
import os
import io
import base64
import warnings
import pandas as pd
import numpy as np
from langchain_core.tools import tool

warnings.filterwarnings("ignore")


def _load_df(file_path: str, sample_size: int, sheet_name=0) -> pd.DataFrame | None:
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".csv":
            for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
                try:
                    return pd.read_csv(file_path, nrows=sample_size, encoding=enc, on_bad_lines="skip")
                except Exception:
                    pass
        elif ext in (".xlsx", ".xls", ".xlsm"):
            return pd.read_excel(file_path, sheet_name=sheet_name, nrows=sample_size, engine="openpyxl")
    except Exception:
        pass
    return None


def _to_plotly_json(fig) -> str:
    try:
        return fig.to_json()
    except Exception:
        return None


def _to_png_b64(fig) -> str:
    try:
        img_bytes = fig.to_image(format="png", width=1200, height=700, scale=2)
        return base64.b64encode(img_bytes).decode()
    except Exception:
        # Fallback: matplotlib
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            return base64.b64encode(buf.read()).decode()
        except Exception:
            return None


@tool
def chart_generator_tool(
    file_path: str,
    chart_type: str,
    x_column: str = "",
    y_column: str = "",
    title: str = "",
    group_by: str = "",
    agg_func: str = "sum",
    sample_size: int = 3000,
    sheet_name: str = "0",
) -> str:
    """Generate interactive Plotly charts. Returns plotly_json + png_b64.
    Use exact column names from data_profiling_tool or python_code_executor output.

    Args:
        file_path: absolute path to data file
        chart_type: bar | horizontal_bar | line | scatter | pie | histogram | box | heatmap | area | violin | correlation
        x_column: column for x-axis / categories (exact name from df.columns)
        y_column: column for y-axis / values (exact name from df.columns)
        title: chart title
        group_by: column to color/group by (optional)
        agg_func: sum | mean | count | max | min (default sum)
        sample_size: rows to load (default 3000)
        sheet_name: Excel sheet name or index (default "0")
    """
    params = {
        "chart_type": chart_type, "file_path": file_path,
        "x_column": x_column, "y_column": y_column, "title": title,
        "group_by": group_by, "agg_func": agg_func,
        "sample_size": sample_size, "sheet_name": sheet_name,
    }

    import plotly.express as px
    import plotly.graph_objects as go

    chart_type = params["chart_type"].lower()
    file_path = params["file_path"]
    x_col = params.get("x_column", "")
    y_col = params.get("y_column", "")
    title = params.get("title", "") or f"{chart_type.title()} Chart"
    group_by = params.get("group_by", "")
    agg_func = params.get("agg_func", "sum")
    sample_size = int(params.get("sample_size", 3000))
    sheet_raw = params.get("sheet_name", "0")
    sheet_name = int(sheet_raw) if str(sheet_raw).isdigit() else sheet_raw

    if not file_path:
        return json.dumps({"error": "file_path is required"})

    df = _load_df(file_path, sample_size, sheet_name)
    if df is None:
        return json.dumps({"error": f"Could not load file: {file_path}"})

    # Clean column names
    if x_col and x_col not in df.columns:
        # Try case-insensitive match
        for col in df.columns:
            if str(col).lower() == x_col.lower():
                x_col = col
                break

    if y_col and y_col not in df.columns:
        for col in df.columns:
            if str(col).lower() == y_col.lower():
                y_col = col
                break

    TEMPLATE = "plotly_white"
    COLOR_SEQ = px.colors.qualitative.Bold

    try:
        fig = None

        if chart_type == "bar":
            if x_col and y_col:
                agg_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index().sort_values(y_col, ascending=False).head(25)
                fig = px.bar(agg_df, x=x_col, y=y_col, title=title, color_discrete_sequence=COLOR_SEQ, template=TEMPLATE)
            elif x_col:
                vc = df[x_col].value_counts().head(25).reset_index()
                vc.columns = [x_col, "count"]
                fig = px.bar(vc, x=x_col, y="count", title=title, template=TEMPLATE)

        elif chart_type == "horizontal_bar":
            if x_col and y_col:
                agg_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index().sort_values(y_col).tail(20)
                fig = px.bar(agg_df, x=y_col, y=x_col, orientation="h", title=title, template=TEMPLATE, color_discrete_sequence=COLOR_SEQ)

        elif chart_type == "line":
            if x_col and y_col:
                try:
                    df[x_col] = pd.to_datetime(df[x_col])
                    df = df.sort_values(x_col)
                except Exception:
                    pass
                color_arg = group_by if group_by and group_by in df.columns else None
                fig = px.line(df, x=x_col, y=y_col, color=color_arg, title=title, template=TEMPLATE)

        elif chart_type == "scatter":
            if x_col and y_col:
                color_arg = group_by if group_by and group_by in df.columns else None
                fig = px.scatter(df, x=x_col, y=y_col, color=color_arg, title=title, template=TEMPLATE, opacity=0.6)

        elif chart_type == "histogram":
            if x_col:
                d = pd.to_numeric(df[x_col], errors="coerce").dropna()
                fig = px.histogram(pd.DataFrame({x_col: d}), x=x_col, title=title, template=TEMPLATE, nbins=50)
                fig.add_vline(x=d.mean(), line_dash="dash", line_color="red", annotation_text=f"Mean: {d.mean():.2f}")

        elif chart_type == "box":
            if y_col:
                x_arg = x_col if x_col and df[x_col].nunique() <= 20 else None
                fig = px.box(df, x=x_arg, y=y_col, title=title, template=TEMPLATE, color_discrete_sequence=COLOR_SEQ)

        elif chart_type == "heatmap" or chart_type == "correlation":
            nc = df.select_dtypes(include="number").columns.tolist()
            if len(nc) >= 2:
                corr = df[nc].corr().round(2)
                fig = px.imshow(corr, text_auto=True, title=title or "Correlation Heatmap",
                                color_continuous_scale="RdYlGn", template=TEMPLATE, aspect="auto")

        elif chart_type == "pie":
            if x_col:
                if y_col:
                    pd_ = df.groupby(x_col)[y_col].agg(agg_func).head(10).reset_index()
                    fig = px.pie(pd_, names=x_col, values=y_col, title=title, template=TEMPLATE)
                else:
                    vc = df[x_col].value_counts().head(10).reset_index()
                    vc.columns = [x_col, "count"]
                    fig = px.pie(vc, names=x_col, values="count", title=title, template=TEMPLATE)

        elif chart_type == "area":
            if x_col and y_col:
                fig = px.area(df, x=x_col, y=y_col, title=title, template=TEMPLATE)

        elif chart_type == "violin":
            if y_col:
                x_arg = x_col if x_col and df[x_col].nunique() <= 10 else None
                fig = px.violin(df, x=x_arg, y=y_col, box=True, title=title, template=TEMPLATE)

        elif chart_type == "stacked_bar":
            if x_col and y_col and group_by:
                pv = df.groupby([x_col, group_by])[y_col].agg(agg_func).unstack(fill_value=0).head(15)
                fig = px.bar(pv.reset_index().melt(id_vars=x_col), x=x_col, y="value", color=group_by, barmode="stack", title=title, template=TEMPLATE)

        if fig is None:
            return json.dumps({"error": f"Could not generate {chart_type} chart with given columns. Verify column names exist in file."})

        # Export
        plotly_json = _to_plotly_json(fig)
        png_b64 = _to_png_b64(fig)

        return json.dumps({
            "status": "success",
            "chart_type": chart_type,
            "title": title,
            "row_count": len(df),
            "plotly_json": plotly_json,
            "png_b64": png_b64,
            "chart_data": True,  # Signals to tool_execution_node to store as chart artifact
        })

    except Exception as e:
        import traceback
        return json.dumps({"error": str(e), "trace": traceback.format_exc()})
