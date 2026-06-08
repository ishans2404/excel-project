from tools.file_discovery import file_discovery_tool
from tools.data_profiling import data_profiling_tool
from tools.code_executor import python_code_executor_tool
from tools.chart_generator import chart_generator_tool
from tools.statistical_analysis import statistical_analysis_tool
from tools.auto_insights import auto_insights_tool
from tools.data_filter import data_filter_tool
from tools.multi_file_ops import multi_file_ops_tool
from tools.kpi_calculator import kpi_calculator_tool
from tools.forecasting import forecasting_tool
from tools.data_quality import data_quality_tool
from tools.column_analysis import column_analysis_tool
from tools.report_generator import report_generator_tool
from tools.hil_tool import request_human_input_tool

ALL_TOOLS = [
    file_discovery_tool,
    data_profiling_tool,
    python_code_executor_tool,
    chart_generator_tool,
    statistical_analysis_tool,
    auto_insights_tool,
    data_filter_tool,
    multi_file_ops_tool,
    kpi_calculator_tool,
    forecasting_tool,
    data_quality_tool,
    column_analysis_tool,
    report_generator_tool,
    request_human_input_tool,
]


def get_all_tools() -> list:
    return ALL_TOOLS


def get_tools_by_name() -> dict:
    return {tool.name: tool for tool in ALL_TOOLS}
