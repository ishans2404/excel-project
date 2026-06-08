"""
The HIL tool signals to the analyst agent that it needs human input.
The actual interrupt() is handled in analyst_agent.tool_execution_node.
This tool's schema tells the LLM HOW to request user input.
"""
import json
from langchain_core.tools import tool


@tool
def request_human_input(component_type: str, component: str) -> str:
    """Request specific input from the user via an interactive UI component.
    
    Use this when you need the user to:
    - Select columns for analysis (component_type: 'column_picker')
    - Choose a chart type (component_type: 'chart_type')  
    - Make a choice between options (component_type: 'choice_picker')
    - Confirm before a slow operation (component_type: 'confirmation')
    - Fill a form with parameters (component_type: 'form')
    - Choose which file to analyze (component_type: 'file_picker')
    
    Args:
        component_type: One of: column_picker, chart_type, choice_picker, confirmation, form, file_picker
        component: JSON string with the component config. Examples:
        
        choice_picker: {"id":"pick1","title":"Choose analysis","options":[{"value":"trend","label":"Trend Analysis","emoji":"📈"},{"value":"distribution","label":"Distribution","emoji":"📊"}]}
        
        column_picker: {"id":"col1","title":"Select columns","columns":["col_a","col_b","col_c"],"multi":true,"subtitle":"Choose columns to analyze"}
        
        chart_type: {"id":"chart1","title":"Choose chart type"}
        
        confirmation: {"id":"conf1","title":"Run full scan?","message":"This will scan 500k rows and may take 30s","confirm_label":"Run it","cancel_label":"Skip"}
        
        form: {"id":"form1","title":"Set parameters","fields":[{"name":"top_n","label":"Show top N","type":"number","default":10,"min":1,"max":100}]}
        
        file_picker: {"id":"file1","title":"Select file","files":["/path/a.csv","/path/b.xlsx"],"multi":false}
    
    Returns:
        The user's selection as JSON string
    """
    # This function body never actually runs —
    # tool_execution_node intercepts calls to this tool
    # and calls interrupt() instead.
    return json.dumps({"error": "This tool should be intercepted by tool_execution_node"})


request_human_input_tool = request_human_input
