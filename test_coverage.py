import coverage
import sys
import tempfile
import subprocess
import os
import io
import types
from typing import List, Dict, Any
from log_probs import Function
from tqdm import tqdm
import re

def calculate_average_coverage(coverage_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculates the average line and branch coverage from per-function coverage data.

    Args:
        coverage_data (Dict[str, Dict[str, Any]]): Per-function coverage data.

    Returns:
        Dict[str, float]: A dictionary with average 'line_coverage_percent' and 'branch_coverage_percent'.
    """
    total_line_coverage = 0.0
    total_branch_coverage = 0.0
    count_line = 0
    count_branch = 0

    for prompt, data in coverage_data.items():
        if 'error' in data:
            # Skip functions that had errors during coverage measurement
            continue
        if 'line_coverage_percent' in data:
            total_line_coverage += data['line_coverage_percent']
            count_line += 1
        if 'branch_coverage_percent' in data and isinstance(data['branch_coverage_percent'], (int, float)):
            total_branch_coverage += data['branch_coverage_percent']
            count_branch += 1

    average_line_coverage = (total_line_coverage / count_line) if count_line > 0 else 0.0
    average_branch_coverage = (total_branch_coverage / count_branch) if count_branch > 0 else 0.0

    return {
        'average_line_coverage_percent': round(average_line_coverage, 2),
        'average_branch_coverage_percent': round(average_branch_coverage, 2) if count_branch > 0 else 'N/A'
    }


def compute_line_coverage(code_str):
    """
    Executes the given Python code string and computes its line coverage.

    Args:
        code_str (str): The Python code to execute.

    Returns:
        dict: A dictionary where keys are line numbers (1-based) and values are booleans
              indicating whether the line was executed (True) or not (False).
    """
    is_error = False
    executed_lines = set()
    filename = "<string_code>"

    def tracer(frame, event, arg):
        if event == 'line' and frame.f_code.co_filename == filename:
            lineno = frame.f_lineno
            executed_lines.add(lineno)
        return tracer

    # Compile the code with the specified filename
    try:
        compiled_code = compile(code_str, filename, 'exec')
    except SyntaxError as e:
        return 0, 10

    # Set the trace function
    sys.settrace(tracer)
    try:
        # Execute the compiled code in a new global namespace
        exec(compiled_code, {})
    except Exception as e:
        is_error = True
        # Optionally, handle or log exceptions from the executed code
        # print(f"Error during execution: {e}")
        pass
    finally:
        # Remove the trace function to avoid affecting other code
        sys.settrace(None)

    # Split the code into lines
    lines = code_str.splitlines()
    total_lines = len(lines)
    # Build the coverage dictionary
    return len(executed_lines), total_lines


def remove_comments_and_empty_lines(code: str) -> str:
    # Remove multi-line comments enclosed in triple quotes
    code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)
    # Split the code into lines
    lines = code.splitlines()
    # Filter out lines that are single-line comments or empty
    filtered_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    # Join the remaining lines back into a string
    return '\n'.join(filtered_lines)


def measure_coverage(functions: List[Function]):
    coverage_results = []
    for idx, func in enumerate(functions):
        sol = remove_comments_and_empty_lines(func.solution) + '\n'
        for test_case in func.testcases:
            sol += test_case.text + "\n"

        total_executed_lines, total_lines = compute_line_coverage(sol)
        coverage_results.append(total_executed_lines/total_lines)

        # intersection_results.append(len(set.union(*coverage_results[idx])) / total_lines)
    return coverage_results


