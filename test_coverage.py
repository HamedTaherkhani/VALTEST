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
import coverage
import subprocess
import tempfile
import os
import sys
import uuid
import json
import resource  # Note: Only available on Unix-based systems
from functools import partial
from multiprocessing import Pool, cpu_count

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


def compute_line_coverage(code_str, idx=None):
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
        # print(executed_lines)
        return executed_lines, None

    # Set the trace function
    sys.settrace(tracer)
    try:
        # Execute the compiled code in a new global namespace
        exec(compiled_code, {})
    except Exception as e:
        is_error = True
        # Optionally, handle or log exceptions from the executed code
        # print()
        # print(f"Error during execution for idx {idx}: {e}")
        pass
    finally:
        # Remove the trace function to avoid affecting other code
        sys.settrace(None)

    # Split the code into lines
    lines = code_str.splitlines()
    total_lines = len(lines)
    # Build the coverage dictionary
    return executed_lines, total_lines


def remove_comments_and_empty_lines(code: str) -> str:
    # Remove multi-line comments enclosed in triple quotes
    code = re.sub(r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', '', code)
    # Split the code into lines
    lines = code.splitlines()
    # Filter out lines that are single-line comments or empty
    filtered_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
    # Join the remaining lines back into a string
    return '\n'.join(filtered_lines)


def limit_resources(max_memory_mb):
    """
    Set resource limits for the subprocess.

    Args:
        max_memory_mb (int): Maximum memory in megabytes.
    """
    # Convert megabytes to bytes
    max_memory_bytes = max_memory_mb * 1024 * 1024
    # Set maximum address space (virtual memory)
    resource.setrlimit(resource.RLIMIT_AS, (max_memory_bytes, max_memory_bytes))
    # Optionally, set other limits like CPU time if needed
    # resource.setrlimit(resource.RLIMIT_CPU, (timeout_seconds, timeout_seconds))



def get_line_coverage_unittest(code_str, test_case_strings):
    """
    Calculate the line coverage of the given code against provided test cases using coverage.py via subprocess.

    Args:
        code_str (str): The code under test as a string.
        test_case_strings (list of str): List of test case class definitions as strings.

    Returns:
        float: Line coverage percentage.
    """
    actual_tests = []
    for test in test_case_strings:
        if 'import subprocess' not in test:
            actual_tests.append(test)
    max_memory_mb = 5120
    # Generate a unique identifier for this test run to avoid module caching issues
    unique_id = uuid.uuid4().hex
    test_module_name = f'test_code_{unique_id}'
    bigcode_venv_path = os.getcwd() + '/.bigcode_venv/bin'
    # Use TemporaryDirectory to ensure isolation
    with tempfile.TemporaryDirectory() as temp_dir:
        # temp_dir = '/home/hamed/PycharmProjects/hallucination/temp_dir2'
        # Paths for the code under test and the test module
        code_path = os.path.join(temp_dir, 'code_under_test.py')
        test_path = os.path.join(temp_dir, f'{test_module_name}.py')
        # Write the code under test to 'code_under_test.py'
        with open(code_path, 'w') as code_file:
            code_file.write(code_str)

        # Write the test cases to the unique test module
        with open(test_path, 'w') as test_file:
            # Import the code under test
            test_file.write('from code_under_test import *\n\n')
            # Write each test case class
            for test_case in actual_tests:
                test_file.write(test_case.strip() + '\n\n')

        # Prepare environment variables
        env = os.environ.copy()
        # Optionally, you can set COVERAGE_FILE to ensure coverage data is stored uniquely
        coverage_file = os.path.join(temp_dir, f'.coverage_{unique_id}')
        env['COVERAGE_FILE'] = coverage_file

        # Change the working directory to the temporary directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:

            # Execute the coverage run command
            try:
                env = os.environ.copy()
                env['MPLBACKEND'] = 'Agg'
                subprocess.run(
                    'coverage run -m unittest discover',
                    preexec_fn=lambda: limit_resources(max_memory_mb),
                    timeout=120,
                    cwd=temp_dir, shell=True, check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                print(f"Test run timed out")
                return 0
            except subprocess.CalledProcessError as e:
                pass
                # print(f"Test run failed: {e.stderr.decode()}")
            except KeyboardInterrupt as e:
                print(f"Test interrupted")
                pass
            subprocess.run(
                'python -m coverage json -o coverage.json',
                cwd=temp_dir, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Read the coverage.json file
            coverage_json_path = os.path.join(temp_dir, 'coverage.json')
            if not os.path.exists(coverage_json_path):
                # If coverage.json does not exist, return 0 coverage
                return 0.0

            with open(coverage_json_path, 'r') as json_file:
                coverage_data = json.load(json_file)
            # Extract coverage data for 'code_under_test.py'
            files = coverage_data.get('files', {})
            # if code_file_key not in files:
            #     print('jere')
            #     # If the code file is not in the coverage report, return 0 coverage
            #     return 0.0
            file_coverage = files['code_under_test.py']
            summary = file_coverage.get('summary', {})
            coverage_percent = summary.get('percent_covered', 0.0)

            return coverage_percent

        finally:
            # Restore the original working directory
            os.chdir(original_cwd)

def _measure_coverage_for_function(func):
    """
    Helper function to encapsulate the coverage call
    for a single Function object.
    """
    tests_list = [tc.text for tc in func.testcases if tc.is_valid==1]
    return get_line_coverage_unittest(func.solution, tests_list)

def measure_coverage(functions: List[Function], dataset):
    coverage_results = []
    if 'BigCodeBench' in dataset:
        with Pool() as pool:
            # imap gives you an iterator, so wrap in list or iterate to collect results
            coverage_results = list(
                tqdm(
                    pool.imap(_measure_coverage_for_function, functions),
                    total=len(functions),
                    desc="Calculating coverage"
                )
            )
        return coverage_results

    for idx, func in tqdm(enumerate(functions)):
        # print(idx)
        lines_per_testcase = []
        total_lines = 1
        for test_case in func.testcases:
            if test_case.is_valid == 0:
                continue
            sol = remove_comments_and_empty_lines(func.solution) + '\n'
            sol += test_case.text + "\n"
            executed_lines, tot = compute_line_coverage(sol, idx)
            if tot is not None:
                total_lines = tot
            lines_per_testcase.append(executed_lines)
        # print(lines_per_testcase)
        coverage_results.append(len(set().union(*lines_per_testcase))/total_lines)

        # intersection_results.append(len(set.union(*coverage_results[idx])) / total_lines)
    # print(coverage_results)
    return coverage_results


