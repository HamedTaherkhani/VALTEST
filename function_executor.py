import concurrent.futures
import resource
import signal
import multiprocessing
import unittest
import sys
import io
import os
import tempfile
import subprocess
from typing import List, Tuple, Union

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    return False


def target(func_str):
    try:
        exec(func_str, {})
    except Exception as e:
        raise e

import multiprocessing

def run_test_cases(func_code, test_cases, timeout=5):
    """
    Runs test cases concurrently on the provided function code.

    Args:
        func_code (str): The function code as a string.
        test_cases (list): A list of assertion statements as strings.
        timeout (int): Timeout in seconds for each test case.

    Returns:
        list: A list of booleans indicating if each test case passed (True) or failed (False).
    """
    def run_test_case(func_code, test_case_str, output_queue):
        # Create a namespace dictionary for the function code
        func_namespace = {}
        try:
            exec(func_code, func_namespace)
        except Exception as e:
            output_queue.put(False)
            return

        test_namespace = func_namespace.copy()
        try:
            exec(test_case_str, test_namespace)
            output_queue.put(True)
        except AssertionError:
            output_queue.put(False)
        except Exception:
            output_queue.put(False)

    results = []
    for test_case in test_cases:
        output_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=run_test_case,
            args=(func_code, test_case, output_queue)
        )
        process.start()
        process.join(timeout)

        if process.is_alive():
            process.terminate()
            process.join()
            results.append(False)  # Timeout counts as failure
        else:
            try:
                result = output_queue.get_nowait()
                results.append(result)
            except Exception:
                results.append(False)
    return results


def run_testcase(func_str, timeout=5) -> int:
    """
    Executes the function definition and test case from the string with a specified timeout.

    Args:
        func_str (str): A string containing the function definition and test cases.
        timeout (int): The number of seconds to wait before timing out (default is 5 seconds).

    Returns:
        bool: True if the code executes successfully within the timeout,
              False if there's an error or a timeout occurs.
    """
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit the target function with func_str as an argument
        future = executor.submit(target, func_str)
        try:
            # Wait for the function to complete within the timeout
            future.result(timeout=timeout)
            return 1
        except concurrent.futures.TimeoutError:
            # print(f"Function execution timed out for \n : {func_str}")
            return 0
        except Exception as e:
            # Catch any other exceptions that may occur
            # print(f"An unexpected error occurred: {e} for \n : {func_str}")
            return 0

def run_single_test_subprocess(code_str: str, test_str: str) -> Tuple[bool, int, int, str]:
    # if 'import subprocess' in code_str:
    #     return (False, 0, 1, 'subprocess')
    with tempfile.TemporaryDirectory() as temp_dir:
        # temp_dir = '/home/hamed/PycharmProjects/hallucination/temp_dir2/'
        code_path = os.path.join(temp_dir, 'code.py')
        test_path = os.path.join(temp_dir, 'test.py')
        # Write the code under test to code.py
        with open(code_path, 'w') as f:
            f.write(code_str)

        # Ensure that test.py can import code.py by adjusting sys.path
        injected_import = "from code import *\n\n"
        # Write the test cases to test.py
        with open(test_path, 'w') as f:
            f.write(injected_import)
            f.write(test_str)

        # Define resource limits
        def set_resource_limits():
            try:
                # Limit CPU time (e.g., 120 seconds)
                cpu_time_limit = 30  # seconds
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))

                # Limit memory usage (e.g., 1 GB)
                ram_limit = 4096 * 1024 * 1024  # bytes
                resource.setrlimit(resource.RLIMIT_AS, (ram_limit, ram_limit))
            except Exception as e:
                print(f"Error setting resource limits: {e}", file=sys.stderr)
                sys.exit(1)

        try:
            current_dir = os.getcwd()
            bigcode_python_path = current_dir + '/.bigcode_venv/bin/python3'
            # Execute the tests using subprocess
            completed_process = subprocess.run(
                f'{bigcode_python_path} -m unittest test.py',
                shell=True,
                # check=True,
                cwd=temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=set_resource_limits,
                timeout=300  # Wall-clock timeout in seconds
            )
            output = completed_process.stdout.decode('utf-8')
            output_stderr = completed_process.stderr.decode("utf-8")
            # print(output_stderr)
            # print(output)
            # Print the captured output
            # print("Subprocess Output:")
            # print(output)

            # Parse the unittest output to determine pass/fail counts
            all_passed = completed_process.returncode == 0
            passed, failed = parse_unittest_output(output_stderr)
            # print(passed, failed)
            # print('*'*100)
            return (all_passed, passed, failed, output_stderr)

        except subprocess.TimeoutExpired:
            return (False, 0, 1, "Test execution timed out.")
        except Exception as e:
            return (False, 0, 1, f"Exception during test execution: {e}")

def parse_unittest_output(output: str) -> Tuple[int, int]:
    """
    Parses the unittest output to extract the number of tests passed and failed.

    Args:
        output (str): The stdout and stderr output from the unittest execution.

    Returns:
        Tuple[int, int]: Number of tests passed and failed.
    """
    passed = 0
    failed = 0
    try:
        # Example lines to parse:
        # OK (tests=3)
        # FAILED (failures=1, errors=0)
        if "OK" in output:
            # Extract the number of tests
            start = output.find('(')
            end = output.find(')', start)
            if start != -1 and end != -1:
                tests_info = output[start+1:end]
                tests_run = int(tests_info.split('=')[1].strip(')'))
                passed = tests_run
                failed = 0
        elif "FAILED" in output:
            # Extract failures and errors
            start = output.find('(')
            end = output.find(')', start)
            if start != -1 and end != -1:
                failures_info = output[start+1:end]
                parts = failures_info.split(',')
                failures = int(parts[0].split('=')[1])
                errors = int(parts[1].split('=')[1])
                failed = failures + errors
                # Assuming all other tests passed
                # You might need to adjust this if more details are available
                # For simplicity, set passed as tests_run - failed
                # Here, tests_run is not directly available, so set passed to 0
                passed = 0  # Alternatively, improve parsing to get exact counts
    except Exception:
        # If parsing fails, default to 0 passed and 1 failed
        passed = 0
        failed = 1
    return (passed, failed)

def worker_wrapper_subprocess(test_str: str, code_str: str) -> Tuple[bool, str]:
    """
    A wrapper that calls run_single_test_subprocess and returns whether all tests passed.

    Args:
        test_str (str): The test code string.
        code_str (str): The code string under test.

    Returns:
        tuple: (all_passed, output) where `all_passed` is True if all tests passed.
    """
    all_passed, passed, failed, output = run_single_test_subprocess(code_str, test_str)
    return all_passed, passed, failed, output

def pool_worker_subprocess(arg_tuple):
    """
    Unpack arguments and call worker_wrapper_subprocess.

    Args:
        arg_tuple (tuple): Tuple of (test_str, code_str)

    Returns:
        tuple: Result from worker_wrapper_subprocess.
    """
    test_str, code_str = arg_tuple
    return worker_wrapper_subprocess(test_str, code_str)


def run_unit_tests_sequential(code_str: str, test_list: List[str]):
    results = []
    for test in test_list:
        result = run_single_test_subprocess(code_str, test)
        results.append(result)
    return results

def run_unit_tests_parallel(code_str: str, test_list: List[str]):
    """
    Runs multiple test case strings in parallel for a given code snippet using subprocess.

    Args:
        code_str (str): The Python code under test.
        test_list (List[str]): List of test case code strings.

    Returns:
        List[tuple]: A list where each element is a tuple (all_passed, output)
                     indicating if the corresponding test case passed and containing
                     the detailed test output.
    """
    # multiprocessing.set_start_method('spawn')
    # print(f'Processing {len(test_list)} test cases...')
    args = [(test_str, code_str) for test_str in test_list]
    with multiprocessing.Pool(5) as pool:
        results = pool.map_async(pool_worker_subprocess, args)
        try:
            # Set a reasonable timeout for all tests to complete
            results = results.get(timeout=40)  # e.g., 5 minutes
        except multiprocessing.TimeoutError:
            print("Timeout while waiting for worker processes to finish.")
            pool.terminate()
            pool.join()
            # Assign failure to all pending tests
            results = [(False,0, 1, "Test execution timed out.") for _ in test_list]
        except BrokenPipeError:
            print('Broken pipe error')
            return [(False, 0, 1, "Broken Pipe Error") for _ in test_list]
        except Exception as e:
            return [(False, 0, 1, e.__str__()) for _ in test_list]
    return results
