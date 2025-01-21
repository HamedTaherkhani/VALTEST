import concurrent.futures
import resource
import signal
import multiprocessing
import unittest
import sys
import io
import inspect
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

def run_test_cases(func_code, test_cases, timeout=5): ## correct version
    """
    Runs test cases concurrently on the provided function code.

    Args:
        func_code (str): The function code as a string.
        test_cases (list): A list of assertion statements as strings.

    Returns:
        list: A list of booleans indicating if each test case passed (True) or failed (False).
        :param timeout:
    """
    # Create a namespace dictionary for the function code
    func_namespace = {}
    exec(func_code, func_namespace)

    def run_test_case(test_case_str):
        # Use a separate namespace for each test case to avoid side effects
        test_namespace = func_namespace.copy()
        try:
            # Execute the test case in the combined namespace
            exec(test_case_str, test_namespace)
            return True
        except AssertionError:
            return False
        except Exception as e:
            # print(f"Exception in test case '{test_case_str}': {e}")
            return False

    # Run test cases concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        try:
            futures = [executor.submit(run_test_case, test_case) for test_case in test_cases]
            results = [future.result(timeout=timeout) for future in futures]
        except TimeoutError as e:
            results = []
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

def run_single_test_inline(code_str: str, test_str: str) -> Tuple[bool, int, int, str]:
    """
    Executes provided Python code and its unittest test cases inline (without spawning
    another process). This function is intended to be run from a process pool worker.

    It also sets OS-level resource limits:
      - Timeout (wall clock) using SIGALRM.
      - CPU time (in seconds) via RLIMIT_CPU.
      - Memory (virtual address space) limit via RLIMIT_AS.

    Args:
        code_str (str): The Python code containing the function(s) to test.
        test_str (str): The Python code containing unittest test cases.

    Returns:
        Tuple[bool, int, int, str]:
            - A boolean indicating if all tests passed.
            - Number of tests passed.
            - Number of tests failed.
            - Detailed test output.
    """
    def alarm_handler(signum, frame):
        raise TimeoutError("Timeout reached during test execution.")

    # Set resource limits here:
    try:
        # Set wall-clock timeout (e.g., 10 seconds).
        timeout_seconds = 120
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(timeout_seconds)  # The alarm is scheduled

        # Limit CPU time (e.g., 5 seconds). The limit applies to CPU seconds.
        cpu_time_limit = 120  # seconds
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time_limit, cpu_time_limit))

        # Limit memory usage (e.g., 256 MB). This sets both soft and hard limits.
        ram_limit = 2048 * 1024 * 1024  # in bytes
        resource.setrlimit(resource.RLIMIT_AS, (ram_limit, ram_limit))
    except Exception as e:
        return (False, 0, 1, f"Exception setting resource limits: {e}")

    try:
        # Redirect stdout to capture unittest output.
        new_stdout = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = new_stdout

        # Create a local namespace.
        local_namespace = {}

        # Execute the code under test.
        exec(code_str, local_namespace)

        # Execute the test code.
        exec(test_str, local_namespace)

        # Discover all unittest.TestCase subclasses.
        test_cases = [
            cls for name, cls in local_namespace.items()
            if inspect.isclass(cls) and issubclass(cls, unittest.TestCase)
        ]
        if not test_cases:
            raise ValueError("No subclasses of unittest.TestCase found in test code.")

        # Load tests from discovered test cases.
        suite = unittest.TestSuite()
        for test_case in test_cases:
            suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_case))

        # Run the tests.
        runner = unittest.TextTestRunner(stream=new_stdout, verbosity=2)
        result = runner.run(suite)

        # Calculate passed/failed numbers.
        passed = result.testsRun - len(result.failures) - len(result.errors)
        failed = len(result.failures) + len(result.errors)
        all_passed = (failed == 0)

        # Collect output.
        output = new_stdout.getvalue()

        # Restore stdout.
        sys.stdout = old_stdout

        # Cancel the alarm now that execution is complete.
        signal.alarm(0)
        return (all_passed, passed, failed, output)
    except Exception as e:
        # Restore stdout and cancel alarm in case of exception.
        sys.stdout = sys.__stdout__
        signal.alarm(0)
        return (False, 0, 1, f"Exception during test execution: {e}")


def worker_wrapper(test_str: str, code_str: str) -> (bool, str):
    """
    A wrapper that calls run_single_test_inline and returns whether all tests passed.

    Args:
        test_str (str): The test code string.
        code_str (str): The code string under test.

    Returns:
        tuple: (all_passed, output) where `all_passed` is True if all tests passed.
    """
    all_passed, passed, failed, output = run_single_test_inline(code_str, test_str)

    # Optionally, log output here.
    return all_passed, output


def pool_worker(arg_tuple):
    """
    Unpack arguments and call worker_wrapper.

    Args:
        arg_tuple (tuple): Tuple of (test_str, code_str)

    Returns:
        tuple: Result from worker_wrapper.
    """
    test_str, code_str = arg_tuple
    return worker_wrapper(test_str, code_str)


def run_unit_tests_parallel(code_str: str, test_list: List[str]):
    """
    Runs multiple test case strings in parallel for a given code snippet.

    Args:
        code_str (str): The Python code under test.
        test_list (List[str]): List of test case code strings.

    Returns:
        List[tuple]: A list where each element is a tuple (all_passed, output)
                     indicating if the corresponding test case passed and containing
                     the detailed test output.
    """
    if 'matplotlib' in code_str:
        print('HEEEre')
        results = []
        for test_str in test_list:
            results.append(worker_wrapper(test_str, code_str))
    else:
        processes = len(test_list)
        args = [(test_str, code_str) for test_str in test_list]
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map_async(pool_worker, args)
            try:
                results = results.get(timeout=60)  # adjust timeout as needed
            except multiprocessing.TimeoutError:
                print("Timeout while waiting for worker processes to finish.")
                pool.terminate()
                pool.join()
    return results


