import concurrent.futures
import sys
import signal
from typing import List
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