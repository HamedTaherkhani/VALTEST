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

def run_testcases(func_str_list: List[str], timeout=5) -> List[int]:
    """
    Executes multiple function definitions and test cases from strings with a specified timeout.

    Args:
        func_str_list (List[str]): A list of strings, each containing the function definition and test cases.
        timeout (int): The number of seconds to wait before timing out (default is 5 seconds).

    Returns:
        List[int]: A list of results corresponding to each func_str.
                   1 if the code executes successfully within the timeout,
                   0 if there's an error or a timeout occurs.
    """
    results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit all tasks to the executor
        futures = {executor.submit(target, func_str): func_str for func_str in func_str_list}

        # Collect the results as they complete
        for future in concurrent.futures.as_completed(futures):
            func_str = futures[future]
            try:
                # Get the result with the specified timeout
                future.result(timeout=timeout)
                results.append(1)
            except concurrent.futures.TimeoutError:
                # The function timed out
                results.append(0)
            except Exception as e:
                # An unexpected error occurred
                results.append(0)
    return results

