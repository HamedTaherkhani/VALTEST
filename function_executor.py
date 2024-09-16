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