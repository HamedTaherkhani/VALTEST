
import signal
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    return False

def run_testcase(func_str, timeout=5):
    """
    Executes the function definition from the string with a specified timeout.

    Args:
    - func_str (str): A string containing the function definition.
    - timeout (int): The number of seconds to wait before timing out (default is 5 seconds).

    Returns:
    - bool: True if the function is defined successfully, False if there's an error or a timeout occurs.
    """
    # Set up the alarm signal to handle timeouts
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)  # Set the timeout
    # print(func_str)
    local_vars = {}
    try:
        # print(func_str)
        # print(func_str)
        # func_str = '''def first_repeated_char(str1):
        #     for index, c in enumerate(str1):
        #         if str1[:index + 1].count(c) > 1:
        #             return c'''
        # print(func_str)
        exec(func_str, {}, local_vars)
        signal.alarm(0)  # Disable the alarm after successful execution
        return True
    except TimeoutException as e:
        print(e)
        raise False
    except Exception as e:
        print(e)
        return False
    finally:
        signal.alarm(0)  # Ensure the alarm is disabled