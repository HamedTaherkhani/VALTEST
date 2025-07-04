# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Dict
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile
from function_executor import run_unit_tests_parallel, run_unit_tests_sequential

def _pack_test_cases(test_cases, timeout):
    blank_4 = ' ' * 4
    blank_8 = ' ' * 8
    blank_12 = ' ' * 12
    result = f'def check():\n    pass_result = []\n'
    for idx, tc in enumerate(test_cases):
        multi_line_assertion = tc.strip().replace('\n', f'\n{blank_12}')
        result += f'\n{blank_4}try:\n{blank_8}with time_limit({timeout}):\n{blank_12}{multi_line_assertion}\
                    \n{blank_12}pass_result.append(True)\n{blank_4}except Exception as e:\n{blank_8}pass_result.append(False)\n'
    result += '\n    return pass_result\n'
    result += f'\nglobal final_result\nfinal_result = check()'
    return result


def check_correctness_with_test_cases(task_id, prompt, completion, test_cases, timeout,is_unittest=False):
    """
    Evaluates the functional correctness of a solution_content by running the test
    suite provided in the problem. 
    """
    extend_timeout = timeout*len(test_cases)

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()
            actual_tests = [test[0] for test in test_cases]
            # Construct the check program and run it.
            check_program = (
                prompt + "\n" +completion + "\n" +
                _pack_test_cases(actual_tests, timeout)
            )

            try:
                exec_globals = {'time_limit': time_limit}
                with swallow_io():
                    exec(check_program, exec_globals)
                result.append(exec_globals['final_result'])
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
    if is_unittest:
        results = run_unit_tests_parallel(code_str=completion, test_list=test_cases)
        print([a[0] for a in results])
        return dict(
            task_id=task_id,
            test_cases=test_cases,
            completion=completion,
            passed= [res[0] for res in results],
            result=[res[3] for res in results],
        )
    else:
        manager = multiprocessing.Manager()
        result = manager.list()

        p = multiprocessing.Process(target=unsafe_execute)
        p.start()
        p.join(timeout=extend_timeout + 0.1)
        if p.is_alive():
            p.kill()

        if not result:
            result.append("timed out")

        return dict(
            task_id=task_id,
            test_cases=test_cases,
            completion=completion,
            passed=(type(result[0]) == list) and len(result[0]) > 0,
            result=result[0],
        )

def check_correctness(task_id: str, prompt: str, completion: str, test: str, entry_point: str, timeout: float, is_unittest:bool) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 
    """

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            a = prompt + completion + "\n" + test
            check_program = (
                prompt + '\n' + completion + "\n" + test
            )

            # print(check_program)
            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        exec(check_program, exec_globals)
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")
            # print(result[-1])
            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
    if is_unittest:
        # results = run_unit_tests_sequential(code_str=completion, test_list=[test])
        results = run_unit_tests_parallel(code_str=completion, test_list=[test])
        # print(results)
        return dict(
            task_id=task_id,
            passed = results[0][0],
            completion=completion,
            result=results[0][3],
        )
    else:
        manager = multiprocessing.Manager()
        result = manager.list()

        p = multiprocessing.Process(target=unsafe_execute)
        p.start()
        p.join(timeout=timeout+1)
        if p.is_alive():
            p.kill()

        if not result:
            result.append("timed out")

        return dict(
            task_id=task_id,
            passed=result[0] == "passed",
            result=result[0],
            completion=completion,
        )

@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = 'stdin'


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the 
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == 'Darwin':
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins
    builtins.exit = None
    builtins.quit = None

    import os
    os.environ['OMP_NUM_THREADS'] = '1'

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil
    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess
    subprocess.Popen = None  # type: ignore

    __builtins__['help'] = None

    import sys
    sys.modules['ipdb'] = None
    sys.modules['joblib'] = None
    sys.modules['resource'] = None
    sys.modules['psutil'] = None
    sys.modules['tkinter'] = None
