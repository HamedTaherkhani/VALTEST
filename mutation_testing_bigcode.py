import os
import tempfile
import subprocess
import sys
import shutil
import re
import multiprocessing
import ast
from tqdm import tqdm

def show_mutant_diff(mutants_list):
    """Display the diffs for all generated mutants."""

    # print(mutants_list)
    print("\nGenerated Mutants and Their Diffs:\n")
    for mutant_id in mutants_list:
        print(f"Mutant ID: {mutant_id}")
        # Show the diff for each mutant
        diff_output = subprocess.run(['mutmut', 'show', mutant_id], capture_output=True, text=True)
        print(diff_output.stdout)  # This shows the actual diff of the mutant
        print("\n" + "=" * 50 + "\n")


class MutationScore:
    def __init__(self, killed, timeout, suspicious, survived, skipped):
        self.killed = killed
        self.timeout = timeout
        self.suspicious = suspicious
        self.survived = survived
        self.skipped = skipped
        self.total = killed + timeout + suspicious + survived + skipped
    def get_mutation_score(self):
        return self.killed / self.total

    def get_total_mutations(self):
        return self.total
    def __str__(self):
        return f"killed = {self.killed}\nsurvived = {self.survived} \nsuspicious = {self.suspicious}\nskipped= {self.skipped}\ntimeout = {self.timeout}"
def find_mut_score(log:str):
    lines = log.strip().split('\n')

    # Define a regex pattern to match the target lines
    pattern = re.compile(
        r'^[^\s]+\s+(\d+/\d+)\s+🎉\s+(\d+)\s+⏰\s+(\d+)\s+🤔\s+(\d+)\s+🙁\s+(\d+)\s+🔇\s+(\d+)$'
    )

    # Initialize a variable to store the last match
    last_match = None

    # Iterate through each line to find matches
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            last_match = match

    # Check if a match was found and extract the numbers
    if last_match:
        step, killed, timeout, suspicious, survived, skipped = last_match.groups()
        # print(step, killed, timeout, suspicious, survived, skipped)
        score = MutationScore(int(killed), int(timeout), int(suspicious), int(survived), int(skipped))
        return score
    return None

def worker_func(args):
    """
    A simple wrapper that unpacks arguments and calls
    `perform_mutation_testing_for_functions`.
    """
    func, dataset = args
    return perform_mutation_testing_for_functions_bigcode(func, dataset)


def perform_overall_mutation_testing_bigcodebench(functions_with_tests, dataset, chunk_size=10):
    # Split your functions into chunks

    total_mutants = 0
    total_killed = 0
    total_survived = 0
    total_timeout = 0
    total_suspicious = 0

    # 2) Use the top-level worker in the Pool
    #    For Windows compatibility, ensure this is guarded under "if __name__ == '__main__'"
    with multiprocessing.Pool(5) as pool:
        # imap gives an iterator of results we can loop over in sync with tqdm
        for (mutants, killed, survived, timeout, suspicious) in tqdm(
            pool.imap(worker_func, [(func,dataset) for func in functions_with_tests]),
            total=len(functions_with_tests),
            desc="Mutation Testing Chunks"
        ):
            # print(mutants, killed, survived, timeout, suspicious)
            total_mutants += mutants
            total_killed += killed
            total_survived += survived
            total_timeout += timeout
            total_suspicious += suspicious

    if total_mutants == 0:
        overall_mutation_score = 0.0
    else:
        overall_mutation_score = (total_killed / total_mutants) * 100

    print("Final Mutation Testing Results:")
    print(f"Total mutants: {total_mutants}")
    print(f"Killed mutants: {total_killed}")
    print(f"Survived mutants: {total_survived}")
    print(f"Timeout mutants: {total_timeout}")
    print(f"Suspicious mutants: {total_suspicious}")
    print(f"Overall Mutation Score: {overall_mutation_score:.2f}%")


def make_files_for_testing(temp_dir, function_with_tests, dataset):

    # Create project structure
    src_dir = os.path.join(temp_dir, 'src')
    tests_dir = os.path.join(temp_dir, 'tests')
    os.makedirs(src_dir)
    os.makedirs(tests_dir)
    # print(function_with_tests)
    func_names, solution, testcases = function_with_tests[0] , function_with_tests[1] , function_with_tests[2]
    # Write each function to its own module in src/
    function_module = []

    if len(testcases) == 0:
        return

    module_names = func_names
    module_file_name = f'{module_names[0]}.py'
    module_file_path = os.path.join(src_dir, module_file_name)
    with open(module_file_path, 'w') as f:
        f.write(solution.strip() + '\n')

    test_file_name = f'test_{module_names[0]}.py'
    test_file_path = os.path.join(tests_dir, test_file_name)
    with open(test_file_path, 'w') as f:
        for module_name in module_names:
            f.write(f'from src.{module_names[0]} import {module_name}\n')
        for test_case in testcases:
            f.write(test_case.strip() + '\n')
    ###
    module_file_name = f'__init__.py'
    src_path = os.path.join(src_dir, module_file_name)
    test_file_path = os.path.join(tests_dir, module_file_name)
    with open(src_path, 'w') as f:
        f.write('\n')
    with open(test_file_path, 'w') as f:
        f.write('\n')
    with open(temp_dir + module_file_name, 'w') as f:
        f.write('\n')
    os.chdir(temp_dir)
    return module_names[0]

def perform_mutation_testing_for_functions_bigcode(function_with_tests, dataset):
    # Create a temporary directory for the project
    temp_dir = tempfile.mkdtemp(prefix="mutation_test_")
    original_cwd = os.getcwd()
    original_pythonpath = os.environ.get('PYTHONPATH', '')

    # Ensure cleanup happens
    try:
        os.chdir(temp_dir)
        os.environ['PYTHONPATH'] = temp_dir
        mutation_counters = {
            'total_mutants': 0,
            'total_killed': 0,
            'total_timeout': 0,
            'total_suspicious': 0,
            'total_survived': 0
        }
        # Set up environment and create necessary files for testing
        module_name = make_files_for_testing(temp_dir, function_with_tests, dataset)

        # Run pytest to check tests, but proceed even if tests fail
        # print("Running pytest to check tests...")
        pytest_result = subprocess.run(
            [sys.executable, '-m', 'pytest', '-x', '--assert=plain'],
            capture_output=True,
            text=True,
            cwd=temp_dir
        )

        if pytest_result.returncode != 0:
            return (
            mutation_counters['total_mutants'],
            mutation_counters['total_killed'],
            mutation_counters['total_survived'],
            mutation_counters['total_timeout'],
            mutation_counters['total_suspicious']
            )
            # print("Tests failed. Output:")
            # print(pytest_result.stdout)
            # print(pytest_result.stderr)
            # print("Proceeding with mutation testing despite test failures.\n")
        else:
            pass
            # print("All tests passed.\n")

        # Initialize mutation testing counters

        # Helper function to clean mutmut cache
        def clean_mutmut_cache():
            cache_path = os.path.join(temp_dir, '.mutmut-cache')
            if os.path.exists(cache_path):
                os.remove(cache_path)

        # Run mutation testing for each function module

        try:
            env = os.environ.copy()
            env['MPLBACKEND'] = 'Agg'
            mutmut_run = subprocess.run(f'mutmut run --paths-to-mutate=src/{module_name}.py',
                                        capture_output=True, text=True, cwd=temp_dir, shell=True, timeout=240, env=env)
        except subprocess.TimeoutExpired:
            print(f"Timeout expired for mutmut on module {module_name}.")
            clean_mutmut_cache()
            mutation_counters['total_timeout'] += 1
            return (
                mutation_counters['total_mutants'],
                mutation_counters['total_killed'],
                mutation_counters['total_survived'],
                mutation_counters['total_timeout'],
                mutation_counters['total_suspicious']
            )

        result = mutmut_run.stdout
        # Parse the results
        scores = find_mut_score(result)
        if scores is None:
            print(f"Failed to parse mutmut results for module {module_name}.")
            clean_mutmut_cache()
            return (
                mutation_counters['total_mutants'],
                mutation_counters['total_killed'],
                mutation_counters['total_survived'],
                mutation_counters['total_timeout'],
                mutation_counters['total_suspicious']
            )


        # Update mutation counters
        mutation_counters['total_mutants'] += scores.total
        mutation_counters['total_killed'] += scores.killed
        mutation_counters['total_timeout'] += scores.timeout
        mutation_counters['total_suspicious'] += scores.suspicious
        mutation_counters['total_survived'] += scores.survived
        # print(mutation_counters)

        # Clean mutmut cache after processing
        clean_mutmut_cache()

    # Compute overall mutation score
        total_mutants = mutation_counters['total_mutants']
        if total_mutants == 0:
            overall_mutation_score = 0.0
        else:
            overall_mutation_score = (mutation_counters['total_killed'] / total_mutants) * 100

        # Display final mutation testing results
        # print("Final Mutation Testing Results:")
        # print(f"Total mutants: {total_mutants}")
        # print(f"Killed mutants: {mutation_counters['total_killed']}")
        # print(f"Survived mutants: {mutation_counters['total_survived']}")
        # print(f"Timeout mutants: {mutation_counters['total_timeout']}")
        # print(f"Suspicious mutants: {mutation_counters['total_suspicious']}")
        # print(f"Overall Mutation Score: {overall_mutation_score:.2f}%")

        return (
            mutation_counters['total_mutants'],
            mutation_counters['total_killed'],
            mutation_counters['total_survived'],
            mutation_counters['total_timeout'],
            mutation_counters['total_suspicious']
        )

    finally:
        # Restore original environment and working directory
        os.environ['PYTHONPATH'] = original_pythonpath
        os.chdir(original_cwd)
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)
        # print(f"Cleaned up temporary directory: {temp_dir}")



def get_top_level_function_names(source_code):
    """
    Parse the given source code and return a list of top-level function names.

    Parameters:
        source_code (str): A string containing Python source code.

    Returns:
        List[str]: A list of names for all top-level function definitions.
    """
    # Parse the source code into an AST
    tree = ast.parse(source_code)

    # Iterate through the top-level nodes in the AST
    top_level_funcs = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]

    return top_level_funcs