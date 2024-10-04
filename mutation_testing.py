import os
import tempfile
import subprocess
import sys
import shutil
import re
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


def perform_mutation_testing_for_functions(functions_with_tests):
    import os
    import tempfile
    import subprocess
    import sys
    import shutil
    import re
    # Create a temporary directory for the project
    # temp_dir = tempfile.mkdtemp()
    cwd = os.getcwd()
    temp_dir = cwd+'/temp2'
    try:
        # Set up environment variables
        os.environ['PYTHONPATH'] = temp_dir

        # Create project structure
        src_dir = os.path.join(temp_dir, 'src')
        tests_dir = os.path.join(temp_dir, 'tests')
        os.makedirs(src_dir)
        os.makedirs(tests_dir)

        # Write each function to its own module in src/
        function_modules = []
        for func_names, func_code, test_cases in functions_with_tests:
            if len(test_cases) == 0:
                continue
            if func_names[0].startswith('test_'):
                continue
            # if func_names[0] in ('factorize', 'get_odd_collatz', 'minPath'):
            #     print('here')
            #     continue
            module_names = func_names
            module_file_name = f'{module_names[0]}.py'
            module_file_path = os.path.join(src_dir, module_file_name)
            with open(module_file_path, 'w') as f:
                f.write(func_code.strip() + '\n')
            function_modules.append((module_names, test_cases))

        # Write test cases for each function in tests/
        for module_names, test_cases in function_modules:
            test_file_name = f'test_{module_names[0]}.py'
            test_file_path = os.path.join(tests_dir, test_file_name)
            with open(test_file_path, 'w') as f:
                # Collect imports from test cases
                imports = set()
                test_code_lines = []
                for test_case in test_cases:
                    lines = test_case.strip().split('\n')
                    for line in lines:
                        if line.strip().startswith('import ') or line.strip().startswith('from '):
                            imports.add(line.strip())
                        else:
                            test_code_lines.append(line.rstrip())
                imports.add('import math')
                imports.add('import numpy as np')
                imports.add('import scipy')
                imports.add('import pandas as pd')
                imports.add('import cmath')
                # Write imports
                for module_name in module_names:
                    f.write(f'from src.{module_names[0]} import {module_name}\n')

                for imp in imports:
                    f.write(f'{imp}\n')
                f.write('\n')
                # print(imports)
                # Write test function
                idx = 0
                f.write(f'def test_{module_name}():\n')
                for line in test_code_lines:
                    f.write(f'    {line}\n')


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

        # Run pytest to check tests, but proceed even if tests fail
        print("Running pytest to check tests...")
        pytest_result = subprocess.run([sys.executable, '-m', 'pytest' ,'-x', '--assert=plain'], capture_output=True, text=True, cwd=temp_dir)
        if pytest_result.returncode != 0:
            print("Tests failed. Output:")
            print(pytest_result.stdout)
            print(pytest_result.stderr)
            print("Proceeding with mutation testing despite test failures.\n")
        else:
            print("All tests passed.\n")
        mutmut_run = subprocess.run(f'mutmut run --paths-to-mutate=src/',
                                    capture_output=True, text=True, cwd=temp_dir, shell=True)
        result = mutmut_run.stdout
        # print(result)
        scores = find_mut_score(result)

        subprocess.run('rm -f .mutmut-cache', cwd=temp_dir, shell=True)

        print("Final Mutation Testing Results:")
        print(f"Total mutants: {scores.total}")
        print(f"Killed mutants: {scores.killed}")
        print(f"Survived mutants: {scores.survived}")
        print(f"Timeout mutants: {scores.timeout}")
        print(f"Suspicious mutants: {scores.suspicious}")
        print(f"Overall Mutation Score: {scores.get_mutation_score():.2f}%")
    finally:
        # sys.exit()
        # pass
        # Change back to original working directory
        os.chdir(cwd)
        # # # Clean up temporary directory
        shutil.rmtree(temp_dir)


def get_top_level_function_names(func_str):
    # Regular expression to match top-level function definitions (without leading indentation)
    pattern = r"^def\s+(\w+)\s*\((.*?)\)(?:\s*->\s*.*)?\s*:"

    # Find all matches
    matches = re.findall(pattern, func_str, re.MULTILINE)

    # Extract the function names from the matches
    if matches:
        func_names = [match[0] for match in matches]
        return func_names
    else:
        return []