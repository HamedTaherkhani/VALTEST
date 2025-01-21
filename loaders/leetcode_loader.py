# from log_probs import run_testcase
from sys import implementation
from tqdm import tqdm
from datasets import load_dataset
from function_executor import run_testcase, TimeoutException
import re
import ast
import os
import textwrap
from evaluate import load
import pickle
import signal


def normalize_name(name):
    """Normalize function names for comparison."""
    return name.lower().replace('_', '')

class FunctionFinder(ast.NodeVisitor):
    def __init__(self, function_names, source_lines):
        # Normalize all function names for comparison
        self.function_names = set(normalize_name(name) for name in function_names)
        self.source_lines = source_lines
        self.functions = []

    def visit_FunctionDef(self, node):
        if normalize_name(node.name) in self.function_names:
            # Get the function code
            start_line = node.lineno - 1  # lineno is 1-indexed
            if hasattr(node, 'end_lineno'):
                end_line = node.end_lineno
            else:
                end_line = self.find_end_line(node)
            function_lines = self.source_lines[start_line:end_line]
            # Normalize indentation
            function_code = self.dedent_code(function_lines)
            self.functions.append((node.name, function_code))
        # Continue visiting nested functions and methods
        self.generic_visit(node)

    def find_end_line(self, node):
        """Recursively find the maximum end line number of a node."""
        max_lineno = node.lineno
        for child in ast.iter_child_nodes(node):
            if hasattr(child, 'lineno'):
                max_lineno = max(max_lineno, self.find_end_line(child))
        return max_lineno

    def dedent_code(self, lines):
        """Remove common leading whitespace from each line in lines."""
        dedented_code = textwrap.dedent('\n'.join(lines))
        return dedented_code

def find_functions_with_names(directory, function_names):
    functions_found = {}
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                # print(filepath)
                with open(filepath, 'r', encoding='utf-8') as f:
                    source = f.read()
                source_lines = source.splitlines()
                # Parse the source code
                try:
                    tree = ast.parse(source, filename=filepath)
                except SyntaxError as e:
                    print(f"Syntax error in file {filepath}: {e}")
                    continue
                # Create a FunctionFinder instance
                finder = FunctionFinder(function_names, source_lines)
                finder.visit(tree)
                # Append found functions to the list
                for func_name, function_code in finder.functions:
                    functions_found[func_name] = (filepath, function_code)
    return functions_found


def convert_camel_to_snake(code):
    # Function to convert a camel case string to snake case
    def camel_to_snake(name):
        return re.sub(r'([a-z])([A-Z])', r'\1_\2', name).lower()

    # Find all function names in the code
    function_names = re.findall(r'def (\w+)\(', code)

    # Convert each function name found to snake case
    converted_names = {name: camel_to_snake(name) for name in function_names}

    # Replace function definitions
    for old_name, new_name in converted_names.items():
        code = re.sub(rf'\b{old_name}\b', new_name, code)

    return code


def validate_test_and_remove_keywords_from_code(code):
    import ast

    class KeywordRemover(ast.NodeTransformer):
        def visit_Call(self, node):
            # Recursively process all child nodes first
            self.generic_visit(node)
            # Append the values of keyword arguments to args
            for kw in node.keywords:
                node.args.append(kw.value)
            node.keywords = []
            return node
    try:
        # Parse the code into an AST
        tree = ast.parse(code)
        # Transform the AST to remove keywords
        tree = KeywordRemover().visit(tree)
        # Fix any locations in the AST that might have been disrupted
        ast.fix_missing_locations(tree)
        # Unparse the AST back into code
        new_code = ast.unparse(tree)
    except SyntaxError as e:
        print(f"Syntax error")
        return None
    return new_code


class LeetCodeLoader:

    def __init__(self):
        leet_code = load_dataset('greengerong/leetcode')
        self.dataset = leet_code['train'] ## 2360
        print(len(self.dataset))
        self.dataset = [s for idx,s in enumerate(self.dataset) if 'python' in s['python'] and idx not in [2129, 2128]]## 2350
        print(len(self.dataset))
        function_names = []
        prompts = []
        solutions = []
        testcases = []
        for idx, instance in enumerate(self.dataset):
            if idx in [1355, 2017, 2127, 2172, 2206, 2271]:
                continue
            print(idx)
            prompt = self.extract_function_signature(instance['python']) + "\n" + "\t\"\"\" \n" + instance['content'] + "\"\"\""
            func_name = self.extract_function_name(instance['python'])
            testcase =self.get_test_cases(func_name, instance['content'])
            solution = self.extract_python_code(instance['python'])
            code = solution + '\n\n' + '\n'.join(testcase)
            is_passed = run_testcase(code, 5)
            if solution is not None and len(testcase) != 0 and is_passed:
                solutions.append(solution)
                function_names.append(func_name)
                testcases.append(testcase)
                prompts.append(prompt)
        # function_with_names = find_functions_with_names('LeetCode/solutions/', function_names)
        print(len(function_names))
        self.testcases = testcases
        self.prompts = prompts
        self.solutions = solutions
        self.function_names = function_names
    @staticmethod
    def extract_python_code(text):
        """
        Extracts Python code enclosed between ```python and ``` markers.
        """
        import_list = """from typing import List, Dict, Tuple
from collections import defaultdict
import math
import sys
import os
import collections
"""
        match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if match:
            implementation1 =  match.group(1).strip()
            return import_list + '\n' +  implementation1
        return None
        # implementation2 = None
        # implementation1 = None
        # # Use regex to find the code between ```python and ```
        # try:
        #     implementation1 = function_with_names[func_name][1]
        # except KeyError:
        #     match = re.search(r'```python(.*?)```', text, re.DOTALL)
        #     # If a match is found, return the code without the markers
        #     if match:
        #         implementation2 =  match.group(1).strip()
        # # print(implementation1)
        # # print(implementation2)
        #
        # return implementation2
        # if implementation2 is not None:
        #     code = implementation2 + '\n\n' + '\n'.join(testcases)
        #     is_passed = run_testcase(code, 5)
        #     if is_passed:
        #         return implementation2
        # return implementation1


    @staticmethod
    def extract_function_signature(code):
        # Regex pattern to capture the entire function signature, including multi-line definitions and type hints
        pattern = r'def\s+[a-zA-Z_]\w*\s*\([^()]*?(?:\([^()]*\)[^()]*?)*\)(?:\s*->\s*[\w\[\], \'"]+)?\s*:'
        # Find all matches of the pattern in the code
        signatures = re.findall(pattern, code, re.DOTALL)
        # Exclude __init__ if it's not desired and select the first valid match
        signature = [f for f in signatures if '__init__' not in f]
        return signature[0] if signature else None

    @staticmethod
    def extract_function_name(code):
        # Regex pattern to match function names in Python
        pattern = r'def\s+([a-zA-Z_]\w*)\s*\('
        # Find all matches of the pattern in the code
        function_names = re.findall(pattern, code)
        # Exclude __init__ if it's not desired and select the first valid match
        function_name = [f for f in function_names if '__init__' not in f]
        return function_name[0] if function_name else None

    @staticmethod
    def get_test_cases(function_name, context):
        input_sections = re.findall(r"\*\*Input:\*\*(.*?)\*\*Output:\*\*", context, re.DOTALL)
        output_sections = re.findall(r"\*\*Output:\*\*(.*?)\*\*", context, re.DOTALL)

        test_cases = []
        for inputs, output in zip(input_sections, output_sections):
            inputs = inputs.strip()
            output = output.strip()
            if inputs.endswith(' \"'):
                inputs = inputs[:-2] + '\"'
            if output.endswith(' \"'):
                output = output[:-2] + '\"'
            test_case = f"assert {function_name}({inputs}) == {output}"
            cleaned_test = test_case.replace("\\", "")
            testcase = validate_test_and_remove_keywords_from_code(cleaned_test)
            if testcase is not None:
                test_cases.append(testcase)

        return test_cases

    # def validate_testcases(self):


    def get_testcases(self):
        return self.testcases

    # def get_dataset(self):
    #     return self.dataset

    def get_prompts(self):
        return self.prompts

    def get_solutions(self):
        return self.solutions

    def get_function_names(self):
        return self.function_names

    def validate_dataset(self):
        falses = 0
        trues = 0
        for index, sol in enumerate(self.solutions):
            if index in (2002,2044, 2139):
                continue
            print(index)
            code = sol + '\n\n' + '\n'.join(self.testcases[index])

            is_passed = run_testcase(code, 5)
            print(is_passed)
            # print(code)
            if is_passed:
                trues +=1
            else:
                # print('-----------------------------------------------------')
                # print(code)
                falses +=1
            # print(is_passed)
            # if not is_passed:
            #     print(code)
        print(f'total passes :{trues}')
        print(f'total errors :{falses}')