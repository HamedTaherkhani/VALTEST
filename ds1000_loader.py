from datasets import load_dataset
import re
from function_executor import run_testcase
import ast
import builtins
import autopep8



import ast
import astor

def ensure_return_statement(code):
    """
    Ensures that each function in the provided code has a return statement.
    If a function lacks a return statement, appends 'return <last_variable>'
    where <last_variable> is the last assigned variable in the function.

    :param code: str, Python code containing function definitions
    :return: str, Modified Python code with necessary return statements
    """
    class ReturnInjector(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            self.generic_visit(node)  # Process nested functions if any

            # Check if any return statement exists in the function
            has_return = any(isinstance(stmt, ast.Return) for stmt in ast.walk(node))

            if not has_return:
                # Find the last assigned variable in the function body
                last_var = None
                for stmt in reversed(node.body):
                    if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                        last_var = stmt.targets[0].id
                        break
                    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                        last_var = stmt.target.id
                        break

                if last_var:
                    # Create a return statement
                    return_stmt = ast.Return(value=ast.Name(id=last_var, ctx=ast.Load()))
                    node.body.append(return_stmt)
                else:
                    raise ValueError(f"No assignable variable found to return in function '{node.name}'.")

            return node

    try:
        # Parse the original code into an AST
        tree = ast.parse(code)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python code provided: {e}")

    # Transform the AST to inject return statements where needed
    transformer = ReturnInjector()
    modified_tree = transformer.visit(tree)
    ast.fix_missing_locations(modified_tree)

    # Convert the modified AST back to Python code
    modified_code = astor.to_source(modified_tree)
    return modified_code


def find_undeclared_variables(code_str):
    tree = ast.parse(code_str)
    assigned_vars = set()
    imported_vars = set()
    used_vars = set()
    built_in_vars = set(dir(builtins))

    class Analyzer(ast.NodeVisitor):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                assigned_vars.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)
            self.generic_visit(node)

        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_vars.add(name.split('.')[0])
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imported_vars.add(name)
            self.generic_visit(node)

        def visit_For(self, node):
            # Handle loop variables as assigned
            targets = self._get_targets(node.target)
            assigned_vars.update(targets)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # Function name is assigned
            assigned_vars.add(node.name)
            # Function arguments are assigned
            for arg in node.args.args:
                assigned_vars.add(arg.arg)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            # Class name is assigned
            assigned_vars.add(node.name)
            self.generic_visit(node)

        def visit_With(self, node):
            for item in node.items:
                if item.optional_vars:
                    targets = self._get_targets(item.optional_vars)
                    assigned_vars.update(targets)
            self.generic_visit(node)

        def visit_ListComp(self, node):
            for gen in node.generators:
                targets = self._get_targets(gen.target)
                assigned_vars.update(targets)
            self.generic_visit(node)

        def visit_DictComp(self, node):
            for gen in node.generators:
                targets = self._get_targets(gen.target)
                assigned_vars.update(targets)
            self.generic_visit(node)

        def visit_SetComp(self, node):
            for gen in node.generators:
                targets = self._get_targets(gen.target)
                assigned_vars.update(targets)
            self.generic_visit(node)

        def visit_GeneratorExp(self, node):
            for gen in node.generators:
                targets = self._get_targets(gen.target)
                assigned_vars.update(targets)
            self.generic_visit(node)

        def _get_targets(self, node):
            """Helper method to extract variable names from targets."""
            targets = set()
            if isinstance(node, ast.Name):
                targets.add(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for elt in node.elts:
                    targets.update(self._get_targets(elt))
            return targets

    Analyzer().visit(tree)
    undeclared_vars = used_vars - assigned_vars - imported_vars - built_in_vars
    return undeclared_vars

def correct_indentation(code_str, indent_size=4):
    """
    Corrects the indentation of the provided Python code string.

    Parameters:
        code_str (str): The Python code as a string.
        indent_size (int): Number of spaces for indentation (default is 4).

    Returns:
        str: The code with corrected indentation.
    """
    try:
        # Configure autopep8 options
        options = {
            'indent_size': indent_size,
            'aggressive': 1,  # Increase aggressiveness for better formatting
        }
        # Format the code
        formatted_code = autopep8.fix_code(code_str, options=options)
        return formatted_code
    except Exception as e:
        return f"Error formatting code: {e}"


def extract_import_statements(text):
    """
    Extracts all import statements from the given text.

    Parameters:
    text (str): The string containing code or text to search for import statements.

    Returns:
    list: A list of import statements found in the text.
    """
    # Regular expression patterns for 'import' and 'from ... import ...' statements
    import_pattern = r'^\s*import\s+[\w\.,\s]+'
    from_import_pattern = r'^\s*from\s+\w+\s+import\s+[\w\.,\s]+'

    # Find all matches for both patterns
    imports = re.findall(import_pattern, text, re.MULTILINE)
    from_imports = re.findall(from_import_pattern, text, re.MULTILINE)
    valid_imports = []

    for im in imports:
        temp = im.split('\n')
        for t in temp:
            if 'import' in t:
                valid_imports.append(t)

    for im in from_imports:
        temp = im.split('\n')
        for t in temp:
            if 'import' in t:
                valid_imports.append(t)
    valid_imports.extend(['import numpy as np', 'import pandas as pd', 'import datetime', 'import math'])
    return valid_imports


def get_signature(prompt, code_string):
    pattern = r"def\s+\w+\(.*\):"
    match = re.search(pattern, code_string)
    if match:
        prompt2 = match.group(0)
        return prompt2, True
        # print(prompt2)
    else:
        undeclared_variables = find_undeclared_variables(code_string)
        temp = ','.join(undeclared_variables)
        prompt2 = f'def g({temp})'
        return prompt2, False


def get_function_signature_from_string(func_str):
    # Regular expression to match a function definition
    pattern = r"def\s+(\w+)\s*\((.*?)\)\s*:"
    match = re.search(pattern, func_str)

    if match:
        func_name = match.group(1)
        params = match.group(2)
        return f"def {func_name}({params})"
    else:
        return None


def remove_specific_line(input_string):
    """
    Removes the line containing 'result = g(df.copy())' from the input string.

    Parameters:
    input_string (str): The string from which the specific line will be removed.

    Returns:
    str: The modified string without the specific line.
    """
    lines = input_string.splitlines()  # Split the string into lines
    # Filter out lines that exactly match 'result = g(df.copy())'
    filtered_lines = [line for line in lines if line.strip() not in (
    "result = g(df.copy())", "result = g(df.copy(), List)", 'result = g(df.copy(), thresh)', 'result = g(a.__copy__())',
    'result = g(A.__copy__(), B.__copy__())', 'result = g(x.__copy__(), row.__copy__(), col.__copy__())',
    'result = g(a.__copy__(), b.__copy__())', 'result = g(df_a.copy(), df_b.copy())', 'df = g(df.copy())',
    'result = g(corr.copy())', 'df = g(df)', 'result = g(df.copy(), row_list, column_list)')]
    return "\n".join(filtered_lines)

def add_four_spaces(text):
    lines = text.splitlines()
    indented_lines = ['    ' + line for line in lines]
    return '\n'.join(indented_lines)


def extract_code_content(text: str) -> str:
    # Use regular expression to find content between <code> and </code>
    pattern = r"<code>(.*?)<\/code>"
    match = re.search(pattern, text, re.DOTALL)

    if match:
        code = match.group(1).strip()
        lines = code.split('\n')

        # Filter out lines that contain 'load_data()'
        filtered_lines = [line for line in lines if 'load_data()' not in line]

        # Join the filtered lines back into a single string
        return '\n'.join(filtered_lines).strip()
    return ""


def get_code(prompt, code_string):
    code_string = remove_specific_line(code_string)
    code_string = correct_indentation(code_string)
    imports = extract_import_statements(prompt)
    if 'import matplotlib.pyplot as plt' in imports:
        raise Exception
    code = correct_indentation(code_string) + '\n' + '\n'.join(imports)
    signature, flag = get_signature(prompt, code)

    if not flag:
        final_code = '\n'.join(imports) + '\n' + signature.rstrip(' ').rstrip('\n') + ":\n" + add_four_spaces(code_string.lstrip(' ').lstrip('\n'))
    else:
        final_code = '\n'.join(imports) + "\n" + code_string
    final_code = ensure_return_statement(correct_indentation(final_code))
    sig = get_function_signature_from_string(final_code)
    if sig is None:
        sig = get_signature(prompt, code)[0]
        final_code = '\n'.join(imports) + '\n' + sig.rstrip(' ').rstrip('\n') + ":\n" + add_four_spaces(
            code_string.lstrip(' ').lstrip('\n'))
    return extract_code_content(prompt) + '\n' + final_code


def get_prompt(initial_prompt, code_string):
    if '# SOLUTION START' in initial_prompt:
        prompt1 = initial_prompt
    else:
        a1 = initial_prompt.split('\nA:\n<code>')
        if len(a1) == 1:
            a1 = initial_prompt.split('\nA:\n\n')
        if len(a1) == 1:
            a1 = initial_prompt.split('\nA:\n\n\n<code>')
        if len(a1) == 1:
            a1 = initial_prompt.split('A:\n\ncorrected')
        if len(a1) == 1:
            a1 = initial_prompt.split('A:\n\nrunnable code')
        if len(a1) == 1:
            a1 = initial_prompt.split('\nA:\n\nDelete any step')
        assert len(a1) == 2
        prompt1 = a1[0]
    signature = get_function_signature_from_string(code_string)
    return signature + ":\n" + "    \"\"\"" + prompt1 + "\n \"\"\""

class DS1000Loader:
    def __init__(self):
        self.dataset = load_dataset("xlangai/DS-1000")['test']
        function_names = []
        total_errors = 0
        prompts = []
        solutions = []
        testcases = []
        for idx, instance in enumerate(self.dataset):
            try:
                code = get_code(instance['prompt'], instance['reference_code'])
                res = run_testcase(code)
                if res == 0:
                    total_errors += 1
                    continue
                prompt = get_prompt(instance['prompt'], code)
                solutions.append(code)
                prompts.append(prompt)
            except IndentationError:
                total_errors += 1
                continue
            except Exception as e:
                total_errors += 1
                continue
        self.prompts = prompts
        self.solutions = solutions
        print(total_errors)

    def get_prompts(self):
        return self.prompts

    def get_solutions(self):
        return  self.solutions