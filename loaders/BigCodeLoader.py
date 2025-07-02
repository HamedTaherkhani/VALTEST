import os

from datasets import load_dataset
import re
import json

def find_import_statements(python_code: str) -> list:
    # Regular expressions to match both 'import' and 'from ... import ...' statements
    import_pattern = r'^\s*import\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?'
    from_import_pattern = r'^\s*from\s+[a-zA-Z_][a-zA-Z0-9_]*\s+import\s+[a-zA-Z_][a-zA-Z0-9_]*'

    # Find all matches using the regex patterns
    imports = re.findall(import_pattern, python_code, re.MULTILINE)
    from_imports = re.findall(from_import_pattern, python_code, re.MULTILINE)

    # Combine both types of imports into a single list
    return imports + from_imports + ['import matplotlib as plt', 'import numpy as np', 'import pandas as pd']

def extract_function_signature(code_string):
    # Regular expression to capture the function signature
    signature_regex = r"def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\([^)]*\)"

    # Search for the function signature in the given code string
    match = re.search(signature_regex, code_string)

    if match:
        return match.group(0)
    else:
        return None

class BigCodeLoader:
    def __init__(self,hard=1):
        # ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
        if hard == 1:
            ds = load_dataset("bigcode/bigcodebench-hard", split="v0.1.2")
        else:
            data = []
            with open(f'{os.getcwd()}/loaders/bigcodebench_subset.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    data.append(json.loads(line.strip()))
            ids = [item['name'] for item in data]
            all_ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
            ds = []
            ids_ = []
            for item in all_ds:
                if item['task_id'] in ids:
                    ds.append(item)
                    ids_.append(item['task_id'])
            self.ids = ids_
        self.prompts = []
        self.dataset = ds
        self.solutions = []
        self.libs = []
        self.all_imports = []
        self.tests = []
        for item in ds:
            # if 'file' in item['complete_prompt']:
            #     continue
            self.prompts.append(item['complete_prompt'])
            # print(extract_function_signature(item['complete_prompt']))
            imports = find_import_statements(item['complete_prompt'])
            self.libs.append(item['libs'])
            for imp in imports:
                if imp not in self.all_imports:
                    self.all_imports.append(imp)
            # self.all_imports.extend(imports)
            # print(imports)
            # print('-----------------------------')
            if len(imports) == 0:
                print(item['complete_prompt'])
            # self.solutions.append('\n'.join(imports) + '\n' + extract_function_signature(item['complete_prompt'])+ ":\n" + item['canonical_solution'])
            imports_text = '\n'.join(imports) + '\n'
            try:
                sol = imports_text + item['instruct_prompt'].split('```')[1] + item['canonical_solution']
            except Exception as e:
                sol = imports_text + item['instruct_prompt'] + item['canonical_solution']
            self.solutions.append(sol)
            self.tests.append(item['test'])
    def get_prompts(self):
        return self.prompts

    def get_tests(self):
        return self.tests
    #
    # def get_func_names(self):
    #     return self.func_names

    def get_dataset(self):
        return self.dataset

    def get_solutions(self):
        return self.solutions