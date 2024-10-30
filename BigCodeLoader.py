from datasets import load_dataset
import re

def find_import_statements(python_code: str) -> list:
    # Regular expressions to match both 'import' and 'from ... import ...' statements
    import_pattern = r'^\s*import\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s+as\s+[a-zA-Z_][a-zA-Z0-9_]*)?'
    from_import_pattern = r'^\s*from\s+[a-zA-Z_][a-zA-Z0-9_]*\s+import\s+[a-zA-Z_][a-zA-Z0-9_]*'

    # Find all matches using the regex patterns
    imports = re.findall(import_pattern, python_code, re.MULTILINE)
    from_imports = re.findall(from_import_pattern, python_code, re.MULTILINE)

    # Combine both types of imports into a single list
    return imports + from_imports + ['import matplotlib as plt']

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
    def __init__(self):
        ds = load_dataset("bigcode/bigcodebench", split="v0.1.2")
        self.prompts = []
        self.dataset = ds
        self.solutions = []
        self.libs = []
        self.all_imports = []
        for item in ds:
            if 'file' in item['complete_prompt']:
                continue
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
            self.solutions.append('\n'.join(imports) + '\n' + extract_function_signature(item['complete_prompt'])+ ":\n" + item['canonical_solution'])
    def get_prompts(self):
        return self.prompts

    # def get_tests(self):
    #     return self.tests
    #
    # def get_func_names(self):
    #     return self.func_names

    def get_dataset(self):
        return self.dataset

    def get_solutions(self):
        return self.solutions