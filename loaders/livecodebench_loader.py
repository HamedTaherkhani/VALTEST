from datasets import load_dataset
from tqdm import tqdm
import ast
from function_executor import run_testcase


def extract_function_signature(func_str):
    try:
        # Parse the string into an AST node
        tree = ast.parse(func_str)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                # Get the function name
                func_name = node.name
                # Get the function arguments
                args = [
                    f"{arg.arg}: {ast.unparse(arg.annotation)}" if arg.annotation else arg.arg
                    for arg in node.args.args
                ]
                # Get the return type if available
                return_annotation = (
                    f" -> {ast.unparse(node.returns)}" if node.returns else ""
                )
                # Construct the signature
                signature = f"def {func_name}({', '.join(args)}){return_annotation}"
                return signature
    except Exception as e:
        raise e
    return None
imports = """
from typing import List, Tuple, DefaultDict, Dict, Any, Optional, Set
import math
from collections import Counter, defaultdict, deque, namedtuple
"""
def extract_function_name(func_str):
    try:
        # Parse the string into an AST node
        tree = ast.parse(func_str)
        # Iterate through the body of the parsed tree
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                return node.name  # Return the first function name found
    except Exception as e:
        raise e
    return None

class LiveCodeBenchLoader:
    def __init__(self, instances=None):
        print('loading livecodebench...')
        # self.execution = load_dataset('livecodebench/execution-v2', split='test')
        self.codegen = load_dataset("livecodebench/code_generation", version_tag="release_v6", split='test')
        print(self.codegen)
        if instances is not None:
            if len(instances) > 0:
                # self.execution = [inst for idx, inst in enumerate(self.execution) if idx in instances]
                self.codegen = [inst for idx, inst in enumerate(self.codegen) if idx in instances]

        prompts = []
        solutions = []
        for cod in tqdm(self.codegen):
                try:
                    func_name = extract_function_name(cod['code'])
                except Exception as e:
                    print('error')
                    continue
                if func_name is None:
                    print('error')
                    continue
                # print(ex['code'])
                # print(func_name)
                # print('---------------------------------------------')
                if str(cod['question_id']) == str(cod['question_id']):
                    try:
                        func_sig = extract_function_signature(cod['code'])
                    except Exception as e:
                        print('error in extract_function_signature')
                        continue
                    sol = imports + cod['code']
                    # print(res)
                    prompts.append(imports + func_sig + ":\n" + " \"\"\"" +cod['question_content'] + "\"\"\"")
                    solutions.append(sol)
                    # print('found')
                    break
        print(len(prompts))
        print(len(solutions))
        self.prompts = prompts
        self.solutions = solutions

    def get_prompts(self):
        return self.prompts

    def get_solutions(self):
        return self.solutions