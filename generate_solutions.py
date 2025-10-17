import ast
import argparse
import sys
import re
import os
import multiprocessing
from tqdm import tqdm
from typing import List
from loaders.humaneval_loader import HumanEvalLoader
from loaders.MBPPLoader import MBPPLoader
from loaders.leetcode_loader import LeetCodeLoader
from log_probs import Function
from llm_requester import OpenaiRequester, HuggingfaceRequester, GeminiRequester, LLamaAPIRequester, AntropicRequester, FireworksAPIRequester
from loaders.livecodebench_loader import LiveCodeBenchLoader
from loaders.livecodebench_loader2 import LiveCodeBenchLoader2
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from loaders.BigCodeLoader import BigCodeLoader
import pickle
IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\n"

def correct_indentation(code):
    try:
        # Parse the code into an Abstract Syntax Tree
        tree = ast.parse(code)

        # Convert the AST back to code with proper indentation
        corrected_code = ast.unparse(tree)

        return corrected_code
    except SyntaxError as e:
        print("Syntax Error while parsing the code:", e)
        return None
    except Exception as e:
        print("An error occurred:", e)
        return None

def get_function_name(func_str):
    pattern = r"def\s+(\w+)\s*\((.*?)\)\s*:"
    match = re.search(pattern, func_str)

    if match:
        func_name = match.group(1)
        return func_name
    else:
        return None


def separate_python_code_blocks(text: str) -> List[str]:
    """
    Extracts all Python code blocks from a string.

    Args:
        text (str): The input text containing multiple Python code blocks.

    Returns:
        List[str]: A list of strings, each string being the content of a Python code block.
    """
    # Regular expression to match code blocks that start with ```python and end with ```
    pattern = r"```python\s*(.*?)```"

    # Use re.DOTALL to allow '.' to match newline characters
    code_blocks = re.findall(pattern, text, re.DOTALL)

    # Optionally, you can strip trailing or leading whitespace for each block
    return [block.strip() for block in code_blocks]


def is_syntax_valid(code: str, func_name) -> bool:
    try:
        ast.parse(code)
        if func_name is not None:
            if func_name in code:
                return True
            else:
                return False
        return True
    except Exception:
        return False


def is_syntax_valid2(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except Exception:
        return False

def parse_tests(tests: str):
    return [test.strip() for test in tests.splitlines() if "assert" in test]

def find_tests(prompt, functions):
    for func in functions:
        if func.prompt == prompt:
            return [t.text for t in func.testcases if t.prediction_is_valid == 1]
    return ['No Tests']

def process_dataset(dataset):
    if dataset in VALID_DATASETS:
        index = VALID_DATASETS.index(dataset)
        print(f"Processing {dataset} dataset... (Index: {index})")
        return index
    else:
        print(f"Error: Invalid dataset. Please choose from {', '.join(VALID_DATASETS)}.")
        sys.exit(1)

def generate_solutions(dataset_name, llm_name):
    test_type = 0  ## 0 for assertions, 1 for python unittest
    path = f'unfiltered_testcases/{dataset_name}_{llm_name}_processed.pkl'
    print(path)
    out_path = f'unfiltered_testcases/with_generated_solutions/{dataset_name}_{llm_name}_processed.pkl'
    with open(path, 'rb') as file:
        data:List[Function]= pickle.load(file)
    # data = data[:10]
    import ast

    if llm_name == 'gpt-4':
        llm_requester = OpenaiRequester('gpt-4')
    elif llm_name == 'o3-mini':
        llm_requester = OpenaiRequester('o3-mini')
    elif llm_name == 'deepseek':
        # llm_requester = OpenaiRequester('deepseek-reasoner', backend="https://api.deepseek.com")
        llm_requester = FireworksAPIRequester('deepseek-r1')
    elif llm_name == 'gpt-4o':
        llm_requester = OpenaiRequester('gpt-4o-2024-08-06')
    elif llm_name == 'gpt-3.5-turbo':
        llm_requester = OpenaiRequester('gpt-3.5-turbo')
    elif llm_name == 'llama3':
        llm_requester = HuggingfaceRequester('meta-llama/Meta-Llama-3.1-8B-Instruct')
    elif llm_name == 'codellama':
        llm_requester = HuggingfaceRequester('codellama/CodeLlama-7b-Instruct-hf')
    elif llm_name == 'codeqwen':
        llm_requester = FireworksAPIRequester('qwen2p5-coder-32b-instruct')
    elif llm_name == 'codestral':
        llm_requester = OpenaiRequester(name='mistralai/codestral-2501', backend="https://api.aimlapi.com/v1")
    elif llm_name == 'magiccoder':
        llm_requester = HuggingfaceRequester('ise-uiuc/Magicoder-S-DS-6.7B')
    elif llm_name == 'GeminiPro':
        llm_requester = GeminiRequester()
    elif llm_name == 'gemini-1.5-flash-002':
        llm_requester = VertexAIRequester('gemini-1.5-flash-002')
    elif llm_name == 'mistral':
        llm_requester = HuggingfaceRequester('mistralai/Mistral-7B-Instruct-v0.3')
    elif llm_name == 'claude-3-7-sonnet-20250219':
        llm_requester = AntropicRequester('claude-3-7-sonnet-20250219')
    else:
        print(f"LLM {llm_name} not supported.")
        return
    for func in data:
        responses = llm_requester.get_completion(
            messages=[
                {
                    "role": "user",
                    "content": func.prompt,
                }
            ],
            logprobs=False,
            top_logprobs=0,
            temperature=0.6,
            n=7
        )['text']
        solutions = []
        for s in responses:
            sols = IMPORT_HEADER + '\n'.join(separate_python_code_blocks(s))
            print(sols)
            print('*'*100)
            solutions.append(sols)
        func.generated_solutions = solutions
    with open(out_path, "wb") as file:
        pickle.dump(data, file)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process a specified dataset.')

    # Add dataset argument
    parser.add_argument('--dataset', type=str, required=True, choices=VALID_DATASETS, help=f'The dataset to process. Options: {VALID_DATASETS}')

    # Add LLM argument
    parser.add_argument('--llm', type=str, required=True, choices=VALID_LLMS, help=f'The LLM to use. Options are {VALID_LLMS}')
    # Parse arguments
    args = parser.parse_args()

    llm_name = args.llm
    generate_solutions(args.dataset, llm_name)

if __name__ == '__main__':
    main()