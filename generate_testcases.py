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
from llm_requester import OpenaiRequester, HuggingfaceRequester, GeminiRequester, VertexAIRequester
from loaders.livecodebench_loader import LiveCodeBenchLoader
from loaders.livecodebench_loader2 import LiveCodeBenchLoader2
# from BigCodeLoader import BigCodeLoader
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from loaders.BigCodeLoader import BigCodeLoader
from log_probs import RawLogProbs ,Function, TestCase, LogProb
from prompts import PY_TEST_GENERATION_FEW_SHOT, PY_TEST_GENERATION_CHAT_INSTRUCTION, \
    PY_TEST_GENERATION_CHAT_INSTRUCTION_BigCodeBench, PY_TEST_GENERATION_FEW_SHOT_BigCodeBench, PY_TEST_GENERATION_CHAT_INSTRUCTION_BigCodeBench_second_run
# Define an abstract base class for LLM requesters


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

def generate_testcases(dataset_choice, llm_name, extra_run):
    """
    :param dataset_choice: 0 for MBPP, 1 for HumanEval, 2 for LeetCode
    :param llm_name: Name of the LLM to use
    :return:
    """
    test_type = 0 ## 0 for assertions, 1 for python unittest
    few_shot = PY_TEST_GENERATION_FEW_SHOT
    instruction = PY_TEST_GENERATION_CHAT_INSTRUCTION
    if dataset_choice == 0:
        loader = MBPPLoader()
        prompts = loader.get_prompts()
        solutions = loader.get_solutions()
        solutions = [correct_indentation(sol) for sol in solutions]
        dataset = zip(prompts, solutions)
        dataset_name = 'MBPP'
    elif dataset_choice == 1:
        he = HumanEvalLoader().get_human_eval()
        data = he['test']
        prompt = [d['prompt'] for d in data]
        solutions = [d['prompt'] + d['canonical_solution'] for d in data]
        dataset = zip(prompt, solutions)
        dataset_name = 'HumanEval'
    elif dataset_choice == 2:
        leetcode = LeetCodeLoader()
        prompts = leetcode.get_prompts()
        solutions = leetcode.get_solutions()
        dataset = zip(prompts, solutions)
        dataset_name = 'LeetCode'
    elif dataset_choice == 3:
        lvb = LiveCodeBenchLoader()
        prompts = lvb.get_prompts()
        solutions = lvb.get_solutions()
        dataset = zip(prompts, solutions)
        dataset_name = 'LiveCodeBench'
    elif dataset_choice == 4:
        lvb2 = LiveCodeBenchLoader2()
        prompts = lvb2.get_prompts()
        solutions = lvb2.get_solutions()
        dataset = zip(prompts, solutions)
        dataset_name = 'LiveCodeBench2'
    elif dataset_choice == 5:
        few_shot = PY_TEST_GENERATION_FEW_SHOT_BigCodeBench
        instruction = PY_TEST_GENERATION_CHAT_INSTRUCTION_BigCodeBench
        bigcode = BigCodeLoader(hard=1)
        prompts = bigcode.get_prompts()
        solutions = bigcode.get_solutions()
        dataset = zip(prompts, solutions)
        dataset_name = 'BigCodeBenchHard'
        test_type = 1
    elif dataset_choice == 6:
        few_shot = PY_TEST_GENERATION_FEW_SHOT_BigCodeBench
        instruction = PY_TEST_GENERATION_CHAT_INSTRUCTION_BigCodeBench
        bigcode = BigCodeLoader(hard=0)
        prompts = bigcode.get_prompts()
        solutions = bigcode.get_solutions()
        dataset = zip(prompts, solutions)
        dataset_name = 'BigCodeBench'
        test_type = 1
    # elif dataset_choice == 3:
    #     ds = DS1000Loader()
    #     prompts = ds.get_prompts()
    #     solutions = ds.get_solutions()
    #     dataset = zip(prompts, solutions)
    #     dataset_name = 'DS1000'
    #     few_shot = PY_TEST_GENERATION_FEW_SHOT_DS1000
    #     instruction = PY_TEST_GENERATION_CHAT_INSTRUCTION_DS1000
    # elif dataset_choice == 3:
    #     bigcode = BigCodeLoader()
    #     prompts = bigcode.get_prompts()
    #     solutions = bigcode.get_solutions()
    #     dataset = zip(prompts, solutions)
    #     dataset_name = 'BigCodeBench'
    #     instruction = PY_TEST_GENERATION_CHAT_INSTRUCTION_DS1000
    else:
        print('Not a valid dataset selected.')
        return

    from tqdm import tqdm
    import pickle
    testcases = []
    import ast
    raw_probs = []

    if llm_name == 'gpt-4':
        llm_requester = OpenaiRequester('gpt-4')
    elif llm_name == 'gpt-4o':
        llm_requester = OpenaiRequester('gpt-4o')
    elif llm_name == 'gpt-3.5-turbo':
        llm_requester = OpenaiRequester('gpt-3.5-turbo')
    elif llm_name == 'llama3':
        llm_requester = HuggingfaceRequester('meta-llama/Meta-Llama-3.1-8B-Instruct')
    elif llm_name == 'magiccoder':
        llm_requester = HuggingfaceRequester('ise-uiuc/Magicoder-S-DS-6.7B')
    elif llm_name == 'GeminiPro':
        llm_requester = GeminiRequester()
    elif llm_name == 'gemini-1.5-flash-002':
        llm_requester = VertexAIRequester('gemini-1.5-flash-002')
    elif llm_name == 'mistral':
        llm_requester = HuggingfaceRequester('mistralai/Mistral-7B-Instruct-v0.3')
    else:
        print(f"LLM {llm_name} not supported.")
        return
    if extra_run:
        print('second run...')
        processed_file_name = f'filtered_testcases/BigCodeBenchHard_gpt-4o.pkl'
        with open(processed_file_name, 'rb') as f:
            functions: List[Function] = pickle.load(f)
    def process_gemini(text):
        added_text = "import unittest\ntestcase = unittest.TestCase()\n"
        return added_text + f'testcase.{text}'
    for idx, (prompt, solution) in tqdm(enumerate(list(dataset))):
        if extra_run:
            tests = find_tests(prompt, functions)
            content = PY_TEST_GENERATION_CHAT_INSTRUCTION_BigCodeBench_second_run + "\nFunction description:\n" + prompt + '\nTests:\n\n' + "\n".join(
                tests) + "\nExtra Tests:"
        else:
            content = instruction + few_shot + prompt
        # print(content)
        API_RESPONSE = llm_requester.get_completion(
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            logprobs=True,
            temperature=0
        )
        # print(API_RESPONSE)
        # print('-------------------------------------------')
        # print(API_RESPONSE['text'])
        # print('-------------------------------------------')
        if test_type == 1: ## python unittests
            all_tests = separate_python_code_blocks(API_RESPONSE['text'])
        else:
            all_tests = parse_tests(API_RESPONSE['text'])
        if solution != '':
            func_name = get_function_name(solution)
            # print(f'func_name is {func_name}')
            valid_tests = [test for test in all_tests if is_syntax_valid(test, func_name)]
        else:  ## this is when we don't have solutions
            valid_tests = [test for test in all_tests if is_syntax_valid2(test)]
        testcases.append(valid_tests)

        print(valid_tests)
        print('-' * 100)
        raw_prob = RawLogProbs(prompt=prompt, logprobs=API_RESPONSE['logprobs'], dataset=dataset_name, id=idx,
                               testcases=valid_tests,
                               solution=solution,
                               test_type=test_type)
        raw_probs.append(raw_prob)
    # print(raw_probs)
    os.makedirs('unfiltered_testcases', exist_ok=True)
    with open(f'unfiltered_testcases/{dataset_name}_{llm_name}.pkl', 'wb') as f:
        pickle.dump(raw_probs, f)


# Function that processes the dataset and returns its index
def process_dataset(dataset):
    if dataset in VALID_DATASETS:
        index = VALID_DATASETS.index(dataset)
        print(f"Processing {dataset} dataset... (Index: {index})")
        return index
    else:
        print(f"Error: Invalid dataset. Please choose from {', '.join(VALID_DATASETS)}.")
        sys.exit(1)

def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Process a specified dataset.')

    # Add dataset argument
    parser.add_argument('--dataset', type=str, required=True, choices=VALID_DATASETS, help=f'The dataset to process. Options: {VALID_DATASETS}')

    # Add LLM argument
    parser.add_argument('--llm', type=str, required=True, choices=VALID_LLMS, help=f'The LLM to use. Options are {VALID_LLMS}')
    parser.add_argument('--extra_run', type=bool, required=False, default=False, help=f'If true, run in extra mode. Default is False.', choices=[True, False])
    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided dataset and LLM
    dataset = process_dataset(args.dataset)
    llm_name = args.llm
    extra_run = args.extra_run
    generate_testcases(dataset, llm_name, extra_run)

if __name__ == '__main__':
    main()
