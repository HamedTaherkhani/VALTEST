import ast
import textwrap
import argparse
import sys
from math import exp
import numpy as np
from IPython.display import display, HTML
import os
from typing import List
from humaneval_loader import HumanEvalLoader
from MBPPLoader import MBPPLoader
from leetcode_loader import LeetCodeLoader
from llm_requester import OpenaiRequester, HuggingfaceRequester
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
# Define an abstract base class for LLM requesters

class RawLogProbs:
    def __init__(self, prompt: str, logprobs: dict, dataset: str, id: str, testcases: List[str], solution: str):
        self.prompt = prompt
        self.logprobs = logprobs
        self.dataset = dataset
        self.id = id
        self.testcases = testcases
        self.solution = solution

    def __str__(self):
        return (
            f"RawLogProbs(\n"
            f"  Prompt: {self.prompt}\n"
            f"  logprobs: {[s[0] for s in self.logprobs]}\n"
            f"  Dataset: {self.dataset}\n"
            f"  ID: {self.id}\n"
            f"  Testcases: {self.testcases}\n"
            f"  Solution: {self.solution}\n"
            f")"
        )

    def __repr__(self):
        return self.__str__()


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

def generate_testcases(dataset_choice, llm_name):
    """
    :param dataset_choice: 0 for MBPP, 1 for HumanEval, 2 for LeetCode
    :param llm_name: Name of the LLM to use
    :return:
    """
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
    else:
        print('Not a valid dataset selected.')
        return

    PY_TEST_GENERATION_CHAT_INSTRUCTION = """You are an AI coding assistant that can write unique, diverse, and intuitive unit tests for functions given the signature and docstring. Do not make any comments on the test cases. Generate 10 to 20 test cases.\n"""
    PY_TEST_GENERATION_FEW_SHOT = """Examples:
    func signature:
    def add3Numbers(x, y, z):
        \"\"\" Add three numbers together.
        This function takes three numbers as input and returns the sum of the three numbers.
        \"\"\"
    unit tests:
    assert add3Numbers(1, 2, 3) == 6
    assert add3Numbers(-1, 2, 3) == 4
    assert add3Numbers(1, -2, 3) == 2
    assert add3Numbers(1, 2, -3) == 0
    assert add3Numbers(-3, -2, -1) == -6
    assert add3Numbers(0, 0, 0) == 0\n
    """
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
    elif llm_name == 'codellama':
        llm_requester = HuggingfaceRequester('codellama/CodeLlama-7b-Instruct-hf')
    elif llm_name == 'magiccoder':
        llm_requester = HuggingfaceRequester('ise-uiuc/Magicoder-S-DS-6.7B')
    elif llm_name == 'gemini':
        llm_requester = HuggingfaceRequester('OpenGemini/Gemini-7B')
    else:
        print(f"LLM {llm_name} not supported.")
        return

    def parse_tests(tests: str):
        return [test.strip() for test in tests.splitlines() if "assert" in test]

    def is_syntax_valid(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    for idx, (prompt, solution) in tqdm(enumerate(list(dataset))):
        API_RESPONSE = llm_requester.get_completion(
            messages=[
                {
                    "role": "user",
                    "content": PY_TEST_GENERATION_CHAT_INSTRUCTION + PY_TEST_GENERATION_FEW_SHOT + prompt
                }
            ],
            logprobs=True,
            temperature=0
        )
        # print(API_RESPONSE)
        # print('-------------------------------------------')
        # print(API_RESPONSE['text'])
        # print('-------------------------------------------')
        all_tests = parse_tests(API_RESPONSE['text'])
        valid_tests = [test for test in all_tests if is_syntax_valid(test)]
        testcases.append(valid_tests)
        raw_prob = RawLogProbs(prompt=prompt, logprobs=API_RESPONSE['logprobs'], dataset=dataset_name, id=idx, testcases=valid_tests,
                               solution=solution)
        raw_probs.append(raw_prob)
    # print(raw_probs)
    os.makedirs('raw_logprobs', exist_ok=True)
    with open(f'raw_logprobs/{dataset_name}_{llm_name}.pkl', 'wb') as f:
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

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided dataset and LLM
    dataset = process_dataset(args.dataset)
    llm_name = args.llm
    generate_testcases(dataset, llm_name)

if __name__ == '__main__':
    main()
