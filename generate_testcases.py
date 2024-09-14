
import ast
import textwrap
import argparse
import sys
from math import exp
import numpy as np
from IPython.display import display, HTML
import os
from openai_requester import OpenaiRequester
from typing import List
from humaneval_loader import HumanEvalLoader
from MBPPLoader import MBPPLoader
from leetcode_loader import LeetCodeLoader
# from swebench import generate_test_cases_for_swebench
class RawLogProbs:
    def __init__(self, prompt: str, API_Response: str, dataset: str, id: str, testcases: List[str], solution: str):
        self.prompt = prompt
        self.API_Response = API_Response
        self.dataset = dataset
        self.id = id
        self.testcases = testcases
        self.solution = solution

    def __str__(self):
        return (
            f"RawLogProbs(\n"
            f"  Prompt: {self.prompt}\n"
            # f"  API_Response: {self.API_Response}\n"
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

def generate_testcases(dataset_choice):
    """
    :param dataset_choice: 0 for MBPP, 1 for HumanEval
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
        print('not valid dataset selected')
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
    open_requester = OpenaiRequester()
    def parse_tests(tests: str):
        return [test.strip() for test in tests.splitlines() if "assert" in test]

    def is_syntax_valid(code: str) -> bool:
        try:
            ast.parse(code)
            return True
        except Exception:
            return False

    for idx, (prompt, solution) in tqdm(enumerate(list(dataset)[0:10])):
        # if idx not in errors:
        #     continue
        API_RESPONSE = open_requester.get_completion(
            [
                {
                    "role": "user",
                    "content": PY_TEST_GENERATION_CHAT_INSTRUCTION + PY_TEST_GENERATION_FEW_SHOT + prompt
                }
            ],
            model="gpt-4o",
            logprobs=True,
            # top_logprobs=3,
            temperature=0
        )
        all_tests = parse_tests(API_RESPONSE.choices[0].message.content)  # type: ignore
        valid_tests = [test for test in all_tests if is_syntax_valid(test)]
        testcases.append(valid_tests)
        raw_prob = RawLogProbs(prompt=prompt, API_Response=API_RESPONSE, dataset=dataset_name, id=idx, testcases=valid_tests,
                               solution=solution)
        raw_probs.append(raw_prob)
        # print(raw_prob)
    with open(f'raw_logprobs/{dataset_name}.plk', 'wb') as f:
        pickle.dump(raw_probs, f)



VALID_DATASETS = ['MBPP', 'HumanEval', 'Leetcode']
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
    parser.add_argument('--dataset', type=str, required=True, choices=VALID_DATASETS, help='The dataset to process. Options: MBPP, HumanEval')

    # Parse arguments
    args = parser.parse_args()

    # Call the function with the provided dataset
    dataset = process_dataset(args.dataset)
    generate_testcases(dataset)

if __name__ == '__main__':
    main()