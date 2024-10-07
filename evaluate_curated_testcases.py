import argparse
from main_train import evaluate_function
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from log_probs import Function, TestCase
import argparse
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
import pickle
from typing import List
import sys
import os
from dotenv import load_dotenv
from function_executor import run_test_cases


def evaluate_curated_testcases(dataset, llm):
    with open(f'curated_testcases/{dataset}_{llm}.pkl', 'rb') as f:
        functions: List[Function] = pickle.load(f)
    evaluate_function(functions, do_mutation=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run testcase curation with specified dataset and LLM.")

    # Add the 'dataset' argument with restricted choices
    parser.add_argument(
        "--dataset",
        type=str,
        choices=VALID_DATASETS,
        required=True,
        help=f"Specify the dataset to use. Choices are: {VALID_DATASETS}."
    )

    # Add the 'LLM' argument with restricted choices, allowing future extensions
    parser.add_argument(
        "--llm",
        type=str,
        choices=VALID_LLMS,
        required=True,
        help=f"Specify the LLM to use. Choices are: {VALID_LLMS}."
    )
    args = parser.parse_args()
    file_name = f'output/{args.dataset}_{args.llm}_curated.txt'
    print(f'Writing the output to {file_name}')
    with open(file_name, 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        evaluate_curated_testcases(args.dataset, args.llm)
