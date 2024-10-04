from log_probs import Function, TestCase
import argparse
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
import pickle
from typing import List


def curate_testcases(dataset, llm):
    with open(f'filtered_testcases/{dataset}_{llm}.pkl', 'rb') as f:
        functions: List[Function] = pickle.load(f)
    total_predicted_1 = 0
    total_predicted_0 = 0
    for f in functions:
        for testcase in f.testcases:
            if testcase.prediction_is_valid == 1:
                total_predicted_1 += 1
            elif testcase.prediction_is_valid == 0:
                total_predicted_0 += 1
    print(total_predicted_1)
    print(total_predicted_0)



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
    curate_testcases(args.dataset, args.llm)