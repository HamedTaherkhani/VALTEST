from pyexpat.errors import messages
from main_train import evaluate_function
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from log_probs import Function, TestCase
import argparse
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
import pickle
from typing import List
import os
from dotenv import load_dotenv
from function_executor import run_test_cases
from tqdm import tqdm
load_dotenv()


def curate_testcases(dataset, llm, threshold=0.85):
    with open(f'filtered_testcases/{dataset}_{llm}.pkl', 'rb') as f:
        functions: List[Function] = pickle.load(f)
    total_valid = 0
    total_invalid = 0
    below_threshold = 0
    for f in functions:
        for testcase in f.testcases:
            if testcase.is_valid == 1:
                total_valid += 1
            elif testcase.is_valid == 0:
                total_invalid += 1
            if testcase.prediction_y_prob < threshold:
                below_threshold += 1
    print(total_valid)
    print(total_invalid)
    print(f'below_threshold : {below_threshold}')

    client = OpenAI(api_key=os.getenv('openai_key'))
    functions = validate_test_cases_concurrently(functions, threshold, client, llm)
    curated_function_file_name = f'curated_testcases/{dataset}_{llm}.pkl'
    print(f'Saving curated functions to {curated_function_file_name}...')
    with open(curated_function_file_name, 'wb') as f:
        pickle.dump(functions, f)

    # evaluate_function(functions=functions, do_mutation=True)


def process_testcase(f, testcase, client, model_name):
    prompt = f"""
You will receive a function description and an assertion designed to test that function. Your task is to verify whether the assertion correctly tests the function according to its specification. Use detailed, step-by-step reasoning to check the correctness of the assertion. After validating, adjust the assertion output as necessary. Finally, present the validated assertion enclosed within three asterisks (***). Do not include anything else inside the asterisks—only the assertion.

Function Description:
{f.prompt}

Test Case:
{testcase.text}
"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1000,
        temperature=0,
        seed=123
    )

    # Extract the assistant's reply
    reply = response.choices[0].message.content.strip()

    # Parse the reply to get the assertion between *** and ***
    start_idx = reply.find('***')
    if start_idx != -1:
        end_idx = reply.find('***', start_idx + 3)
        if end_idx != -1:
            validated_assertion = reply[start_idx + 3:end_idx].strip()
            # Store the validated assertion in the test case
            testcase.validated_text = validated_assertion
            print('------------------------')
            print(testcase.text)
            print(testcase.validated_text)
            print('*************************')
        else:
            print("Closing *** not found in GPT-4 response.")
    else:
        print("No *** found in GPT-4 response.")


def validate_test_cases_concurrently(functions, threshold, client, llm):
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for f in functions:
            for testcase in f.testcases:
                if testcase.prediction_y_prob < threshold:
                    # Submit the processing of each test case to the thread pool
                    futures.append(executor.submit(process_testcase, f, testcase, client, llm))
        # Wait for all threads to complete
        for future in tqdm(futures):
            future.result()

    for f in functions:
        testcases = []
        for index, testcase in enumerate(f.testcases):
            try:
                if testcase.validated_text is not None:
                    if testcase.text.replace(' ', '').replace('\n', '') != testcase.validated_text.replace(' ', '').replace('\n', ''):
                        # print(testcase.text)
                        testcase.text = testcase.validated_text
                        # print(testcase.validated_text)
                        # print('----------------------------')
            except Exception as e:
                pass
            testcases.append(testcase.text)
        is_passed_list = run_test_cases(f.solution, testcases, timeout=5)
        for index, testcase in enumerate(f.testcases):
            if is_passed_list[index]:
                testcase.is_valid = 1
            else:
                testcase.is_valid = 0

    return functions


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