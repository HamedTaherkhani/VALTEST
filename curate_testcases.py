import argparse
import ast
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from typing import List
from openai import OpenAI
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from function_executor import run_test_cases, run_unit_tests_parallel, run_single_test_subprocess
from log_probs import Function, TestCase
import multiprocessing
from main_train import evaluate_function
from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.generative_models._generative_models import ResponseValidationError
load_dotenv()


class LLMClient:
    """Abstract base class for LLM clients."""

    def chat_completion(self, model_name, messages, max_tokens, temperature, seed):
        raise NotImplementedError("chat_completion method not implemented.")


class OpenAIClient(LLMClient):
    """Client for interacting with OpenAI's API."""

    def __init__(self, name):
        self.key = os.getenv('openai_key')
        self.client = OpenAI(api_key=self.key)
        self.name = name

    def chat_completion(self, model_name, messages, max_tokens, temperature, seed):
        params = {
            "model": self.name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        completion = self.client.chat.completions.create(**params)
        text = completion.choices[0].message.content
        return text.strip()


class VertexAIClient(LLMClient):
    """Client for interacting with Vertex AI's API."""

    def __init__(self, project, location, model_name):
        from google.cloud import aiplatform
        vertexai.init(project=project, location=location)
        self.config = GenerationConfig(temperature=0, max_output_tokens=1500)
        generative_model = GenerativeModel(model_name)
        self.chat_session = generative_model.start_chat()

    def chat_completion(self, model_name, messages, max_tokens, temperature=0, seed=123,):
        prompt = messages[-1]['content']
        try:
            responses = self.chat_session.send_message(prompt, stream=False, generation_config=self.config)
            res = responses.candidates[0].content.parts[0].text
            return res.strip()
        except ResponseValidationError:
            return None

def create_llm_client(llm):
    """Creates an LLM client based on the specified LLM."""
    if llm == 'gemini-1.5-flash-002':
        client = VertexAIClient(
            project=os.getenv('GCP_PROJECT'),
            location=os.getenv('GCP_LOCATION'),
            model_name=llm,
        )
    else:
        client = OpenAIClient(llm)
    return client


def curate_testcases(dataset, llm, threshold=0.7):
    """Curates test cases by validating and adjusting them using an LLM."""
    with open(f'filtered_testcases/{dataset}_{llm}.pkl', 'rb') as f:
        functions: List[Function] = pickle.load(f)
    # functions = functions[:10]
    total_valid = sum(
        1 for f in functions for testcase in f.testcases if testcase.is_valid == 1
    )
    total_invalid = sum(
        1 for f in functions for testcase in f.testcases if testcase.is_valid == 0
    )
    below_threshold = sum(
        1
        for f in functions
        for testcase in f.testcases
        if testcase.prediction_y_prob < threshold
    )

    print(f'Total valid: {total_valid}')
    print(f'Total invalid: {total_invalid}')
    print(f'Below threshold: {below_threshold}')

    client = create_llm_client(llm)
    functions = validate_test_cases_concurrently(functions, threshold, client, llm, dataset)

    curated_function_file_name = f'curated_testcases/{dataset}_{llm}.pkl'
    print(f'Saving curated functions to {curated_function_file_name}...')
    with open(curated_function_file_name, 'wb') as f:
        pickle.dump(functions, f)

import re

def extract_python_code(text):
    # Regex to find text within ```python and ``` tags
    try:
        pattern = r"```python(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
    except Exception:
        return text
    return matches[0]


def process_unit_testcases(f, testcase, client, llm):
    prompt = f"""
    You will receive a function description and a python unittest designed to test the the function. Your task is to find any issues with the test case and verify it's correctness according to the function specification.
    Use detailed, step-by-step reasoning to check the correctness of the unit test. After validating, adjust the test case as necessary. Finally, present the validated test case enclosed within exactly ***```python and ```*** tags.
    Do not include anything else inside the asterisks—only the testcase. Don't use subprocess, multiprocessing or concurrent libraries. Also don't add any more assertions in the test case. And don't make the test stricter. Loosen the test case to make them assertions less restrictive if you are not sure about the expected output of the assertions in the test case.
    Function Description:
    {f.prompt}

    Test Case:
    {testcase.text}
    """
    response_text = client.chat_completion(
        model_name=llm,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0,
        seed=123,
    )
    if response_text is None:
        return
    # Extract the validated assertion from the LLM's response
    reply = response_text.strip()
    # print(f'Response: {reply}')
    code = extract_python_code(reply)
    # print(f'Code: {code}')
    try:
        ast.parse(code)
        # print(code)
        testcase.validated_text = code
    except SyntaxError:
        print("Syntax error in validated assertion.")

def process_testcase(f, testcase, client, llm):
    """Processes a single test case using the specified LLM client."""

    prompt = f"""
You will receive a function description and an assertion designed to test that function. Your task is to verify whether the assertion correctly tests the function according to its specification. Use detailed, step-by-step reasoning to check the correctness of the assertion. After validating, adjust the assertion output as necessary. Finally, present the validated assertion enclosed within three asterisks (***). Do not include anything else inside the asterisks—only the assertion.

Function Description:
{f.prompt}

Test Case:
{testcase.text}
"""

    response_text = client.chat_completion(
        model_name=llm,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0,
        seed=123,
    )
    if response_text is None:
        return
    # Extract the validated assertion from the LLM's response
    reply = response_text.strip()
    start_idx = reply.find('***')
    if start_idx != -1:
        end_idx = reply.find('***', start_idx + 3)
        if end_idx != -1:
            validated_assertion = reply[start_idx + 3:end_idx].strip()
            try:
                ast.parse(validated_assertion)
                testcase.validated_text = validated_assertion
                print('------------------------')
                print(f'Original Test Case: {testcase.text}')
                print(f'Validated Test Case: {testcase.validated_text}')
            except SyntaxError:
                print("Syntax error in validated assertion.")
        else:
            print("Closing *** not found in LLM response.")
    else:
        print("No *** found in LLM response.")


def validate_test_cases_concurrently(functions, threshold, client, llm, dataset):
    """Validates test cases concurrently using a thread pool."""
    process_func = process_unit_testcases if 'BigCodeBench' in dataset else process_testcase
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(process_func, f, testcase, client, llm)
            for f in functions
            for testcase in f.testcases
            if testcase.prediction_y_prob < threshold
        ]
        for future in tqdm(futures):
            future.result()

    for f in functions:
        old_tests = []
        new_tests = []
        testcases = []
        for testcase in f.testcases:
            if getattr(testcase, 'validated_text', None):
                if (
                    testcase.text.replace(' ', '').replace('\n', '')
                    != testcase.validated_text.replace(' ', '').replace('\n', '')
                ):
                    old_tests.append(testcase.text)
                    new_tests.append(testcase.validated_text)
                    testcase.text = testcase.validated_text
                else:
                    old_tests.append(testcase.text)
                    new_tests.append(testcase.text)
            else:
                old_tests.append(testcase.text)
                new_tests.append(testcase.text)
                # print(testcase.validated_text)
            testcases.append(testcase.text)
        if 'BigCodeBench' not in dataset:
            is_passed_list = run_test_cases(f.solution, testcases, timeout=5)
        else:
            # is_passed_list = []
            # for t in testcases:
            #     print(t)
            #     is_passed_list.append(run_single_test_subprocess(test_str=t, code_str=f.solution)[0])
            is_passed_list = run_unit_tests_parallel(f.solution, testcases)
            is_passed_list = [passed[0] for passed in is_passed_list]

        orig_passed_list = [t.is_valid for t in f.testcases]
        # print('orignal passed', orig_passed_list)
        # print('passed after validating', is_passed_list)
        # print(f.solution)
        # for index, t in enumerate(f.testcases):
            # if t.is_valid == 1 and not is_passed_list[index]:
            #     print(old_tests[index])
            #     print(new_tests[index])
            #     print('*'*100)
        for index, testcase in enumerate(f.testcases):
            testcase.is_valid = 1 if is_passed_list[index] else 0
        # print('*'*100)
    return functions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run testcase curation with specified dataset and LLM."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=VALID_DATASETS,
        required=True,
        help=f"Specify the dataset to use. Choices are: {VALID_DATASETS}.",
    )
    parser.add_argument(
        "--llm",
        type=str,
        choices=VALID_LLMS,
        required=True,
        help=f"Specify the LLM to use. Choices are: {VALID_LLMS}.",
    )

    args = parser.parse_args()
    curate_testcases(args.dataset, args.llm)
