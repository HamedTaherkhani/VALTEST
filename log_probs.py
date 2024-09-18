import sys
from typing import List
import pickle
import os
from function_executor import run_testcases, TimeoutException
import numpy as np
from tqdm import tqdm
import re
from generate_testcases import RawLogProbs

class LogProb:
    def __init__(self, type, token, prob):
        """

        :param type: 1 for input 2 for output
        :param token:
        :param prob:
        """
        self.type = type
        self.token = token
        self.prob = prob

    def __str__(self):
        return f'{self.token}: {self.prob}'


class TestCase:
    def __init__(self, text, input_logprobs: List[LogProb], output_logprobs: List[LogProb],
                 second_output_logprobs, second_input_logprobs, is_valid: int = None):
        self.text = text
        self.input_logprobs = input_logprobs
        self.output_logprobs = output_logprobs
        self.second_input_logprobs = second_input_logprobs
        self.second_output_logprobs = second_output_logprobs
        self.is_valid = is_valid

    def __str__(self):
        input_logprobs_str = ", ".join([str(lp) for lp in self.input_logprobs])
        output_logprobs_str = ", ".join([str(lp) for lp in self.output_logprobs]) if self.output_logprobs else "None"
        second_input_logprobs = ", ".join([str(lp) for lp in self.second_input_logprobs]) if self.second_input_logprobs else "None"
        second_output_logprobs = ", ".join([str(lp) for lp in self.second_output_logprobs]) if self.second_output_logprobs else "None"

        return f'Text: "{self.text}"\nIs Valid: {self.is_valid}\nInput LogProbs: [{input_logprobs_str}]\nOutput LogProbs: [{output_logprobs_str}]\nsecond_input_logprobs: [{second_input_logprobs}]\nsecond_output_logprobs: [{second_output_logprobs}]\n'

    def __repr__(self):
        return self.__str__()


class Function:
    def __init__(self, prompt: str, testcases: list[TestCase], solution: str, original_tests: str = None):
        self.prompt = prompt
        self.testcases = testcases
        self.solution = solution
        self.original_tests = original_tests

    def __str__(self):
        # Create a string representation of test cases
        testcases_str = "\n".join([str(tc) for tc in self.testcases])
        original_tests_str = self.original_tests if self.original_tests else "None"

        return (
            f"Prompt:\n{self.prompt}\n"
            f"Solution:\n{self.solution}\n"
            f"Original Tests:\n{original_tests_str}\n"
            f"Test Cases:\n{testcases_str}"
        )

    def __repr__(self):
        return self.__str__()

def getter(logprobs):
    for lp in logprobs:
        yield lp


# def find_test_case(res, testcase):
#     temp_test = testcase.copy()
#     for lp in res.choices[0].logprobs.content:
#         if testcase[0] == lp.token:
#
#         for t in testcase:
#             if t == lp.token:

def validate_string(input_string):
    # List of unacceptable tokens
    unacceptable_tokens = ('==', ' ', '  ', ',', '[', ']', '\'', '\"' ,')', '(', "\"")
    pattern = '|'.join(re.escape(token) for token in unacceptable_tokens)
    matches = re.findall(pattern, input_string)
    validated_string = ''.join(matches)
    if validated_string == input_string:
        return False
    else:
        return True

def get_logprobs(logprobs, testcases, method_name, ground_truth) -> List[TestCase]:
    gen = getter(logprobs)
    all_tests: List[TestCase] = []
    func_str_list = []
    testcase_data_list = []
    textcase_index = 0

    for logprob in gen:
        if 'assert' in logprob[0]:
            for logprob2 in gen:
                is_break = False
                if '(' in logprob2[0]:
                    input_logprobs: List[LogProb] = []
                    output_logprobs: List[LogProb] = []
                    second_input_logprobs: List[LogProb] = []
                    second_output_logprobs: List[LogProb] = []

                    for logprob3 in gen:
                        if '==' in logprob3[0]:
                            for logprob4 in gen:
                                if not validate_string(logprob4[0]):
                                    continue
                                elif '#' in logprob4[0] or '\n' in logprob4[0]:  # comment or newline
                                    break
                                output_logprobs.append(LogProb(
                                    type=2,
                                    token=logprob4[0],
                                    prob=np.round(np.exp(logprob4[1]) * 100, 2)
                                ))
                                second_output_logprobs.append(LogProb(
                                    type=2,
                                    token=logprob4[2][1][0],
                                    prob=np.round(np.exp(logprob4[2][1][1]) * 100, 2)
                                ))

                            # Collect the function string and associated data
                            func_str = ground_truth + '\n' + testcases[textcase_index]
                            func_str_list.append(func_str)
                            testcase_data_list.append({
                                'text': testcases[textcase_index],
                                'input_logprobs': input_logprobs,
                                'output_logprobs': output_logprobs,
                                'second_input_logprobs': second_input_logprobs,
                                'second_output_logprobs': second_output_logprobs
                            })
                            textcase_index += 1
                            is_break = True
                            break
                        elif logprob3[0] == ',':
                            continue
                        else:
                            if validate_string(logprob3[0]):
                                input_logprobs.append(LogProb(
                                    type=1,
                                    token=logprob3[0],
                                    prob=np.round(np.exp(logprob3[1]) * 100, 2)
                                ))
                                second_input_logprobs.append(LogProb(
                                    type=1,
                                    token=logprob3[2][1][0],
                                    prob=np.round(np.exp(logprob3[2][1][1]) * 100, 2)
                                ))
                    if is_break:
                        break
            if is_break:
                continue

    # Run all collected function strings concurrently
    is_passed_list = run_testcases(func_str_list, timeout=5)

    # Create TestCase instances with the results
    for i, is_passed in enumerate(is_passed_list):
        data = testcase_data_list[i]
        testcase = TestCase(
            text=data['text'],
            input_logprobs=data['input_logprobs'],
            output_logprobs=data['output_logprobs'],
            is_valid=is_passed,
            second_input_logprobs=data['second_input_logprobs'],
            second_output_logprobs=data['second_output_logprobs']
        )
        all_tests.append(testcase)

    return all_tests


def get_all_tests(dataset: str, llm: str) -> List[Function]:
    # Define the file paths for raw and processed data
    raw_file_name = f'raw_logprobs/{dataset}_{llm}.pkl'
    processed_file_name = f'raw_logprobs/{dataset}_{llm}_processed.pkl'

    # Check if the processed file exists
    if os.path.exists(processed_file_name):
        print(f'Loading processed functions from {processed_file_name}...')
        with open(processed_file_name, 'rb') as f:
            functions: List[Function] = pickle.load(f)
            if dataset == "LeetCode":
                functions = [r for idx, r in enumerate(functions) if idx != 60]
                # for idx, f in enumerate(functions):
                #     if 'fraction_to_decimal' in f.prompt:
                #         print(idx)
                # sys.exit()
        return functions

    # Load raw data if processed data does not exist
    try:
        with open(raw_file_name, 'rb') as f:
            raw_probs: List[RawLogProbs] = pickle.load(f)
    except FileNotFoundError:
        print(f'File {raw_file_name} not found')
        raise FileNotFoundError

    # Special handling for HumanEval dataset
    if dataset == 'HumanEval':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx != 39]
    if dataset == 'MBPP':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx not in (184, 244)]
    if dataset == 'LeetCode':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx not in (186, 481)]
    functions: List['Function'] = []
    for idx,prob in tqdm(enumerate(raw_probs)):
        # if idx == 182:
        #     continue
        # print(prob)
        # print(idx)
        try:
            testcases: List['TestCase'] = get_logprobs(prob.logprobs, prob.testcases, prob.prompt, prob.solution)
        except Exception as e:
            print(e)
            print('Error processing test cases, skipping...')
            continue

        # Create Function object and append to the list
        f = Function(prompt=prob.prompt, testcases=testcases, solution=prob.solution)
        functions.append(f)

    # Save processed functions for future use
    print(f'Saving processed functions to {processed_file_name}...')
    with open(processed_file_name, 'wb') as f:
        pickle.dump(functions, f)

    return functions