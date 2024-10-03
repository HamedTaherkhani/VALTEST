import sys
from typing import List
import pickle
import os
from function_executor import run_testcases, TimeoutException
import numpy as np
from tqdm import tqdm
import re
import ast
from typing import List, Tuple, Optional
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
    unacceptable_tokens = ('==', ' ', '  ', ',', '[', ']', '\'', '\"' ,')', '(', "\"", '\n')
    pattern = '|'.join(re.escape(token) for token in unacceptable_tokens)
    matches = re.findall(pattern, input_string)
    validated_string = ''.join(matches)
    if validated_string == input_string:
        return False
    else:
        return True


def contains_number(s: str) -> bool:
    return bool(re.search(r'\d', s))


import ast
from typing import Optional, Tuple, List

def separate_assertion(assertion_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Separates an assertion string into input and output parts.

    Parameters:
    - assertion_str (str): The assertion statement as a string.

    Returns:
    - Tuple[Optional[str], Optional[str]]: A tuple containing the input and output parts as strings.
      Returns (None, None) if separation fails.
    """
    # Remove 'assert ' prefix
    if assertion_str.startswith("assert "):
        expr_str = assertion_str[len("assert "):]
    else:
        expr_str = assertion_str

    try:
        # Try parsing as a comparison first (e.g., a == b)
        expr_ast = ast.parse(expr_str, mode='exec').body[0].value
        if isinstance(expr_ast, ast.Compare):
            if len(expr_ast.ops) == 1 and isinstance(expr_ast.ops[0], ast.Eq):
                input_expr = expr_ast.left
                output_expr = expr_ast.comparators[0]
            else:
                print(f"Unsupported comparison operator in assertion: {assertion_str}")
                return None, None
        elif isinstance(expr_ast, ast.Call):
            # Handle function or method calls
            func = expr_ast.func
            args = expr_ast.args

            if isinstance(func, ast.Attribute):
                if len(args) == 1:
                    # Method call (e.g., a.equals(b))
                    input_expr = func.value
                    output_expr = args[0]
                else:
                    # Function call with multiple arguments (e.g., torch.equal(a, b))
                    input_expr = args[0]
                    output_expr = args[1]
            else:
                # Function call without attribute (e.g., some_function(a, b))
                if len(args) >= 2:
                    input_expr = args[0]
                    output_expr = args[1]
                else:
                    print(f"Function call with insufficient arguments in assertion: {assertion_str}")
                    return None, None
        else:
            print(f"Unsupported assertion format: {assertion_str}")
            return None, None

        # Convert AST nodes back to code strings
        if input_expr and output_expr:
            try:
                # If input_expr is a function call, extract its arguments
                if isinstance(input_expr, ast.Call):
                    if len(input_expr.args) == 1:
                        input_arg_expr = input_expr.args[0]
                        input_str = ast.unparse(input_arg_expr).strip()
                    else:
                        # Handle multiple arguments if necessary
                        args_list = [ast.unparse(arg).strip() for arg in input_expr.args]
                        input_str = f"({', '.join(args_list)})"
                else:
                    input_str = ast.unparse(input_expr).strip()

                output_str = ast.unparse(output_expr).strip()
                return input_str, output_str
            except Exception as e:
                print(f"Failed to unparse expressions in assertion: {assertion_str}\nError: {e}")
                return None, None
        else:
            return None, None

    except Exception as e:
        print(f"Failed to parse assertion: {assertion_str}\nError: {e}")
        return None, None

def separate_assertions(assertions: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Processes a list of assertion strings and separates each into input and output parts.

    Parameters:
    - assertions (List[str]): A list of assertion statements as strings.

    Returns:
    - List[Tuple[Optional[str], Optional[str]]]: A list of tuples, each containing the input and output parts.
    """
    separated = []
    for assertion in assertions:
        input_part, output_part = separate_assertion(assertion)
        if input_part is None or output_part is None:
            continue
        separated.append((input_part, output_part))
    return separated


def token_iterator(s):
    for aa in s:
        yield aa


def remove_unnecessary_tokens(results):
    temp_results = []
    for test in results:
        tt = []
        for in_or_output in test:
            tokens = []
            for token in in_or_output:
                if validate_string(token[0]):
                    tokens.append(token)
            tt.append(tokens)
        temp_results.append(tt)

    for idx1,test in enumerate(temp_results):
        for idx2,in_or_output in enumerate(test):
            if len(in_or_output) == 0:
                temp_results[idx1][idx2] = results[idx1][idx2]
    return temp_results

def match_strings_to_tokens(testcases, tokens, dataset):
    results = []
    token_itr = token_iterator(tokens)
    while True:
        temp = next(token_itr)
        if 'assert' in temp[0]:
            break
    for tc in testcases:
        tt_temp = []
        for testcase in tc:
            index = 0
            testcase_len = len(testcase)
            # print(f'testcase_len is {testcase_len}')
            found_tokens = []
            testcase = testcase.replace('\"','\'')
            # print(f'testcase is {testcase}')
            for idx,token in enumerate(token_itr):
                temp = token[0].replace('\"','\'')
                # print(f'remaining : {testcase[index:]}')
                if index == 0:
                    temp = temp.lstrip()
                    if not testcase.startswith('('):
                        temp = temp.lstrip('(')
                # print(temp)
                if len(temp) > len(testcase[index:]):
                    if temp.startswith(testcase[index:testcase_len]):
                        # if validate_string(token[0]):
                        if dataset == 'DS1000':
                            if contains_number(token[0]):
                                found_tokens.append(token)
                        else:
                            found_tokens.append(token)
                        index += len(temp)
                        # print(f'index is {index}')
                else:
                    if testcase[index:].startswith(temp):
                        # if validate_string(token[0]):
                        if dataset == 'DS1000':
                            if contains_number(token[0]):
                                found_tokens.append(token)
                        else:
                            found_tokens.append(token)
                        index += len(temp)
                        # print(f'index is {index}')
                if index > testcase_len:
                    break
            tt_temp.append(found_tokens)
        results.append(tt_temp)

    results = remove_unnecessary_tokens(results)
    return results


def get_logprobs_dynamic(logprobs, testcases, method_name, ground_truth, dataset='HumanEval') -> List[TestCase]:
    separated_assertions = separate_assertions(testcases)
    # print('-----------------------------------------')
    # print(separated_assertions)

    assertions_to_tokens =  match_strings_to_tokens(separated_assertions, logprobs, dataset)
    # if len(assertions_to_tokens) == 0:
    # print(separated_assertions)
    # print(logprobs)
    textcase_index = 0
    func_str_list = []
    all_tests: List[TestCase] = []
    testcase_data_list = []
    for instance in assertions_to_tokens:
        input_logprobs: List[LogProb] = []
        output_logprobs: List[LogProb] = []
        second_input_logprobs: List[LogProb] = []
        second_output_logprobs: List[LogProb] = []
        inputs = instance[0]
        outputs = instance[1]
        for ins in inputs:
            input_logprobs.append(LogProb(
                type = 1,
                token = ins[0],
                prob=np.round(np.exp(ins[1]) * 100, 2)
            ))
            second_input_logprobs.append(LogProb(
                type = 1,
                token=ins[2][1][0],
                prob=np.round(np.exp(ins[2][1][1]) * 100, 2)
            ))

        for ins in outputs:
            output_logprobs.append(LogProb(
                type=2,
                token=ins[0],
                prob=np.round(np.exp(ins[1]) * 100, 2)
            ))
            second_output_logprobs.append(LogProb(
                type=2,
                token=ins[2][1][0],
                prob=np.round(np.exp(ins[2][1][1]) * 100, 2)
            ))
        func_str = ground_truth + '\n' + testcases[textcase_index]
        func_str_list.append(func_str)
        testcase_data_list.append({
            'text': testcases[textcase_index],
            'input_logprobs': input_logprobs,
            'output_logprobs': output_logprobs,
            'second_input_logprobs': second_input_logprobs,
            'second_output_logprobs': second_output_logprobs
        })
        textcase_index +=1
    # print(separated_assertions)
    # print(assertions_to_tokens[0])
    # print('-------------------------------------------------------------------')
    is_passed_list = run_testcases(func_str_list, timeout=5)
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
            if dataset == "LeetCode" and llm == "llama3":
                functions = [r for idx, r in enumerate(functions) if idx != 286]
            if dataset == "MBPP" and llm == "gpt-4o":
                functions = [r for idx, r in enumerate(functions) if idx != 400]
            if dataset == "HumanEval" and llm == "llama3":
                functions = [r for idx, r in enumerate(functions) if idx != 79]
            if dataset == "HumanEval" and llm == "gpt-4o":
                functions = [r for idx, r in enumerate(functions) if idx != 146]
            if dataset == 'HumanEval' and llm == "gpt-3.5-turbo":
                # print('here')
                functions = [r for idx, r in enumerate(functions) if idx != 56]
        return functions

    # Load raw data if processed data does not exist
    try:
        with open(raw_file_name, 'rb') as f:
            raw_probs: List[RawLogProbs] = pickle.load(f)
    except FileNotFoundError:
        print(f'File {raw_file_name} not found')
        raise FileNotFoundError

    # Special handling for HumanEval dataset
    if dataset == 'HumanEval' and llm == 'gpt-4o':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx != 39]
    if dataset == 'MBPP' and llm == 'gpt-4o':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx not in (184, 244)]
    if dataset == 'LeetCode':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx not in (186, 481)]
    if dataset == 'MBPP' and llm == "llama3":
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx != 250]
    # print(raw_probs[50])
    functions: List['Function'] = []
    for idx,prob in tqdm(enumerate(raw_probs)):
        # print(idx)
        # if idx == 182:
        #     continue
        # print(prob)
        # print(idx)
        # if idx != 38:
        #     continue
        try:
            if len(prob.testcases) == 0:
                continue
            # print(prob.testcases)
            testcases: List['TestCase'] = get_logprobs_dynamic(prob.logprobs, prob.testcases, prob.prompt,
                                                               prob.solution, dataset)
            # for t in testcases:
            #     if t.output_logprobs[0] is None:
            #         print(idx)
            #         sys.exit()
            print(testcases)

        except Exception as e:
            # print(e)
            # print('Error processing test cases, skipping...')
            continue

        # Create Function object and append to the list
        f = Function(prompt=prob.prompt, testcases=testcases, solution=prob.solution)
        functions.append(f)
    # sys.exit()

    # Save processed functions for future use
    print(f'Saving processed functions to {processed_file_name}...')
    with open(processed_file_name, 'wb') as f:
        pickle.dump(functions, f)

    return functions