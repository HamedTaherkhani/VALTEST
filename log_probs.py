from typing import List
import numpy as np
from function_executor import run_testcase, TimeoutException
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

        return f'Text: "{self.text}"\nIs Valid: {self.is_valid}\nInput LogProbs: [{input_logprobs_str}]\nOutput LogProbs: [{output_logprobs_str}]\n'

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

    # Create a pattern to match the unacceptable tokens
    pattern = '|'.join(re.escape(token) for token in unacceptable_tokens)

    # Find all occurrences of the unacceptable tokens in the string
    matches = re.findall(pattern, input_string)

    # Reconstruct the validated part of the string using the unacceptable tokens
    validated_string = ''.join(matches)

    # Check if the input string contains only the unacceptable tokens
    if validated_string == input_string:
        return False
    else:
        return True

def get_logprobs(logprobs, testcases, method_name, ground_truth) -> List[TestCase]:
    # unacceptable_tokens = (' ==', '==', ' == ', ' ', '  ', ' ,', ',', ' [', '[', ']' , ' ]', '\'',' \'', '\"', ' \"')
    gen = getter(logprobs)
    all_tests: List[TestCase] = []
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
                        # print(logprob3.token)
                        if '==' in logprob3[0]:
                            for logprob4 in gen:
                                if not validate_string(logprob4[0]):
                                    continue
                                elif '#' in logprob4[0] or '\n' in logprob4[0]:  # comment here
                                    break
                                output_logprobs.append(LogProb(type=2, token=logprob4[0],
                                                               prob=np.round(np.exp(logprob4[1]) * 100, 2)))

                                second_output_logprobs.append(LogProb(type=2,token=logprob4[2][0][0], prob=np.round(np.exp(logprob4[2][0][1]) * 100, 2)))
                            try:
                                is_passed = run_testcase(ground_truth + '\n' + testcases[textcase_index] , 5)
                                testcase = TestCase(text=testcases[textcase_index], input_logprobs=input_logprobs,
                                                    output_logprobs=output_logprobs, is_valid=is_passed, second_input_logprobs=second_input_logprobs, second_output_logprobs=second_output_logprobs)
                                # print(testcase)
                                # print('******************************************************')
                                all_tests.append(testcase)
                            except TimeoutException as e:
                                print(1)
                                pass
                            finally:
                                textcase_index += 1
                                is_break = True
                                break
                        elif logprob3[0] == ',':
                            pass
                        else:
                            # print(logprob2.token)
                            if validate_string(logprob3[0]):
                                lp = LogProb(type=1, token=logprob3[0], prob=np.round(np.exp(logprob3[1]) * 100, 2))
                                second_lp = LogProb(type=1, token=logprob3[2][0][0], prob=np.round(np.exp(logprob3[2][0][1]) * 100, 2))
                                input_logprobs.append(lp)
                                second_input_logprobs.append(second_lp)
                if is_break:
                    break
    return all_tests


def get_all_tests(dataset:str, llm:str):
    import pickle
    file_name = f'raw_logprobs/{dataset}_{llm}.pkl'
    try:
        with open(file_name, 'rb') as f:
            raw_probs: List[RawLogProbs] = pickle.load(f)
    except FileNotFoundError:
        print(f'file {file_name} not found')
        raise FileNotFoundError
    if dataset == 'HumanEval':
        raw_probs = [r for idx,r in enumerate(raw_probs) if idx!=39]
    functions: List[Function] = []
    for prob in tqdm(raw_probs):
        # print(prob.API_Response.choices[0].message.content)
        # print(prob.solution + prob.testcases[0])
        try:
            testcases: List[TestCase] = get_logprobs(prob.logprobs, prob.testcases, prob.prompt, prob.solution)
        except Exception as e:
            # raise e
            print(e)
            print('here')
            continue
        # for a in prob.API_Response.choices[0].logprobs.content:
        #     print(a.token)
        # print(testcases)
        # print(prob.testcases)
        # print('----------------------------------------------')
        f = Function(prompt=prob.prompt, testcases=testcases, solution=prob.solution)
        functions.append(f)
        print(f)
    return functions