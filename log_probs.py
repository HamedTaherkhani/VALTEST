from typing import List
import numpy as np
from function_executor import run_testcase, TimeoutException
import numpy as np
from tqdm import tqdm
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
    def __init__(self, text, input_logprobs: List[LogProb], output_logprobs: List[LogProb] = None,
                 is_valid: bool = None):
        self.text = text
        self.input_logprobs = input_logprobs
        self.output_logprobs = output_logprobs
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

def getter(res):
    for lp in res.choices[0].logprobs.content:
        yield lp


# def find_test_case(res, testcase):
#     temp_test = testcase.copy()
#     for lp in res.choices[0].logprobs.content:
#         if testcase[0] == lp.token:
#
#         for t in testcase:
#             if t == lp.token:


def get_logprobs(response, testcases, method_name, ground_truth) -> List[TestCase]:
    unacceptable_tokens = (' ==', '==', ' == ', ' ', '  ', ' ,', ',')
    gen = getter(response)
    all_tests: List[TestCase] = []
    textcase_index = 0
    for logprob in gen:
        if logprob.token == 'assert':
            for logprob2 in gen:
                is_break = False
                if '(' in logprob2.token:
                    input_logprobs: List[LogProb] = []
                    output_logprobs: List[LogProb] = []
                    for logprob3 in gen:
                        # print(logprob3.token)
                        if ')' in logprob3.token:
                            for logprob4 in gen:
                                if logprob4.token in unacceptable_tokens:
                                    continue
                                elif '#' in logprob4.token or '\n' in logprob4.token:  # comment here
                                    break
                                output_logprobs.append(LogProb(type=2, token=logprob4.token,
                                                               prob=np.round(np.exp(logprob4.logprob) * 100, 2)))
                            try:
                                is_passed = run_testcase(ground_truth + '\n' + testcases[textcase_index] , 5)
                                testcase = TestCase(text=testcases[textcase_index], input_logprobs=input_logprobs,
                                                    output_logprobs=output_logprobs, is_valid=is_passed)
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
                        elif logprob3.token == ',':
                            pass
                        else:
                            # print(logprob2.token)
                            lp = LogProb(type=1, token=logprob3.token, prob=np.round(np.exp(logprob3.logprob) * 100, 2))
                            input_logprobs.append(lp)
                if is_break:
                    break
    return all_tests


def get_all_tests(llm:str):
    import pickle
    file_name = f'raw_logprobs/{llm}.plk'
    try:
        with open(file_name, 'rb') as f:
            raw_probs: List[RawLogProbs] = pickle.load(f)
    except FileNotFoundError:
        print(f'file {file_name} not found')
        raise FileNotFoundError
    functions: List[Function] = []
    # print(raw_probs)
    for prob in tqdm(raw_probs):
        # print(prob.API_Response.choices[0].message.content)
        # print(prob.solution + prob.testcases[0])
        try:
            testcases: List[TestCase] = get_logprobs(prob.API_Response, prob.testcases, prob.prompt, prob.solution)
        except Exception as e:
            # raise e
            print('here')
            continue
        # for a in prob.API_Response.choices[0].logprobs.content:
        #     print(a.token)
        # print(testcases)
        # print(prob.testcases)
        # print('----------------------------------------------')
        f = Function(prompt=prob.prompt, testcases=testcases, solution=prob.solution)
        functions.append(f)
    return functions