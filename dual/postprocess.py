# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from log_probs import Function

STOP_TOKEN = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']


import ast

def extract_function_name(code):
    # Parse the code into an AST
    tree = ast.parse(code)

    # Traverse the AST to find the first function definition
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node.name

    return None

class PostProcessor:
    @staticmethod
    def map_task_id_for_solution(functions:list[Function], dataset):
        # database = dict()
        # raw_problems = Tools.load_tasks(source_path)
        # for task_id in raw_problems.keys():
        #     database[raw_problems[task_id]['prompt']] = raw_problems[task_id]
        #
        # result = []
        # predictions = Tools.load_jsonl(predict_path)
        # for pre in predictions:
        #     task = database[pre['prompt']]
        #     if not pre['samples']:
        #         result.append({
        #             'task_id': task['task_id'],
        #             'prompt': pre['prompt'],
        #             'test': task['test'],
        #             'entry_point': task['entry_point'],
        #             'completion': 'empty solution here, execution will fail'
        #         })
        #     for sample in pre['samples']:
        #         # processed_code = PostProcessor.solution_extract(sample)
        #         result.append({
        #             'task_id': task['task_id'],
        #             'prompt': pre['prompt'],
        #             'test': task['test'],
        #             'entry_point': task['entry_point'],
        #             'completion': sample
        #         })
        result = []
        for ix, func in enumerate(functions):
            tests = None
            # print(ix)
            for d in dataset:
                if d['prompt'] == func.prompt:
                    tests = d['tests']
            if tests is None:
                raise 'BAD'
            for comp in func.generated_solutions:
                result.append({
                    'task_id': f'{ix}',
                    'prompt': func.prompt,
                    'test':tests,
                    'entry_point': extract_function_name(func.solution),
                    'completion': comp
                })
        return result

    @staticmethod
    def map_task_id_for_test_case(functions: list[Function], valtest: list[Function], dataset):
        # database = dict()
        # raw_problems = Tools.load_tasks(source_path)
        # for task_id in raw_problems.keys():
        #     database[raw_problems[task_id]['prompt']] = raw_problems[task_id]
        #
        test_cases_by_task = defaultdict(list)
        # predictions = Tools.load_jsonl(predict_path)
        # for pre in predictions:
        #     task = database[pre['prompt']]
        #     for sample in pre['samples']:
        #         test_cases = PostProcessor.test_case_extract(sample, task['entry_point'])
        #         test_cases_by_task[task['task_id']].append(test_cases)

        for idx, func in enumerate(functions):
            testcases = None
            for item in valtest:
                if func.prompt == item.prompt:
                    testcases = [(tt.text, tt.prediction_y_prob) for tt in item.testcases]
            if testcases is None:
                print('too bad')
                testcases = []
            test_cases_by_task[f'{idx}'].append(testcases)
        return test_cases_by_task

    @staticmethod
    def solution_extract(content):
        for identifier in STOP_TOKEN:
            if identifier in content:
                content = content.split(identifier)[0]
        return content
    
    @staticmethod
    def test_case_extract(content, entry_point):
        def _truncate(content):
            for identifier in STOP_TOKEN:
                if identifier in content:
                    content = content.split(identifier)[0]
            return content.strip()
        
        split_by_assert = [f'assert {part}'.strip() for part in f'assert {content}'.split('assert ') if (entry_point.strip() in part) and len(part.strip()) > 0]
        truncated_test_cases = [_truncate(i) for i in split_by_assert]
        checked_assertions = [i for i in truncated_test_cases if PostProcessor._check_test_case_validation(i)]
        return checked_assertions

    @staticmethod
    def _check_test_case_validation(test_case):
        if len(test_case.strip()) < 1:
            return False
        if 'assert' not in test_case:
            return False
        try:
            multi_line_test_case = test_case.replace("\n", "\n    ")
            assert_in_a_block = f'try:\n    {multi_line_test_case}\nexcept:\n    pass\n'
            compile(assert_in_a_block, '', 'exec')
            return True
        except Exception:
            return False