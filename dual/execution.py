# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import ctypes
libgcc_s = ctypes.CDLL('libgcc_s.so.1')

from collections import defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
import logging

from _execution import check_correctness, check_correctness_with_test_cases

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def evaluate_with_test_code(
    samples,
    timeout,
    dataset_name,
):
    logger.info(f'Start evaluation with test code, timeout={timeout}')
    # Check the generated samples against test suites.
    if dataset_name == 'BigCodeBenchHard':
        existed_completion = defaultdict(set)
        results = defaultdict(defaultdict)
        for ij,sample in enumerate(samples):
            task_id = sample["task_id"]
            prompt = sample['prompt']
            test = sample['test']
            entry_point = sample['entry_point']
            completion = sample["completion"]
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            if dataset_name == 'BigCodeBenchHard' or dataset_name == 'BigCodeBench':
                is_unittest = True
            else:
                is_unittest = False
            args = (task_id, prompt, completion, test, entry_point, timeout, is_unittest)
            res = check_correctness(*args)
            print(res)
            logger.info(f'Evaluation result: {res}')
            logger.info(f'{ij} execution requests are submitted')
            results[res["task_id"]][res["completion"]] = res
    else:
        with ProcessPoolExecutor() as executor:

            futures = []
            existed_completion = defaultdict(set)
            results = defaultdict(defaultdict)

            for sample in samples:
                task_id = sample["task_id"]
                prompt = sample['prompt']
                test = sample['test']
                entry_point = sample['entry_point']
                completion = sample["completion"]
                if completion in existed_completion[task_id]:
                    continue
                existed_completion[task_id].add(completion)
                if dataset_name == 'BigCodeBenchHard' or dataset_name == 'BigCodeBench':
                    is_unittest = True
                else:
                    is_unittest = False
                args = (task_id, prompt, completion, test, entry_point, timeout, is_unittest)
                # print(prompt)
                # print(completion)
                # print(test)
                future = executor.submit(check_correctness, *args)
                futures.append(future)
            logger.info(f'{len(futures)} execution requests are submitted')

            for idx, future in enumerate(as_completed(futures)):
                logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
                result = future.result()
                print(result)
                results[result["task_id"]][result["completion"]] = result

    logger.info('execution finished! start parsing results')
    samples_with_result = []
    for sample in samples:
        task_id = sample["task_id"]
        completion = sample["completion"]
        result = results[task_id][completion]
        sample["result"] = result["result"]
        sample["passed"] = result["passed"]
        samples_with_result.append(sample)

    assert len(samples_with_result) == len(samples), "Some problems are not attempted."

    return samples_with_result

def evaluate_with_test_cases(
    solutions,
    test_cases_dict,
    timeout,
    limit,
    dataset_name
):
    logger.info(f'Start evaluation with test cases, timeout={timeout}, limit={limit}')
    # Check the generated solutions against test suites.
    if dataset_name == 'BigCodeBenchHard':
        futures = []
        results_list = []
        existed_completion = defaultdict(set)
        print('inside evaluate_with_test_cases')
        for idx, solution in enumerate(solutions):
            task_id = solution['task_id']
            print(f'task_id: {task_id}')
            prompt = solution['prompt']
            completion = solution['completion']
            if completion in existed_completion[task_id]:
                continue
            existed_completion[task_id].add(completion)
            task_test_cases = test_cases_dict[task_id]
            if not task_test_cases:
                continue
            # get limited test cases
            limited_task_test_cases = [cases_per_sample[:limit] for cases_per_sample in task_test_cases]
            limited_task_test_cases = sum(limited_task_test_cases, [])
            limited_task_test_cases = [a[0] for a in limited_task_test_cases]

            args = (task_id, prompt, completion, list(set(limited_task_test_cases)), timeout, True)
            result = check_correctness_with_test_cases(*args)
            results_list.append(result)
            print(f'execution result for {task_id}: {idx}')
            logger.info('[{}/{}] execution completed'.format(idx + 1, task_id))
    else:
        with ProcessPoolExecutor() as executor:
            futures = []
            results_list = []
            existed_completion = defaultdict(set)

            for solution in solutions:
                task_id = solution['task_id']
                prompt = solution['prompt']
                completion = solution['completion']
                if completion in existed_completion[task_id]:
                    continue
                existed_completion[task_id].add(completion)
                task_test_cases = test_cases_dict[task_id]
                if not task_test_cases:
                    continue
                # get limited test cases
                limited_task_test_cases = [cases_per_sample[:limit] for cases_per_sample in task_test_cases]
                limited_task_test_cases = sum(limited_task_test_cases, [])
                limited_task_test_cases = [a[0] for a in limited_task_test_cases]

                args = (task_id, prompt, completion, list(set(limited_task_test_cases)), timeout, False)
                future = executor.submit(check_correctness_with_test_cases, *args)
                futures.append(future)

            logger.info(f'{len(futures)} execution requests are submitted')
            for idx, future in enumerate(as_completed(futures)):
                logger.info('[{}/{}] execution completed'.format(idx+1, len(futures)))
                result = future.result()
                results_list.append(result)

    logger.info('execution finished!')
    return results_list

