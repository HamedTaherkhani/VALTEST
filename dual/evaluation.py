# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import statistics
import numpy as np
from collections import defaultdict
import logging
from typing import List, Union
import itertools
from function_executor_codet import check_correctness_with_test_cases
import pickle
from function_executor import run_unit_tests_parallel, run_unit_tests_sequential
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

logging.basicConfig(
    format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)

def _dictionized_ground_truth_results(ground_truth_exec_results):
    ground_truth_results_by_task_and_solution = defaultdict(defaultdict)
    for result in ground_truth_exec_results:
        ground_truth_results_by_task_and_solution[result['task_id']][result['completion']] = result['passed']
    return ground_truth_results_by_task_and_solution

def _turn_solution_scores_into_choose_count(sorted_solution_scores, topk):
    # sorted_solution_scores: list of (solution, score)
    # if wrapped, sorted_solution_scores is list of ([solutions], score)
    # return list of (solution, choose_count)
    wrapped = True if type(sorted_solution_scores[0][0]) == list else False
    result = []
    if wrapped:
        last_score = sorted_solution_scores[0][1]
        merged_solutions_and_score = [sorted_solution_scores[0]]
        for solutions, score in sorted_solution_scores[1:]:
            if score == last_score:
                last_solutions = merged_solutions_and_score[-1][0]
                merged_solutions_and_score[-1] = (last_solutions + solutions, score)
            else:
                merged_solutions_and_score.append((solutions, score))
                last_score = score
        for solutions_and_score in merged_solutions_and_score:
            result.append((solutions_and_score[0], 1))  # choose one from solutions_and_score
    else:
        topk_scores = sorted(list(set([i[1] for i in sorted_solution_scores])), reverse=True)
        for score in topk_scores:
            solutions = [s[0] for s in sorted_solution_scores if s[1] == score]
            result.append((solutions, 1))

    if len(result) >= topk:
        return result[:topk]
    else:
        intial_choose_count = [1]*len(result)
        for i in range(topk-len(result)):
            intial_choose_count[i%len(result)] += 1
        for i, choose_count in enumerate(intial_choose_count):
            result[i] = (result[i][0], choose_count)
        return result

def compute_vr_func(func, best_sol, idx, is_unittest=False):
    # c) Run all generated testcases on best solution
    gen_test_texts = [tc.text for tc in func.testcases]
    # gen_results = run_test_cases(best_sol, gen_test_texts, timeout=300)
    if is_unittest:
        gen_results = run_unit_tests_parallel(code_str=best_sol, test_list=gen_test_texts)
        gen_results = [gen[0] for gen in gen_results]
    else:
        gen_results = check_correctness_with_test_cases(idx, func.prompt, best_sol, gen_test_texts, timeout=5, is_unittest=is_unittest)['result']
    print(gen_results)
    # print(best_sol)
    # for a in gen_test_texts:
    #     print(a)
    # print(gen_test_texts)
    # print(gen_results)
    # print('*'*100)
    # Update each TestCase.prediction_is_valid and accumulate TP/FP/TN/FN
    tp = fp = tn = fn = 0
    for tc, passed in zip(func.testcases, gen_results):
        tc.prediction_is_valid = 1 if passed else 0
        print(passed)
        # Only count if tc.is_valid is not None
        if tc.is_valid is not None:
            if tc.is_valid == 1 and tc.prediction_is_valid == 1:
                tp += 1
            elif tc.is_valid == 0 and tc.prediction_is_valid == 1:
                fp += 1
            elif tc.is_valid == 0 and tc.prediction_is_valid == 0:
                tn += 1
            elif tc.is_valid == 1 and tc.prediction_is_valid == 0:
                fn += 1
            # If is_valid not in {0,1}, skip
        else:
            print('tc.is_valid is none')
    return tp, fp, tn, fn, func

def save_updated_functions(output_pickle_path, functions, tp, fp, tn, fn):
    with open(output_pickle_path, "wb") as f:
        pickle.dump(functions, f)
    # print(f'all groups size : {all_groups}')
    # Print summary
    total_functions = len(functions)
    print(f"Total Functions processed: {total_functions}")
    # print(f"Functions whose best solution passed all original tests: {num_funcs_passing_original}")
    print()
    print("Across all generated testcases:")
    print(f"  True Positives (is_valid=1 & pred=1):  {tp}")
    print(f"  False Positives (is_valid=0 & pred=1): {fp}")
    print(f"  True Negatives (is_valid=0 & pred=0):  {tn}")
    print(f"  False Negatives (is_valid=1 & pred=0): {fn}")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    print(f'precision is {precision}')
    print(f'recall is {recall}')
    print(f'f1 score is {2*(precision*recall)/(precision+recall)}')

def find_scores(func, dual_res, valtest):
    tests_scores = {}
    actual_scores = {}
    for idx,test in enumerate(valtest.testcases):
        tests_scores[test.text] = test.prediction_y_prob
        actual_scores[test.text] = test.is_valid
    all_codes_len = len(set([r[0] for r in dual_res]))
    print(f"all_codes_len: {all_codes_len}")
    all_passed_tests = {}
    if dual_res is not None:
        for pair in dual_res:
            if pair[1] in all_passed_tests.keys():
                all_passed_tests[pair[1]] +=1
            else:
                all_passed_tests[pair[1]] = 1

        for key,value in all_passed_tests.items():
            all_passed_tests[key] = all_passed_tests[key] / all_codes_len
        for key,value in tests_scores.items():
            if key not in all_passed_tests.keys():
                all_passed_tests[key] = 0
    else:
        all_passed_tests = tests_scores.copy()
    if len(all_passed_tests.keys()) != len(tests_scores.keys()) or len(all_passed_tests.keys()) != len(actual_scores.keys()) or len(tests_scores.keys()) != len(actual_scores.keys()):
        logger.info(all_passed_tests)
        logger.info(tests_scores)
        logger.info(actual_scores)
    return all_passed_tests, tests_scores, actual_scores


def compute_validity_rate(ranked_result, ground_truth_exec_result, functions, output_pickle_path, strategy, is_unittest,passed_solution_test_case_pairs_by_task, valtest):
    print(f'ranked_result len: {len(ranked_result)}')
    print(ranked_result['0'])
    print(f'functions len: {len(functions)}')
    ground_truth_exec_result = _dictionized_ground_truth_results(ground_truth_exec_result)

    print(f'ground_truth_exec_result len: {len(ground_truth_exec_result)}')
    # print(ground_truth_exec_result['0'])
    tp = fp = tn = fn = 0
    if strategy in ('one-shot', 'dual'):
        for idd,func in enumerate(functions):
            if strategy == 'one-shot':
                best_sol = func.generated_solutions[0]
                tp_, fp_, tn_, fn_, func = compute_vr_func(func, best_sol, idd, is_unittest)
            elif strategy == 'dual':
                if f'{idd}' in ranked_result and ranked_result[f'{idd}']:
                    best_sol = ranked_result[f'{idd}'][0][0][0]
                    # print(idd)
                    # print(f'best_sol: \n{best_sol}')
                else:
                    best_sol = list(ground_truth_exec_result[f'{idd}'].keys())[0]
                    # print(f'best sol is \n{best_sol}')
                tp_, fp_, tn_, fn_, func = compute_vr_func(func, best_sol, idd, is_unittest)
            tp += tp_
            fp += fp_
            tn += tn_
            fn += fn_
        save_updated_functions(output_pickle_path, functions, tp, fp, tn, fn)
    elif strategy == 'dual_average_weights':
        all_dual_scores = {}
        all_valtest_scores = {}
        all_actual_scores = {}
        for idd, func in enumerate(functions):
            found = None
            for val in valtest:
                if val.prompt == func.prompt:
                    found = val
            if found is None:
                logger.info(f'{func.prompt} is not in valtest')
                logger.info(f'{idd} not found in valtest')
                continue
            if f'{idd}' in passed_solution_test_case_pairs_by_task.keys():
                dual_res = passed_solution_test_case_pairs_by_task[f'{idd}']
            else:
                dual_res = None
            dual_scores, valtest_scores, actual_scores = find_scores(func, dual_res, found)
            all_dual_scores = {**all_dual_scores, **dual_scores}
            all_valtest_scores = {**all_valtest_scores, **valtest_scores}
            all_actual_scores = {**all_actual_scores, **actual_scores}
        # Build DataFrame
        logger.info(len(list(all_dual_scores.keys())))
        logger.info(len(list(all_dual_scores.values())))
        logger.info(len(list(all_valtest_scores.values())))
        logger.info(len(list(all_actual_scores.values())))
        df = pd.DataFrame({
            "test": list(all_dual_scores.keys()),
            "agreement": list(all_dual_scores.values()),
            "token_prob": list(all_valtest_scores.values()),
            "actual_validity": list(all_actual_scores.values())
        })

        # Features and labels
        X = df[["agreement", "token_prob"]].values
        y_true = df["actual_validity"].values

        # 5-fold cross-validation: predict probabilities
        model = GaussianNB()
        cv_probs = cross_val_predict(model, X, y_true, cv=5, method="predict_proba")[:, 1]

        # Threshold probabilities at 0.5 to get predicted labels
        y_pred = (cv_probs >= 0.5).astype(int)

        # Add predictions to DataFrame
        df["valid_prob_cv"] = cv_probs
        df["predicted_validity"] = y_pred

        # Compute metrics
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        # Print results
        print(df[["test", "agreement", "token_prob", "valid_prob_cv", "actual_validity", "predicted_validity"]])
        print("\nPrecision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        # print("\nClassification Report:\n", classification_report(y_true, y_pred))
    else:
        raise ValueError('Strategy not recognized')



def get_result_of_sorted_solutions(ground_truth_results_list, sorted_solutions_by_task, topks=[1,2,10]):
    # sorted_solutions_by_task {task_id: [([solutions], score), ...]}
    def _count_correct(solutions: list, ground_truth_results: dict) -> int:
        return sum([ground_truth_results[s] for s in solutions])
    
    ground_truth_results = _dictionized_ground_truth_results(ground_truth_results_list)
    topk_results = dict()
    for topk in topks:
        random_pass_at_k_by_task = pass_at_K_by_task(ground_truth_results_list, k=topk)
        pass_rates = []
        for task_id in ground_truth_results.keys():
            all_wrong_probability = 1
            if task_id in sorted_solutions_by_task and sorted_solutions_by_task[task_id]:
                solutions_and_probability = _turn_solution_scores_into_choose_count(sorted_solutions_by_task[task_id], topk)
                for solutions, choose_count in solutions_and_probability:
                    current_wrong_prob = _estimator(len(solutions), _count_correct(solutions, ground_truth_results[task_id]), 1)
                    repeat_current_wrong_prob = pow(current_wrong_prob, choose_count)
                    all_wrong_probability *= repeat_current_wrong_prob
                pass_rates.append(1-all_wrong_probability)
            else:
                pass_rates.append(random_pass_at_k_by_task[task_id])
        
        # the avg rate of all tasks
        topk_results[f'pass@{topk}'] = round(statistics.mean(pass_rates), 4)
    logger.info(topk_results)

def pass_at_K_by_task(results, k):
    result_dict = defaultdict(list)
    for line in results:
        result_dict[line['task_id']].append(line['passed'])
    result = dict()
    for task_id in result_dict.keys():
        total = len(result_dict[task_id])
        correct = sum(result_dict[task_id])
        score = _estimate_pass_at_k(total, [correct], k)[0]
        result[task_id] = score
    return result

def pass_at_K(results, k = [1, 10, 100]):
    def _turn_list_into_dict(result_lines):
        result_dict = defaultdict(list)
        for line in result_lines:
            result_dict[line['task_id']].append(line['passed'])
        return result_dict

    # Calculate pass@k.
    total, correct = [], []
    for passed in _turn_list_into_dict(results).values():
        total.append(len(passed))
        correct.append(sum(passed))

    total = np.array(total)
    correct = np.array(correct)

    ks = k
    pass_at_k = {f"pass@{k}": round(_estimate_pass_at_k(total, correct, k).mean(), 4)
                 for k in ks if (total >= k).all()}
    logger.info(pass_at_k)

def _estimator(n: int, c: int, k: int) -> float:
    """
    Calculates comb(n - c, k) / comb(n, k).
    """
    if n - c < k:
        return 0
    return np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def _estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([1.0 - _estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])