import pickle
import sys
import traceback
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Any
from log_probs import Function
from function_executor import run_test_cases
from generate_solutions import IMPORT_HEADER
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from function_executor_codet import check_correctness_with_test_cases
from tqdm import tqdm
import math
from loaders import HumanEvalLoader, MBPPLoader, BigCodeLoader
from evaluation import compute_validity_rate, compute_vr_func, save_updated_functions


def _evaluate_solution_on_generated_tests(
    func: Function, solution_code: str
) -> List[int]:
    """
    Given a Function object and one candidate solution (as a string),
    run all its generated TestCase.text assertions using run_test_cases.

    Returns a list of indices of generated testcases that passed.
    """
    solution_code = IMPORT_HEADER + solution_code
    test_texts: List[str] = [tc.text for tc in func.testcases]
    results: List[bool] = check_correctness_with_test_cases(task_id=1, prompt=func.prompt, completion=solution_code, test_cases=test_texts, timeout=5)['result']
    passed_indices: List[int] = [i for i, passed in enumerate(results) if passed]
    return passed_indices


# -------------------------------------------------------------------------------
# Run original tests (assumed to be one block of assertions) on a candidate solution
# -------------------------------------------------------------------------------
def _passes_original_tests(solution_code: str, original_tests: str) -> bool:
    """
    Given solution code and a block of original tests (assertions),
    return True if all original tests pass, False otherwise.
    """
    if not original_tests:
        # No original tests provided; treat as passing by default
        return True

    namespace: Dict[str, Any] = {}
    try:
        # First, define the function
        exec(solution_code, namespace)
    except Exception:
        return False

    try:
        # Then run the entire original_tests string in the same namespace
        exec(original_tests, namespace)
        return True
    except AssertionError:
        return False
    except Exception:
        return False


# -------------------------------------------------------------------------------
# Pick best solution according to dual execution agreement (“CODET”)
# -------------------------------------------------------------------------------
def find_best_solution_for_function_v2(func: Function, valtest: List[Function]) -> str:
    """
    Implements CODET's dual execution agreement to pick the single best solution
    from func.generated_solutions. Returns the chosen solution string. If
    generated_solutions is empty, returns an empty string.
    """
    test_scores = None
    for aa in valtest:
        if aa.prompt == func.prompt:
            val = aa
            test_scores = [a.prediction_y_prob for a in aa.testcases]

    candidates = func.generated_solutions
    if not candidates:
        return ""

    # Map from frozenset of passed-generated-test indices -> list of solution strings
    grouping: Dict[frozenset, List[str]] = defaultdict(list)
    passed_set_scores: Dict[frozenset, int] = {}

    for sol in candidates:
        passed_indices = _evaluate_solution_on_generated_tests(func, sol)
        if test_scores is None:
            total_score = len(passed_indices)
        else:
            total_score = sum(tt for ix, tt in enumerate(test_scores) if ix in passed_indices)
        passed_key = frozenset(passed_indices)
        grouping[passed_key].append(sol)
        passed_set_scores[passed_key] = total_score
        # print(f'total score: {total_score}')

    # Score each group: score = (#solutions in group) * (#generated-tests passed)
    best_key = None
    best_score = -1

    for key, sols_in_group in grouping.items():
        num_solutions = len(sols_in_group)
        # print(f'num solutions: {num_solutions}')
        passed_set_score = passed_set_scores[key]
        score = num_solutions * passed_set_score
        print(f'score: {score}')
        if score > best_score:
            best_score = score
            best_key = key
    print('*'*100)
    # Fallback if something went wrong
    if best_key is None:
        return candidates[0]

    return grouping[best_key][0]


def find_best_solution_for_function_v1(func: Function) -> (str,int):
    """
    Implements CODET's dual execution agreement to pick the single best solution
    from func.generated_solutions. Returns the chosen solution string. If
    generated_solutions is empty, returns an empty string.
    """
    candidates = func.generated_solutions
    if not candidates:
        return ""

    # Map from frozenset of passed-generated-test indices -> list of solution strings
    grouping: Dict[frozenset, List[str]] = defaultdict(list)
    passed_set_sizes: Dict[frozenset, int] = {}

    for sol in candidates:
        passed_indices = _evaluate_solution_on_generated_tests(func, sol)
        passed_key = frozenset(passed_indices)
        grouping[passed_key].append(sol)
        passed_set_sizes[passed_key] = len(passed_indices)

    # Score each group: score = (#solutions in group) * (#generated-tests passed)
    best_key = None
    best_score = -1

    for key, sols_in_group in grouping.items():
        num_solutions = len(sols_in_group)
        num_tests_passed = passed_set_sizes[key]
        score = math.sqrt(num_solutions) * num_tests_passed
        if score > best_score:
            best_score = score
            best_key = key
    print(f'best_score: {best_score}')
    # Fallback if something went wrong
    if best_key is None:
        return candidates[0]

    return grouping[best_key][0] , len(grouping)


# -------------------------------------------------------------------------------
# Main processing: load, select best, run both original & generated tests, update, and compute metrics
# -------------------------------------------------------------------------------
def perform_dual_agreement(
    input_pickle_path: str, output_pickle_path: str, valtest_scores_dir:str, approach:str, dataset, timeout, test_case_limit, dataset_name
) -> None:
    with open(input_pickle_path, "rb") as f:
        functions: List[Function] = pickle.load(f)

    with open(valtest_scores_dir, "rb") as f:
        valtest: List[Function] = pickle.load(f)
    # functions = functions[:10]
    # valtest = valtest[:10]

    print(f'functions len: {len(functions)}')
    print(f'valtest len: {len(valtest)}')
    print(f'dataset len: {len(dataset)}')
    import logging

    from postprocess import PostProcessor
    from execution import evaluate_with_test_code, evaluate_with_test_cases
    from io_utils import Tools
    from agreement import DataManager, DualAgreement
    from evaluation import pass_at_K, get_result_of_sorted_solutions

    logging.basicConfig(
        format="SystemLog: [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    if dataset_name == 'BigCodeBenchHard':
        is_unittest=True
    else:
        is_unittest=False
    logger = logging.getLogger(__name__)

    handled_solutions = PostProcessor.map_task_id_for_solution(functions, dataset)
    handled_test_cases = PostProcessor.map_task_id_for_test_case(functions, valtest, dataset)

    ground_truth_exec_result = evaluate_with_test_code(samples=handled_solutions, timeout=timeout, dataset_name=dataset_name)
    # ground_truth_exec_result = [00]
    count = 0
    for aa in ground_truth_exec_result:
        if aa['passed']:
            count += 1
    print(f'total count: {len(ground_truth_exec_result)}')
    print(f'total pass: {count}')
    dual_exec_result = evaluate_with_test_cases(handled_solutions, handled_test_cases, timeout=timeout,
                                                limit=test_case_limit, dataset_name=dataset_name)
    # print(ground_truth_exec_result[0])
    print('dual result')
    print(f'len dual result: {len(dual_exec_result)}')
    print(dual_exec_result[0])
    count_duel = 0
    for aa in dual_exec_result:
        for ii in aa['passed']:
            if ii:
                count_duel += 1
    print(f'total correct dual result: {count_duel}')
    # Tools.dump_pickle(os.path.join(args.cache_dir, 'ground_truth_exec_result.pkl'), ground_truth_exec_result)
    # Tools.dump_pickle(os.path.join(args.cache_dir, 'dual_exec_result.pkl'), dual_exec_result)
    print('*'*100)
    print(len(dual_exec_result))
    print(len(handled_solutions))
    print(len(handled_test_cases))
    print('*'*100)
    data_manager = DataManager(dual_exec_result, handled_solutions, handled_test_cases, test_case_limit)
    set_consistency = DualAgreement(data_manager)
    ranked_result = set_consistency.get_sorted_solutions_without_iter(use_valtest_scores=False)
    compute_validity_rate(ranked_result, ground_truth_exec_result, functions, output_pickle_path,'other',is_unittest=is_unittest)
    # print(ranked_result)
    # logger.info('pass rates of ranked solutions')
    # get_result_of_sorted_solutions(ground_truth_exec_result, ranked_result)
    # logger.info('pass rates of random solutions')
    # pass_at_K(ground_truth_exec_result)



def process_functions(
    input_pickle_path: str, output_pickle_path: str, valtest_scores_dir:str, approach:str
) -> None:
    """
    1. Load a list of Function objects from input_pickle_path.
    2. For each Function:
         a. Find best generated solution via dual agreement.
         b. Run original tests on that best solution to see if it passes them all.
         c. Run all generated testcases on that best solution; set each TestCase.prediction_is_valid.
    3. Count:
         - Number of Functions whose best solution passes all original tests.
         - True Positives, False Positives, True Negatives, False Negatives across all generated testcases.
    4. Write the updated list of Function objects (with updated prediction_is_valid fields) to output_pickle_path.
    5. Print summary statistics.
    """
    # Load
    with open(input_pickle_path, "rb") as f:
        functions: List[Function] = pickle.load(f)

    with open(valtest_scores_dir, "rb") as f:
        valtest: List[Function] = pickle.load(f)

    num_funcs_passing_original = 0
    tp = fp = tn = fn = 0
    all_groups = []
    for idx, func in enumerate(tqdm(functions)):
        print(f'idx is {idx}')

        # a) Find best solution
        if approach == 'dual_agreement':
            best_sol, grp = find_best_solution_for_function_v1(func)
        elif approach == 'dual_agreement_valtest':
            best_sol = find_best_solution_for_function_v2(func, valtest)
        else:
            raise "approach not recognized"
        # all_groups.append(grp)
        # best_sol = func.generated_solutions[0]
        # best_sol = func.solution
        best_sol = IMPORT_HEADER + best_sol
        # b) Check original tests
        if _passes_original_tests(best_sol, func.original_tests):
            num_funcs_passing_original += 1
        tp_, fp_, tn_, fn_, func = compute_vr_func(func, best_sol, idx)
        tp += tp_
        fp += fp_
        tn += tn_
        fn += fn_

    # Save updated Functions back to pickle
    save_updated_functions(output_pickle_path, functions, tp, fp, tn, fn)

# -------------------------------------------------------------------------------
# Command-line entry point
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the script with specified dataset and LLM.")

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
    parser.add_argument(
        "--approach",
        type=str,
        default='dual_agreement',
        choices=['dual_agreement', 'dual_agreement_valtest'],
        required=False,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        required=False,
    )
    parser.add_argument(
        "--testcase_limit",
        type=int,
        default=20,
        required=False,
    )
    args = parser.parse_args()
    file_name = f'output/dual_agreement/{args.dataset}_{args.llm}_{args.approach}.txt'
    # file_name = f'output/RQ2/{args.dataset}_{args.llm}_{args.threshold}_{args.topN}.txt'
    # file_name = f'output/RQ3/second_output/{args.dataset}_{args.llm}_{args.features}.txt'
    print(f'Writing the output to {file_name}')
    with open(file_name, 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        input_path = f'unfiltered_testcases/with_generated_solutions/{args.dataset}_{args.llm}_processed.pkl'
        output_pickle = f'unfiltered_testcases/with_generated_solutions/filtered/{args.dataset}_{args.llm}_processed.pkl'
        valtest_scores_dir = f'filtered_testcases/{args.dataset}_{args.llm}.pkl'
        real_dataset = []
        if args.dataset == 'MBPP':
            tests = MBPPLoader().get_tests()
            prompts = MBPPLoader().get_prompts()
            print(f'tests is {len(tests)}')
            print(f'prompts is {len(prompts)}')
            for idx, d in enumerate(tests):
                real_dataset.append({
                    'tests': d,
                    'prompt': prompts[idx],
                })
        elif args.dataset == 'HumanEval':
            he = HumanEvalLoader().get_human_eval()
            tests = HumanEvalLoader().get_final_test_cases()
            prompts = [p['prompt'] for p in he['test']]
            for idx, d in enumerate(tests):
                real_dataset.append({
                    'tests': d,
                    'prompt': prompts[idx],
                })
        elif args.dataset == 'BigCodeBenchHard':
            bg = BigCodeLoader(hard=1)
            tests = bg.get_tests()
            prompts = bg.get_prompts()
            for idx, d in enumerate(tests):
                real_dataset.append({
                    'tests': d,
                    'prompt': prompts[idx],
                })

        # process_functions(input_path, output_pickle, valtest_scores_dir, args.approach)
        perform_dual_agreement(input_path, output_pickle, valtest_scores_dir, args.approach, real_dataset, args.timeout, args.testcase_limit, args.dataset)