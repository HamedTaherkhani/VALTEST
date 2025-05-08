from log_probs import get_all_tests
from main_train import remove_unnecessary_functions, StatisticalFeatureExtraction, extract_features
VALID_DATASETS = ['MBPP', 'HumanEval', 'LeetCode']
VALID_LLMS = ['gpt-4o', 'gpt-3.5-turbo', 'llama3', 'codeqwen',]
import numpy as np
from scipy.stats import mannwhitneyu

if __name__ == "__main__":
    feature_file_name = 'output/features.txt'
    with open(feature_file_name, 'w') as a_file:
        for dataset in VALID_DATASETS:
            for llm in VALID_LLMS:
                functions = get_all_tests(dataset, llm)
                functions = remove_unnecessary_functions(functions)
                # print(functions)
                # return
                all_testcases = []
                function_ids = []  # List to store function IDs
                test_case_ids = []  # List to store unique test case identifiers
                for func_id, f in enumerate(functions):
                    for test_idx, test_case in enumerate(f.testcases):
                        all_testcases.append(test_case)
                        function_ids.append(func_id)
                        test_case_ids.append((func_id, test_idx))  # Assign unique ID

                strategy = StatisticalFeatureExtraction()
                # print(all_testcases)
                # Modify extract_features to also handle test_case_ids if necessary
                features = extract_features(all_testcases, function_ids, strategy)

                input_mean_valid = []
                input_mean_invalid = []

                output_mean_valid = []
                output_mean_invalid = []
                valid = 0
                invalid = 0
                ##extract features statistics on valid and invalid test cases
                for i in features:
                    if i['is_valid']:
                        input_mean_valid.append(i['input_mean'])
                        output_mean_valid.append(i['output_mean'])
                        valid += 1
                    else:
                        input_mean_invalid.append(i['input_mean'])
                        output_mean_invalid.append(i['output_mean'])
                        invalid += 1
                a_file.write(f'Features for {dataset} and {llm} for valid instances:\n')
                a_file.write(f'input_mean average: {round(np.mean(input_mean_valid),3)}\n')
                a_file.write(f'output_mean average: {round(np.mean(output_mean_valid),3)}\n')

                a_file.write(f'Features for {dataset} and {llm} for invalid instances:\n')
                a_file.write(f'input_mean average: {round(np.mean(input_mean_invalid),3)}\n')
                a_file.write(f'output_mean average: {round(np.mean(output_mean_invalid),3)}\n')

                statistic1, p_value1 = mannwhitneyu(input_mean_valid, input_mean_invalid, alternative='two-sided')
                statistic2, p_value2 = mannwhitneyu(output_mean_valid, output_mean_invalid, alternative='two-sided')

                a_file.write(f'mann whitney u test for input_mean: statistics: {statistic1} p-value:{p_value1}\n')
                a_file.write(f'mann whitney u test for output_mean: statistics: {statistic2} p-value:{p_value2}\n')
                a_file.write('--------------------------------------------------------------------------------\n')

