import argparse
import copy
import numpy as np
import pandas as pd
from main_train import StatisticalFeatureExtraction, extract_features, prepare_data, balance_data, train_and_evaluate, remove_unnecessary_functions, evaluate_function
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import joblib

from log_probs import get_all_tests, RawLogProbs


def evaluate_dataset_with_model(dataset: str, llm: str, mutation:False,
                                threshold: float = 0.75, topN: int = 3,
                                feature_sets: str = 'all'):
    """
    Evaluate a dataset (with a specific LLM) using a pre-trained model.
    Mimics the selection logic from `train_and_evaluate` but does not retrain.
    Uses a saved model for predictions.

    Parameters:
    - dataset (str): The dataset name.
    - llm (str): The LLM name.
    - model_path (str): Path to the saved model (e.g. 'models/trained_model_{llm}.pkl').
    - threshold (float): The classification threshold for selecting test cases.
    - topN (int): The minimum number of instances to select per group.
    - feature_sets (str): Which features to use ('all', 'input', 'output').

    Returns:
    selected_ids_per_group, total_selected, ratio (valid test case ratio), all_y_prob_with_groups
    """
    if dataset not in VALID_DATASETS:
        raise ValueError(f"Dataset {dataset} not in VALID_DATASETS: {VALID_DATASETS}")
    if llm not in VALID_LLMS:
        raise ValueError(f"LLM {llm} not in VALID_LLMS: {VALID_LLMS}")
    model_path = f"models/trained_model_{llm}_{dataset}.pkl"
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    print(f"Loading tests for dataset={dataset}, llm={llm}")
    functions = get_all_tests(dataset, llm)
    functions = remove_unnecessary_functions(functions)

    # Prepare test data
    all_testcases = []
    function_ids = []
    test_case_ids = []
    for func_id, f in enumerate(functions):
        for test_idx, test_case in enumerate(f.testcases):
            all_testcases.append(test_case)
            function_ids.append(func_id)
            test_case_ids.append((func_id, test_idx))

    print('calculating initial coverage of the functions and mutation score....')
    # coverage = evaluate_function(copy.deepcopy(functions), do_mutation=False, dataset=dataset)
    # print('Initial coverage:')
    # print(coverage)
    models_performance = {}

    strategy = StatisticalFeatureExtraction()
    features = extract_features(all_testcases, function_ids, strategy, feature_sets)
    X, y, groups = prepare_data(features)
    test_case_ids_series = pd.Series(test_case_ids, name='test_case_id')
    X = pd.concat([X.reset_index(drop=True), test_case_ids_series], axis=1)

    true_count_balanced = y.sum()
    false_count_balanced = len(y) - true_count_balanced
    print(f"Balanced dataset size - Number of valid testcases: {true_count_balanced}")
    print(f"Balanced dataset size - Number of invalid testcases: {false_count_balanced}")
    print(f'Valid Testcase Ratio" {round(true_count_balanced / (false_count_balanced + true_count_balanced), 2)}')

    X_features = X.drop(columns=['test_case_id'])

    # Predict probabilities using the pre-trained model
    if hasattr(model.named_steps['classifier'], "predict_proba"):
        y_prob = model.predict_proba(X_features)[:, 1]
    elif hasattr(model.named_steps['classifier'], "decision_function"):
        y_scores = model.decision_function(X_features)
        y_prob = 1 / (1 + np.exp(-y_scores))  # Sigmoid
    else:
        raise AttributeError("The classifier does not have predict_proba or decision_function methods.")

    # Apply threshold
    y_pred = (y_prob >= threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    cm = confusion_matrix(y, y_pred)

    print("=== Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print("Confusion Matrix:")
    print(cm)

    # Similar logic to train_and_evaluate for selection
    # Create a DataFrame with y_prob, group, test_case_id, and true label
    eval_df = pd.DataFrame({
        'y_prob': y_prob,
        'group': groups,
        'test_case_id': X['test_case_id'],
        'y_true': y
    })

    # Function to select top instances per group as in train_and_evaluate
    def select_top_instances(group_df):
        # Select instances above the threshold
        selected = group_df[group_df['y_prob'] >= threshold]
        if len(selected) >= topN:
            return selected
        else:
            needed = topN - len(selected)
            # Select top 'needed' instances from the group by probability
            remaining = group_df[~group_df.index.isin(selected.index)]
            additional = remaining.sort_values('y_prob', ascending=False).head(needed)
            return pd.concat([selected, additional])

    final_selected_df = eval_df.groupby('group').apply(select_top_instances).reset_index(drop=True)

    # Compute total number of selected instances
    total_selected = final_selected_df.shape[0]

    # Compute ratio of valid test cases in the selected instances
    label_counts = final_selected_df['y_true'].value_counts()
    label_0_count = label_counts.get(0, 0)
    label_1_count = label_counts.get(1, 0)
    ratio = round(label_1_count / total_selected, 3) if total_selected > 0 else 0

    # Format selected test_case_ids per group
    selected_ids_per_group = final_selected_df.groupby('group')['test_case_id'].apply(lambda ids: tuple(idx for _, idx in ids)).to_dict()

    print("=== Selection Statistics ===")
    print(f"Total selected instances: {total_selected}")
    print(f"Ratio of valid test cases (selected): {ratio}")
    print("============================")
    temp = copy.deepcopy(functions)
    for group, ids in selected_ids_per_group.items():
        temp[group].testcases = [te for idx, te in enumerate(temp[group].testcases) if idx in ids]
    print(f'Calculating coverage and mutation score using filtered test cases...')
    models_performance = {}

    coverage = evaluate_function(temp, do_mutation=False, dataset=dataset)
    models_performance['ensemble'] = {
        'coverage': coverage,
        'total_selected': total_selected,
        'valid_test_case_ration': ratio
    }
    print(models_performance)

if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run the script with specified dataset and LLM.")

    # Add the 'dataset' argument with restricted choices
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
        "--mutation",
        type=int,
        default=0,
        help=f"Specify if mutation should be performed or not. Its takes a lot of time to do mutation, so be careful. Choices are: {0, 1}.",
        choices=[0, 1],
        required=False
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help=f"Specify the threshold for filtering out the test cases. Choices are: {0.5, 0.65, 0.75, 0.8, 0.85, 0.9}.",
        choices=[0.5, 0.65, 0.75, 0.8, 0.85, 0.9],
        required=False
    )
    parser.add_argument(
        "--topN",
        type=int,
        default=3,
        choices=[1, 3, 5, 7],
        help=f"Specify the top N test cases. Choices are: {1, 3, 5, 7}.",
        required=False
    )
    parser.add_argument(
        "--features",
        type=str,
        default=['all'],
        choices=['all', 'input', 'output'],
        help=f"Specify the feature sets to use. Choices are: {'all', 'input', 'output'}.",
        required=False
    )
    args = parser.parse_args()

    evaluate_dataset_with_model(args.dataset, args.llm, args.mutation, args.threshold, args.topN, args.features)