import argparse
import sys
from pyexpat import features
import copy
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier  # For XGBoost
from lightgbm import LGBMClassifier  # For LightGBM
from log_probs import LogProb, TestCase, get_all_tests, RawLogProbs, Function
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.impute import SimpleImputer
from test_coverage import measure_coverage
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.python.keras.layers import Dense, Dropout
from mutation_testing import perform_overall_mutation_testing, get_top_level_function_names
from functools import partial
import matplotlib.pyplot as plt
import warnings
import pickle
import joblib
warnings.filterwarnings("ignore")

# from catboost import CatBoostClassifier
class FeatureExtractionStrategy:
    def extract_features(self, log_probs: List[LogProb]) -> dict:
        raise NotImplementedError

import math

def calculate_perplexity(probabilities):
    # Ensure that all probabilities are greater than zero
    probabilities = [p for p in probabilities if p > 0]
    N = len(probabilities)
    if N == 0:
        return 0  # Avoid division by zero if the list is empty or all probabilities are zero

    # Calculate the average log probability
    avg_log_prob = sum(math.log2(p) for p in probabilities) / N

    # Calculate the perplexity
    perplexity = 2 ** (-avg_log_prob)
    return perplexity


# Concrete Strategy for Statistical Feature Extraction
class StatisticalFeatureExtraction(FeatureExtractionStrategy):
    def extract_features(self, log_probs: List[LogProb]) -> dict:
        probs = [lp.prob for lp in log_probs]
        if not probs:
            return {'mean': 0, 'max': 0, 'min': 0, 'sum': 0, 'total':0,
                    'variance':0,
                    # 'kurtosis':0,
                    # 'entropy':0
                    }
        return {
            'mean': np.mean(probs),
            'max': np.max(probs),
            'min': np.min(probs),
            'sum': np.sum(probs),
            'total': len(probs),
            'variance': np.var(probs) if np.var(probs) is not None else 0,
            # 'kurtosis': scipy.stats.kurtosis(probs) if scipy.stats.kurtosis(probs) is not None else 0,
            # 'entropy': scipy.stats.entropy(probs) if scipy.stats.entropy(probs) is not None else 0,

        }


# Factory for ML Models
class ModelFactory:
    @staticmethod
    def get_model(model_name: str, input_dim: int = None):
        """
        Returns a scikit-learn compatible model based on the provided model name.

        Parameters:
        - model_name: str, name identifier for the model.
        - input_dim: int, number of input features (required for deep_nn).

        Returns:
        - model: scikit-learn estimator.
        """
        if model_name == 'ensemble':
            # *** Added Ensemble Model ***
            # Define the base models for the ensemble
            base_models = [
                ('lr', LogisticRegression()),
                ('svm', SVC(probability=True)),
                ('rf', RandomForestClassifier()),
                ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                ('lgbm', LGBMClassifier()),
                ('adaboost', AdaBoostClassifier()),
                ('gradient_boosting', GradientBoostingClassifier())

            ]
            # Create a VotingClassifier with soft voting
            ensemble_model = VotingClassifier(estimators=base_models, voting='soft')
            return ensemble_model
        else:
            models = {
                'logistic_regression': LogisticRegression(),
                'svm': SVC(probability=True),
                'decision_tree': DecisionTreeClassifier(),
                'random_forest': RandomForestClassifier(),
                'gradient_boosting': GradientBoostingClassifier(),
                'xgboost': XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                'lightgbm': LGBMClassifier(),
                # 'catboost': CatBoostClassifier(verbose=0),
                'knn': KNeighborsClassifier(),
                'naive_bayes': GaussianNB(),
                'mlp': MLPClassifier(max_iter=500),
                'adaboost': AdaBoostClassifier(),
                'deep_nn': KerasClassifier(
                    build_fn=partial(create_deep_nn, input_dim=input_dim),
                    loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'],
                    epochs=150,
                    batch_size=32,
                    verbose=0
                )
            }
            return models.get(model_name)


# Function to extract features from TestCases
def extract_features(test_cases: List[TestCase], function_ids: List[int], strategy: FeatureExtractionStrategy,
                     feature_sets: str = 'all') -> List[dict]:
    features = []

    for test_case, func_id in zip(test_cases, function_ids):
        # Extract input features and add a prefix to each key
        input_features = {f'input_{k}': v for k, v in strategy.extract_features(test_case.input_logprobs).items()}
        second_input_features = {f'second_input_{k}': v for k, v in
                                 strategy.extract_features(test_case.second_input_logprobs).items()}

        # Extract output features and add a prefix to each key
        output_features = {f'output_{k}': v for k, v in strategy.extract_features(test_case.output_logprobs).items()}
        second_output_features = {f'second_output_{k}': v for k, v in
                                  strategy.extract_features(test_case.second_output_logprobs).items()}

        # Based on feature_sets argument, combine relevant features
        if feature_sets == 'input':
            combined_features = {**input_features, **second_input_features}
        elif feature_sets == 'output':
            combined_features = {**output_features, **second_output_features}
        else:  # 'all'
            combined_features = {**input_features, **output_features, **second_input_features, **second_output_features}

        # Add 'is_valid' and 'function_id' to the feature set
        combined_features['is_valid'] = test_case.is_valid
        combined_features['function_id'] = func_id

        features.append(combined_features)

    return features


# Function to prepare data for ML training
def prepare_data(features: List[dict]):
    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(features)
    X = df.drop(columns=['is_valid', 'function_id'])
    y = df['is_valid']
    groups = df['function_id']  # Extract groups based on function IDs
    return X, y, groups



# Function to train and evaluate models
def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model_name: str,
    threshold: float = 0.8,
    min_instances_per_group: int = 5
):
    """
    Train and evaluate a model with a custom classification threshold,
    capturing the association between predicted probabilities and groups,
    and selecting instances based on threshold and group constraints.

    Parameters:
    - X: Feature dataframe (includes 'test_case_id').
    - y: Target series.
    - groups: Group labels for GroupKFold.
    - model_name: Name identifier for the model.
    - threshold: Classification threshold for the positive class.
    - min_instances_per_group: Minimum number of instances to select per group.

    Returns:
    - None (prints selection statistics).
    """
    # Extract test_case_id from X
    # input_dim = X.shape[1]
    test_case_ids = X['test_case_id']
    X_features = X.drop(columns=['test_case_id'])
    input_dim = X_features.shape[1]
    # Initialize the pipeline with scaling and classifier
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', ModelFactory.get_model(model_name, input_dim=input_dim))
    ])

    # Set up GroupKFold cross-validation
    group_kfold = GroupKFold(n_splits=5)
    fold = 1
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    confusion_matrices = []
    y_prob_with_groups = []  # List to store y_prob, group labels, and test_case_ids

    for train_index, test_index in group_kfold.split(X_features, y, groups=groups):
        # Split the data
        X_train, X_test = X_features.iloc[train_index], X_features.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        groups_test = groups.iloc[test_index].reset_index(drop=True)  # Reset index for alignment
        test_case_ids_test = test_case_ids.iloc[test_index].reset_index(drop=True)

        pipeline.fit(X_train, y_train)

        # Obtain predicted probabilities
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        elif hasattr(pipeline.named_steps['classifier'], "decision_function"):
            y_scores = pipeline.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_scores))  # Sigmoid to convert scores to probabilities
        else:
            raise AttributeError("The classifier does not have predict_proba or decision_function methods.")

        # Apply threshold to get predictions
        y_pred = (y_prob >= threshold).astype(int)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        cm = confusion_matrix(y_test, y_pred)

        # Store metrics
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        confusion_matrices.append(cm)

        # Store y_prob with corresponding group labels, test_case_ids, and true labels
        fold_y_prob_df = pd.DataFrame({
            'y_prob': y_prob,
            'group': groups_test,
            'test_case_id': test_case_ids_test,
            'y_true': y_test.reset_index(drop=True),
            'fold': fold
        })
        y_prob_with_groups.append(fold_y_prob_df)

        fold += 1

    # Concatenate all y_prob_with_groups into a single DataFrame
    all_y_prob_with_groups = pd.concat(y_prob_with_groups, ignore_index=True)

    def select_top_instances(group_df):
        # Select instances above the threshold
        selected = group_df[group_df['y_prob'] >= threshold]
        if len(selected) >= min_instances_per_group:
            return selected
        else:
            needed = min_instances_per_group - len(selected)
            # Select top 'needed' instances from the group, excluding already selected
            remaining = group_df[~group_df.index.isin(selected.index)]
            additional = remaining.sort_values('y_prob', ascending=False).head(needed)
            # Combine selected and additional
            return pd.concat([selected, additional])

    # Apply selection per group
    final_selected_df = all_y_prob_with_groups.groupby('group').apply(select_top_instances).reset_index(drop=True)

    # Compute total number of selected instances
    total_selected = final_selected_df.shape[0]

    # Compute total number of instances with label 0 and 1
    label_counts = final_selected_df['y_true'].value_counts()
    label_0_count = label_counts.get(0, 0)
    label_1_count = label_counts.get(1, 0)

    # Format selected test_case_ids per group
    selected_ids_per_group = final_selected_df.groupby('group')['test_case_id'].apply(
        lambda ids: tuple(idx for _, idx in ids)
    ).to_dict()

    # Print selection statistics
    print(f"Model: {model_name}")
    print(f"Threshold: {threshold}")
    print("=== Selection Statistics ===")
    print(f"Total selected instances: {total_selected}")
    print(f"The ratio valid test cases: {round(label_1_count / total_selected, 3)}")
    print("============================")

    # pipeline.fit(X_features, y)
    #
    # joblib.dump(pipeline, f'saved_models/{model_name}_final_model.pkl')
    #
    # print(f"Final model saved as saved_models/{model_name}_final_model.pkl")
    # Print selected test case IDs per group
    # print("\n=== Selected Test Cases per Function ===")
    # for group, ids in selected_ids_per_group.items():
    #     print(f"Function {group}: {ids}")
    # print("========================================\n")
    return selected_ids_per_group, total_selected, round(label_1_count / total_selected, 3), all_y_prob_with_groups



def balance_data(X, y, groups):
    df = X.copy()
    df['is_valid'] = y
    df['function_id'] = groups

    valid_cases = df[df['is_valid'] == 1]
    invalid_cases = df[df['is_valid'] == 0]

    min_size = min(len(valid_cases), len(invalid_cases))

    valid_cases_resampled = resample(valid_cases, replace=False, n_samples=min_size, random_state=42)
    invalid_cases_resampled = resample(invalid_cases, replace=False, n_samples=min_size, random_state=42)

    balanced_df = pd.concat([valid_cases_resampled, invalid_cases_resampled])

    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    X_balanced = balanced_df.drop(columns=['is_valid', 'function_id'])
    y_balanced = balanced_df['is_valid']
    groups_balanced = balanced_df['function_id']

    return X_balanced, y_balanced, groups_balanced



def perform_mutation_testing(functions: List[Function]):
    functions_tests = []
    for f in functions:
        func_names = get_top_level_function_names(f.solution)
        if len(func_names) == 0:
            print(f.solution)
            print('here')
            continue
        # functions_tests.append((f.solution, [ff.text for ff in f.testcases if ff.is_valid]))
        functions_tests.append((func_names, f.solution, [ff.text for ff in f.testcases if ff.is_valid]))
    all_tests = []
    for f in functions_tests:
        all_tests.extend(f[2])
    print(f'The total number of tests for mutation testing: {len(all_tests)}')
    # Run mutation testing
    # print(functions_tests[0:1])
    print('running mutation testing...')

    #perform custom mutation
    # mutation_scores, total_mutation_score, all_mutant_codes, mutation_scores_per_operator = mutation_testing(functions_tests)
    # print(mutation_scores, total_mutation_score, all_mutant_codes)
    # Display results
    # for idx, (function_code, test_cases) in enumerate(functions_tests):
    #     print(f"Function {idx + 1} Mutation Score: {mutation_scores[idx]:.2f}")
        # print("Generated Mutants:")
        # for mutant_code in all_mutant_codes[idx]:
            # print(mutant_code)
            # print("-" * 40)
    # print(f"Mutation Score: {total_mutation_score}")
    # print(f'Mutation scores per operator:{mutation_scores_per_operator}.3f')

    ## perform mutmut mutation testing
    perform_overall_mutation_testing(functions_tests)


def downsample_tests(tests):
    # Desired number of elements after downsampling
    desired_length = 5

    # Calculate the sampling interval
    sampling_interval = len(tests) // desired_length
    if sampling_interval == 0:
        return tests
    # Downsample the list to 10 elements
    downsampled_list = tests[::sampling_interval][:desired_length]
    return downsampled_list


def evaluate_function(functions: List[Function], do_mutation=False):
    # total_tests_before_sample = 0
    # total_tests_after_sample = 0
    # for f in functions:
    #     total_tests_before_sample += len(f.testcases)
    #     f.testcases = downsample_tests(f.testcases)
    #     total_tests_after_sample += len(f.testcases)
    #
    #
    # print(f'total_tests_before_sample: {total_tests_before_sample}')
    # print(f'total_tests_after_sample: {total_tests_after_sample}')

    coverage = measure_coverage(functions=functions)
    coverage = round(sum(coverage) / len(coverage), 3)
    if do_mutation:
        perform_mutation_testing(functions)
    return coverage

def create_deep_nn(input_dim):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.7))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])# Binary classification
    return model


def visualize_features(features):
    if features:
        print("Sample feature:", features[0])
    else:
        print("No features extracted.")
        return

    input_mean_valid = []
    input_mean_invalid = []
    output_mean_valid = []
    output_mean_invalid = []

    valid = 0
    invalid = 0

    # Extract features statistics for valid and invalid test cases
    for i in features:
        if i['is_valid']:
            input_mean_valid.append(min(i['input_mean'], 100))  # Clip to 100
            output_mean_valid.append(min(i['output_mean'], 100))  # Clip to 100
            valid += 1
        else:
            input_mean_invalid.append(min(i['input_mean'], 100))  # Clip to 100
            output_mean_invalid.append(min(i['output_mean'], 100))  # Clip to 100
            invalid += 1

    # Prepare the data for seaborn violin plot
    data_input = {
        'Mean': input_mean_valid + input_mean_invalid,
        'Type': ['Valid'] * len(input_mean_valid) + ['Invalid'] * len(input_mean_invalid)
    }

    data_output = {
        'Mean': output_mean_valid + output_mean_invalid,
        'Type': ['Valid'] * len(output_mean_valid) + ['Invalid'] * len(output_mean_invalid)
    }

    # Create the violin plot for Input Mean
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Type', y='Mean', data=data_input, inner="quartile", palette="Set2", cut=0)  # cut=0 to prevent extending beyond data
    plt.ylim(50, 100)  # Set y-axis limit from 50 to 100

    # Calculate and plot 50th (median) and 75th percentiles for Input Mean
    median_valid_input = np.percentile(input_mean_valid, 50)
    percentile_75_valid_input = np.percentile(input_mean_valid, 75)
    median_invalid_input = np.percentile(input_mean_invalid, 50)
    percentile_75_invalid_input = np.percentile(input_mean_invalid, 75)

    # Plot percentiles
    plt.axhline(median_valid_input, color='blue', linestyle='--', label=f'50th Percentile Valid: {median_valid_input:.2f}')
    plt.axhline(percentile_75_valid_input, color='blue', linestyle='-.', label=f'75th Percentile Valid: {percentile_75_valid_input:.2f}')
    plt.axhline(median_invalid_input, color='orange', linestyle='--', label=f'50th Percentile Invalid: {median_invalid_input:.2f}')
    plt.axhline(percentile_75_invalid_input, color='orange', linestyle='-.', label=f'75th Percentile Invalid: {percentile_75_invalid_input:.2f}')

    plt.title('Violin Plot for Input Mean (Valid vs Invalid)')
    plt.legend()
    plt.show()

    # Create the violin plot for Output Mean
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Type', y='Mean', data=data_output, inner="quartile", palette="Set2", cut=0)  # cut=0 to prevent extending beyond data
    plt.ylim(50, 100)  # Set y-axis limit from 50 to 100

    # Calculate and plot 50th (median) and 75th percentiles for Output Mean
    median_valid_output = np.percentile(output_mean_valid, 50)
    percentile_75_valid_output = np.percentile(output_mean_valid, 75)
    median_invalid_output = np.percentile(output_mean_invalid, 50)
    percentile_75_invalid_output = np.percentile(output_mean_invalid, 75)

    # Plot percentiles
    plt.axhline(median_valid_output, color='blue', linestyle='--', label=f'50th Percentile Valid: {median_valid_output:.2f}')
    plt.axhline(percentile_75_valid_output, color='blue', linestyle='-.', label=f'75th Percentile Valid: {percentile_75_valid_output:.2f}')
    plt.axhline(median_invalid_output, color='orange', linestyle='--', label=f'50th Percentile Invalid: {median_invalid_output:.2f}')
    plt.axhline(percentile_75_invalid_output, color='orange', linestyle='-.', label=f'75th Percentile Invalid: {percentile_75_invalid_output:.2f}')

    plt.title('Violin Plot for Output Mean (Valid vs Invalid)')
    plt.legend()
    plt.show()

    sys.exit()

def remove_unnecessary_functions(functions):
    print(len(functions))
    functions_to_remove = []
    count = 0
    for idx1, f in enumerate(functions):
        to_remove_ids = []
        # print('-------------------------------------------------------------------------------------------------------')
        # print(f)
        # print(idx1)
        for idx2, t in enumerate(f.testcases):
            if len(t.input_logprobs) == 0 and len(t.output_logprobs) == 0:
                to_remove_ids.append(idx2)
                count += 1
            if len(t.output_logprobs) == 1:
                if t.output_logprobs[0] is None:
                    to_remove_ids.append(idx2)
                    count += 1
        f.testcases = [ff for idx, ff in enumerate(f.testcases) if idx not in to_remove_ids]
        if len(f.testcases) == 0:
            functions_to_remove.append(idx1)
    functions = [f for idx, f in enumerate(functions) if idx not in functions_to_remove]
    # print(count)
    print(len(functions))
    return functions

def main(dataset: str, llm: str, mutation:bool=False, threshold=0.8, topN=5, feature_sets='all'):
    print('Extracting testcases and running them...')
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
    features = extract_features(all_testcases, function_ids, strategy, feature_sets)
    print(features[0])
    # visualize_features(features)
    X, y, groups = prepare_data(features)  # Now returns groups

    # Ensure that the order of test_case_ids aligns with X, y, groups
    # If prepare_data shuffles or modifies the order, you'll need to adjust accordingly
    # Here, we assume the order is preserved
    test_case_ids_series = pd.Series(test_case_ids, name='test_case_id')

    # Concatenate test_case_ids with X to maintain alignment
    X = pd.concat([X.reset_index(drop=True), test_case_ids_series], axis=1)
    # Balance the data
    # X_balanced, y_balanced, groups_balanced = balance_data(X, y, groups)
    X_balanced, y_balanced, groups_balanced = X, y, groups
    true_count_balanced = y_balanced.sum()
    false_count_balanced = len(y_balanced) - true_count_balanced
    print(f"Balanced dataset size - Number of valid testcases: {true_count_balanced}")
    print(f"Balanced dataset size - Number of invalid testcases: {false_count_balanced}")
    print(f'Valid Testcase Ratio" {round(true_count_balanced/ (false_count_balanced + true_count_balanced), 2)}')

    # Train and evaluate different models
    model_name = 'ensemble'
    print('calculating initial coverage of the functions and mutation score....')
    coverage = evaluate_function(copy.deepcopy(functions), mutation)
    print('Initial coverage:')
    print(coverage)
    models_performance = {}

    print(f"\nTraining and evaluating model: {model_name}")
    selected_ids_per_group, total_selected, ratio, all_y_prob_with_groups = train_and_evaluate(X_balanced, y_balanced, groups_balanced,
                                                                       model_name, threshold, topN)
    # print(selected_ids_per_group)
    temp = copy.deepcopy(functions)
    for group, ids in selected_ids_per_group.items():
        temp[group].testcases = [te for idx, te in enumerate(temp[group].testcases) if idx in ids]
    print(f'Calculating coverage and mutation score using filtered test cases...')
    coverage = evaluate_function(temp, mutation)
    models_performance[model_name] = {
        'coverage': coverage,
        'total_selected': total_selected,
        'valid_test_case_ration': ratio
    }
    print(models_performance)

    ## Saving functions with prediction values
    functions_with_predictions = copy.deepcopy(functions)
    for group, ids in selected_ids_per_group.items():
        for idx, test in enumerate(functions_with_predictions[group].testcases):
            if idx in ids:
                functions_with_predictions[group].testcases[idx].prediction_is_valid = 1
            else:
                functions_with_predictions[group].testcases[idx].prediction_is_valid = 0


    for index, row in all_y_prob_with_groups.iterrows():
        functions_with_predictions[row['group']].testcases[row['test_case_id'][1]].prediction_y_prob = row['y_prob']

    filtered_function_file_name = f'filtered_testcases/{dataset}_{llm}.pkl'
    print(f'Saving filtered functions to {filtered_function_file_name}...')
    with open(filtered_function_file_name, 'wb') as f:
        pickle.dump(functions_with_predictions, f)


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
        default=0.8,
        help=f"Specify the threshold for filtering out the test cases. Choices are: {0.5, 0.65, 0.7, 0.8, 0.85, 0.9}.",
        choices=[0.5, 0.65, 0.7, 0.8, 0.85, 0.9],
        required=False
    )
    parser.add_argument(
        "--topN",
        type=int,
        default=5,
        choices=[0, 1, 3, 5, 7],
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
    file_name = f'output/{args.dataset}_{args.llm}.txt'
    # file_name = f'output/RQ2/{args.dataset}_{args.llm}_{args.threshold}_{args.topN}.txt'
    # file_name = f'output/RQ3/second_output/{args.dataset}_{args.llm}_{args.features}.txt'
    print(f'Writing the output to {file_name}')
    with open(file_name, 'w') as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        main(args.dataset, args.llm, args.mutation, args.threshold, args.topN, args.features)