import argparse
import sys
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import List
from datasets_and_llms import VALID_DATASETS, VALID_LLMS
from main_train import remove_unnecessary_functions, StatisticalFeatureExtraction, extract_features, prepare_data, ModelFactory
from log_probs import LogProb, TestCase, get_all_tests

def train_llm_model(datasets: List[str], llm: str, model_name: str = 'ensemble', feature_sets: str = 'all'):
    for ds in datasets:
        if ds not in VALID_DATASETS:
            raise ValueError(f"Dataset {ds} not in VALID_DATASETS: {VALID_DATASETS}")

    if llm not in VALID_LLMS:
        raise ValueError(f"LLM {llm} not in VALID_LLMS: {VALID_LLMS}")

    all_testcases = []
    function_ids = []

    combined_functions = []
    # Load functions from all datasets and LLMs
    for ds in datasets:
        print(f"Loading tests for dataset={ds}, llm={llm}")
        funcs = get_all_tests(ds, llm)
        funcs = remove_unnecessary_functions(funcs)
        combined_functions.extend(funcs)

    # Assign IDs and gather test cases
    for func_id, f in enumerate(combined_functions):
        for test_idx, test_case in enumerate(f.testcases):
            all_testcases.append(test_case)
            function_ids.append(func_id)

    strategy = StatisticalFeatureExtraction()
    features = extract_features(all_testcases, function_ids, strategy, feature_sets)
    X, y, groups = prepare_data(features)

    # Create and train the pipeline
    input_dim = X.shape[1]
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', ModelFactory.get_model(model_name, input_dim=input_dim))
    ])

    print("Training the model...")
    pipeline.fit(X, y)
    dato = [item for item in ['MBPP', 'HumanEval', 'LeetCode'] if item not in datasets]
    dato = dato[0]
    # Save the trained model
    output_model_path: str = f'models/trained_model_{llm}_{dato}.pkl'
    joblib.dump(pipeline, output_model_path)
    print(f"Model trained and saved to {output_model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the script with specified dataset and LLM.")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs='+',  # Allow multiple datasets
        required=True,
        help=f"Specify one or more datasets to use. Choices are: {VALID_DATASETS}."
    )
    parser.add_argument(
        "--llm",
        type=str,
        required=True,
        help=f"Specify one LLM to use. Choices are: {VALID_LLMS}."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='ensemble',
        help="Specify the model name to train. For example: ensemble, logistic_regression, svm, etc."
    )
    parser.add_argument(
        "--features",
        type=str,
        default='all',
        choices=['all', 'input', 'output'],
        help="Specify which feature sets to use: all, input, or output."
    )
    args = parser.parse_args()

    train_llm_model(args.dataset, args.llm, args.model_name, args.features)