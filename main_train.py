import argparse
import sys
import scipy
import numpy as np
import pandas as pd
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from xgboost import XGBClassifier  # For XGBoost
from lightgbm import LGBMClassifier  # For LightGBM
from log_probs import LogProb, TestCase, get_all_tests, RawLogProbs

# from catboost import CatBoostClassifier
class FeatureExtractionStrategy:
    def extract_features(self, log_probs: List[LogProb]) -> dict:
        raise NotImplementedError


# Concrete Strategy for Statistical Feature Extraction
class StatisticalFeatureExtraction(FeatureExtractionStrategy):
    def extract_features(self, log_probs: List[LogProb]) -> dict:
        probs = [lp.prob for lp in log_probs]
        if not probs:
            return {'mean': 0, 'max': 0, 'min': 0, 'sum': 0, 'total':0,
                    'variance':0,
                    # 'kurtosis':0,
                    'entropy':0
                    }
        return {
            'mean': np.mean(probs),
            'max': np.max(probs),
            'min': np.min(probs),
            'sum': np.sum(probs),
            'total': len(probs),
            'variance': np.var(probs) if np.var(probs) is not None else 0,
            # 'kurtosis': scipy.stats.kurtosis(probs) if scipy.stats.kurtosis(probs) is not None else 0,
            'entropy': scipy.stats.entropy(probs) if scipy.stats.entropy(probs) is not None else 0,

        }


# Factory for ML Models
class ModelFactory:
    @staticmethod
    def get_model(model_name: str):
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
            'adaboost': AdaBoostClassifier()
        }
        return models.get(model_name)


# Function to extract features from TestCases
def extract_features(test_cases: List[TestCase], strategy: FeatureExtractionStrategy) -> List[dict]:
    features = []
    for test_case in test_cases:
        # Extract input features and add a prefix to each key
        input_features = {f'input_{k}': v for k, v in strategy.extract_features(test_case.input_logprobs).items()}

        # Extract output features and add a prefix to each key
        output_features = {f'output_{k}': v for k, v in strategy.extract_features(
            test_case.output_logprobs).items()}

        # Combine input and output features with is_valid flag
        combined_features = {**input_features, **output_features, 'is_valid': test_case.is_valid}
        features.append(combined_features)

    return features


# Function to prepare data for ML training
def prepare_data(features: List[dict]):
    # Convert list of dicts to a DataFrame
    df = pd.DataFrame(features)
    X = df.drop(columns=['is_valid'])
    y = df['is_valid']
    return X, y


# Function to train and evaluate models
def train_and_evaluate(X, y, model_name: str):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scaling and model pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ModelFactory.get_model(model_name))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    return accuracy


def balance_data(X, y):
    df = pd.concat([X, y], axis=1)
    valid_cases = df[df['is_valid'] == True]
    invalid_cases = df[df['is_valid'] == False]
    # Determine the smaller class size
    min_size = min(len(valid_cases), len(invalid_cases))

    # Undersample the larger class to match the smaller class size
    valid_cases_resampled = resample(valid_cases, replace=False, n_samples=min_size, random_state=42)
    invalid_cases_resampled = resample(invalid_cases, replace=False, n_samples=min_size, random_state=42)

    # Combine the resampled data
    balanced_df = pd.concat([valid_cases_resampled, invalid_cases_resampled])

    # Shuffle the data to avoid ordering bias
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the balanced data back into features and target
    X_balanced = balanced_df.drop(columns=['is_valid'])
    y_balanced = balanced_df['is_valid']

    return X_balanced, y_balanced


# Main function tying everything together
def main(dataset:str, llm:str):
    # Extract features
    functions = get_all_tests(dataset)
    all_testcases = []
    for f in functions:
        all_testcases.extend(f.testcases)
    strategy = StatisticalFeatureExtraction()

    features = extract_features(all_testcases, strategy)
    print(features[0])

    # print(all_testcases)

    # Prepare data
    X, y = prepare_data(features)
    X_balanced, y_balanced = balance_data(X, y)

    true_count = y_balanced.sum()
    false_count = len(y_balanced) - true_count
    print(f"Number of rows where is_valid is True: {true_count}")
    print(f"Number of rows where is_valid is False: {false_count}")

    # Train and evaluate different models
    models = [
        'logistic_regression',
        'svm',
        'decision_tree',
        'random_forest',
        'gradient_boosting',
        'xgboost',
        'lightgbm',
        # 'catboost',
        'knn',
        'naive_bayes',
        'mlp',
        'adaboost'
    ]
    for model_name in models:
        train_and_evaluate(X_balanced, y_balanced, model_name)


if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="Run the script with specified dataset and LLM.")

    # Add the 'dataset' argument with restricted choices
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["HumanEval", "MBPP", "LeetCode"],
        required=True,
        help="Specify the dataset to use. Choices are: 'HumanEval' or 'MBPP' or 'LeetCode'."
    )

    # Add the 'LLM' argument with restricted choices, allowing future extensions
    parser.add_argument(
        "--llm",
        type=str,
        choices=["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        required=True,
        help="Specify the LLM to use. Choices are: 'gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-3.5-turbo'."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.dataset, args.llm)