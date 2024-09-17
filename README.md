# VALTEST

This project is designed to generate test cases for functions using various datasets and Large Language Models (LLMs) and then train machine learning models on the generated data to improve the validity and coverage of test cases. The two main scripts, `generate_testcases.py` and `main_train.py`, handle test case generation and machine learning training/evaluation, respectively.

This project automates the generation of test cases for functions using different datasets such as MBPP, HumanEval, and LeetCode, alongside various LLMs like GPT-4, GPT-3.5-turbo, and CodeLlama. It uses these generated test cases to train multiple machine learning models, aiming to select the most effective test cases based on statistical features and improve overall function coverage.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```
2. Navigate into the project directory:
   ```bash
   cd your-repo
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Generating Test Cases

Use the `generate_testcases.py` script to generate test cases for functions from a chosen dataset and LLM.

```bash
python generate_testcases.py --dataset <dataset_name> --llm <llm_name>
```

- `--dataset`: Specify the dataset to use (e.g., `MBPP`, `HumanEval`, `LeetCode`).
- `--llm`: Specify the LLM to use (e.g., `gpt-4`, `gpt-3.5-turbo`, `codellama`).

Example:
```bash
python generate_testcases.py --dataset MBPP --llm gpt-4
```

### Training and Evaluating Models

Use the `main_train.py` script to train machine learning models on the generated test cases.

```bash
python main_train.py --dataset <dataset_name> --llm <llm_name>
```

- `--dataset`: Specify the dataset to use.
- `--llm`: Specify the LLM to use.

Example:
```bash
python main_train.py --dataset MBPP --llm gpt-4
```

## Files

- **`generate_testcases.py`**: Handles the generation of test cases using the specified dataset and LLM. It parses the dataset, communicates with the LLM to generate test cases, and saves raw log probabilities and test cases.
  
- **`main_train.py`**: Responsible for extracting features from generated test cases, training multiple machine learning models, and evaluating test case validity and coverage. The script balances the dataset, applies feature extraction strategies, and uses cross-validation techniques to assess model performance.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any improvements or bugs.

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Open a pull request.
