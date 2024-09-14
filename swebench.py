import requests
import openai_requester
from datasets import load_dataset
import re
from typing import List, Dict
from openai_requester import OpenaiRequester
from generate_testcases import RawLogProbs
# GitHub API token for accessing private repositories or increasing rate limits (optional)
github_token = "your_github_token"  # Replace with your GitHub token or leave empty if not needed


def extract_file_paths_from_problem(problem_statement: str) -> List[str]:
    """
    Extract file paths from the problem statement using regex patterns.

    Args:
        problem_statement (str): The problem statement containing file paths and context.

    Returns:
        List[str]: A list of file paths mentioned in the problem statement.
    """
    # Regex pattern to extract paths like `repo_name/file_path`
    file_paths = re.findall(r'[\w\-_]+/[\w\-_/.]+\.py', problem_statement)
    return list(set(file_paths))  # Return unique paths


def fetch_file_from_github(repo: str, file_path: str, branch: str = 'main') -> str:
    """
    Fetch the content of a file from a GitHub repository.

    Args:
        repo (str): The GitHub repository in the format 'owner/repo'.
        file_path (str): The path to the file in the repository.
        branch (str): The branch from which to fetch the file (default is 'main').

    Returns:
        str: The content of the file or an empty string if the fetch fails.
    """
    url = f"https://api.github.com/repos/{repo}/contents/{file_path}?ref={branch}"
    headers = {"Authorization": f"token {github_token}"} if github_token else {}

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        file_data = response.json()
        # Decode the file content from base64
        file_content = requests.utils.unquote(file_data['content']).decode('utf-8')
        return file_content
    else:
        print(f"Failed to fetch {file_path} from GitHub: {response.status_code}")
        return ""


def generate_prompt(issue_description: str, extracted_code: str) -> str:
    """
    Construct a detailed prompt combining the issue description and extracted code context.

    Args:
        issue_description (str): The problem statement from the issue.
        extracted_code (str): Extracted relevant code and context from the repository.

    Returns:
        str: A detailed prompt for generating test cases.
    """
    prompt = f"""
    Based on the given issue description and the following code from the repository, generate new test cases.

    ### Issue Description:
    {issue_description}

    ### Extracted Code:
    {extracted_code}

    ### Task:
    - Generate new test cases that test the changes made in the issue.
    - Ensure the test cases cover various scenarios, including edge cases, to validate the modified functions.
    - Do not use the existing test cases in the provided code; create new ones.
    """
    return prompt


def request_test_cases_from_openai(prompt: str, open_requester) -> (str,str):
    """
    Send the constructed prompt to OpenAI API to generate test cases.

    Args:
        prompt (str): The constructed prompt for generating test cases.

    Returns:
        str: Generated test cases from the OpenAI API.
    """
    API_RESPONSE = open_requester.get_completion(
        [
            {
                "role": "user",
                "content": prompt
            }
        ],
        model="gpt-4o",
        logprobs=True,
        # top_logprobs=3,
        temperature=0
    )
    all_tests = API_RESPONSE.choices[0].message.content
    return all_tests, API_RESPONSE
    return API_RESPONSE.choices[0].text.strip()


def generate_test_cases_for_swebench():
    swebench_test = load_dataset('princeton-nlp/SWE-bench', split='test')
    """
    Iterates over each instance in the SWE-bench dataset and generates test cases.

    Args:
        dataset: The loaded dataset containing the SWE-bench test split.
    """
    open_requester = OpenaiRequester()
    for instance in swebench_test:
        repo = instance['repo'].split()[0]  # Extract repo, e.g., 'astropy/astropy'
        problem_statement = instance['problem_statement']
        base_commit = instance['base_commit']  # Use the base commit as a branch if needed

        # Extract file paths from the problem statement
        file_paths = extract_file_paths_from_problem(problem_statement)

        # Fetch relevant code from GitHub
        extracted_code = ""
        for file_path in file_paths:
            code = fetch_file_from_github(repo, file_path, branch=base_commit)
            extracted_code += f"\n# File: {file_path}\n{code}\n"

        # Generate the detailed prompt
        prompt = generate_prompt(problem_statement, extracted_code)

        # Request test cases from OpenAI
        generated_test_cases, API_RESPONSE = request_test_cases_from_openai(prompt, open_requester)
        raw_prob = RawLogProbs(prompt=prompt, API_Response=API_RESPONSE, dataset='swe_bench', id=2,
                               testcases=generated_test_cases,
                               solution=solution)
        # Output the generated test cases
        print(f"Generated Test Cases for {instance['instance_id']}:\n", generated_test_cases)
        print("=" * 80)

