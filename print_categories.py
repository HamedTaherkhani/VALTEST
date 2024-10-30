import pickle


def print_categorized_results(pickle_file='output/categorized_results.pkl'):
    """
    Reads the categorized results from a pickle file and prints the test case counts per category
    along with detailed results for each function.
    """
    # Load the categorized results from the pickle file
    with open(pickle_file, 'rb') as f:
        categorized_results = pickle.load(f)

    # Count the number of test cases per category
    category_counts = {}
    category_counts_correct = {}
    category_counts_incorrect = {}
    for result in categorized_results:
        for test_case in result['test_case_results']:
            category_name = test_case['category_name']
            category_counts[category_name] = category_counts.get(category_name, 0) + 1
            if test_case['predicted'] == 0:
                category_counts_correct[category_name] = category_counts_correct.get(category_name, 0) + 1
            else:
                category_counts_incorrect[category_name] = category_counts_incorrect.get(category_name, 0) + 1
    # Print the counts
    print("\nTest Case Counts per Category:")
    for category_name, count in category_counts.items():
        print(f"  {category_name}: {count}")

    print('Incorrect Categories:')
    for category_name, count in category_counts_incorrect.items():
        print(f"  {category_name}: {count}")

    print('Correct Categories:')
    for category_name, count in category_counts_correct.items():
        print(f"  {category_name}: {count}")
    # for result in categorized_results:
    #     print(f"\nFunction Prompt: {result['function_prompt']}")
    #     print(f"Function Implementation: {result['function_implementation']}")
    #     for test_case in result['test_case_results']:
    #         print(f"  Test Case: {test_case['test_case_text']}")
    #         print(f"    Category Index: {test_case['category_index']}")
    #         print(f"    Category Name: {test_case['category_name']}")

# Call the function to print the results
if __name__ == '__main__':
    print_categorized_results()
