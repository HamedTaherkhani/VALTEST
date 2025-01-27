import sys
from typing import List
import pickle
import os
from function_executor import run_test_cases, TimeoutException, run_unit_tests_parallel
import numpy as np
from tqdm import tqdm
import re
import ast
from typing import List, Tuple, Optional
from generate_testcases import RawLogProbs
from dataclasses import dataclass

@dataclass
class ErrorTypes:
    assertion_errors:int = 0
    syntax_errors:int = 0
    indentation_errors:int = 0
    type_errors:int = 0
    value_errors:int = 0
    module_not_found_errors:int = 0
    import_errors:int = 0
    attribute_errors:int = 0
    name_error:int = 0
    file_not_found:int = 0
    def __add__(self, other):
        if not isinstance(other, ErrorTypes):
            return NotImplemented
        return ErrorTypes(self.assertion_errors + other.assertion_errors,self.syntax_errors + other.syntax_errors,
                          self.indentation_errors + other.indentation_errors,self.type_errors + other.type_errors,
                          self.value_errors + other.value_errors,self.module_not_found_errors + other.module_not_found_errors,
                          self.import_errors + other.import_errors,self.attribute_errors + other.attribute_errors,
                          self.name_error + other.name_error,self.file_not_found + other.file_not_found)
    def __str__(self):
        print('assertion errors:',self.assertion_errors)
        print('syntax errors:',self.syntax_errors)
        print('indentation errors:',self.indentation_errors)
        print('type errors:',self.type_errors)
        print('value errors:',self.value_errors)
        print('module not found:',self.module_not_found_errors)
        print('import error:',self.import_errors)
        print('attribute error:',self.attribute_errors)
        print('name error:',self.name_error)
        print('file not found:',self.file_not_found)
        return ''

class LogProb:
    def __init__(self, type, token, prob):
        """

        :param type: 1 for input 2 for output
        :param token:
        :param prob:
        """
        self.type = type
        self.token = token
        self.prob = prob

    def __str__(self):
        return f'{self.token}: {self.prob}'


class TestCase:
    def __init__(self, text, input_logprobs: List[LogProb], output_logprobs: List[LogProb],
                 second_output_logprobs, second_input_logprobs, is_valid: int = None, prediction_is_valid: int = None, prediction_y_prob: int = None, validated_text=None):
        self.text = text
        self.input_logprobs = input_logprobs
        self.output_logprobs = output_logprobs
        self.second_input_logprobs = second_input_logprobs
        self.second_output_logprobs = second_output_logprobs
        self.is_valid = is_valid
        self.prediction_is_valid = prediction_is_valid
        self.prediction_y_prob = prediction_y_prob
        self.validated_text = validated_text

    def __str__(self):
        input_logprobs_str = ", ".join([str(lp) for lp in self.input_logprobs])
        output_logprobs_str = ", ".join([str(lp) for lp in self.output_logprobs]) if self.output_logprobs else "None"
        second_input_logprobs = ", ".join([str(lp) for lp in self.second_input_logprobs]) if self.second_input_logprobs else "None"
        second_output_logprobs = ", ".join([str(lp) for lp in self.second_output_logprobs]) if self.second_output_logprobs else "None"

        return f'Text: "{self.text}"\nIs Valid: {self.is_valid}\nInput LogProbs: [{input_logprobs_str}]\nOutput LogProbs: [{output_logprobs_str}]\nsecond_input_logprobs: [{second_input_logprobs}]\nsecond_output_logprobs: [{second_output_logprobs}]\n'

    def __repr__(self):
        return self.__str__()


class Function:
    def __init__(self, prompt: str, testcases: list[TestCase], solution: str, original_tests: str = None):
        self.prompt = prompt
        self.testcases = testcases
        self.solution = solution
        self.original_tests = original_tests

    def __str__(self):
        # Create a string representation of test cases
        testcases_str = "\n".join([str(tc) for tc in self.testcases])
        original_tests_str = self.original_tests if self.original_tests else "None"

        return (
            f"Prompt:\n{self.prompt}\n"
            f"Solution:\n{self.solution}\n"
            f"Original Tests:\n{original_tests_str}\n"
            f"Test Cases:\n{testcases_str}"
        )

    def __repr__(self):
        return self.__str__()

def getter(logprobs):
    for lp in logprobs:
        yield lp


# def find_test_case(res, testcase):
#     temp_test = testcase.copy()
#     for lp in res.choices[0].logprobs.content:
#         if testcase[0] == lp.token:
#
#         for t in testcase:
#             if t == lp.token:

def validate_string(input_string):
    # List of unacceptable tokens
    unacceptable_tokens = ('==', ' ', '  ', ',', '[', ']', '\'', '\"' ,')', '(', "\"", '\n')
    pattern = '|'.join(re.escape(token) for token in unacceptable_tokens)
    matches = re.findall(pattern, input_string)
    validated_string = ''.join(matches)
    if validated_string == input_string:
        return False
    else:
        return True


def contains_number(s: str) -> bool:
    return bool(re.search(r'\d', s))


import ast
from typing import Optional, Tuple, List

def separate_assertion(assertion_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Separates an assertion string into input and output parts.

    Parameters:
    - assertion_str (str): The assertion statement as a string.

    Returns:
    - Tuple[Optional[str], Optional[str]]: A tuple containing the input and output parts as strings.
      Returns (None, None) if separation fails.
    """
    # Remove 'assert ' prefix
    if assertion_str.startswith("assert "):
        expr_str = assertion_str[len("assert "):]
    else:
        expr_str = assertion_str

    try:
        # Try parsing as a comparison first (e.g., a == b)
        expr_ast = ast.parse(expr_str, mode='exec').body[0].value
        if isinstance(expr_ast, ast.Compare):
            if len(expr_ast.ops) == 1 and isinstance(expr_ast.ops[0], ast.Eq):
                input_expr = expr_ast.left
                output_expr = expr_ast.comparators[0]
            else:
                print(f"Unsupported comparison operator in assertion: {assertion_str}")
                return None, None
        elif isinstance(expr_ast, ast.Call):
            # Handle function or method calls
            func = expr_ast.func
            args = expr_ast.args

            if isinstance(func, ast.Attribute):
                if len(args) == 1:
                    # Method call (e.g., a.equals(b))
                    input_expr = func.value
                    output_expr = args[0]
                else:
                    # Function call with multiple arguments (e.g., torch.equal(a, b))
                    input_expr = args[0]
                    output_expr = args[1]
            else:
                # Function call without attribute (e.g., some_function(a, b))
                if len(args) >= 2:
                    input_expr = args[0]
                    output_expr = args[1]
                else:
                    print(f"Function call with insufficient arguments in assertion: {assertion_str}")
                    return None, None
        else:
            print(f"Unsupported assertion format: {assertion_str}")
            return None, None

        # Convert AST nodes back to code strings
        if input_expr and output_expr:
            try:
                # If input_expr is a function call, extract its arguments
                if isinstance(input_expr, ast.Call):
                    if len(input_expr.args) == 1:
                        input_arg_expr = input_expr.args[0]
                        input_str = ast.unparse(input_arg_expr).strip()
                    else:
                        # Handle multiple arguments if necessary
                        args_list = [ast.unparse(arg).strip() for arg in input_expr.args]
                        input_str = f"({', '.join(args_list)})"
                else:
                    input_str = ast.unparse(input_expr).strip()

                output_str = ast.unparse(output_expr).strip()
                return input_str, output_str
            except Exception as e:
                print(f"Failed to unparse expressions in assertion: {assertion_str}\nError: {e}")
                return None, None
        else:
            return None, None

    except Exception as e:
        print(f"Failed to parse assertion: {assertion_str}\nError: {e}")
        return None, None

def separate_assertions(assertions: List[str]) -> List[Tuple[Optional[str], Optional[str]]]:
    """
    Processes a list of assertion strings and separates each into input and output parts.

    Parameters:
    - assertions (List[str]): A list of assertion statements as strings.

    Returns:
    - List[Tuple[Optional[str], Optional[str]]]: A list of tuples, each containing the input and output parts.
    """
    separated = []
    for assertion in assertions:
        input_part, output_part = separate_assertion(assertion)
        if input_part is None or output_part is None:
            continue
        separated.append((input_part, output_part))
    return separated


def token_iterator(s):
    for aa in s:
        yield aa


def remove_unnecessary_tokens(results):
    temp_results = []
    for test in results:
        tt = []
        for in_or_output in test:
            tokens = []
            for token in in_or_output:
                if validate_string(token[0]):
                    tokens.append(token)
            tt.append(tokens)
        temp_results.append(tt)

    for idx1,test in enumerate(temp_results):
        for idx2,in_or_output in enumerate(test):
            if len(in_or_output) == 0:
                temp_results[idx1][idx2] = results[idx1][idx2]
    return temp_results

def match_strings_to_tokens(testcases, tokens, dataset):
    results = []
    token_itr = token_iterator(tokens)
    while True:
        temp = next(token_itr)
        if 'assert' in temp[0]:
            break
    for tc in testcases:
        tt_temp = []
        for testcase in tc:
            index = 0
            testcase_len = len(testcase)
            # print(f'testcase_len is {testcase_len}')
            found_tokens = []
            testcase = testcase.replace('\"','\'')
            # print(f'testcase is {testcase}')
            for idx,token in enumerate(token_itr):
                temp = token[0].replace('\"','\'')
                # print(f'remaining : {testcase[index:]}')
                if index == 0:
                    temp = temp.lstrip()
                    if not testcase.startswith('('):
                        temp = temp.lstrip('(')
                # print(temp)
                if len(temp) > len(testcase[index:]):
                    if temp.startswith(testcase[index:testcase_len]):
                        # if validate_string(token[0]):
                        if dataset == 'DS1000':
                            if contains_number(token[0]):
                                found_tokens.append(token)
                        else:
                            found_tokens.append(token)
                        index += len(temp)
                        # print(f'index is {index}')
                else:
                    if testcase[index:].startswith(temp):
                        # if validate_string(token[0]):
                        if dataset == 'DS1000':
                            if contains_number(token[0]):
                                found_tokens.append(token)
                        else:
                            found_tokens.append(token)
                        index += len(temp)
                        # print(f'index is {index}')
                if index > testcase_len:
                    break
            tt_temp.append(found_tokens)
        results.append(tt_temp)

    results = remove_unnecessary_tokens(results)
    return results


def match_strings_to_tokens_for_unittests_v2(testcases, tokens):
    results = []
    token_itr = token_iterator(tokens)
    for tc in testcases:
        tt_temp = []
        for testcase in tc:
            testcase = testcase.replace(' ','').replace('\n','').replace('\r','').replace('\t','')
            index = 0
            testcase_len = len(testcase)
            # print(f'testcase_len is {testcase_len}')
            found_tokens = []
            # testcase = testcase.replace('\"','\'').lstrip()
            testcase = testcase.lstrip()
            # print(f'testcase is {testcase}')
            for idx,token in enumerate(token_itr):
                # temp = token[0].replace('\"','\'')
                temp = token[0]
                temp = temp.replace(' ','').replace('\n','').replace('\r','').replace('\t','')
                # print(f'remaining : {testcase[index:]}')
                if index == 0:
                    temp = temp.lstrip()
                # print(temp)
                if len(temp) > len(testcase[index:]):
                    if temp.startswith(testcase[index:testcase_len]):
                        # if validate_string(token[0]):
                        found_tokens.append(token)
                        index += len(temp)
                        # print(f'index is {index}')
                else:
                    if testcase[index:].startswith(temp):
                        # if validate_string(token[0]):
                        found_tokens.append(token)
                        index += len(temp)
                        # print(f'index is {index}')
                if index > testcase_len:
                    break
            tt_temp.append(found_tokens)
        results.append(tt_temp)

    results = remove_unnecessary_tokens(results)
    return results


def match_strings_to_tokens_for_unittests(testcases, tokens):
    results = []
    temp_toks = ''
    for tt in tokens:
        temp_toks += tt[0]
    # print(temp_toks)
    token_itr = token_iterator(tokens)
    for tc in testcases:
        tt_temp = []
        for testcase in tc:
            for a_testcase in testcase:
                index = 0
                testcase_len = len(a_testcase)
                # print(f'testcase_len is {a_testcase}')
                found_tokens = []
                a_testcase = a_testcase.replace('\"','\'')
                # print(f'testcase is {a_testcase}')
                for idx,token in enumerate(token_itr):
                    temp = token[0].replace('\"','\'').replace('\n','')
                    # print(f'remaining : {a_testcase[index:]}')
                    # print(temp)
                    if len(temp) > len(a_testcase[index:]):
                        if temp.startswith(a_testcase[index:testcase_len]):
                            # if validate_string(token[0]):
                            found_tokens.append(token)
                            index += len(temp)
                            # print(f'index is {index}')
                        elif index == 0:
                            temp = temp.lstrip(' ').lstrip()
                            if a_testcase[index:].startswith(temp):
                                found_tokens.append(token)
                                index += len(temp)
                            if a_testcase[index:].startswith(temp[1:]):
                                found_tokens.append(token)
                                index += len(temp[1:])
                            elif a_testcase[index:].startswith(temp[2:]):
                                found_tokens.append(token)
                                index += len(temp[2:])
                    else:
                        if a_testcase[index:].startswith(temp):
                            # if validate_string(token[0]):
                            found_tokens.append(token)
                            index += len(temp)
                            # print(f'index is {index}')
                        elif index == 0:
                            temp = temp.lstrip(' ').lstrip()
                            if a_testcase[index:].startswith(temp):
                                found_tokens.append(token)
                                index += len(temp)
                            if a_testcase[index:].startswith(temp[1:]):
                                found_tokens.append(token)
                                index += len(temp[1:])
                            elif a_testcase[index:].startswith(temp[2:]):
                                found_tokens.append(token)
                                index += len(temp[2:])
                            # print(f'index is {index}')
                    if index > testcase_len:
                        break
                tt_temp.append(found_tokens)
        results.append(tt_temp)

    results = remove_unnecessary_tokens(results)
    return results



import ast
from typing import Any, Dict, List, Optional, Tuple, Union

# Container to store inputs and expected outputs for each test case
class TestCaseInfo:
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.inputs: List[str] = []
        self.expected_outputs: List[str] = []

    def add_input(self, inp: str):
        if inp not in self.inputs:
            self.inputs.append(inp)

    def add_output(self, out: str):
        if out not in self.expected_outputs:
            self.expected_outputs.append(out)

    def __repr__(self):
        return f"TestCaseInfo({self.test_name}, inputs={self.inputs}, expected_outputs={self.expected_outputs})"


def get_full_attr_name(node: ast.AST) -> str:
    """
    Recursively reconstruct the full attribute name from an AST node.
    For example, for np.testing.assert_array_equal, it returns the string "np.testing.assert_array_equal".
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return get_full_attr_name(node.value) + "." + node.attr
    else:
        return ""


class InputOutputVisitor(ast.NodeVisitor):
    """
    Visits the AST of unittest.TestCase classes to extract inputs from setUp methods
    and expected outputs from test methods.
    """
    def __init__(self, function_name: str = "task_func") -> None:
        super().__init__()
        self.function_name = function_name
        self.setup_inputs: Dict[str, List[str]] = {}
        self.test_results: Dict[str, TestCaseInfo] = {}
        self.current_class: Optional[str] = None
        self.current_method: Optional[str] = None

        # Lists of recognized NumPy and pandas testing assertion function names:
        self.numpy_assert_names = {
            "np.testing.assert_allclose",
            "np.testing.assert_array_almost_equal_nulp",
            "np.testing.assert_array_max_ulp",
            "np.testing.assert_array_equal",
            "np.testing.assert_array_less",
            "np.testing.assert_equal",
            "np.testing.assert_almost_equal",
            "np.testing.assert_approx_equal",
            "np.testing.assert_array_almost_equal",
            "np.testing.assert_string_equal"
        }
        self.pandas_assert_names = {
            "pandas.testing.assert_frame_equal",
            "pandas.testing.assert_series_equal",
            "pandas.testing.assert_index_equal",
            "pandas.testing.assert_extension_array_equal"
        }

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        # Check if the class inherits from unittest.TestCase
        if any(
            (isinstance(base, ast.Name) and base.id == "TestCase") or
            (isinstance(base, ast.Attribute) and base.attr == "TestCase")
            for base in node.bases
        ):
            self.current_class = node.name
            # Process methods within the TestCase class
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if item.name == "setUp":
                        # Gather all RHS expressions from setUp
                        inputs = self.gather_inputs_from_method(item)
                        self.setup_inputs[node.name] = inputs
                    elif item.name.startswith("test_"):
                        # Initialize TestCaseInfo with setUp inputs
                        test_info = TestCaseInfo(item.name)
                        if node.name in self.setup_inputs:
                            for inp in self.setup_inputs[node.name]:
                                test_info.add_input(inp)
                        self.test_results[item.name] = test_info
                        # Visit the test method
                        self.current_method = item.name
                        self.visit(item)
                        self.current_method = None
            self.current_class = None
        return None

    def gather_inputs_from_method(self, node: ast.FunctionDef) -> List[str]:
        """
        Extracts all RHS expressions from assignments and content written to files in the given method.
        """
        inputs: List[str] = []

        class SetupVisitor(ast.NodeVisitor):
            def __init__(self, outer):
                self.outer = outer
                self.inputs: List[str] = []

            def visit_Assign(self, assign_node: ast.Assign):
                # Capture all RHS expressions from assignments to self.*
                for target in assign_node.targets:
                    if (
                        isinstance(target, ast.Attribute) and
                        isinstance(target.value, ast.Name) and
                        target.value.id == "self"
                    ):
                        try:
                            rhs = ast.unparse(assign_node.value).strip()
                            self.inputs.append(rhs)
                        except Exception:
                            self.inputs.append("<unable to unparse>")
                self.generic_visit(assign_node)

            def visit_With(self, with_node: ast.With):
                """
                Capture content written to files within with statements, e.g.,
                with open('file.txt', 'w') as f:
                    f.write('content')
                """
                for item in with_node.body:
                    if isinstance(item, ast.Expr) and isinstance(item.value, ast.Call):
                        call = item.value
                        if (
                            isinstance(call.func, ast.Attribute) and
                            call.func.attr == "write" and
                            call.args
                        ):
                            try:
                                content = ast.unparse(call.args[0]).strip()
                                self.inputs.append(content)
                            except Exception:
                                self.inputs.append("<unable to unparse write content>")
                self.generic_visit(with_node)

        visitor = SetupVisitor(self)
        visitor.visit(node)
        return visitor.inputs

    def visit_Assign(self, node: ast.Assign) -> Any:
        if self.current_method:
            # Look for assignments like: output = task_func(...)
            if isinstance(node.value, ast.Call) and self.is_target_function_call(node.value):
                inputs = self.get_call_inputs(node.value)
                for inp in inputs:
                    self.test_results[self.current_method].add_input(inp)
        self.generic_visit(node)
        return None

    def visit_Expr(self, node: ast.Expr) -> Any:
        if self.current_method:
            # Look for bare function calls like: task_func(...)
            if isinstance(node.value, ast.Call) and self.is_target_function_call(node.value):
                inputs = self.get_call_inputs(node.value)
                for inp in inputs:
                    self.test_results[self.current_method].add_input(inp)
        self.generic_visit(node)
        return None

    def visit_Call(self, node: ast.Call) -> Any:
        if self.current_method:
            # First, check if this is a call to a write() function (e.g. f.write(...))
            fname = self.match_write_call(node)
            if fname:
                self.test_results[self.current_method].add_input(fname)

            # Check for assertion calls. There are now two cases:
            # (1) Calls of the form self.assert*(...)
            if (
                isinstance(node.func, ast.Attribute) and
                isinstance(node.func.value, ast.Name) and
                node.func.value.id == "self" and
                node.func.attr.startswith("assert")
            ):
                expected = self.extract_expected_output(node.func.attr, node)
                if expected is not None:
                    self.test_results[self.current_method].add_output(expected)

            # (2) Calls to NumPy or pandas testing assertions such as np.testing.assert_array_equal(...)
            else:
                full_name = get_full_attr_name(node.func)
                if full_name in self.numpy_assert_names or full_name in self.pandas_assert_names:
                    expected = self.extract_expected_output(full_name, node)
                    if expected is not None:
                        self.test_results[self.current_method].add_output(expected)
        self.generic_visit(node)
        return None

    def visit_Assert(self, node: ast.Assert) -> Any:
        """
        Handle bare assert statements of the form:
            assert task_func(...args) == expected
        """
        if self.current_method and isinstance(node.test, ast.Compare):
            comp = node.test
            # Check for a simple equality comparison
            if (len(comp.ops) == 1 and isinstance(comp.ops[0], ast.Eq) and
                isinstance(comp.left, ast.Call) and self.is_target_function_call(comp.left)):
                # Extract inputs from the function call
                inputs = self.get_call_inputs(comp.left)
                for inp in inputs:
                    self.test_results[self.current_method].add_input(inp)
                # The expected output is the right-hand side of the comparison
                try:
                    expected = ast.unparse(comp.comparators[0]).strip()
                except Exception:
                    expected = "<unable to unparse expected output>"
                self.test_results[self.current_method].add_output(expected)
        self.generic_visit(node)
        return None

    def is_target_function_call(self, call_node: ast.Call) -> bool:
        """Check if the call is to the target function (e.g., task_func)."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id == self.function_name
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr == self.function_name
        return False

    def get_call_inputs(self, call_node: ast.Call) -> List[str]:
        """Extract all arguments passed to the function call as strings,
        skipping any attribute accesses on 'self'."""
        inputs = []
        for arg in call_node.args:
            # Skip if the argument is an attribute of self (e.g., self.data)
            if (isinstance(arg, ast.Attribute) and
                isinstance(arg.value, ast.Name) and
                arg.value.id == "self"):
                continue
            try:
                s = ast.unparse(arg).strip()
                inputs.append(s)
            except Exception:
                inputs.append("<unable to unparse>")
        return inputs

    def match_write_call(self, node: ast.Call) -> Optional[str]:
        """
        Identify write calls like f.write('content') and extract the content.
        """
        if (
            isinstance(node.func, ast.Attribute) and
            node.func.attr == "write" and
            node.args
        ):
            try:
                content = ast.unparse(node.args[0]).strip()
                return content
            except Exception:
                return "<unable to unparse write content>"
        return None

    def extract_expected_output(self, assert_method: str, call_node: ast.Call) -> Optional[str]:
        """
        Extract the expected output based on the assertion method.
        Recognizes standard unittest assertions as well as common NumPy and pandas testing assertions.
        For assertions with exactly 2 arguments, both arguments are joined as a comma-separated string.
        """
        try:
            # Handle standard unittest assertions using the method name
            if assert_method.startswith("assert"):
                # For common unittest assertions:
                if assert_method in (
                    "assertEqual", "assertNotEqual", "assertIs",
                    "assertIsNot", "assertIsInstance", "assertNotIsInstance",
                    "assertCountEqual"
                ):
                    if len(call_node.args) >= 2:
                        return ast.unparse(call_node.args[1]).strip()
                elif assert_method in ("assertIn", "assertNotIn"):
                    return ", ".join([ast.unparse(arg).strip() for arg in call_node.args])
                elif assert_method in ("assertTrue", "assertFalse"):
                    if len(call_node.args) >= 1:
                        return ast.unparse(call_node.args[0]).strip()
                elif assert_method in ("assertIsNone", "assertIsNotNone"):
                    return "None"
                elif assert_method.startswith("assertRaises"):
                    if len(call_node.args) >= 1:
                        return ast.unparse(call_node.args[0]).strip()
                elif assert_method in (
                    "assertAlmostEqual", "assertNotAlmostEqual",
                    "assertGreater", "assertGreaterEqual",
                    "assertLess", "assertLessEqual",
                    "assertRegex", "assertNotRegex"
                ):
                    if len(call_node.args) >= 2:
                        return ast.unparse(call_node.args[1]).strip()
                elif assert_method in ("assertWarns", "assertWarnsRegex"):
                    return ", ".join([ast.unparse(arg).strip() for arg in call_node.args])

            # For NumPy testing assertions and pandas testing assertions—
            # The assert_method here might be a full dotted name (e.g., "np.testing.assert_array_equal")
            if len(call_node.args) == 2:
                arg0 = ast.unparse(call_node.args[0]).strip()
                arg1 = ast.unparse(call_node.args[1]).strip()
                return f"{arg0}, {arg1}"
            elif len(call_node.args) >= 2:
                return ast.unparse(call_node.args[1]).strip()
            else:
                return ", ".join([ast.unparse(arg).strip() for arg in call_node.args])
        except Exception:
            return None


def separate_unittest_iosources(test_code: str, function_name: str = "task_func") -> Dict[str, Tuple[List[str], List[str]]]:
    """
    Processes Python unittest source code and extracts input and expected output sources.
    For each test method in a TestCase class, returns a tuple: (list of inputs, list of expected outputs).

    Parameters:
      - test_code (str): Python source code containing unittest.TestCase classes.
      - function_name (str): The name of the function under test. Default is "task_func".

    Returns:
      - Dict[str, Tuple[List[str], List[str]]]: A dictionary mapping test method names to tuples of (inputs, expected_outputs).
    """
    try:
        tree = ast.parse(test_code)
    except Exception as e:
        print(f"Error parsing code: {e}")
        return {}
    if function_name is None:
        function_name = "task_func"
    visitor = InputOutputVisitor(function_name)
    visitor.visit(tree)

    # Organize the outputs in a simple dict: test_method -> (inputs, expected_outputs)
    results = {
        test_name: (info.inputs, info.expected_outputs)
        for test_name, info in visitor.test_results.items()
    }
    return results

def extract_function_name(code_str):
    # This regex matches "def" followed by one or more spaces,
    # then captures the function name (one or more word characters),
    # followed by optional whitespace and an opening parenthesis.
    pattern = r"def\s+(\w+)\s*\("

    match = re.search(pattern, code_str)
    if match:
        return match.group(1)
    else:
        return None


def extract_setup_and_test_methods(source: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse a Python source string containing unittest classes and return a dict where keys
    are class names and values are lists of tuples. Each tuple is (setup_code, test_code)
    for each test method in the class.

    If the class does not define a setUp, the setup_code will be an empty string.

    :param source: Python source code as a string.
    :return: Dictionary mapping class names to a list of (setUp_code, test_method_code) tuples.
    """
    tree = ast.parse(source)

    # Split the source into lines for slicing purposes.
    source_lines = source.splitlines()

    def get_source_segment(node):
        # node.lineno and node.end_lineno are 1-indexed.
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            # Include the entire block, rejoin with newline.
            return "\n".join(source_lines[node.lineno - 1:node.end_lineno])
        return ""

    # This dict will map the test class name to list of (setUp_code, test_code) pairs.
    result: Dict[str, List[Tuple[str, str]]] = {}

    for node in tree.body:
        # Look for class definitions.
        if isinstance(node, ast.ClassDef):
            # Check if the class inherits from unittest.TestCase.
            inherits_testcase = False
            for base in node.bases:
                # The base might be a Name (e.g. TestCase) or an Attribute (e.g. unittest.TestCase)
                if isinstance(base, ast.Name) and base.id == "TestCase":
                    inherits_testcase = True
                elif isinstance(base, ast.Attribute):
                    # For an Attribute, check if it is unittest.TestCase (or similar)
                    parts = []
                    curr = base
                    while isinstance(curr, ast.Attribute):
                        parts.insert(0, curr.attr)
                        curr = curr.value
                    if isinstance(curr, ast.Name):
                        parts.insert(0, curr.id)
                    # Compare the joined name. You may need to adjust this check.
                    if ".".join(parts) == "unittest.TestCase":
                        inherits_testcase = True
            if not inherits_testcase:
                continue

            class_name = node.name
            # Get the setUp method code (if present)
            setup_code = ""
            for class_body_item in node.body:
                if isinstance(class_body_item, ast.FunctionDef) and class_body_item.name == "setUp":
                    setup_code = get_source_segment(class_body_item)
                    # If you want only the inner part, you might remove the header.
                    # Here, we'll extract everything but the first line (the signature) and dedent.
                    setup_lines = setup_code.splitlines()
                    if len(setup_lines) > 1:
                        # Assume the first line is "def setUp(self):" and remove it.
                        # Also remove one level of indentation.
                        dedented = [line[4:] if line.startswith("    ") else line for line in setup_lines[1:]]
                        setup_code = "\n".join(dedented)
                    break

            # For each test method in the class, extract its code.
            tests = []
            for class_body_item in node.body:
                if isinstance(class_body_item, ast.FunctionDef) and class_body_item.name.startswith("test_"):
                    test_code = get_source_segment(class_body_item)
                    test_lines = test_code.splitlines()
                    if len(test_lines) > 1:
                        dedented = [line[4:] if line.startswith("    ") else line for line in test_lines[1:]]
                        test_code = "\n".join(dedented)
                    tests.append((setup_code, test_code))

            if tests:
                result[class_name] = tests

    return result


def get_logprobs_dynamic(logprobs, testcases, method_name, ground_truth, test_type=0, dataset='HumanEval') -> (List[TestCase], Dict):
    # print(f'test type: {test_type}')
    # print(len(testcases))
    # print(testcases)
    if test_type == 1: ## python unittests
        sep_tests = []
        ## MEthod 1
        # function_name = extract_function_name(ground_truth)
        # # print('python unittest')
        # # print(function_name)
        # for test in testcases:
        #     separated_tests = separate_unittest_iosources(test, function_name)
        #     # print(f'len separated tests: {len(separated_tests)}')
        #     for test_name, (inputs, outputs) in separated_tests.items():
        #         sep_tests.append((inputs, outputs))
        ## Method 2
        for test in testcases:
            extracted = extract_setup_and_test_methods(test)
            for cls, tests in extracted.items():
                setup, test = tests[0]
                sep_tests.append((setup, test))
                # for idx, (setup, test) in enumerate(tests, 1):
                #     print(f'--- Test Method {idx} ---')
                #     print("setUp:")
                #     print(setup)
                #     print("Test method:")
                #     print(test)
                #     print("--------------------")

    else:
        sep_tests = separate_assertions(testcases)
    # print('-----------------------------------------')
    # print(separated_assertions)
    if test_type == 1:
        # Method1
        # assertions_to_tokens = match_strings_to_tokens_for_unittests(sep_tests, logprobs)
        # Method2
        assertions_to_tokens = match_strings_to_tokens_for_unittests_v2(sep_tests, logprobs)
    else:
        assertions_to_tokens =  match_strings_to_tokens(sep_tests, logprobs, dataset)
    # print(assertions_to_tokens)
    # if len(assertions_to_tokens) == 0:
    # print(separated_assertions)
    # print(logprobs)
    textcase_index = 0
    test_list = []
    all_tests: List[TestCase] = []
    testcase_data_list = []
    assertion_errors, syntax_errors, indentation_errors, type_errors, value_errors, attribute_errors, module_not_found_errors, import_errors, name_error, file_not_found = 0, 0, 0, 0, 0, 0, 0, 0,0,0
    for instance in assertions_to_tokens:
        input_logprobs: List[LogProb] = []
        output_logprobs: List[LogProb] = []
        second_input_logprobs: List[LogProb] = []
        second_output_logprobs: List[LogProb] = []
        inputs = instance[0]
        outputs = instance[1]
        for ins in inputs:
            input_logprobs.append(LogProb(
                type = 1,
                token = ins[0],
                prob=np.round(np.exp(ins[1]) * 100, 2)
            ))
            second_input_logprobs.append(LogProb(
                type = 1,
                token=ins[2][1][0],
                prob=np.round(np.exp(ins[2][1][1]) * 100, 2)
            ))

        for ins in outputs:
            output_logprobs.append(LogProb(
                type=2,
                token=ins[0],
                prob=np.round(np.exp(ins[1]) * 100, 2)
            ))
            second_output_logprobs.append(LogProb(
                type=2,
                token=ins[2][1][0],
                prob=np.round(np.exp(ins[2][1][1]) * 100, 2)
            ))
        test_list.append(testcases[textcase_index])
        testcase_data_list.append({
            'text': testcases[textcase_index],
            'input_logprobs': input_logprobs,
            'output_logprobs': output_logprobs,
            'second_input_logprobs': second_input_logprobs,
            'second_output_logprobs': second_output_logprobs
        })
        textcase_index +=1

    # print(separated_assertions)
    # print(assertions_to_tokens[0])
    # print('-------------------------------------------------------------------')
    if test_type == 1:
        res = run_unit_tests_parallel(ground_truth, test_list)
        is_passed_list = [r[0] for r in res]
        outputs = [r[1] for r in res]
        for output in outputs:
            # print(output)
            # print('*'*100)
            if "AssertionError" in output:
                assertion_errors += 1
            elif "SyntaxError" in output:
                syntax_errors += 1
            elif "IndentationError" in output:
                indentation_errors += 1
            elif "TypeError" in output:
                type_errors += 1
            elif "ValueError" in output:
                value_errors += 1
            elif "AttributeError" in output:
                attribute_errors += 1
            elif "ModuleNotFoundError" in output:
                module_not_found_errors += 1
            elif "ImportError" in output:
                print(output)
                print('*'*100)
                import_errors += 1
            elif "NameError" in output:
                # print(output)
                name_error += 1
            elif "FileNotFoundError" in output:
                file_not_found += 1
    else:
        is_passed_list = run_test_cases(ground_truth, test_list, timeout=5)
    # print(f'the pass list is passed: {is_passed_list}')
    for i, is_passed in enumerate(is_passed_list):
        if is_passed:
            is_passed = 1
        else:
            is_passed = 0
        data = testcase_data_list[i]
        testcase = TestCase(
            text=data['text'],
            input_logprobs=data['input_logprobs'],
            output_logprobs=data['output_logprobs'],
            is_valid=is_passed,
            second_input_logprobs=data['second_input_logprobs'],
            second_output_logprobs=data['second_output_logprobs']
        )
        all_tests.append(testcase)

    return all_tests, ErrorTypes(assertion_errors=assertion_errors, syntax_errors=syntax_errors,
                                 indentation_errors=indentation_errors, type_errors=type_errors,
                                 name_error=name_error, file_not_found=file_not_found,
                                 attribute_errors=attribute_errors, module_not_found_errors=module_not_found_errors,
                                 value_errors=value_errors,import_errors=import_errors)


def get_logprobs(logprobs, testcases, method_name, ground_truth) -> List[TestCase]:
    gen = getter(logprobs)
    all_tests: List[TestCase] = []
    test_list = []
    testcase_data_list = []
    textcase_index = 0

    for logprob in gen:
        if 'assert' in logprob[0]:
            for logprob2 in gen:
                is_break = False
                if '(' in logprob2[0]:
                    input_logprobs: List[LogProb] = []
                    output_logprobs: List[LogProb] = []
                    second_input_logprobs: List[LogProb] = []
                    second_output_logprobs: List[LogProb] = []

                    for logprob3 in gen:
                        if '==' in logprob3[0]:
                            for logprob4 in gen:
                                if not validate_string(logprob4[0]):
                                    continue
                                elif '#' in logprob4[0] or '\n' in logprob4[0]:  # comment or newline
                                    break
                                output_logprobs.append(LogProb(
                                    type=2,
                                    token=logprob4[0],
                                    prob=np.round(np.exp(logprob4[1]) * 100, 2)
                                ))
                                second_output_logprobs.append(LogProb(
                                    type=2,
                                    token=logprob4[2][1][0],
                                    prob=np.round(np.exp(logprob4[2][1][1]) * 100, 2)
                                ))

                            # Collect the function string and associated data
                            test_list.append(testcases[textcase_index])
                            testcase_data_list.append({
                                'text': testcases[textcase_index],
                                'input_logprobs': input_logprobs,
                                'output_logprobs': output_logprobs,
                                'second_input_logprobs': second_input_logprobs,
                                'second_output_logprobs': second_output_logprobs
                            })
                            textcase_index += 1
                            is_break = True
                            break
                        elif logprob3[0] == ',':
                            continue
                        else:
                            if validate_string(logprob3[0]):
                                input_logprobs.append(LogProb(
                                    type=1,
                                    token=logprob3[0],
                                    prob=np.round(np.exp(logprob3[1]) * 100, 2)
                                ))
                                second_input_logprobs.append(LogProb(
                                    type=1,
                                    token=logprob3[2][1][0],
                                    prob=np.round(np.exp(logprob3[2][1][1]) * 100, 2)
                                ))
                    if is_break:
                        break
            if is_break:
                continue

    # Run all collected function strings concurrently
    is_passed_list = run_test_cases(ground_truth, test_list, timeout=5)

    # Create TestCase instances with the results
    for i, is_passed in enumerate(is_passed_list):
        data = testcase_data_list[i]
        testcase = TestCase(
            text=data['text'],
            input_logprobs=data['input_logprobs'],
            output_logprobs=data['output_logprobs'],
            is_valid=is_passed,
            second_input_logprobs=data['second_input_logprobs'],
            second_output_logprobs=data['second_output_logprobs']
        )
        all_tests.append(testcase)

    return all_tests


def get_all_tests(dataset: str, llm: str) -> List[Function]:
    # Define the file paths for raw and processed data
    raw_file_name = f'unfiltered_testcases/{dataset}_{llm}.pkl'
    processed_file_name = f'unfiltered_testcases/{dataset}_{llm}_processed.pkl'

    # Check if the processed file exists
    if os.path.exists(processed_file_name):
        print(f'Loading processed functions from {processed_file_name}...')
        with open(processed_file_name, 'rb') as f:
            functions: List[Function] = pickle.load(f)
            if dataset == "LeetCode":
                functions = [r for idx, r in enumerate(functions) if idx != 60]
                # for idx, f in enumerate(functions):
                #     if 'fraction_to_decimal' in f.prompt:
                #         print(idx)
                # sys.exit()
            if dataset == "LeetCode" and llm == "llama3":
                functions = [r for idx, r in enumerate(functions) if idx != 286]
            if dataset == "MBPP" and llm == "gpt-4o":
                functions = [r for idx, r in enumerate(functions) if idx != 400]
            if dataset == "HumanEval" and llm == "llama3":
                functions = [r for idx, r in enumerate(functions) if idx != 79]
            if dataset == "HumanEval" and llm == "gpt-4o":
                functions = [r for idx, r in enumerate(functions) if idx != 146]
            if dataset == 'HumanEval' and llm == "gpt-3.5-turbo":
                # print('here')
                functions = [r for idx, r in enumerate(functions) if idx != 56]
        return functions

    # Load raw data if processed data does not exist
    try:
        with open(raw_file_name, 'rb') as f:
            raw_probs: List[RawLogProbs] = pickle.load(f)
    except FileNotFoundError:
        print(f'File {raw_file_name} not found')
        raise FileNotFoundError

    # Special handling for HumanEval dataset
    if dataset == 'HumanEval' and llm == 'gpt-4o':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx not in (39, 147)]
    elif dataset == 'MBPP' and llm == 'gpt-4o':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx not in (184, 244)]
    elif dataset == 'LeetCode':
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx not in (70, 186, 481)]
    elif dataset == 'MBPP' and llm == "llama3":
        raw_probs = [r for idx, r in enumerate(raw_probs) if idx != 250]
    # print(raw_probs[50])
    functions: List['Function'] = []
    all_errors = ErrorTypes()
    print(f'the dataset length is {len(raw_probs)}')
    for idx,prob in tqdm(enumerate(raw_probs)):
        try:
            if len(prob.testcases) == 0:
                continue
            try:
                test_type = prob.test_type
            except AttributeError:
                test_type = 0
            testcases, errors = get_logprobs_dynamic(prob.logprobs, prob.testcases, prob.prompt,
                                                               prob.solution, test_type,dataset)
            # if errors.import_errors !=0 :
            #     print(prob.solution)
            #     for t in testcases:
            #         print(t.text)
            #     print('*'*100)
            all_errors += errors

        except Exception as e:
            # print(e)
            # print('Error processing test cases, skipping...')
            continue

        # Create Function object and append to the list
        f = Function(prompt=prob.prompt, testcases=testcases, solution=prob.solution)
        functions.append(f)
    # print(all_errors)
    # Save processed functions for future use
    print(f'Saving processed functions to {processed_file_name}...')
    with open(processed_file_name, 'wb') as f:
        pickle.dump(functions, f)

    return functions