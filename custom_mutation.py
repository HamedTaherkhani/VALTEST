import ast
import copy
import sys

from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import List, Tuple, Set
import concurrent.futures
import signal
TIMEOUT = 5  # Timeout in seconds for each mutant execution

class MutationOperator(ABC):
    @abstractmethod
    def generate_mutants(self, node: ast.AST) -> List[ast.AST]:
        pass

class NodeIDAssigner(ast.NodeVisitor):
    def __init__(self):
        self.current_id = 0

    def generic_visit(self, node):
        node.node_id = self.current_id
        self.current_id += 1
        super().generic_visit(node)

class AOROperator(MutationOperator):
    """Arithmetic Operator Replacement (AOR):
    Replaces arithmetic operators like +, -, *, / with others."""
    def generate_mutants(self, node: ast.AST) -> List[ast.AST]:
        mutants = []
        arithmetic_ops = [ast.Add, ast.Sub, ast.Mult, ast.Div]
        binop_nodes = [n for n in ast.walk(node) if isinstance(n, ast.BinOp)]
        for binop_node in binop_nodes:
            if isinstance(binop_node.op, tuple(arithmetic_ops)):
                original_op_class = type(binop_node.op)
                for new_op_class in arithmetic_ops:
                    if new_op_class != original_op_class:
                        mutated_tree = copy.deepcopy(node)
                        for n in ast.walk(mutated_tree):
                            if hasattr(n, 'node_id') and n.node_id == binop_node.node_id:
                                n.op = new_op_class()
                                break
                        mutants.append(mutated_tree)
        return mutants

class ROROperator(MutationOperator):
    """Relational Operator Replacement (ROR):
    Changes relational operators like >, <, >=, <=, ==, !=."""
    def generate_mutants(self, node: ast.AST) -> List[ast.AST]:
        mutants = []
        relational_ops = [ast.Gt, ast.Lt, ast.GtE, ast.LtE, ast.Eq, ast.NotEq]
        compare_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Compare)]
        for compare_node in compare_nodes:
            for idx, op in enumerate(compare_node.ops):
                if isinstance(op, tuple(relational_ops)):
                    original_op_class = type(op)
                    for new_op_class in relational_ops:
                        if new_op_class != original_op_class:
                            mutated_tree = copy.deepcopy(node)
                            for n in ast.walk(mutated_tree):
                                if hasattr(n, 'node_id') and n.node_id == compare_node.node_id:
                                    n.ops[idx] = new_op_class()
                                    break
                            mutants.append(mutated_tree)
        return mutants

class NOIOperator(MutationOperator):
    """Negation Operator Insertion (NOI):
    Adds or removes negation operators."""
    def generate_mutants(self, node: ast.AST) -> List[ast.AST]:
        mutants = []
        compare_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Compare)]
        for compare_node in compare_nodes:
            # Add 'not' to the comparison
            mutated_tree = copy.deepcopy(node)
            for n in ast.walk(mutated_tree):
                if hasattr(n, 'node_id') and n.node_id == compare_node.node_id:
                    new_node = ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(n))
                    ast.copy_location(new_node, n)
                    parent = self.get_parent(mutated_tree, n.node_id)
                    if parent:
                        for idx, field in enumerate(parent._fields):
                            if getattr(parent, field) is n:
                                setattr(parent, field, new_node)
                                break
                        else:
                            for field, value in ast.iter_fields(parent):
                                if isinstance(value, list):
                                    for i, item in enumerate(value):
                                        if item is n:
                                            value[i] = new_node
                                            break
                    break
            mutants.append(mutated_tree)
        return mutants

    def get_parent(self, node, target_id):
        for n in ast.walk(node):
            for child in ast.iter_child_nodes(n):
                if hasattr(child, 'node_id') and child.node_id == target_id:
                    return n
        return None

class VROperator(MutationOperator):
    """Variable Replacement (VR):
    Replaces a variable with another of the same scope."""
    def generate_mutants(self, node: ast.AST) -> List[ast.AST]:
        mutants = []
        # Collect all variable names
        variable_names = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                variable_names.add(n.id)
        variable_names = list(variable_names)
        name_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Name)]
        for name_node in name_nodes:
            for var_name in variable_names:
                if var_name != name_node.id:
                    mutated_tree = copy.deepcopy(node)
                    for n in ast.walk(mutated_tree):
                        if hasattr(n, 'node_id') and n.node_id == name_node.node_id:
                            n.id = var_name
                            break
                    mutants.append(mutated_tree)
        return mutants

class CROperator(MutationOperator):
    """Constant Replacement (CR):
    Replaces constant values with other constants."""
    def generate_mutants(self, node: ast.AST) -> List[ast.AST]:
        mutants = []
        const_nodes = [n for n in ast.walk(node) if isinstance(n, ast.Constant)]
        for const_node in const_nodes:
            if isinstance(const_node.value, (int, float)):
                new_values = [0, 1, -1, const_node.value + 1, const_node.value - 1]
                for new_value in new_values:
                    if new_value != const_node.value:
                        mutated_tree = copy.deepcopy(node)
                        for n in ast.walk(mutated_tree):
                            if hasattr(n, 'node_id') and n.node_id == const_node.node_id:
                                n.value = new_value
                                break
                        mutants.append(mutated_tree)
            elif isinstance(const_node.value, bool):
                mutated_tree = copy.deepcopy(node)
                for n in ast.walk(mutated_tree):
                    if hasattr(n, 'node_id') and n.node_id == const_node.node_id:
                        n.value = not const_node.value
                        break
                mutants.append(mutated_tree)
        return mutants

class LOROperator(MutationOperator):
    """Logical Operator Replacement (LOR):
    Alters logical operators like 'and', 'or'."""
    def generate_mutants(self, node: ast.AST) -> List[ast.AST]:
        mutants = []
        boolop_nodes = [n for n in ast.walk(node) if isinstance(n, ast.BoolOp)]
        for boolop_node in boolop_nodes:
            original_op_class = type(boolop_node.op)
            if isinstance(boolop_node.op, ast.And):
                new_op = ast.Or()
            elif isinstance(boolop_node.op, ast.Or):
                new_op = ast.And()
            else:
                continue
            mutated_tree = copy.deepcopy(node)
            for n in ast.walk(mutated_tree):
                if hasattr(n, 'node_id') and n.node_id == boolop_node.node_id:
                    n.op = new_op
                    break
            mutants.append(mutated_tree)
        return mutants

def process_mutant(mutant_code, function_name, test_cases):
    # Compile mutant function
    mutant_namespace = {}
    try:
        exec(mutant_code, mutant_namespace)
    except Exception as e:
        return True  # Mutant is killed

    mutant_function = mutant_namespace.get(function_name)
    if not mutant_function:
        return True  # Mutant is killed

    # Run all test cases
    all_test_cases = '\n'.join(test_cases) + '\n'
    test_namespace_mutant = mutant_namespace.copy()
    mutant_passed = run_test_case(all_test_cases, test_namespace_mutant)
    if not mutant_passed:
        return True  # Mutant is killed

    return False  # Mutant survived

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Execution timed out!")


def run_test_case(test_case_code, namespace, timeout_seconds=5):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)

    try:
        # Attempt to execute the test case
        exec(test_case_code, namespace)
        return True  # Test passed
    except AssertionError:
        return False  # Test failed
    except TimeoutException:
        return False  # Timeout exceeded
    except Exception as e: ## not expected
        raise e  # Other exceptions
    finally:
        # Cancel the alarm
        signal.alarm(0)

class MutationTestFramework:
    def __init__(self):
        self.mutation_operators = []
        self.mutants_per_operation = 10

    def add_mutation_operator(self, operator: MutationOperator):
        self.mutation_operators.append(operator)

    def run_tests(self, functions_and_tests: List[Tuple[str, List[str]]]):
        total_mutants = 0
        total_killed = 0
        mutation_scores = []
        all_mutant_codes = []

        for function_code, test_cases in tqdm(functions_and_tests):
            try:
                function_ast = ast.parse(function_code)
            except SyntaxError as e:
                print(f"Error parsing function code: {e}")
                continue

            # Assign unique IDs to nodes
            assigner = NodeIDAssigner()
            assigner.visit(function_ast)

            # Extract function name
            function_name = None
            for node in function_ast.body:
                if isinstance(node, ast.FunctionDef):
                    function_name = node.name
                    break
            if not function_name:
                print("No function definition found in code.")
                continue

            # Compile original function
            namespace = {}
            try:
                exec(function_code, namespace)
            except Exception as e:
                print(f"Error executing original function code: {e}")
                continue
            original_function = namespace.get(function_name)
            if not original_function:
                print(f"Function {function_name} not found after execution.")
                continue

            mutants = []
            mutant_codes_set: Set[str] = set()
            mutant_codes = []

            # Generate mutants using all operators
            for operator in self.mutation_operators:
                operator_mutants = operator.generate_mutants(function_ast)[:self.mutants_per_operation]
                mutants.extend(operator_mutants)

            # Filter out duplicates and mutants identical to the original function
            unique_mutants = []
            original_code = ast.unparse(function_ast)
            for mutant_ast in mutants:
                mutant_code = ast.unparse(mutant_ast)
                if mutant_code != original_code and mutant_code not in mutant_codes_set:
                    mutant_codes_set.add(mutant_code)
                    unique_mutants.append(mutant_ast)
                    mutant_codes.append(mutant_code)

            num_mutants = len(unique_mutants)
            num_killed = 0

            # Use ProcessPoolExecutor to run mutants concurrently with a timeout
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_mutant, mutant_code, function_name, test_cases) for mutant_code in mutant_codes]
                for idx, future in enumerate(concurrent.futures.as_completed(futures)):
                    # print(idx)
                    try:
                        killed = future.result(timeout=TIMEOUT)
                        if killed:
                            num_killed += 1
                    except concurrent.futures.TimeoutError:
                        num_killed += 1  # If timeout, mutant is considered killed
                        print("Mutant execution timed out and is considered killed.")
                    except Exception as e:
                        num_killed += 1  # If exception, mutant is considered killed
                        # print(f"Exception occurred: {e}")
                        # print(function_code)
                        # print(mutant_codes)
                        # print(test_cases)
                        # sys.exit()

            mutation_score = (num_killed / num_mutants) if num_mutants > 0 else 1.0
            mutation_scores.append(mutation_score)
            total_mutants += num_mutants
            total_killed += num_killed
            all_mutant_codes.append(mutant_codes)

        total_mutation_score = total_killed / total_mutants

        return mutation_scores, total_mutation_score, all_mutant_codes

def mutation_testing(functions_and_tests):
    framework = MutationTestFramework()
    framework.add_mutation_operator(AOROperator())
    framework.add_mutation_operator(ROROperator())
    framework.add_mutation_operator(NOIOperator())
    # framework.add_mutation_operator(VROperator())
    framework.add_mutation_operator(CROperator())
    framework.add_mutation_operator(LOROperator())
    # Add other mutation operators as needed
    mutation_scores, total_mutation_score, all_mutant_codes = framework.run_tests(functions_and_tests)
    return mutation_scores, total_mutation_score, all_mutant_codes
