import ast
import sys

import ast
import sys


def extract_functions_from_solution(code_string, return_signatures=False):
    """
    Extract functions from Solution class and convert to standalone functions.
    Keeps all other classes untouched.

    Args:
        code_string: Python code as a string
        return_signatures: If True, returns (transformed_code, signatures_list)
                          If False, returns just transformed_code

    Returns:
        If return_signatures=False: Transformed code as a string
        If return_signatures=True: Tuple of (transformed_code, list of signature strings)
    """

    # Parse the code
    tree = ast.parse(code_string)

    # Helper function to build signature string from FunctionDef node
    def build_signature(func_node):
        """Build function signature string without body"""
        # Get function name
        sig_parts = [f"def {func_node.name}("]

        # Get arguments (excluding 'self')
        args = func_node.args
        params = []

        # Regular positional arguments
        num_defaults = len(args.defaults)
        num_args = len(args.args)

        for i, arg in enumerate(args.args):
            if arg.arg == 'self':
                continue

            param_str = arg.arg

            # Add type annotation if exists
            if arg.annotation:
                if sys.version_info >= (3, 9):
                    param_str += f": {ast.unparse(arg.annotation)}"

            # Add default value if exists
            default_idx = i - (num_args - num_defaults)
            if default_idx >= 0:
                if sys.version_info >= (3, 9):
                    param_str += f" = {ast.unparse(args.defaults[default_idx])}"

            params.append(param_str)

        # Add *args if exists
        if args.vararg:
            vararg_str = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                if sys.version_info >= (3, 9):
                    vararg_str += f": {ast.unparse(args.vararg.annotation)}"
            params.append(vararg_str)

        # Add keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            param_str = arg.arg
            if arg.annotation:
                if sys.version_info >= (3, 9):
                    param_str += f": {ast.unparse(arg.annotation)}"
            if args.kw_defaults[i]:
                if sys.version_info >= (3, 9):
                    param_str += f" = {ast.unparse(args.kw_defaults[i])}"
            params.append(param_str)

        # Add **kwargs if exists
        if args.kwarg:
            kwarg_str = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                if sys.version_info >= (3, 9):
                    kwarg_str += f": {ast.unparse(args.kwarg.annotation)}"
            params.append(kwarg_str)

        sig_parts.append(", ".join(params))
        sig_parts.append(")")

        # Add return type annotation if exists
        if func_node.returns:
            if sys.version_info >= (3, 9):
                sig_parts.append(f" -> {ast.unparse(func_node.returns)}")

        sig_parts.append(":")

        return "".join(sig_parts)

    # Visitor to remove 'self' parameter and replace self.method() calls
    class SelfRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.args.args and node.args.args[0].arg == 'self':
                node.args.args.pop(0)
            self.generic_visit(node)
            return node

        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == 'self':
                    node.func = ast.Name(id=node.func.attr, ctx=ast.Load())
            self.generic_visit(node)
            return node

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'self':
                return ast.Name(id=node.attr, ctx=node.ctx)
            self.generic_visit(node)
            return node

    # Collect new body for the module and signatures
    new_body = []
    signatures = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Keep all import statements
            new_body.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.name == 'Solution':
                # Extract methods from Solution class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if return_signatures:
                            sig_str = build_signature(item)
                            signatures.append(sig_str)

                        transformer = SelfRemover()
                        transformed_func = transformer.visit(item)
                        new_body.append(transformed_func)
            else:
                # Keep all other classes untouched
                new_body.append(node)
        elif isinstance(node, ast.FunctionDef):
            # Keep other functions that are not in any class
            new_body.append(node)

    # Create new module with transformed body
    new_tree = ast.Module(body=new_body, type_ignores=[])

    # Convert back to source code
    if sys.version_info >= (3, 9):
        transformed_code = ast.unparse(new_tree)
    else:
        try:
            import astor
            transformed_code = astor.to_source(new_tree)
        except ImportError:
            raise RuntimeError("Python 3.9+ required for ast.unparse, or install 'astor' package")

    if return_signatures:
        return transformed_code, signatures
    else:
        return transformed_code


import json
class TestEvalLoader:
    def __init__(self):
        with open('loaders/testeval.jsonl', 'r') as f:
            data = [json.loads(line) for line in f]
        sols = []
        prompts = []
        orig_sols = []
        for d in data:
            sol, sig = extract_functions_from_solution(d['python_solution'],True)
            orig_sols.append(d['python_solution'])
            sols.append(sol)
            prompt = f"""
{sig[0]}
    \"\"\"
    {d['description']}
    \"\"\"
"""
            prompts.append(prompt)
        self.sols = sols
        self.prompts = prompts
        self.orig_sols = orig_sols

    def get_prompts(self):
        return self.prompts
    def get_sols(self):
        return self.sols
    def get_orig_sols(self):
        return self.orig_sols