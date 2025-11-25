import os
import json
import pandas as pd
import ast
import re
from typing import Dict, Any, List
from datetime import datetime
import subprocess
import tempfile


class DocumentSearchTool:
    """Tool 1: RAG-based document search (UNCHANGED - keep working)"""
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def execute(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search documents and return relevant chunks"""
        try:
            results = self.rag.retrieve(query, top_k=top_k)
            
            chunks = []
            context = ""
            for i, result in enumerate(results):
                chunks.append({
                    "chunk_id": i,
                    "text": result["text"],
                    "score": result["score"]
                })
                context += f"\n[Chunk {i+1}]: {result['text']}\n"
            
            return {
                "success": True,
                "chunks": chunks,
                "context": context,
                "query": query
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class CodeAnalyzerTool:
    """Tool 2: Advanced Python code analysis with AST"""
    
    def execute(self, code_file_path: str) -> Dict[str, Any]:
        """Analyze Python code for bugs, complexity, and best practices"""
        try:
            with open(code_file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {
                    "success": False,
                    "error": f"Syntax Error: {e.msg} at line {e.lineno}"
                }
            
            # Analyze code
            issues = self._analyze_ast(tree, code)
            metrics = self._calculate_metrics(tree, code)
            suggestions = self._generate_suggestions(issues, metrics)
            
            return {
                "success": True,
                "file": os.path.basename(code_file_path),
                "lines": len(code.split('\n')),
                "issues": issues,
                "metrics": metrics,
                "suggestions": suggestions,
                "severity": self._calculate_severity(issues)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _analyze_ast(self, tree: ast.AST, code: str) -> List[Dict]:
        """Analyze AST for common issues"""
        issues = []
        code_lines = code.split('\n')
        
        for node in ast.walk(tree):
            # Check for bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "type": "Bare except clause",
                        "line": node.lineno,
                        "severity": "MEDIUM",
                        "message": "Using bare 'except:' catches all exceptions including system exits",
                        "suggestion": "Use 'except Exception:' or specify exception types"
                    })
            
            # Check for unused variables
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                var_name = node.id
                if var_name.startswith('_') and not var_name.startswith('__'):
                    issues.append({
                        "type": "Potentially unused variable",
                        "line": node.lineno,
                        "severity": "LOW",
                        "message": f"Variable '{var_name}' may be unused (starts with _)",
                        "suggestion": "Remove if truly unused, or rename without underscore"
                    })
            
            # Check for mutable default arguments
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        issues.append({
                            "type": "Mutable default argument",
                            "line": node.lineno,
                            "severity": "HIGH",
                            "message": f"Function '{node.name}' uses mutable default argument",
                            "suggestion": "Use None as default and create mutable object inside function"
                        })
            
            # Check for long functions
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    issues.append({
                        "type": "Long function",
                        "line": node.lineno,
                        "severity": "MEDIUM",
                        "message": f"Function '{node.name}' is {func_lines} lines long",
                        "suggestion": "Consider breaking into smaller functions"
                    })
            
            # Check for deeply nested code
            if isinstance(node, (ast.For, ast.While, ast.If)):
                depth = self._get_nesting_depth(node)
                if depth > 3:
                    issues.append({
                        "type": "Deep nesting",
                        "line": node.lineno,
                        "severity": "MEDIUM",
                        "message": f"Code nested {depth} levels deep",
                        "suggestion": "Refactor to reduce nesting (early returns, extract functions)"
                    })
        
        return issues
    
    def _get_nesting_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate nesting depth of control structures"""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.For, ast.While, ast.If)):
                child_depth = self._get_nesting_depth(child, depth + 1)
                max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _calculate_metrics(self, tree: ast.AST, code: str) -> Dict:
        """Calculate code metrics"""
        metrics = {
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "lines_of_code": len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]),
            "comment_lines": len([l for l in code.split('\n') if l.strip().startswith('#')]),
            "blank_lines": len([l for l in code.split('\n') if not l.strip()]),
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["functions"] += 1
            elif isinstance(node, ast.ClassDef):
                metrics["classes"] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics["imports"] += 1
        
        total_lines = metrics["lines_of_code"] + metrics["comment_lines"] + metrics["blank_lines"]
        metrics["comment_ratio"] = round(metrics["comment_lines"] / max(total_lines, 1) * 100, 1)
        
        return metrics
    
    def _generate_suggestions(self, issues: List[Dict], metrics: Dict) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        
        if metrics["comment_ratio"] < 10:
            suggestions.append("Consider adding more comments (current: {:.1f}%)".format(metrics["comment_ratio"]))
        
        high_severity = len([i for i in issues if i["severity"] == "HIGH"])
        if high_severity > 0:
            suggestions.append(f"Fix {high_severity} high-severity issues first")
        
        if metrics["functions"] == 0 and metrics["lines_of_code"] > 50:
            suggestions.append("Consider organizing code into functions")
        
        return suggestions
    
    def _calculate_severity(self, issues: List[Dict]) -> str:
        """Calculate overall severity"""
        if any(i["severity"] == "HIGH" for i in issues):
            return "HIGH"
        elif any(i["severity"] == "MEDIUM" for i in issues):
            return "MEDIUM"
        elif issues:
            return "LOW"
        return "CLEAN"


class SecurityScannerTool:
    """Tool 3: Security vulnerability scanner"""
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Scan code for security vulnerabilities"""
        try:
            vulnerabilities = []
            
            # Check for SQL injection
            if self._check_sql_injection(code):
                vulnerabilities.append({
                    "type": "SQL Injection Risk",
                    "severity": "CRITICAL",
                    "description": "Potential SQL injection via string formatting",
                    "pattern": "String concatenation or formatting in SQL queries",
                    "fix": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"
                })
            
            # Check for hardcoded secrets
            secrets = self._check_hardcoded_secrets(code)
            if secrets:
                vulnerabilities.append({
                    "type": "Hardcoded Credentials",
                    "severity": "CRITICAL",
                    "description": f"Found {len(secrets)} hardcoded secrets",
                    "patterns": secrets,
                    "fix": "Use environment variables: os.getenv('API_KEY')"
                })
            
            # Check for eval/exec usage
            if 'eval(' in code or 'exec(' in code:
                vulnerabilities.append({
                    "type": "Code Injection Risk",
                    "severity": "CRITICAL",
                    "description": "Use of eval() or exec() with user input",
                    "fix": "Avoid eval/exec entirely, use safe alternatives like ast.literal_eval()"
                })
            
            # Check for pickle usage
            if 'pickle.load' in code or 'pickle.loads' in code:
                vulnerabilities.append({
                    "type": "Insecure Deserialization",
                    "severity": "HIGH",
                    "description": "Pickle can execute arbitrary code during deserialization",
                    "fix": "Use json or msgpack for untrusted data"
                })
            
            # Check for weak cryptography
            if 'md5' in code.lower() or 'sha1' in code.lower():
                vulnerabilities.append({
                    "type": "Weak Cryptography",
                    "severity": "MEDIUM",
                    "description": "MD5/SHA1 are cryptographically weak",
                    "fix": "Use SHA-256 or bcrypt for password hashing"
                })
            
            # Check for open file without context manager
            if re.search(r'\bopen\s*\([^)]+\)(?!\s*with)', code):
                vulnerabilities.append({
                    "type": "Resource Leak",
                    "severity": "LOW",
                    "description": "Files opened without context manager may not close properly",
                    "fix": "Use 'with open() as f:' to ensure file closes"
                })
            
            risk_level = self._calculate_risk_level(vulnerabilities)
            
            return {
                "success": True,
                "vulnerabilities": vulnerabilities,
                "total_issues": len(vulnerabilities),
                "risk_level": risk_level,
                "scan_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_sql_injection(self, code: str) -> bool:
        """Check for SQL injection patterns"""
        patterns = [
            r'execute\s*\([^)]*%[^)]*\)',  # % formatting
            r'execute\s*\([^)]*\+[^)]*\)',  # + concatenation
            r'execute\s*\([^)]*f["\'][^"\']*\{',  # f-string
        ]
        return any(re.search(pattern, code) for pattern in patterns)
    
    def _check_hardcoded_secrets(self, code: str) -> List[str]:
        """Check for hardcoded credentials"""
        patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'aws_secret_access_key\s*=\s*["\'][^"\']+["\']',
        ]
        found = []
        for pattern in patterns:
            matches = re.findall(pattern, code, re.IGNORECASE)
            found.extend(matches)
        return found
    
    def _calculate_risk_level(self, vulnerabilities: List[Dict]) -> str:
        """Calculate overall risk level"""
        if any(v["severity"] == "CRITICAL" for v in vulnerabilities):
            return "CRITICAL"
        elif any(v["severity"] == "HIGH" for v in vulnerabilities):
            return "HIGH"
        elif any(v["severity"] == "MEDIUM" for v in vulnerabilities):
            return "MEDIUM"
        elif vulnerabilities:
            return "LOW"
        return "SECURE"


class SummarizerTool:
    """Tool 4: Document summarizer (UNCHANGED - keep working)"""
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def execute(self, query: str, style: str = "brief") -> Dict[str, Any]:
        """Summarize retrieved content"""
        try:
            # Retrieve relevant chunks
            results = self.rag.retrieve(query, top_k=10)
            
            # Combine text
            full_text = "\n\n".join([r["text"] for r in results])
            
            # Extract key points (simple extractive summarization)
            sentences = full_text.split('.')
            highlights = [s.strip() + '.' for s in sentences if len(s.strip()) > 50][:5]
            
            return {
                "success": True,
                "full_text": full_text[:2000],
                "highlights": highlights,
                "style": style,
                "word_count": len(full_text.split())
            }
        except Exception as e:
            return {"success": False, "error": str(e)}



class TestGeneratorTool:
    """Tool 5: Automatic pytest test generation"""
    
    def execute(self, code_file_path: str) -> Dict[str, Any]:
        """Generate pytest test cases for Python functions"""
        try:
            with open(code_file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {"success": False, "error": f"Syntax Error: {e.msg}"}
            
            # Extract functions
            functions = self._extract_functions(tree)
            
            if not functions:
                return {
                    "success": False,
                    "error": "No testable functions found in the code"
                }
            
            # Generate test code
            test_code = self._generate_test_code(functions, code_file_path)
            test_file_path = self._save_test_file(test_code, code_file_path)
            
            return {
                "success": True,
                "file": os.path.basename(code_file_path),
                "test_file": test_file_path,
                "functions_found": len(functions),
                "test_cases_generated": sum(len(f['test_cases']) for f in functions),
                "test_code_preview": test_code[:800],  # Preview
                "functions": [f['name'] for f in functions],
                "message": f"Generated {len(functions)} test functions covering {sum(len(f['test_cases']) for f in functions)} test cases"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function signatures and metadata"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private/magic functions
                if node.name.startswith('_'):
                    continue
                
                # Extract parameters
                params = []
                for arg in node.args.args:
                    if arg.arg != 'self':  # Skip self parameter
                        param_type = self._infer_type(arg.arg)
                        params.append({
                            'name': arg.arg,
                            'type': param_type
                        })
                
                # Generate test cases
                test_cases = self._generate_test_cases(node.name, params)
                
                functions.append({
                    'name': node.name,
                    'params': params,
                    'test_cases': test_cases,
                    'line': node.lineno
                })
        
        return functions
    
    def _infer_type(self, param_name: str) -> str:
        """Infer parameter type from name"""
        name_lower = param_name.lower()
        
        # Check for list/array types first (before numbers)
        if any(x in name_lower for x in ['list', 'items', 'array', 'values', 'numbers']):
            return 'list'
        # Numeric types
        elif any(x in name_lower for x in ['num', 'count', 'size', 'length', 'age', 'id', 'a', 'b', 'x', 'y']):
            return 'int'
        elif any(x in name_lower for x in ['price', 'rate', 'ratio', 'percent', 'score']):
            return 'float'
        # String types
        elif any(x in name_lower for x in ['name', 'text', 'message', 'str', 'title', 'word']):
            return 'str'
        # Dict types
        elif any(x in name_lower for x in ['dict', 'map', 'data', 'config']):
            return 'dict'
        # Boolean types
        elif any(x in name_lower for x in ['flag', 'is_', 'has_', 'can_', 'should_']):
            return 'bool'
        else:
            return 'Any'
    
    def _generate_test_cases(self, func_name: str, params: List[Dict]) -> List[Dict]:
        """Generate test cases for a function"""
        test_cases = []
        
        # Normal case
        test_cases.append({
            'name': 'normal',
            'inputs': self._generate_normal_inputs(params),
            'type': 'normal'
        })
        
        # Edge cases
        if params:
            # Empty/zero values
            test_cases.append({
                'name': 'edge_empty',
                'inputs': self._generate_empty_inputs(params),
                'type': 'edge'
            })
            
            # Large values
            test_cases.append({
                'name': 'edge_large',
                'inputs': self._generate_large_inputs(params),
                'type': 'edge'
            })
        
        return test_cases
    
    def _generate_normal_inputs(self, params: List[Dict]) -> Dict:
        """Generate normal test inputs"""
        inputs = {}
        for param in params:
            param_type = param['type']
            if param_type == 'int':
                inputs[param['name']] = 5
            elif param_type == 'float':
                inputs[param['name']] = 10.5
            elif param_type == 'str':
                inputs[param['name']] = "test_string"
            elif param_type == 'list':
                inputs[param['name']] = [1, 2, 3]
            elif param_type == 'dict':
                inputs[param['name']] = {"key": "value"}
            elif param_type == 'bool':
                inputs[param['name']] = True
            else:
                inputs[param['name']] = "test"
        return inputs
    
    def _generate_empty_inputs(self, params: List[Dict]) -> Dict:
        """Generate empty/zero test inputs"""
        inputs = {}
        for param in params:
            param_type = param['type']
            if param_type == 'int':
                inputs[param['name']] = 0
            elif param_type == 'float':
                inputs[param['name']] = 0.0
            elif param_type == 'str':
                inputs[param['name']] = ""
            elif param_type == 'list':
                inputs[param['name']] = []
            elif param_type == 'dict':
                inputs[param['name']] = {}
            elif param_type == 'bool':
                inputs[param['name']] = False
            else:
                inputs[param['name']] = None
        return inputs
    
    def _generate_large_inputs(self, params: List[Dict]) -> Dict:
        """Generate large value test inputs"""
        inputs = {}
        for param in params:
            param_type = param['type']
            if param_type == 'int':
                inputs[param['name']] = 1000000
            elif param_type == 'float':
                inputs[param['name']] = 999999.99
            elif param_type == 'str':
                inputs[param['name']] = "x" * 100  # Changed from 1000 to 100
            elif param_type == 'list':
                inputs[param['name']] = list(range(100))
            elif param_type == 'dict':
                inputs[param['name']] = {f"key{i}": i for i in range(10)}
            elif param_type == 'bool':
                inputs[param['name']] = True
            else:
                inputs[param['name']] = "large_test"
        return inputs
    
    def _generate_test_code(self, functions: List[Dict], original_file: str) -> str:
        """Generate pytest test code"""
        module_name = os.path.basename(original_file).replace('.py', '')
        
        code = f'''"""
Auto-generated tests for {module_name}.py
Generated by IntelliCode RAG Assistant on {datetime.now().strftime("%Y-%m-%d %H:%M")}

⚠️ Note: These are template tests. Update assertions with expected values.
"""

import pytest
from {module_name} import {", ".join([f["name"] for f in functions])}


'''
        
        for func in functions:
            # Normal test
            code += f'''def test_{func["name"]}_normal():
    """Test {func["name"]} with normal inputs"""
'''
            if func['params']:
                inputs_str = ", ".join([f"{p['name']}={repr(func['test_cases'][0]['inputs'][p['name']])}" 
                                       for p in func['params']])
                code += f'''    result = {func["name"]}({inputs_str})
    assert result is not None  # TODO: Add specific assertion
    print(f"Normal test passed: {{result}}")


'''
            else:
                code += f'''    result = {func["name"]}()
    assert result is not None  # TODO: Add specific assertion


'''
            
            # Edge case - empty inputs
            if len(func['test_cases']) > 1 and func['params']:
                code += f'''def test_{func["name"]}_edge_empty():
    """Test {func["name"]} with empty/zero inputs"""
'''
                empty_inputs = ", ".join([f"{p['name']}={repr(func['test_cases'][1]['inputs'][p['name']])}"
                                         for p in func['params']])
                code += f'''    # May raise exception - adjust as needed
    result = {func["name"]}({empty_inputs})
    # TODO: Add assertions based on expected behavior


'''
            
            # Edge case - large inputs
            if len(func['test_cases']) > 2 and func['params']:
                code += f'''def test_{func["name"]}_edge_large():
    """Test {func["name"]} with large inputs"""
'''
                large_inputs = ", ".join([f"{p['name']}={repr(func['test_cases'][2]['inputs'][p['name']])}"
                                         for p in func['params']])
                code += f'''    result = {func["name"]}({large_inputs})
    # TODO: Add assertions for large input handling


'''
            
            # Parametrized test template
            if func['params']:
                param_names = [p['name'] for p in func['params']]
            
                code += f'''@pytest.mark.parametrize("{','.join(param_names)},expected", [
    # TODO: Add test cases here
    # Example test cases:
    # (normal_values..., expected_output),
    # (edge_case_values..., expected_output),
])
def test_{func["name"]}_parametrized({', '.join(param_names)}, expected):
    """Parametrized tests for {func["name"]}"""
    result = {func["name"]}({', '.join(param_names)})
    assert result == expected


'''
    
        code += '''
# Run tests with: pytest -v test_*.py
'''
        return code
    
    def _save_test_file(self, test_code: str, original_file: str) -> str:
        """Save generated test file"""
        test_dir = "test"
        os.makedirs(test_dir, exist_ok=True)
        
        module_name = os.path.basename(original_file).replace('.py', '')
        test_file = os.path.join(test_dir, f"test_{module_name}.py")
        
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        return test_file


class CodeExecutionTool:
    """Tool 6: Safe Python code execution in isolated subprocess"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.max_output_size = 5000  # Limit output size
    
    def execute(self, code: str, timeout: int = None) -> Dict[str, Any]:
        """Execute Python code safely in subprocess"""
        timeout = timeout or self.timeout
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name
            
            start_time = datetime.now()
            
            try:
                # Execute in subprocess with timeout
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir(),  # Isolate working directory
                    env={'PYTHONIOENCODING': 'utf-8'}  # Ensure UTF-8 encoding
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # Limit output size
                stdout = result.stdout[:self.max_output_size]
                stderr = result.stderr[:self.max_output_size]
                
                if len(result.stdout) > self.max_output_size:
                    stdout += "\n... (output truncated)"
                if len(result.stderr) > self.max_output_size:
                    stderr += "\n... (output truncated)"
                
                return {
                    "success": result.returncode == 0,
                    "output": stdout if stdout else "(no output)",
                    "error": stderr if stderr else "",
                    "exit_code": result.returncode,
                    "execution_time": f"{execution_time:.2f}s",
                    "timeout": False,
                    "message": "Code executed successfully" if result.returncode == 0 else "Code execution failed"
                }
                
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "output": "",
                    "error": f"⏱️ Execution timeout after {timeout} seconds. Code may have infinite loop or is too slow.",
                    "exit_code": -1,
                    "execution_time": f"{timeout}s (timeout)",
                    "timeout": True,
                    "message": "Execution timeout"
                }
            
            finally:
                # Cleanup temporary file
                try:
                    os.unlink(temp_file)
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Execution error: {str(e)}",
                "output": "",
                "exit_code": -1,
                "execution_time": "0s",
                "timeout": False,
                "message": f"Failed to execute code: {str(e)}"
            }


def get_tools(rag_pipeline):
    """Initialize all tools"""
    return {
        "document_search": DocumentSearchTool(rag_pipeline),
        "code_analyzer": CodeAnalyzerTool(),
        "security_scanner": SecurityScannerTool(),
        "summarizer": SummarizerTool(rag_pipeline),
        "test_generator": TestGeneratorTool(),      # NEW
        "code_executor": CodeExecutionTool()        # NEW
    }

