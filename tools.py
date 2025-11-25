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
    """Tool 1: RAG-based document search"""
    
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
                    "text": result.get("text", ""),
                    "score": result.get("score", 0)
                })
                context += f"\n[Chunk {i+1}]: {result.get('text', '')}\n"
            
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
        """Analyze Python code for bugs, complexity, and quality"""
        try:
            with open(code_file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {"success": False, "error": f"Syntax Error: {e.msg}"}
            
            issues = []
            
            # Check for common issues
            issues.extend(self._check_mutable_defaults(tree))
            issues.extend(self._check_bare_except(tree))
            issues.extend(self._check_unused_variables(tree))
            issues.extend(self._check_deep_nesting(tree))
            issues.extend(self._check_long_functions(tree))
            
            # Calculate metrics
            metrics = self._calculate_metrics(code, tree)
            
            # Determine severity
            high_count = len([i for i in issues if i.get("severity") == "HIGH"])
            severity = "HIGH" if high_count > 0 else "MEDIUM" if len(issues) > 3 else "LOW"
            
            return {
                "success": True,
                "file": os.path.basename(code_file_path),
                "issues": issues,
                "metrics": metrics,
                "severity": severity,
                "total_lines": len(code.split('\n'))
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_mutable_defaults(self, tree: ast.AST) -> List[Dict]:
        """Check for mutable default arguments"""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for default in node.args.defaults:
                    if isinstance(default, (ast.List, ast.Dict)):
                        issues.append({
                            "type": "Mutable default argument",
                            "severity": "HIGH",
                            "line": node.lineno,
                            "message": f"Function '{node.name}' uses mutable default argument",
                            "suggestion": "Use None as default and initialize inside function"
                        })
        return issues
    
    def _check_bare_except(self, tree: ast.AST) -> List[Dict]:
        """Check for bare except clauses"""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "type": "Bare except clause",
                        "severity": "MEDIUM",
                        "line": node.lineno,
                        "message": "Bare 'except:' catches all exceptions",
                        "suggestion": "Use 'except Exception as e:' instead"
                    })
        return issues
    
    def _check_unused_variables(self, tree: ast.AST) -> List[Dict]:
        """Check for unused variables"""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assigned = set()
                used = set()
                
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                assigned.add(target.id)
                    elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        used.add(child.id)
                
                unused = assigned - used
                for var in unused:
                    if not var.startswith('_'):
                        issues.append({
                            "type": "Unused variable",
                            "severity": "LOW",
                            "line": node.lineno,
                            "message": f"Variable '{var}' is assigned but never used",
                            "suggestion": "Remove unused variable or prefix with underscore"
                        })
        return issues
    
    def _check_deep_nesting(self, tree: ast.AST) -> List[Dict]:
        """Check for deep nesting"""
        issues = []
        
        def get_depth(node, depth=0):
            max_depth = depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.With)):
                    child_depth = get_depth(child, depth + 1)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                depth = get_depth(node)
                if depth >= 4:
                    issues.append({
                        "type": "Deep nesting",
                        "severity": "MEDIUM",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' has nesting depth of {depth}",
                        "suggestion": "Reduce nesting using early returns or guard clauses"
                    })
        return issues
    
    def _check_long_functions(self, tree: ast.AST) -> List[Dict]:
        """Check for long functions"""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                if func_lines > 50:
                    issues.append({
                        "type": "Long function",
                        "severity": "LOW",
                        "line": node.lineno,
                        "message": f"Function '{node.name}' is {func_lines} lines long",
                        "suggestion": "Break into smaller, single-purpose functions"
                    })
        return issues
    
    def _calculate_metrics(self, code: str, tree: ast.AST) -> Dict[str, Any]:
        """Calculate code metrics"""
        lines = code.split('\n')
        total_lines = len(lines)
        code_lines = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
        comment_lines = len([l for l in lines if l.strip().startswith('#')])
        
        functions = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        classes = len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
        
        return {
            "total_lines": total_lines,
            "code_lines": code_lines,
            "comment_lines": comment_lines,
            "comment_ratio": (comment_lines / total_lines * 100) if total_lines > 0 else 0,
            "functions": functions,
            "classes": classes
        }


class SecurityScannerTool:
    """Tool 3: Security vulnerability scanner"""
    
    def execute(self, code: str) -> Dict[str, Any]:
        """Scan code for security vulnerabilities"""
        try:
            vulnerabilities = []
            
            # Scan for various security issues
            vulnerabilities.extend(self._check_hardcoded_secrets(code))
            vulnerabilities.extend(self._check_sql_injection(code))
            vulnerabilities.extend(self._check_weak_crypto(code))
            vulnerabilities.extend(self._check_dangerous_functions(code))
            vulnerabilities.extend(self._check_insecure_deserialization(code))
            
            # Determine risk level
            critical_count = len([v for v in vulnerabilities if v.get("severity") == "CRITICAL"])
            high_count = len([v for v in vulnerabilities if v.get("severity") == "HIGH"])
            
            if critical_count > 0:
                risk_level = "CRITICAL"
            elif high_count > 0:
                risk_level = "HIGH"
            elif len(vulnerabilities) > 0:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            return {
                "success": True,
                "vulnerabilities": vulnerabilities,
                "risk_level": risk_level,
                "total_issues": len(vulnerabilities)
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _check_hardcoded_secrets(self, code: str) -> List[Dict]:
        """Check for hardcoded secrets"""
        vulnerabilities = []
        patterns = {
            r'password\s*=\s*["\'][^"\']+["\']': "Hardcoded password",
            r'api[_-]?key\s*=\s*["\'][^"\']+["\']': "Hardcoded API key",
            r'secret[_-]?key\s*=\s*["\'][^"\']+["\']': "Hardcoded secret key",
            r'token\s*=\s*["\'][^"\']+["\']': "Hardcoded token",
        }
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, vuln_type in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    vulnerabilities.append({
                        "type": vuln_type,
                        "severity": "CRITICAL",
                        "line": line_num,
                        "description": f"Hardcoded credential detected: {line.strip()[:50]}...",
                        "fix": "Use environment variables or secure credential management"
                    })
        return vulnerabilities
    
    def _check_sql_injection(self, code: str) -> List[Dict]:
        """Check for SQL injection vulnerabilities"""
        vulnerabilities = []
        patterns = [
            r'execute\s*\(\s*["\'].*%s.*["\']',
            r'execute\s*\(\s*f["\'].*\{.*\}.*["\']',
            r'execute\s*\(\s*["\'].*\+.*["\']',
        ]
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in patterns:
                if re.search(pattern, line):
                    vulnerabilities.append({
                        "type": "SQL Injection",
                        "severity": "CRITICAL",
                        "line": line_num,
                        "description": "Potential SQL injection vulnerability detected",
                        "fix": "Use parameterized queries or ORM"
                    })
        return vulnerabilities
    
    def _check_weak_crypto(self, code: str) -> List[Dict]:
        """Check for weak cryptography"""
        vulnerabilities = []
        weak_algorithms = ['md5', 'sha1']
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for algo in weak_algorithms:
                if f'hashlib.{algo}' in line:
                    vulnerabilities.append({
                        "type": "Weak Cryptography",
                        "severity": "HIGH",
                        "line": line_num,
                        "description": f"Using weak cryptographic algorithm: {algo.upper()}",
                        "fix": "Use SHA256 or bcrypt for password hashing"
                    })
        return vulnerabilities
    
    def _check_dangerous_functions(self, code: str) -> List[Dict]:
        """Check for dangerous function usage"""
        vulnerabilities = []
        dangerous_funcs = {
            'eval(': ("Code Injection", "Arbitrary code execution via eval()"),
            'exec(': ("Code Injection", "Arbitrary code execution via exec()"),
            'os.system(': ("Command Injection", "Shell command injection via os.system()"),
        }
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            for func, (vuln_type, desc) in dangerous_funcs.items():
                if func in line:
                    vulnerabilities.append({
                        "type": vuln_type,
                        "severity": "CRITICAL",
                        "line": line_num,
                        "description": desc,
                        "fix": f"Avoid using {func.strip('(')} with user input"
                    })
        return vulnerabilities
    
    def _check_insecure_deserialization(self, code: str) -> List[Dict]:
        """Check for insecure deserialization"""
        vulnerabilities = []
        
        lines = code.split('\n')
        for line_num, line in enumerate(lines, 1):
            if 'pickle.loads' in line or 'pickle.load' in line:
                vulnerabilities.append({
                    "type": "Insecure Deserialization",
                    "severity": "HIGH",
                    "line": line_num,
                    "description": "Pickle deserialization can execute arbitrary code",
                    "fix": "Use JSON or safer serialization formats"
                })
        return vulnerabilities


class SummarizerTool:
    """Tool 4: Document summarizer"""
    
    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline
    
    def execute(self, query: str) -> Dict[str, Any]:
        """Generate document summary"""
        try:
            results = self.rag.retrieve(query, top_k=5)
            
            if not results:
                return {"success": False, "error": "No documents to summarize"}
            
            full_text = " ".join([r.get("text", "") for r in results])
            highlights = [r.get("text", "")[:100] for r in results[:3]]
            
            return {
                "success": True,
                "highlights": highlights,
                "full_text": full_text[:2000],
                "num_chunks": len(results)
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
            
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {"success": False, "error": f"Syntax Error: {e.msg}"}
            
            functions = self._extract_functions(tree)
            
            if not functions:
                return {
                    "success": False,
                    "error": "No testable functions found in the code"
                }
            
            test_code = self._generate_test_code(functions, code_file_path)
            test_file_path = self._save_test_file(test_code, code_file_path)
            
            return {
                "success": True,
                "file": os.path.basename(code_file_path),
                "test_file": test_file_path,
                "functions_found": len(functions),
                "test_cases_generated": sum(len(f.get('test_cases', [])) for f in functions),
                "test_code_preview": test_code[:800],
                "functions": [f.get('name', '') for f in functions],
                "message": f"Generated {len(functions)} test functions"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict]:
        """Extract function signatures"""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith('_'):
                    continue
                
                params = []
                for arg in node.args.args:
                    if arg.arg != 'self':
                        param_type = self._infer_type(arg.arg)
                        params.append({'name': arg.arg, 'type': param_type})
                
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
        
        if any(x in name_lower for x in ['list', 'items', 'array', 'values', 'numbers']):
            return 'list'
        elif any(x in name_lower for x in ['num', 'count', 'size', 'length', 'age', 'id', 'a', 'b', 'x', 'y']):
            return 'int'
        elif any(x in name_lower for x in ['price', 'rate', 'ratio', 'percent', 'score']):
            return 'float'
        elif any(x in name_lower for x in ['name', 'text', 'message', 'str', 'title', 'word']):
            return 'str'
        elif any(x in name_lower for x in ['dict', 'map', 'data', 'config']):
            return 'dict'
        elif any(x in name_lower for x in ['flag', 'is_', 'has_', 'can_', 'should_']):
            return 'bool'
        else:
            return 'Any'
    
    def _generate_test_cases(self, func_name: str, params: List[Dict]) -> List[Dict]:
        """Generate test cases"""
        test_cases = []
        
        test_cases.append({'name': 'normal', 'inputs': self._generate_normal_inputs(params), 'type': 'normal'})
        
        if params:
            test_cases.append({'name': 'edge_empty', 'inputs': self._generate_empty_inputs(params), 'type': 'edge'})
            test_cases.append({'name': 'edge_large', 'inputs': self._generate_large_inputs(params), 'type': 'edge'})
        
        return test_cases
    
    def _generate_normal_inputs(self, params: List[Dict]) -> Dict:
        """Generate normal test inputs"""
        inputs = {}
        for param in params:
            param_type = param.get('type', 'Any')
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
        """Generate empty test inputs"""
        inputs = {}
        for param in params:
            param_type = param.get('type', 'Any')
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
        """Generate large test inputs"""
        inputs = {}
        for param in params:
            param_type = param.get('type', 'Any')
            if param_type == 'int':
                inputs[param['name']] = 1000000
            elif param_type == 'float':
                inputs[param['name']] = 999999.99
            elif param_type == 'str':
                inputs[param['name']] = "x" * 100
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
"""

import pytest
from {module_name} import {", ".join([f.get("name", "") for f in functions])}


'''
        
        for func in functions:
            func_name = func.get('name', 'unknown')
            params = func.get('params', [])
            test_cases = func.get('test_cases', [])
            
            code += f'''def test_{func_name}_normal():
    """Test {func_name} with normal inputs"""
'''
            if params and test_cases:
                inputs_str = ", ".join([f"{p['name']}={repr(test_cases[0]['inputs'][p['name']])}" 
                                       for p in params])
                code += f'''    result = {func_name}({inputs_str})
    assert result is not None


'''
            else:
                code += f'''    result = {func_name}()
    assert result is not None


'''
        
        code += '# Run tests with: pytest -v test_*.py\n'
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
    """Tool 6: Safe Python code execution"""
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
        self.max_output_size = 5000
    
    def execute(self, code: str, timeout: int = None) -> Dict[str, Any]:
        """Execute Python code safely"""
        timeout = timeout or self.timeout
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name
            
            start_time = datetime.now()
            
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir(),
                    env={'PYTHONIOENCODING': 'utf-8'}
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
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
                    "error": f"Execution timeout after {timeout} seconds",
                    "exit_code": -1,
                    "execution_time": f"{timeout}s (timeout)",
                    "timeout": True,
                    "message": "Execution timeout"
                }
            
            finally:
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": "",
                "exit_code": -1,
                "execution_time": "0s",
                "timeout": False,
                "message": f"Failed to execute: {str(e)}"
            }


class CodeFixerTool:
    """Tool 7: Code fix suggestions"""
    
    def execute(self, code_file_path: str) -> Dict[str, Any]:
        """Analyze code and suggest fixes"""
        try:
            analyzer = CodeAnalyzerTool()
            analysis = analyzer.execute(code_file_path)
            
            if not analysis.get("success"):
                return analysis
            
            issues = analysis.get("issues", [])
            
            if not issues:
                return {
                    "success": True,
                    "message": "No issues found - code looks good!",
                    "fixes_suggested": 0,
                    "issues": [],
                    "original_issues": 0
                }
            
            fixes = []
            for issue in issues[:10]:
                fix_suggestion = self._generate_fix_suggestion(issue)
                if fix_suggestion:
                    fixes.append(fix_suggestion)
            
            return {
                "success": True,
                "message": f"Generated {len(fixes)} fix suggestions for {len(issues)} issues",
                "fixes_suggested": len(fixes),
                "original_issues": len(issues),
                "fixes": fixes,
                "severity": analysis.get("severity", "UNKNOWN")
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _generate_fix_suggestion(self, issue: Dict) -> Dict[str, Any]:
        """Generate fix suggestion"""
        issue_type = issue.get('type', '').lower()
        line = issue.get('line', 0)
        message = issue.get('message', '')
        
        suggestion = {
            'issue_type': issue.get('type', 'Unknown'),
            'line': line,
            'severity': issue.get('severity', 'UNKNOWN'),
            'original_message': message,
            'fix': '',
            'explanation': '',
            'code_example': ''
        }
        
        if 'mutable default' in issue_type:
            suggestion['fix'] = "Replace mutable default (list/dict) with None"
            suggestion['explanation'] = "Mutable defaults are shared across function calls."
            suggestion['code_example'] = """# Before:
def function(items=[]):
    items.append(x)

# After:
def function(items=None):
    if items is None:
        items = []
    items.append(x)"""
        
        elif 'bare except' in issue_type:
            suggestion['fix'] = "Replace 'except:' with 'except Exception as e:'"
            suggestion['explanation'] = "Bare except catches all exceptions including system exits."
            suggestion['code_example'] = """# Before:
try:
    operation()
except:
    handle_error()

# After:
try:
    operation()
except Exception as e:
    print(f"Error: {e}")
    handle_error()"""
        
        elif 'unused variable' in issue_type:
            suggestion['fix'] = "Remove unused variable or prefix with underscore"
            suggestion['explanation'] = "Unused variables clutter code."
            suggestion['code_example'] = """# Before:
unused_var = calculate()

# After:
_unused_var = calculate()  # or remove"""
        
        elif 'deep nesting' in issue_type:
            suggestion['fix'] = "Reduce nesting using early returns"
            suggestion['explanation'] = "Deep nesting makes code hard to read."
            suggestion['code_example'] = """# Before:
if condition1:
    if condition2:
        do_something()

# After:
if not condition1:
    return
if not condition2:
    return
do_something()"""
        
        else:
            suggestion['fix'] = f"Review and fix: {message}"
            suggestion['explanation'] = issue.get('suggestion', 'Review the code.')
            suggestion['code_example'] = "# See issue details"
        
        return suggestion


def get_tools(rag_pipeline):
    """Initialize all tools"""
    return {
        "document_search": DocumentSearchTool(rag_pipeline),
        "code_analyzer": CodeAnalyzerTool(),
        "security_scanner": SecurityScannerTool(),
        "summarizer": SummarizerTool(rag_pipeline),
        "test_generator": TestGeneratorTool(),
        "code_executor": CodeExecutionTool(),
        "code_fixer": CodeFixerTool()
    }
