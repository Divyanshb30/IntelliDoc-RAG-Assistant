import os
import json
import pandas as pd
import ast
import re
from typing import Dict, Any, List
from datetime import datetime


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


def get_tools(rag_pipeline):
    """Initialize all tools"""
    return {
        "document_search": DocumentSearchTool(rag_pipeline),
        "code_analyzer": CodeAnalyzerTool(),
        "security_scanner": SecurityScannerTool(),
        "summarizer": SummarizerTool(rag_pipeline)
    }
