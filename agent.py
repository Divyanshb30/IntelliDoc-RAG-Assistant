from llama_cpp import Llama
from typing import Dict, Any, List
import json

class RAGAgent:
    """AI Agent with intent detection and tool routing"""
    
    def __init__(self, llm: Llama, tools: Dict):
        self.llm = llm
        self.tools = tools
    
    def detect_intent(self, query: str, file_context: Dict = None) -> str:
        """Detect which tool to use based on query and context"""
        query_lower = query.lower()
        
        # Code execution (HIGHEST PRIORITY for explicit run commands)
        if any(phrase in query_lower for phrase in ["run this", "execute this", "run the code", "execute the code", "run code", "execute code"]):
            if file_context and file_context.get('type') == 'code':
                return "code_executor"
        
        # Security scanning (HIGH PRIORITY for security-related queries)
        if any(word in query_lower for word in ["security", "vulnerability", "vulnerabilities", "secure", "exploit", "injection", "hack", "threat", "risk", "scan security"]):
            if file_context and file_context.get('type') == 'code':
                return "security_scanner"
        
        # Test generation (MEDIUM PRIORITY - specific phrases only, no generic "test")
        if any(phrase in query_lower for phrase in ["generate test", "create test", "write test", "make test", "pytest", "unit test", "test case", "test generation", "test file"]):
            if file_context and file_context.get('type') == 'code':
                return "test_generator"
        
        # Code analysis (MEDIUM PRIORITY - for general code quality)
        if file_context and file_context.get('type') == 'code':
            if any(word in query_lower for word in ["analyze", "analyse", "check", "review", "find bugs", "issues", "quality", "problems", "lint", "inspect", "find issues"]):
                return "code_analyzer"
        
        # Document operations
        if any(word in query_lower for word in ["summary", "summarize", "overview", "tl;dr", "brief"]):
            return "summarizer"
        
        # Default: document search
        return "document_search"

    
    def execute(self, query: str, file_context: Dict = None) -> Dict[str, Any]:
        """Execute query through appropriate tool"""
        # Detect intent
        tool_name = self.detect_intent(query, file_context)
        
        # Get tool
        tool = self.tools.get(tool_name)
        if not tool:
            return {"success": False, "error": f"Tool '{tool_name}' not found"}
        
        # Execute tool based on type
        if tool_name == "code_analyzer" and file_context:
            tool_output = tool.execute(file_context['path'])
        elif tool_name == "security_scanner" and file_context:
            with open(file_context['path'], 'r') as f:
                code = f.read()
            tool_output = tool.execute(code)
        elif tool_name == "test_generator" and file_context:  # NEW
            tool_output = tool.execute(file_context['path'])
        elif tool_name == "code_executor" and file_context:  # NEW
            with open(file_context['path'], 'r') as f:
                code = f.read()
            tool_output = tool.execute(code)
        elif tool_name in ["document_search", "summarizer"]:
            tool_output = tool.execute(query)
        else:
            tool_output = {"success": False, "error": "Invalid tool execution"}
        
        # Generate natural language response
        if tool_output.get("success"):
            answer = self._generate_answer(query, tool_output, tool_name)
        else:
            answer = f"Error: {tool_output.get('error', 'Unknown error')}"
        
        return {
            "answer": answer,
            "tool_used": tool_name,
            "raw_output": tool_output,
            "success": tool_output.get("success", False)
        }
    
    def _generate_answer(self, query: str, tool_output: Dict, tool_name: str) -> str:
        """Generate natural language answer from tool output"""
        
        # ===== TEMPLATE-BASED RESPONSES (No LLM - Fast & Clean) =====
        
        if tool_name == "test_generator":
            functions_found = tool_output.get("functions_found", 0)
            test_cases = tool_output.get("test_cases_generated", 0)
            test_file = tool_output.get("test_file", "")
            funcs = tool_output.get("functions", [])
            
            # Create concise function list
            if len(funcs) <= 3:
                func_list = ', '.join(funcs)
            else:
                func_list = f"{', '.join(funcs[:3])} and {len(funcs) - 3} more"
            
            return f"✅ **Test Generation Complete**\n\n• Functions tested: {func_list}\n• Total test cases: {test_cases} (normal, edge cases, parametrized)\n• File saved: `{test_file}`\n• Run with: `pytest {test_file} -v`"
        
        elif tool_name == "code_executor":
            output = tool_output.get("output", "")
            error = tool_output.get("error", "")
            exec_time = tool_output.get("execution_time", "")
            exit_code = tool_output.get("exit_code", 0)
            timeout = tool_output.get("timeout", False)
            
            if timeout:
                return f"⏱️ **Execution Timeout**\n\nCode execution exceeded {exec_time}. Possible causes:\n• Infinite loop\n• Heavy computation\n• Long-running operations\n\nConsider optimizing or breaking into smaller parts."
            
            if exit_code == 0:
                output_display = output.strip() if output.strip() else "(no output)"
                if len(output_display) > 400:
                    output_display = output_display[:400] + "\n... (truncated)"
                return f"✅ **Execution Successful** ({exec_time})\n\n``````"
            else:
                error_display = error.strip() if error.strip() else "Unknown error"
                if len(error_display) > 400:
                    error_display = error_display[:400] + "\n... (truncated)"
                return f"❌ **Execution Failed** (exit code {exit_code}, {exec_time})\n\n``````"
        
        # ===== LLM-BASED RESPONSES (For Analysis Tools) =====
        
        if tool_name == "code_analyzer":
            issues = tool_output.get("issues", [])
            metrics = tool_output.get("metrics", {})
            severity = tool_output.get("severity", "UNKNOWN")
            
            high = len([i for i in issues if i["severity"] == "HIGH"])
            medium = len([i for i in issues if i["severity"] == "MEDIUM"])
            low = len([i for i in issues if i["severity"] == "LOW"])
            
            # Get top 2 critical issues
            top_issues = sorted(issues, key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x["severity"], 0), reverse=True)[:2]
            issues_summary = ", ".join([f"{i['type']} (line {i['line']})" for i in top_issues]) if top_issues else "None"
            
            prompt = f"""Write a concise 2-sentence code quality summary. Be direct and specific.

Issues: {len(issues)} total ({high} high, {medium} medium, {low} low)
Severity: {severity}
Top problems: {issues_summary}
Code quality: {metrics.get('comment_ratio', 0):.0f}% commented

Summary:"""

        elif tool_name == "security_scanner":
            vulnerabilities = tool_output.get("vulnerabilities", [])
            risk_level = tool_output.get("risk_level", "UNKNOWN")
            
            critical = len([v for v in vulnerabilities if v["severity"] == "CRITICAL"])
            high = len([v for v in vulnerabilities if v["severity"] == "HIGH"])
            medium = len([v for v in vulnerabilities if v["severity"] == "MEDIUM"])
            
            # Get top 2 critical vulnerabilities
            top_vulns = [v for v in vulnerabilities if v["severity"] in ["CRITICAL", "HIGH"]][:2]
            vuln_summary = ", ".join([v['type'] for v in top_vulns]) if top_vulns else "None critical"
            
            prompt = f"""Write a concise 2-sentence security assessment. Be direct and specific.

Risk: {risk_level}
Vulnerabilities: {len(vulnerabilities)} total ({critical} critical, {high} high, {medium} medium)
Critical issues: {vuln_summary}

Assessment:"""

        elif tool_name == "document_search":
            chunks = tool_output.get("chunks", [])
            if not chunks:
                return "No relevant information found in the documents."
            
            # Get most relevant chunk
            context = chunks[0].get("text", "")[:1000]
            
            prompt = f"""Answer in 2-3 concise sentences using only the context below. Be direct.

Question: {query}

Context: {context}

Answer:"""

        else:  # summarizer
            highlights = tool_output.get("highlights", [])
            if highlights:
                return "**Key Points:**\n" + "\n".join([f"• {h}" for h in highlights[:5]])
            
            text = tool_output.get("full_text", "")[:1000]
            prompt = f"""Summarize in 3 bullet points. Be concise.

Text: {text}

Summary:"""

        # Generate LLM response with strict constraints
        try:
            response = self.llm(
                prompt,
                max_tokens=200,  # Reduced from 512 for conciseness
                temperature=0.5,  # Lower temperature for more focused output
                top_p=0.9,
                stop=["Question:", "\n\n\n", "Human:", "Assistant:", "Context:", "Text:", "Summary:", "Assessment:", "Answer:"],
                repeat_penalty=1.2  # Prevent repetition
            )
            
            answer = response['choices'][0]['text'].strip()
            
            # Aggressive cleaning
            for prefix in ["Assistant:", "Human:", "AI:", "Response:", "Direct answer:", "Summary:", "Assessment:", "Answer:"]:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
            
            # Remove leading dashes or bullets
            answer = answer.lstrip('•-– ')
            
            # Ensure it ends with punctuation
            if answer and answer[-1] not in '.!?':
                last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
                if last_period > 50:  # Keep if we have a reasonable sentence
                    answer = answer[:last_period + 1]
                else:
                    answer += "."
            
            # Fallback if response is too verbose or short
            if len(answer) > 500 or len(answer) < 20:
                return self._fallback_answer(tool_output, tool_name)
            
            return answer
            
        except Exception as e:
            print(f"LLM generation error: {e}")
            return self._fallback_answer(tool_output, tool_name)
    
    def _fallback_answer(self, tool_output: Dict, tool_name: str) -> str:
        """Fallback template-based answer when LLM fails"""
        
        if tool_name == "code_analyzer":
            issues = tool_output.get("issues", [])
            metrics = tool_output.get("metrics", {})
            severity = tool_output.get("severity", "UNKNOWN")
            
            high = len([i for i in issues if i["severity"] == "HIGH"])
            medium = len([i for i in issues if i["severity"] == "MEDIUM"])
            low = len([i for i in issues if i["severity"] == "LOW"])
            
            return f"**Code Analysis:** Found {len(issues)} issues (Severity: {severity}). {high} high priority, {medium} medium, {low} low. Code has {metrics.get('comment_ratio', 0):.1f}% comment coverage. Review detailed findings below."
        
        elif tool_name == "security_scanner":
            vulns = tool_output.get("vulnerabilities", [])
            risk = tool_output.get("risk_level", "UNKNOWN")
            
            critical = len([v for v in vulns if v["severity"] == "CRITICAL"])
            high = len([v for v in vulns if v["severity"] == "HIGH"])
            
            return f"**Security Scan:** Risk level {risk}. Detected {len(vulns)} vulnerabilities: {critical} critical, {high} high severity. Immediate attention required for critical issues."
        
        elif tool_name == "test_generator":
            functions_found = tool_output.get("functions_found", 0)
            test_file = tool_output.get("test_file", "")
            
            return f"**Tests Generated:** Created {functions_found} test functions in `{test_file}`. Run with: `pytest {test_file} -v`"
        
        elif tool_name == "code_executor":
            output = tool_output.get("output", "")[:200]
            exit_code = tool_output.get("exit_code", 0)
            
            if exit_code == 0:
                return f"**Execution Success:** Code ran successfully.\n\n``````"
            else:
                return f"**Execution Failed:** Exit code {exit_code}. Check error details below."
        
        elif tool_name == "document_search":
            chunks = tool_output.get("chunks", [])
            if chunks:
                return chunks[0].get("text", "")[:300] + "..."
            return "No relevant information found."
        
        else:  # summarizer
            highlights = tool_output.get("highlights", [])
            if highlights:
                return "**Key Points:**\n" + "\n".join([f"• {h}" for h in highlights[:4]])
            return "Unable to generate summary from provided content."
