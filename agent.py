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
        
        # IF CODE FILE IS UPLOADED - Check code-specific tools first
        if file_context and file_context.get('type') == 'code':
            
            # Code execution (HIGHEST PRIORITY)
            if any(phrase in query_lower for phrase in ["run this", "execute this", "run the code", "execute the code", "run code"]):
                return "code_executor"
            
            # Security scanning (HIGH PRIORITY)
            if any(word in query_lower for word in ["security", "vulnerability", "vulnerabilities", "secure", "exploit", "injection", "scan"]):
                return "security_scanner"
            
            # Code fixing (MEDIUM-HIGH PRIORITY)
            if any(phrase in query_lower for phrase in ["fix", "fix issue", "fix code", "fix bug", "suggest fix", "repair", "fix this", "fix the"]):
                return "code_fixer"
            
            # Test generation (MEDIUM PRIORITY)
            if any(phrase in query_lower for phrase in ["generate test", "create test", "write test", "pytest", "unit test", "test case"]):
                return "test_generator"
            
            # Code analysis (DEFAULT for code files if any analysis word is present)
            if any(word in query_lower for word in ["analyze", "analyse", "check", "review", "find bugs", "issues", "quality", "inspect", "metric"]):
                return "code_analyzer"
            
            # If code file but no specific keywords, default to analyzer
            return "code_analyzer"
        
        # Document operations (when no code file)
        if any(word in query_lower for word in ["summary", "summarize", "overview", "brief"]):
            return "summarizer"
        
        # Default: document search
        return "document_search"
    
    def execute(self, query: str, file_context: Dict = None) -> Dict[str, Any]:
        """Execute query through appropriate tool"""
        try:
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
                with open(file_context['path'], 'r', encoding='utf-8') as f:
                    code = f.read()
                tool_output = tool.execute(code)
            elif tool_name == "test_generator" and file_context:
                tool_output = tool.execute(file_context['path'])
            elif tool_name == "code_executor" and file_context:
                with open(file_context['path'], 'r', encoding='utf-8') as f:
                    code = f.read()
                tool_output = tool.execute(code)
            elif tool_name == "code_fixer" and file_context:
                tool_output = tool.execute(file_context['path'])
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
        except Exception as e:
            return {
                "answer": f"Error executing query: {str(e)}",
                "tool_used": "error",
                "raw_output": {"error": str(e)},
                "success": False
            }
    
    def _generate_answer(self, query: str, tool_output: Dict, tool_name: str) -> str:
        """Generate natural language answer from tool output"""
        
        # TEMPLATE-BASED RESPONSES (No LLM)
        
        if tool_name == "test_generator":
            functions_found = tool_output.get("functions_found", 0)
            test_cases = tool_output.get("test_cases_generated", 0)
            test_file = tool_output.get("test_file", "")
            funcs = tool_output.get("functions", [])
            
            if len(funcs) <= 3:
                func_list = ', '.join(funcs)
            else:
                func_list = f"{', '.join(funcs[:3])} and {len(funcs) - 3} more"
            
            return f"Test Generation Complete\n\nFunctions tested: {func_list}\nTotal test cases: {test_cases}\nFile saved: {test_file}\nRun with: pytest {test_file} -v"
        
        elif tool_name == "code_executor":
            output = tool_output.get("output", "")
            error = tool_output.get("error", "")
            exec_time = tool_output.get("execution_time", "")
            exit_code = tool_output.get("exit_code", 0)
            timeout = tool_output.get("timeout", False)
            
            if timeout:
                return f"Execution Timeout\n\nCode execution exceeded {exec_time}. Possible infinite loop or heavy computation."
            
            if exit_code == 0:
                output_display = output.strip() if output.strip() else "(no output)"
                if len(output_display) > 400:
                    output_display = output_display[:400] + "\n... (truncated)"
                return f"Execution Successful ({exec_time})\n\nOutput:\n{output_display}"
            else:
                error_display = error.strip() if error.strip() else "Unknown error"
                if len(error_display) > 400:
                    error_display = error_display[:400] + "\n... (truncated)"
                return f"Execution Failed (exit code {exit_code}, {exec_time})\n\nError:\n{error_display}"
        
        elif tool_name == "code_fixer":
            fixes = tool_output.get("fixes", [])
            fixes_count = tool_output.get("fixes_suggested", 0)
            original_issues = tool_output.get("original_issues", 0)
            
            if fixes_count == 0:
                return "No Issues to Fix\n\nYour code looks good! No automatic fixes needed."
            
            fix_summary = ""
            for i, fix in enumerate(fixes[:3], 1):
                line = fix.get('line', 'N/A')
                issue_type = fix.get('issue_type', 'Unknown')
                fix_text = fix.get('fix', 'No fix available')
                fix_summary += f"\n{i}. Line {line} - {issue_type}\n   Fix: {fix_text}\n"
            
            if len(fixes) > 3:
                fix_summary += f"\n... and {len(fixes) - 3} more fixes available in details."
            
            return f"Fix Suggestions Generated\n\nFound {original_issues} issues, generated {fixes_count} fix suggestions.{fix_summary}\n\nExpand 'Detailed Analysis' below for code examples."
        
        # LLM-BASED RESPONSES (For Analysis Tools)
        
        if tool_name == "code_analyzer":
            issues = tool_output.get("issues", [])
            metrics = tool_output.get("metrics", {})
            severity = tool_output.get("severity", "UNKNOWN")
            
            high = len([i for i in issues if i.get("severity") == "HIGH"])
            medium = len([i for i in issues if i.get("severity") == "MEDIUM"])
            low = len([i for i in issues if i.get("severity") == "LOW"])
            
            top_issues = sorted(issues, key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x.get("severity", ""), 0), reverse=True)[:2]
            issues_summary = ", ".join([f"{i.get('type', 'Unknown')} (line {i.get('line', 'N/A')})" for i in top_issues]) if top_issues else "None"
            
            prompt = f"""Write a concise 2-sentence code quality summary. Be direct and specific.

Issues: {len(issues)} total ({high} high, {medium} medium, {low} low)
Severity: {severity}
Top problems: {issues_summary}
Code quality: {metrics.get('comment_ratio', 0):.0f}% commented

Summary:"""

        elif tool_name == "security_scanner":
            vulnerabilities = tool_output.get("vulnerabilities", [])
            risk_level = tool_output.get("risk_level", "UNKNOWN")
            
            critical = len([v for v in vulnerabilities if v.get("severity") == "CRITICAL"])
            high = len([v for v in vulnerabilities if v.get("severity") == "HIGH"])
            medium = len([v for v in vulnerabilities if v.get("severity") == "MEDIUM"])
            
            top_vulns = [v for v in vulnerabilities if v.get("severity") in ["CRITICAL", "HIGH"]][:2]
            vuln_summary = ", ".join([v.get('type', 'Unknown') for v in top_vulns]) if top_vulns else "None critical"
            
            prompt = f"""Write a concise 2-sentence security assessment. Be direct and specific.

Risk: {risk_level}
Vulnerabilities: {len(vulnerabilities)} total ({critical} critical, {high} high, {medium} medium)
Critical issues: {vuln_summary}

Assessment:"""

        elif tool_name == "document_search":
            chunks = tool_output.get("chunks", [])
            if not chunks:
                return "No relevant information found in the documents."
            
            # Get best matching chunks
            context_parts = []
            for chunk in chunks[:3]:
                context_parts.append(chunk.get("text", ""))
            
            context = " ".join(context_parts)[:1800]
            
            # ULTRA STRICT PROMPT - Prevents question generation
            prompt = f"""Based on the information below, provide a direct factual answer to the question. Do NOT generate questions, do NOT ask for clarification, do NOT list numbered points unless specifically asked.

QUESTION: {query}

INFORMATION:
{context}

ANSWER (write 2-3 complete sentences stating facts only):"""

            try:
                response = self.llm(
                    prompt,
                    max_tokens=120,
                    temperature=0.2,  # Very low for factual
                    top_p=0.8,
                    stop=["\n\nQUESTION:", "\nQUESTION:", "INFORMATION:", "\n\n\n", "1.", "2.", "3.", "###", "Question:", "question:"],
                    repeat_penalty=1.5  # High penalty to prevent repetition
                )
                
                answer = response['choices'][0]['text'].strip()
                
                # Clean answer
                for prefix in ["ANSWER:", "Answer:", "Direct Answer:", "Response:", "Based on", "According to"]:
                    if answer.startswith(prefix):
                        answer = answer[len(prefix):].strip()
                
                answer = answer.lstrip('•-– :')
                
                # Remove if starts with number (likely a question)
                if answer and answer[0].isdigit() and '. ' in answer[:5]:
                    return self._fallback_answer(tool_output, tool_name)
                
                # Check for question words at start
                question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'compare', 'list']
                if any(answer.lower().startswith(word) for word in question_words):
                    return self._fallback_answer(tool_output, tool_name)
                
                # Ensure proper ending
                if answer and answer[-1] not in '.!?':
                    last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
                    if last_period > 20:
                        answer = answer[:last_period + 1]
                
                # Length validation
                if len(answer) < 15 or len(answer) > 600:
                    return self._fallback_answer(tool_output, tool_name)
                
                return answer
                
            except Exception as e:
                print(f"LLM generation error: {e}")
                return self._fallback_answer(tool_output, tool_name)


        else:  # summarizer
            highlights = tool_output.get("highlights", [])
            if highlights:
                return "Key Points:\n" + "\n".join([f"- {h}" for h in highlights[:5]])
            
            text = tool_output.get("full_text", "")[:1000]
            prompt = f"""Summarize in 3 bullet points. Be concise.

Text: {text}

Summary:"""

        # Generate LLM response
        try:
            response = self.llm(
                prompt,
                max_tokens=200,
                temperature=0.5,
                top_p=0.9,
                stop=["Question:", "\n\n\n", "Human:", "Assistant:", "Context:", "Text:", "Summary:", "Assessment:", "Answer:"],
                repeat_penalty=1.2
            )
            
            answer = response['choices'][0]['text'].strip()
            
            # Clean prefixes
            for prefix in ["Assistant:", "Human:", "AI:", "Response:", "Answer:", "Summary:", "Assessment:"]:
                if answer.lower().startswith(prefix.lower()):
                    answer = answer[len(prefix):].strip()
            
            answer = answer.lstrip('•-– ')
            
            if answer and answer[-1] not in '.!?':
                last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
                if last_period > 50:
                    answer = answer[:last_period + 1]
                else:
                    answer += "."
            
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
            
            high = len([i for i in issues if i.get("severity") == "HIGH"])
            medium = len([i for i in issues if i.get("severity") == "MEDIUM"])
            low = len([i for i in issues if i.get("severity") == "LOW"])
            
            return f"Code Analysis: Found {len(issues)} issues (Severity: {severity}). {high} high priority, {medium} medium, {low} low. Code has {metrics.get('comment_ratio', 0):.1f}% comment coverage."
        
        elif tool_name == "security_scanner":
            vulns = tool_output.get("vulnerabilities", [])
            risk = tool_output.get("risk_level", "UNKNOWN")
            
            critical = len([v for v in vulns if v.get("severity") == "CRITICAL"])
            high = len([v for v in vulns if v.get("severity") == "HIGH"])
            
            return f"Security Scan: Risk level {risk}. Detected {len(vulns)} vulnerabilities: {critical} critical, {high} high severity."
        
        elif tool_name == "code_fixer":
            fixes_count = tool_output.get("fixes_suggested", 0)
            return f"Code Fixer: Generated {fixes_count} fix suggestions. Review details below."
        
        elif tool_name == "test_generator":
            functions_found = tool_output.get("functions_found", 0)
            test_file = tool_output.get("test_file", "")
            return f"Tests Generated: Created {functions_found} test functions in {test_file}."
        
        elif tool_name == "code_executor":
            output = tool_output.get("output", "")[:200]
            exit_code = tool_output.get("exit_code", 0)
            if exit_code == 0:
                return f"Execution Success: Code ran successfully.\n\nOutput:\n{output}"
            else:
                return f"Execution Failed: Exit code {exit_code}."
        
        elif tool_name == "document_search":
            chunks = tool_output.get("chunks", [])
            if chunks:
                # Return first chunk directly as factual answer
                text = chunks[0].get("text", "")
                # Get first 2-3 sentences
                sentences = text.split('.')[:3]
                return '. '.join(sentences).strip() + '.'
            return "No relevant information found in the documents."

        
        else:  # summarizer
            highlights = tool_output.get("highlights", [])
            if highlights:
                return "Key Points:\n" + "\n".join([f"- {h}" for h in highlights[:4]])
            return "Unable to generate summary."
