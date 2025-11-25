from llama_cpp import Llama
from typing import Dict, Any, List  # ADD List here!
import json


class RAGAgent:
    """AI Agent with intent detection and tool routing"""
    
    def __init__(self, llm: Llama, tools: Dict):
        self.llm = llm
        self.tools = tools
    
    def detect_intent(self, query: str, file_context: Dict = None) -> str:
        """Detect which tool to use based on query and context"""
        query_lower = query.lower()
        
        # Security scanning (HIGHEST PRIORITY for security-related queries)
        if any(word in query_lower for word in ["security", "vulnerability", "vulnerabilities", "secure", "exploit", "injection", "hack", "threat", "risk"]):
            if file_context and file_context.get('type') == 'code':
                return "security_scanner"
        
        # Code analysis (for general code quality, bugs, issues)
        if file_context and file_context.get('type') == 'code':
            if any(word in query_lower for word in ["analyze", "check", "review", "find bugs", "issues", "quality", "problems"]):
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
        
        if tool_name == "code_analyzer":
            issues = tool_output.get("issues", [])
            metrics = tool_output.get("metrics", {})
            severity = tool_output.get("severity", "UNKNOWN")
            
            high = len([i for i in issues if i["severity"] == "HIGH"])
            medium = len([i for i in issues if i["severity"] == "MEDIUM"])
            low = len([i for i in issues if i["severity"] == "LOW"])
            
            # Get top 2 issues
            top_issues = sorted(issues, key=lambda x: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(x["severity"], 0), reverse=True)[:2]
            issues_text = "\n".join([f"- Line {i['line']}: {i['type']}" for i in top_issues])
            
            prompt = f"""You are a code analyzer. Write ONLY a direct 2-3 sentence summary. Do not simulate conversations or include "Assistant:" or "Human:".

Analysis Results:
- Total Issues: {len(issues)} ({high} HIGH, {medium} MEDIUM, {low} LOW)
- Overall Severity: {severity}
- Comment Coverage: {metrics.get('comment_ratio', 0):.1f}%

Top Issues Found:
{issues_text}

Write a direct, professional summary starting with "The code contains" or "Analysis found":"""

        elif tool_name == "security_scanner":
            vulnerabilities = tool_output.get("vulnerabilities", [])
            risk_level = tool_output.get("risk_level", "UNKNOWN")
            
            critical = len([v for v in vulnerabilities if v["severity"] == "CRITICAL"])
            high = len([v for v in vulnerabilities if v["severity"] == "HIGH"])
            
            top_vulns = vulnerabilities[:2]
            vulns_text = "\n".join([f"- {v['type']}" for v in top_vulns])
            
            prompt = f"""You are a security scanner. Write ONLY a direct 2-3 sentence security assessment. Do not simulate conversations.

Scan Results:
- Risk Level: {risk_level}
- Total Vulnerabilities: {len(vulnerabilities)} ({critical} CRITICAL, {high} HIGH)

Critical Findings:
{vulns_text}

Write a direct assessment starting with "Security scan detected" or "Analysis identified":"""

        elif tool_name == "document_search":
            context = tool_output.get("context", "")[:1200]
            prompt = f"""Answer this question directly in 2-3 sentences using the context below. Do not include "Assistant:" or simulate conversations.

Question: {query}

Context:
{context}

Direct answer:"""

        else:  # summarizer
            text = tool_output.get("full_text", "")[:1200]
            prompt = f"""Summarize this text directly in 3-4 sentences. Do not simulate conversations.

Text:
{text}

Summary:"""
    
        # Generate response
        try:
            response = self.llm(
                prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["Question:", "\n\n\n\n", "Human:", "Assistant:", "Context:"]  # Stop on conversation markers
            )
            
            answer = response['choices'][0]['text'].strip()
            
            # Remove any conversational prefixes
            for prefix in ["Assistant:", "Human:", "AI:", "Response:"]:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):].strip()
            
            # Clean trailing incomplete sentences
            if answer and answer[-1] not in '.!?':
                last_period = max(answer.rfind('.'), answer.rfind('!'), answer.rfind('?'))
                if last_period > 0:
                    answer = answer[:last_period + 1]
            
            # Fallback if too short
            if len(answer) < 30:
                return self._fallback_answer(tool_output, tool_name)
            
            return answer
            
        except Exception as e:
            print(f"LLM error: {e}")
            return self._fallback_answer(tool_output, tool_name)


    
    def _format_issues(self, issues: List[Dict]) -> str:
        """Format issues for prompt"""
        formatted = []
        for issue in issues:
            formatted.append(f"- Line {issue['line']}: {issue['type']} ({issue['severity']})")
        return "\n".join(formatted) if formatted else "None"
    
    def _format_vulnerabilities(self, vulns: List[Dict]) -> str:
        """Format vulnerabilities for prompt"""
        formatted = []
        for vuln in vulns:
            formatted.append(f"- {vuln['type']}: {vuln['description']}")
        return "\n".join(formatted) if formatted else "None found"
    
    def _fallback_answer(self, tool_output: Dict, tool_name: str) -> str:
        """Fallback template-based answer when LLM fails"""
        if tool_name == "code_analyzer":
            issues = tool_output.get("issues", [])
            metrics = tool_output.get("metrics", {})
            severity = tool_output.get("severity", "UNKNOWN")
            
            high = len([i for i in issues if i["severity"] == "HIGH"])
            medium = len([i for i in issues if i["severity"] == "MEDIUM"])
            low = len([i for i in issues if i["severity"] == "LOW"])
            
            return f"Code analysis complete. Found {len(issues)} issues (Severity: {severity}). Breakdown: {high} high, {medium} medium, {low} low. Comment ratio: {metrics.get('comment_ratio', 0):.1f}%. See details below for specific issues and recommendations."
        
        elif tool_name == "security_scanner":
            vulns = tool_output.get("vulnerabilities", [])
            risk = tool_output.get("risk_level", "UNKNOWN")
            
            critical = len([v for v in vulns if v["severity"] == "CRITICAL"])
            high = len([v for v in vulns if v["severity"] == "HIGH"])
            
            return f"Security scan complete. Risk Level: {risk}. Found {len(vulns)} vulnerabilities ({critical} critical, {high} high severity). Review the detailed report below for specific vulnerabilities and recommended fixes."
        
        elif tool_name == "document_search":
            chunks = tool_output.get("chunks", [])
            if chunks:
                return chunks[0].get("text", "No information found.")[:250] + "..."
            return "No relevant information found in documents."
        
        else:  # summarizer
            highlights = tool_output.get("highlights", [])
            if highlights:
                return " ".join(highlights[:3])
            return "Unable to generate summary."

