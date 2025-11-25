import streamlit as st
from llama_cpp import Llama
from rag import RAGPipeline
from tools import get_tools
from agent import RAGAgent
import os
import time

# Page config
st.set_page_config(
    page_title="IntelliCode RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'rag' not in st.session_state:
    st.session_state.rag = None
if 'code_file' not in st.session_state:
    st.session_state.code_file = None
if 'trigger_query' not in st.session_state:
    st.session_state.trigger_query = None

def process_uploaded_file(uploaded_file, file_path):
    """Process uploaded files"""
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        import pypdf
        pdf_reader = pypdf.PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        text_path = file_path.replace('.pdf', '.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text)
        return True
    elif file_type in ['txt', 'md', 'csv', 'py']:
        return True
    else:
        st.warning(f"Unsupported file type: {file_type}")
        return False

# Sidebar
with st.sidebar:
    st.markdown("## IntelliCode RAG")
    st.markdown("*AI-powered code analysis*")
    st.divider()
    
    mode = st.radio("**Mode**", ["Documents", "Code Analysis"], key="mode")
    
    if mode == "Documents":
        st.markdown("### Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=['txt', 'pdf', 'csv', 'md'],
            accept_multiple_files=True,
            key="doc_uploader"
        )
        
        if uploaded_files:
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                file_path = os.path.join(data_dir, uploaded_file.name)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                process_uploaded_file(uploaded_file, file_path)
            
            st.success(f"Uploaded {len(uploaded_files)} file(s)")
            
            if st.button("Build Knowledge Base", type="primary"):
                with st.spinner("Building..."):
                    if st.session_state.rag is None:
                        st.session_state.rag = RAGPipeline()
                    st.session_state.rag.build_index(data_dir)
                st.success("Ready!")
    
    else:  # Code Analysis
        st.markdown("### Upload Code")
        code_file = st.file_uploader(
            "Upload Python file",
            type=['py'],
            key="code_uploader"
        )
        
        if code_file:
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, code_file.name)
            
            with open(file_path, 'wb') as f:
                f.write(code_file.getbuffer())
            
            st.session_state.code_file = {
                'name': code_file.name,
                'path': file_path,
                'type': 'code'
            }
            
            st.success(f"Loaded: {code_file.name}")
            
            # Quick Actions
            st.markdown("### Quick Actions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Analyze Code", use_container_width=True):
                    st.session_state.trigger_query = "Analyze this code"
                
                if st.button("Security Scan", use_container_width=True):
                    st.session_state.trigger_query = "Check security vulnerabilities"
                
                if st.button("Generate Tests", use_container_width=True):
                    st.session_state.trigger_query = "Generate pytest tests"
                
                if st.button("Fix Issues", use_container_width=True):
                    st.session_state.trigger_query = "Fix the issues"
            
            with col2:
                if st.button("Run Code", use_container_width=True):
                    st.session_state.trigger_query = "Run this code"
                
                if st.button("Code Metrics", use_container_width=True):
                    st.session_state.trigger_query = "Show code metrics"
                
                if st.button("Find Bugs", use_container_width=True):
                    st.session_state.trigger_query = "Find bugs in code"
                
                if st.button("Explain Code", use_container_width=True):
                    st.session_state.trigger_query = "Explain this code"
    
    st.divider()
    st.caption("Qwen2.5-3B | FAISS | AST Parser")

# Main content
st.markdown('<p class="main-header">IntelliCode RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered code analysis and documentation</p>', unsafe_allow_html=True)

# Initialize agent
if st.session_state.agent is None:
    with st.spinner("Loading AI model..."):
        try:
            llm = Llama(
                model_path="models/qwen2.5-3b-instruct-q4_k_m.gguf",
                n_ctx=4096,
                n_threads=8,
                verbose=False
            )
            
            if st.session_state.rag is None:
                st.session_state.rag = RAGPipeline()
            
            tools = get_tools(st.session_state.rag)
            st.session_state.agent = RAGAgent(llm, tools)
            st.success("Model loaded")
            
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "raw_output" in message and message.get("tool"):
            raw = message["raw_output"]
            tool = message.get("tool")
            
            # CODE ANALYZER
            if tool == "code_analyzer":
                with st.expander("Detailed Analysis"):
                    issues = raw.get("issues", [])
                    metrics = raw.get("metrics", {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Issues", len(issues))
                    with col2:
                        st.metric("Severity", raw.get("severity", "N/A"))
                    with col3:
                        st.metric("Comments", f"{metrics.get('comment_ratio', 0):.1f}%")
                    
                    for issue in issues:
                        severity = issue.get("severity", "UNKNOWN")
                        line = issue.get("line", "N/A")
                        issue_type = issue.get("type", "Unknown")
                        message_text = issue.get("message", "")
                        suggestion = issue.get("suggestion", "")
                        
                        st.markdown(f"**Line {line}: {issue_type}** ({severity})")
                        
                        if severity == "HIGH":
                            st.error(message_text)
                        elif severity == "MEDIUM":
                            st.warning(message_text)
                        else:
                            st.info(message_text)
                        
                        if suggestion:
                            st.success(f"Suggestion: {suggestion}")
                        st.divider()
            
            # SECURITY SCANNER
            elif tool == "security_scanner":
                with st.expander("Security Report"):
                    vulns = raw.get("vulnerabilities", [])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Vulnerabilities", len(vulns))
                    with col2:
                        st.metric("Risk Level", raw.get("risk_level", "N/A"))
                    
                    for vuln in vulns:
                        severity = vuln.get("severity", "UNKNOWN")
                        line = vuln.get("line", "N/A")
                        vuln_type = vuln.get("type", "Unknown")
                        description = vuln.get("description", "No description")
                        fix = vuln.get("fix", "No fix available")
                        
                        st.markdown(f"**Line {line}: {vuln_type}** ({severity})")
                        st.error(description)
                        st.success(f"Fix: {fix}")
                        st.divider()
            
            # CODE EXECUTOR
            elif tool == "code_executor":
                with st.expander("Execution Output"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Exit Code", raw.get("exit_code", 0))
                    with col2:
                        st.metric("Time", raw.get("execution_time", "N/A"))
                    
                    output = raw.get("output", "")
                    error = raw.get("error", "")
                    
                    if output:
                        st.markdown("**Output:**")
                        st.code(output, language="text")
                    
                    if error:
                        st.markdown("**Errors:**")
                        st.code(error, language="text")
            
            # TEST GENERATOR
            elif tool == "test_generator":
                with st.expander("Test Results"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Functions", raw.get("functions_found", 0))
                    with col2:
                        st.metric("Test Cases", raw.get("test_cases_generated", 0))
                    
                    st.markdown("**Functions:**")
                    for func in raw.get("functions", []):
                        st.markdown(f"- `{func}()`")
                    
                    st.info(f"File: {raw.get('test_file', 'N/A')}")
            
            # CODE FIXER
            elif tool == "code_fixer":
                with st.expander("Fix Suggestions"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Issues", raw.get("original_issues", 0))
                    with col2:
                        st.metric("Fixes", raw.get("fixes_suggested", 0))
                    
                    fixes = raw.get("fixes", [])
                    for i, fix in enumerate(fixes[:5], 1):
                        severity = fix.get("severity", "UNKNOWN")
                        line = fix.get("line", "N/A")
                        issue_type = fix.get("issue_type", "Unknown")
                        original_msg = fix.get("original_message", "")
                        fix_text = fix.get("fix", "")
                        explanation = fix.get("explanation", "")
                        code_example = fix.get("code_example", "")
                        
                        st.markdown(f"**Fix #{i}: Line {line}** ({severity})")
                        st.markdown(f"**Issue:** {issue_type}")
                        st.info(original_msg)
                        st.success(f"**Solution:** {fix_text}")
                        st.markdown(f"**Why:** {explanation}")
                        
                        if code_example:
                            st.markdown("**Code Example:**")
                            st.code(code_example, language="python")
                        
                        st.divider()

# Handle triggered queries
if st.session_state.trigger_query:
    prompt = st.session_state.trigger_query
    st.session_state.trigger_query = None
else:
    prompt = st.chat_input("Ask about your code or documents...")

# Process query
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    file_context = None
    if mode == "Code Analysis" and st.session_state.code_file:
        file_context = st.session_state.code_file
    
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            start_time = time.time()
            result = st.session_state.agent.execute(prompt, file_context=file_context)
            elapsed = time.time() - start_time
            
            st.markdown(result["answer"])
            
            # Show detailed analysis for current response
            if result.get("tool_used") and result.get("raw_output"):
                raw = result["raw_output"]
                tool = result["tool_used"]
                
                # CODE ANALYZER
                if tool == "code_analyzer":
                    with st.expander("Detailed Analysis", expanded=False):
                        issues = raw.get("issues", [])
                        metrics = raw.get("metrics", {})
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Issues", len(issues))
                        with col2:
                            st.metric("Severity", raw.get("severity", "N/A"))
                        with col3:
                            st.metric("Comments", f"{metrics.get('comment_ratio', 0):.1f}%")
                        
                        for issue in issues:
                            severity = issue.get("severity", "UNKNOWN")
                            line = issue.get("line", "N/A")
                            issue_type = issue.get("type", "Unknown")
                            msg = issue.get("message", "")
                            suggestion = issue.get("suggestion", "")
                            
                            st.markdown(f"**Line {line}: {issue_type}** ({severity})")
                            
                            if severity == "HIGH":
                                st.error(msg)
                            elif severity == "MEDIUM":
                                st.warning(msg)
                            else:
                                st.info(msg)
                            
                            if suggestion:
                                st.success(f"Suggestion: {suggestion}")
                            st.divider()
                
                # SECURITY SCANNER
                elif tool == "security_scanner":
                    with st.expander("Security Report", expanded=False):
                        vulns = raw.get("vulnerabilities", [])
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Vulnerabilities", len(vulns))
                        with col2:
                            st.metric("Risk", raw.get("risk_level", "N/A"))
                        
                        for vuln in vulns:
                            severity = vuln.get("severity", "UNKNOWN")
                            line = vuln.get("line", "N/A")
                            vuln_type = vuln.get("type", "Unknown")
                            desc = vuln.get("description", "")
                            fix = vuln.get("fix", "")
                            
                            st.markdown(f"**Line {line}: {vuln_type}** ({severity})")
                            st.error(desc)
                            st.success(f"Fix: {fix}")
                            st.divider()
                
                # CODE EXECUTOR
                elif tool == "code_executor":
                    with st.expander("Execution Output", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Exit Code", raw.get("exit_code", 0))
                        with col2:
                            st.metric("Time", raw.get("execution_time", "N/A"))
                        
                        output = raw.get("output", "")
                        error = raw.get("error", "")
                        
                        if output:
                            st.markdown("**Output:**")
                            st.code(output, language="text")
                        
                        if error:
                            st.markdown("**Errors:**")
                            st.code(error, language="text")
                
                # TEST GENERATOR
                elif tool == "test_generator":
                    with st.expander("Test Results", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Functions", raw.get("functions_found", 0))
                        with col2:
                            st.metric("Tests", raw.get("test_cases_generated", 0))
                        
                        st.markdown("**Functions:**")
                        for func in raw.get("functions", []):
                            st.markdown(f"- `{func}()`")
                        
                        st.info(f"File: {raw.get('test_file', 'N/A')}")
                        st.code(f"pytest {raw.get('test_file', '')} -v", language="bash")
                
                # CODE FIXER
                elif tool == "code_fixer":
                    with st.expander("Fix Suggestions", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Issues", raw.get("original_issues", 0))
                        with col2:
                            st.metric("Fixes", raw.get("fixes_suggested", 0))
                        
                        fixes = raw.get("fixes", [])
                        for i, fix in enumerate(fixes, 1):
                            severity = fix.get("severity", "UNKNOWN")
                            line = fix.get("line", "N/A")
                            issue_type = fix.get("issue_type", "Unknown")
                            original_msg = fix.get("original_message", "")
                            fix_text = fix.get("fix", "")
                            explanation = fix.get("explanation", "")
                            code_example = fix.get("code_example", "")
                            
                            st.markdown(f"**Fix #{i}: Line {line}** ({severity})")
                            st.markdown(f"**Issue:** {issue_type}")
                            st.info(original_msg)
                            st.success(f"**Solution:** {fix_text}")
                            st.markdown(f"**Why:** {explanation}")
                            
                            if code_example:
                                st.markdown("**Code Example:**")
                                st.code(code_example, language="python")
                            
                            st.divider()
            
            st.caption(f"Tool: {result.get('tool_used', 'N/A')} | Time: {elapsed:.2f}s")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tool": result.get("tool_used"),
        "raw_output": result.get("raw_output")
    })

# Footer
st.divider()
st.markdown("**Powered by:** Qwen2.5-3B | FAISS | Python AST | v2.0")
